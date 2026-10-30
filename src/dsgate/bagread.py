"""Read one rosbag2 segment (sqlite3 or MCAP storage) without a ROS install.

Damaged or unsupported input becomes an explicit ``Unreadable`` result
(quarantine). Programming errors are not caught here: they fail the worker.

Decoding is deliberately narrow and checked:
* Image: ``rgb8`` only, rows read with their ``step`` (padding dropped), every
  frame of a segment with the same height and width;
* PointCloud2: fields ``x``, ``y``, ``z``, ``intensity`` as FLOAT32 with
  ``count`` 1 at any offsets, any ``point_step`` >= 16, rows read with their
  ``row_step``, little-endian; other fields are ignored; the number of points
  may change from sweep to sweep;
* the message definition embedded in the bag must hash (RIHS01) to the ROS 2
  Humble definition of that type; a divergent definition is refused.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader, AnyReaderError
from rosbags.rosbag2 import ReaderError
from rosbags.typesys import Stores, get_typestore
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from .canon import image_hash, points_hash
from .recipe import CLOUD, IMAGE

TS = get_typestore(Stores.ROS2_HUMBLE)
FLOAT32 = 7


class Unreadable(Exception):
    def __init__(self, kind: str, detail: str, evidence: dict | None = None):
        super().__init__(f"{kind}: {detail}")
        self.kind, self.detail, self.evidence = kind, detail, evidence or {}


@dataclass
class Stream:
    msgtype: str
    log_ns: list[int] = field(default_factory=list)
    stamp_ns: list[int] = field(default_factory=list)
    content: list[str] = field(default_factory=list)  # canonical content hash per message
    payload: list[np.ndarray] = field(default_factory=list)
    valid_ratio: list[float] = field(default_factory=list)  # point clouds only


@dataclass
class Segment:
    capture_id: str
    vehicle_id: str
    storage: str
    streams: dict[str, Stream]
    schemas: dict[str, dict]
    start_log_ns: int
    end_log_ns: int


def _scrub(text: str, bag: Path) -> str:
    """Drop any directory prefix in front of the bag name (temporary paths differ per run)."""
    return re.sub(r"[^\s'\"]*/(?=" + re.escape(bag.name) + r")", "", text)


def read_metadata(bag: Path) -> dict:
    try:
        doc = YAML(typ="safe").load((Path(bag) / "metadata.yaml").read_text())
        meta = doc["rosbag2_bagfile_information"]
        if not isinstance(meta, dict):
            raise TypeError("rosbag2_bagfile_information is not a mapping")
        return meta
    except FileNotFoundError:
        raise Unreadable("metadata_unreadable", "metadata.yaml missing (a recorder stopped before closing the bag "
                                                "leaves none; `ros2 bag reindex` can rebuild it)") from None
    except (OSError, YAMLError, KeyError, TypeError) as exc:
        raise Unreadable("metadata_unreadable", f"metadata.yaml: {type(exc).__name__}: {_scrub(str(exc), Path(bag))}") from None


def capture_identity(meta: dict) -> tuple[str, str]:
    """Fallback source of the capture identity: rosbag2 ``custom_data``.

    ``custom_data`` exists in rosbag2 metadata from version 6 on (recent
    distributions); bags recorded with Humble do not carry it. The primary
    source is the identity declared at upload (see ``batch.upload_bags``).
    Never derived from file names.
    """
    custom = meta.get("custom_data") or {}
    if "capture_id" not in custom or "vehicle_id" not in custom:
        raise Unreadable("capture_identity_missing", "metadata.yaml custom_data lacks capture_id/vehicle_id")
    return str(custom["capture_id"]), str(custom["vehicle_id"])


def _stamp(msg) -> int:
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


def decode_image(msg) -> np.ndarray:
    if msg.encoding != "rgb8":
        raise Unreadable("schema_unsupported", f"image encoding {msg.encoding!r} (only rgb8)")
    h, w, step = int(msg.height), int(msg.width), int(msg.step)
    data = np.asarray(msg.data, dtype=np.uint8)
    if step < w * 3 or data.size < h * step:
        raise Unreadable("message_malformed", f"image {w}x{h} step {step} with {data.size} bytes")
    return data[: h * step].reshape(h, step)[:, : w * 3].reshape(h, w, 3).copy()


def decode_cloud(msg) -> np.ndarray:
    fields = {f.name: f for f in msg.fields}
    names = ("x", "y", "z", "intensity")
    bad = [n for n in names if n not in fields or fields[n].datatype != FLOAT32 or fields[n].count != 1]
    if bad or msg.is_bigendian or msg.point_step < 16:
        raise Unreadable("schema_unsupported", f"point cloud needs float32 x, y, z, intensity (count 1), "
                                               f"little-endian; missing or different: {bad or 'endianness/point_step'}")
    h, w, ps, rs = int(msg.height), int(msg.width), int(msg.point_step), int(msg.row_step)
    data = np.asarray(msg.data, dtype=np.uint8)
    if rs < w * ps or data.size < h * rs or any(fields[n].offset + 4 > ps for n in names):
        raise Unreadable("message_malformed", f"cloud {w}x{h} point_step {ps} row_step {rs} with {data.size} bytes")
    rows = data[: h * rs].reshape(h, rs)[:, : w * ps].reshape(-1)
    dt = np.dtype({"names": list(names), "formats": ["<f4"] * 4,
                   "offsets": [fields[n].offset for n in names], "itemsize": ps})
    pts = np.frombuffer(rows.tobytes(), dtype=dt)
    return np.stack([pts[n] for n in names], axis=1).astype("<f4")


def read_segment(bag: Path, required_topics: dict, identity: tuple[str, str] | None = None) -> Segment:
    bag = Path(bag)
    meta = read_metadata(bag)
    capture_id, vehicle_id = identity if identity else capture_identity(meta)
    storage = meta.get("storage_identifier", "")
    streams: dict[str, Stream] = {}
    schemas: dict[str, dict] = {}
    try:
        with AnyReader([bag], default_typestore=TS) as reader:
            for conn in reader.connections:
                if conn.topic not in required_topics:
                    continue
                want = required_topics[conn.topic]["msgtype"]
                if conn.msgtype != want:
                    raise Unreadable("msgtype_mismatch", f"{conn.topic} is {conn.msgtype}, recipe expects {want}")
                effective, humble = reader.typestore.hash_rihs01(conn.msgtype), TS.hash_rihs01(conn.msgtype)
                if effective != humble:
                    raise Unreadable("schema_mismatch", f"{conn.topic}: definition of {conn.msgtype} in the bag "
                                                        f"hashes to {effective[:20]}..., ROS 2 Humble to {humble[:20]}...")
                streams[conn.topic] = Stream(conn.msgtype)
                schemas[conn.topic] = {"msgtype": conn.msgtype, "rihs01": effective}
            wanted = [c for c in reader.connections if c.topic in streams]
            for conn, log_ns, raw in reader.messages(connections=wanted):
                st = streams[conn.topic]
                try:
                    msg = reader.deserialize(raw, conn.msgtype)
                except (AnyReaderError, ValueError, IndexError) as exc:
                    raise Unreadable("message_malformed", f"{conn.topic}: {exc}") from None
                st.log_ns.append(int(log_ns))
                st.stamp_ns.append(_stamp(msg))
                if conn.msgtype == IMAGE:
                    img = decode_image(msg)
                    if st.payload and st.payload[0].shape != img.shape:
                        raise Unreadable("schema_unsupported", f"{conn.topic}: image size changes within the segment")
                    h, w = img.shape[:2]
                    st.content.append(image_hash("rgb8", h, w, w * 3, img.tobytes()))
                    st.payload.append(img)
                elif conn.msgtype == CLOUD:
                    pts = decode_cloud(msg)
                    st.content.append(points_hash(pts))
                    st.payload.append(pts)
                    finite = np.isfinite(pts[:, :3]).all(axis=1)
                    st.valid_ratio.append(float(finite.mean()) if len(pts) else 0.0)
    except Unreadable:
        raise
    except (AnyReaderError, ReaderError, sqlite3.DatabaseError) as exc:
        text = _scrub(str(exc), bag)
        data_files = [bag / p for p in meta.get("relative_file_paths", [])]
        if any(p.suffix == ".mcap" for p in data_files) and ("magic" in text.lower() or "footer" in text.lower()):
            raise Unreadable("mcap_truncated", text) from None
        if "malformed" in text.lower() or "not a database" in text.lower():
            raise Unreadable("db3_malformed", text) from None
        raise Unreadable("unreadable", text) from None
    logs = [x for s in streams.values() for x in (s.log_ns[:1] + s.log_ns[-1:])]
    return Segment(capture_id, vehicle_id, storage, streams, schemas,
                   min(logs) if logs else 0, max(logs) if logs else 0)
