"""Read one rosbag2 segment (sqlite3 or MCAP storage) without a ROS install.

Storage-level damage is turned into an explicit ``Unreadable`` result.
Programming errors are not caught here: they must fail the worker.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader, AnyReaderError
from rosbags.rosbag2 import ReaderError
from rosbags.typesys import Stores, get_typestore
from ruamel.yaml import YAML

from . import CAMERA, LIDAR, ODOM
from .canon import image_hash, points_hash

TS = get_typestore(Stores.ROS2_HUMBLE)
SUPPORTED = {
    CAMERA: "sensor_msgs/msg/Image",
    LIDAR: "sensor_msgs/msg/PointCloud2",
    ODOM: "nav_msgs/msg/Odometry",
}
XYZI = [("x", 0, 7), ("y", 4, 7), ("z", 8, 7), ("intensity", 12, 7)]


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
    valid_ratio: list[float] = field(default_factory=list)  # lidar only


@dataclass
class Segment:
    capture_id: str
    vehicle_id: str
    storage: str
    streams: dict[str, Stream]
    schemas: dict[str, dict]
    start_log_ns: int


def _metadata(bag: Path) -> dict:
    meta = YAML(typ="safe").load((bag / "metadata.yaml").read_text())["rosbag2_bagfile_information"]
    return meta


def capture_identity(meta: dict) -> tuple[str, str]:
    """Adaptation point: where the capture identity lives.

    Here: rosbag2 ``custom_data`` keys ``capture_id`` and ``vehicle_id``.
    Never derived from file names.
    """
    custom = meta.get("custom_data") or {}
    if "capture_id" not in custom or "vehicle_id" not in custom:
        raise Unreadable("capture_identity_missing", "metadata.yaml custom_data lacks capture_id/vehicle_id")
    return str(custom["capture_id"]), str(custom["vehicle_id"])


def _stamp(msg) -> int:
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


def _mcap_complete_records(path: Path) -> int:
    from mcap.records import Message
    from mcap.stream_reader import StreamReader

    n = 0
    try:
        with open(path, "rb") as fh:
            for rec in StreamReader(fh).records:
                if isinstance(rec, Message):
                    n += 1
    except Exception:  # the cut itself ends the scan
        pass
    return n


def read_segment(bag: Path) -> Segment:
    bag = Path(bag)
    meta = _metadata(bag)
    capture_id, vehicle_id = capture_identity(meta)
    storage = meta.get("storage_identifier", "")
    data_files = [bag / p for p in meta.get("relative_file_paths", [])]
    streams: dict[str, Stream] = {}
    schemas: dict[str, dict] = {}
    try:
        with AnyReader([bag], default_typestore=TS) as reader:
            for conn in reader.connections:
                if conn.topic in SUPPORTED:
                    if conn.msgtype != SUPPORTED[conn.topic]:
                        raise Unreadable("schema_unsupported", f"{conn.topic} is {conn.msgtype}, expected {SUPPORTED[conn.topic]}")
                    streams[conn.topic] = Stream(conn.msgtype)
                    schemas[conn.topic] = {"msgtype": conn.msgtype, "rihs01": TS.hash_rihs01(conn.msgtype)}
            wanted = [c for c in reader.connections if c.topic in streams]
            for conn, log_ns, raw in reader.messages(connections=wanted):
                st = streams[conn.topic]
                msg = reader.deserialize(raw, conn.msgtype)
                st.log_ns.append(int(log_ns))
                st.stamp_ns.append(_stamp(msg))
                if conn.topic == CAMERA:
                    if msg.encoding != "rgb8":
                        raise Unreadable("schema_unsupported", f"image encoding {msg.encoding!r}")
                    data = np.asarray(msg.data, dtype=np.uint8)
                    st.content.append(image_hash(msg.encoding, msg.height, msg.width, msg.step, data.tobytes()))
                    st.payload.append(data.reshape(msg.height, msg.width, 3))
                elif conn.topic == LIDAR:
                    fields = [(f.name, f.offset, f.datatype) for f in msg.fields]
                    if fields != XYZI or msg.point_step != 16 or msg.is_bigendian:
                        raise Unreadable("schema_unsupported", f"point layout {fields} step {msg.point_step}")
                    pts = np.frombuffer(np.asarray(msg.data, dtype=np.uint8).tobytes(), dtype="<f4").reshape(-1, 4)
                    st.content.append(points_hash(pts))
                    st.payload.append(pts)
                    finite = np.isfinite(pts[:, :3]).all(axis=1)
                    st.valid_ratio.append(float(finite.mean()) if len(pts) else 0.0)
    except Unreadable:
        raise
    except (AnyReaderError, ReaderError, sqlite3.DatabaseError) as exc:
        text = str(exc).replace(str(bag.parent) + "/", "")
        mcap_files = [p for p in data_files if p.suffix == ".mcap"]
        if mcap_files and ("magic" in text.lower() or "footer" in text.lower() or "end" in text.lower()):
            n = sum(_mcap_complete_records(p) for p in mcap_files)
            raise Unreadable("mcap_truncated", text, {"complete_records_before_cut": n}) from None
        if "malformed" in text.lower() or "not a database" in text.lower():
            raise Unreadable("db3_malformed", text) from None
        raise Unreadable("unreadable", text) from None
    first = [s.log_ns[0] for s in streams.values() if s.log_ns]
    return Segment(capture_id, vehicle_id, storage, streams, schemas, min(first) if first else 0)
