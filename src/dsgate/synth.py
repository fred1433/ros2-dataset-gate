"""Synthetic ROS 2 drive logs with declared, known faults.

Everything here is generated: no third-party or customer data. Content is a
pure function of (capture, message index), so a segment re-cut with a temporal
overlap carries byte-identical payloads for the overlapping instants, exactly
like a real re-upload of the same capture.

The analyzer never imports the ``expect`` fields below; they exist only so the
tests can compare what the analyzer found against what was injected.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rosbags.rosbag2 import StoragePlugin, Writer
from rosbags.typesys import Stores, get_typestore

from . import CAMERA, LIDAR, ODOM
from .canon import sha256

TS = get_typestore(Stores.ROS2_HUMBLE)
T = TS.types

HEIGHT, WIDTH = 30, 40
N_POINTS = 240
LIDAR_PERIOD_NS = 100_000_000  # 10 Hz
CAMERA_PERIOD_NS = 50_000_000  # 20 Hz: each lidar sweep has a camera trigger
ODOM_PERIOD_NS = 20_000_000  # 50 Hz
# recorder arrival latency (log_time - header.stamp), per topic
LATENCY_NS = {CAMERA: 6_000_000, LIDAR: 12_000_000, ODOM: 1_000_000}


@dataclass(frozen=True)
class Fault:
    kind: str
    topic: str = ""
    at_s: float = 0.0
    dur_s: float = 0.0
    value: float = 0.0


@dataclass(frozen=True)
class SegmentSpec:
    start_s: int
    end_s: int
    storage: str  # "sqlite3" | "mcap"
    faults: tuple[Fault, ...] = ()
    damage: str | None = None  # "truncate"
    expect: tuple[str, ...] = ()  # oracle only: expected rules / status


@dataclass(frozen=True)
class CaptureSpec:
    vehicle_id: str
    start_utc: str  # "2026-10-12T08:14:00Z"
    segments: tuple[SegmentSpec, ...] = field(default_factory=tuple)

    @property
    def capture_id(self) -> str:
        return f"{self.vehicle_id}/{self.start_utc.replace('-', '').replace(':', '')}"

    @property
    def t0_ns(self) -> int:
        dt = datetime.strptime(self.start_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return int(dt.timestamp()) * 1_000_000_000

    @property
    def seed(self) -> int:
        return int(sha256(self.capture_id.encode())[:8], 16)


def bag_name(capture: CaptureSpec, seg: SegmentSpec) -> str:
    t = datetime.fromtimestamp(capture.t0_ns // 1_000_000_000 + seg.start_s, tz=timezone.utc)
    return t.strftime("rosbag2_%Y_%m_%d-%H_%M_%S")


S, M = "sqlite3", "mcap"
F = Fault

BATCH1 = (
    CaptureSpec("tractor-01", "2026-10-12T08:14:00Z", (
        SegmentSpec(0, 10, S, (F("flat", CAMERA, at_s=4.0),), expect=("retained",)),
        SegmentSpec(10, 20, M, expect=("retained",)),
        SegmentSpec(20, 30, S, expect=("retained",)),
    )),
    CaptureSpec("tractor-02", "2026-10-12T09:30:00Z", (
        SegmentSpec(0, 10, M, expect=("retained",)),
        # 0.7 s lidar gap: below 1.0 s (recipe v1), above 0.5 s (recipe v2)
        SegmentSpec(10, 20, M, (F("gap", LIDAR, at_s=14.05, dur_s=0.6),),
                    expect=("retained@v1", "excluded@v2", "stream_gap@v2")),
    )),
    CaptureSpec("tractor-03", "2026-10-12T11:15:00Z", (
        SegmentSpec(0, 10, S, (F("flat", CAMERA, at_s=6.0),), expect=("retained",)),
        SegmentSpec(10, 20, M, expect=("retained",)),
    )),
)

BATCH2 = (
    # re-upload of tractor-01: re-cut with overlap over two existing segments
    CaptureSpec("tractor-01", "2026-10-12T08:14:00Z", (
        SegmentSpec(14, 24, M, expect=("deduplicated",)),
    )),
    CaptureSpec("tractor-02", "2026-10-14T06:45:00Z", (
        SegmentSpec(0, 10, M, (F("absent", LIDAR),), expect=("excluded", "required_stream_absent")),
        SegmentSpec(10, 20, S, (F("nan", LIDAR, at_s=13.0, dur_s=1.5, value=0.7),),
                    expect=("excluded", "lidar_degraded")),
        SegmentSpec(20, 30, M, expect=("retained",)),
        SegmentSpec(30, 40, M, (F("gap", LIDAR, at_s=33.05, dur_s=2.4),),
                    expect=("excluded", "stream_gap")),
        SegmentSpec(40, 50, S, (F("rate", CAMERA, value=12),), expect=("excluded", "low_rate")),
    )),
    CaptureSpec("tractor-03", "2026-10-14T13:00:00Z", (
        SegmentSpec(0, 10, M, (F("frozen", CAMERA, at_s=5.0, value=8),),
                    expect=("excluded", "frozen_content")),
        # acquisition shifted by 18 ms, arrival at the recorder unchanged
        SegmentSpec(10, 20, M, (F("acq_skew", CAMERA, value=18),),
                    expect=("excluded", "acquisition_skew")),
        # arrival delayed by 60 ms, acquisition unchanged: must not read as skew
        SegmentSpec(20, 30, S, (F("rec_delay", CAMERA, value=60),),
                    expect=("retained", "recording_delay")),
        SegmentSpec(30, 40, M, expect=("retained",)),
    )),
    CaptureSpec("tractor-04", "2026-10-15T09:10:00Z", (
        SegmentSpec(0, 10, M, damage="truncate", expect=("quarantined", "mcap_truncated")),
        SegmentSpec(10, 20, S, damage="truncate", expect=("quarantined", "db3_malformed")),
        # camera stamped on a boot clock: not comparable with the lidar clock
        SegmentSpec(20, 30, M, (F("boot_clock", CAMERA, value=3600),),
                    expect=("excluded", "clock_not_comparable")),
        SegmentSpec(30, 40, S, expect=("retained",)),
        SegmentSpec(40, 50, M, expect=("retained",)),
    )),
)


# ---------------------------------------------------------------- scene model

def _rng(capture: CaptureSpec, kind: int, index: int) -> np.random.Generator:
    return np.random.default_rng([capture.seed, kind, index])


def _jitter_ns(capture: CaptureSpec, kind: int, index: int) -> int:
    return int(_rng(capture, kind, index).integers(-1000, 1001)) * 1000  # +-1 ms


def range_mm(t_ms: int) -> int:
    """Distance to a parked cart ahead: approach from 40 m to 8 m, repeat."""
    return 8000 + (32000 * (18000 - t_ms % 18000)) // 18000


def lateral_mm(t_ms: int) -> int:
    tri = abs((t_ms % 23000) - 11500)
    return (tri * 3000) // 11500 - 1500


def lidar_stamp(capture: CaptureSpec, n: int) -> int:
    return capture.t0_ns + n * LIDAR_PERIOD_NS + _jitter_ns(capture, 1, n)


def camera_stamp(capture: CaptureSpec, m: int) -> int:
    return capture.t0_ns + m * CAMERA_PERIOD_NS + _jitter_ns(capture, 2, m)


def render_image(capture: CaptureSpec, m: int, t_ms: int, noise: bool = True) -> np.ndarray:
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for row in range(12):
        img[row, :] = (110 + 4 * row, 140 + 4 * row, 190)
    for row in range(12, HEIGHT):
        img[row, :] = (95 - row, 90 - row, 85 - row)
    r, lat = range_mm(t_ms), lateral_mm(t_ms)
    w = min(max(48000 // r, 2), 24)
    h = min(max(36000 // r, 2), 18)
    col = min(max(20 + (lat * 25) // r, 0), WIDTH - 1)
    bottom = 12 + min(max(150000 // r, 0), 17)
    top = max(bottom - h, 0)
    c0, c1 = max(col - w // 2, 0), min(col + (w + 1) // 2, WIDTH)
    img[top:bottom, c0:c1] = (40, 42, 48)
    img[top, c0:c1] = (230, 190, 40)
    if noise:
        img = img + _rng(capture, 3, m).integers(0, 7, size=img.shape, dtype=np.uint8)
    return img


_AZ = np.arange(N_POINTS, dtype=np.float64) * (2 * np.pi / N_POINTS)
_COS, _SIN = np.cos(_AZ), np.sin(_AZ)


def render_points(t_ms: int) -> np.ndarray:
    r, lat = range_mm(t_ms) / 1000.0, lateral_mm(t_ms) / 1000.0
    rng_m = np.full(N_POINTS, 12.0)
    z = np.full(N_POINTS, -1.8)
    inten = np.full(N_POINTS, 20.0)
    center, half = np.arctan2(lat, r), np.arctan2(1.0, r)
    d = np.abs(np.angle(np.exp(1j * (_AZ - center))))
    obstacle = d <= half
    rng_m[obstacle], z[obstacle], inten[obstacle] = r, -0.6, 80.0
    pts = np.stack([rng_m * _COS, rng_m * _SIN, z, inten], axis=1)
    return (np.round(pts * 1000.0) / 1000.0).astype(np.float32)


# ------------------------------------------------------------- message build

def _header(stamp: int, frame: str):
    return T["std_msgs/msg/Header"](
        stamp=T["builtin_interfaces/msg/Time"](sec=stamp // 1_000_000_000, nanosec=stamp % 1_000_000_000),
        frame_id=frame,
    )


def _image_msg(stamp: int, img: np.ndarray) -> bytes:
    msg = T["sensor_msgs/msg/Image"](
        header=_header(stamp, "camera_front"), height=HEIGHT, width=WIDTH, encoding="rgb8",
        is_bigendian=0, step=WIDTH * 3, data=img.reshape(-1).copy(),
    )
    return TS.serialize_cdr(msg, "sensor_msgs/msg/Image")


_PF = T["sensor_msgs/msg/PointField"]
_FIELDS = [_PF(name=n, offset=4 * i, datatype=7, count=1) for i, n in enumerate(("x", "y", "z", "intensity"))]


def _cloud_msg(stamp: int, pts: np.ndarray) -> bytes:
    msg = T["sensor_msgs/msg/PointCloud2"](
        header=_header(stamp, "lidar_top"), height=1, width=pts.shape[0], fields=_FIELDS,
        is_bigendian=False, point_step=16, row_step=16 * pts.shape[0],
        data=np.ascontiguousarray(pts, dtype="<f4").view(np.uint8).reshape(-1).copy(),
        is_dense=False,
    )
    return TS.serialize_cdr(msg, "sensor_msgs/msg/PointCloud2")


def _odom_msg(stamp: int, t_ms: int) -> bytes:
    g = T
    pose = g["geometry_msgs/msg/PoseWithCovariance"](
        pose=g["geometry_msgs/msg/Pose"](
            position=g["geometry_msgs/msg/Point"](x=t_ms * 0.0017, y=0.0, z=0.0),
            orientation=g["geometry_msgs/msg/Quaternion"](x=0.0, y=0.0, z=0.0, w=1.0),
        ),
        covariance=np.zeros(36, dtype=np.float64),
    )
    twist = g["geometry_msgs/msg/TwistWithCovariance"](
        twist=g["geometry_msgs/msg/Twist"](
            linear=g["geometry_msgs/msg/Vector3"](x=1.7, y=0.0, z=0.0),
            angular=g["geometry_msgs/msg/Vector3"](x=0.0, y=0.0, z=0.0),
        ),
        covariance=np.zeros(36, dtype=np.float64),
    )
    msg = g["nav_msgs/msg/Odometry"](header=_header(stamp, "odom"), child_frame_id="base_link", pose=pose, twist=twist)
    return TS.serialize_cdr(msg, "nav_msgs/msg/Odometry")


def _faults(seg: SegmentSpec, kind: str, topic: str) -> list[Fault]:
    return [f for f in seg.faults if f.kind == kind and f.topic in ("", topic)]


def segment_messages(capture: CaptureSpec, seg: SegmentSpec) -> list[tuple[int, str, bytes]]:
    """Messages of one segment, cut on the recorder clock (log_time)."""
    t0 = capture.t0_ns
    lo, hi = t0 + seg.start_s * 1_000_000_000, t0 + seg.end_s * 1_000_000_000
    out: list[tuple[int, str, bytes]] = []
    absent = {f.topic for f in seg.faults if f.kind == "absent"}

    def rel_s(stamp: int) -> float:
        return (stamp - t0) / 1e9

    # lidar
    if LIDAR not in absent:
        n0, n1 = seg.start_s * 10 - 2, seg.end_s * 10 + 2
        for n in range(max(n0, 0), n1):
            stamp = lidar_stamp(capture, n)
            log = stamp + LATENCY_NS[LIDAR]
            if not lo <= log < hi:
                continue
            if any(f.at_s <= rel_s(stamp) < f.at_s + f.dur_s for f in _faults(seg, "gap", LIDAR)):
                continue
            pts = render_points((stamp - t0) // 1_000_000)
            for f in _faults(seg, "nan", LIDAR):
                if f.at_s <= rel_s(stamp) < f.at_s + f.dur_s:
                    k = int(round(f.value * N_POINTS))
                    idx = _rng(capture, 4, n).choice(N_POINTS, size=k, replace=False)
                    pts[idx, :3] = np.nan
            out.append((log, LIDAR, _cloud_msg(stamp, pts)))

    # camera
    if CAMERA not in absent:
        m0, m1 = seg.start_s * 20 - 4, seg.end_s * 20 + 4
        frozen = _faults(seg, "frozen", CAMERA)
        flat = _faults(seg, "flat", CAMERA)
        rate = _faults(seg, "rate", CAMERA)
        for m in range(max(m0, 0), m1):
            if rate:
                hz = int(rate[0].value)
                if (m * hz) // 20 == ((m - 1) * hz) // 20:
                    continue
            stamp = camera_stamp(capture, m)
            log = stamp + LATENCY_NS[CAMERA]
            for f in _faults(seg, "rec_delay", CAMERA):
                log += int(f.value) * 1_000_000
            if not lo <= log < hi:
                continue
            t_ms = (stamp - t0) // 1_000_000
            img = render_image(capture, m, t_ms)
            for f in frozen:
                first = int(np.ceil(f.at_s * 20))
                if first <= m < first + int(f.value):
                    img = render_image(capture, first, (camera_stamp(capture, first) - t0) // 1_000_000)
            for f in flat:
                if m == int(round(f.at_s * 20)):
                    img = np.full((HEIGHT, WIDTH, 3), 128, dtype=np.uint8)
            hstamp = stamp
            for f in _faults(seg, "acq_skew", CAMERA):
                hstamp += int(f.value) * 1_000_000
            for f in _faults(seg, "boot_clock", CAMERA):
                hstamp = stamp - t0 + int(f.value) * 1_000_000_000
            out.append((log, CAMERA, _image_msg(hstamp, img)))

    # odometry
    if ODOM not in absent:
        for k in range(max(seg.start_s * 50 - 2, 0), seg.end_s * 50 + 2):
            stamp = t0 + k * ODOM_PERIOD_NS
            log = stamp + LATENCY_NS[ODOM]
            if lo <= log < hi:
                out.append((log, ODOM, _odom_msg(stamp, (stamp - t0) // 1_000_000)))

    out.sort(key=lambda x: (x[0], x[1]))
    return out


TYPES = {CAMERA: "sensor_msgs/msg/Image", LIDAR: "sensor_msgs/msg/PointCloud2", ODOM: "nav_msgs/msg/Odometry"}


def write_segment(root: Path, capture: CaptureSpec, seg: SegmentSpec, storage: str | None = None,
                  name: str | None = None) -> Path:
    storage = storage or seg.storage
    path = root / capture.vehicle_id / (name or bag_name(capture, seg))
    path.parent.mkdir(parents=True, exist_ok=True)
    plugin = StoragePlugin.MCAP if storage == "mcap" else StoragePlugin.SQLITE3
    msgs = segment_messages(capture, seg)
    with Writer(path, version=9, storage_plugin=plugin) as w:
        w.set_custom_data("capture_id", capture.capture_id)
        w.set_custom_data("vehicle_id", capture.vehicle_id)
        conns = {}
        for topic in (CAMERA, LIDAR, ODOM):
            if any(t == topic for _, t, _ in msgs):
                conns[topic] = w.add_connection(topic, TYPES[topic], typestore=TS)
        for log, topic, data in msgs:
            w.write(conns[topic], log, data)
    if seg.damage == "truncate":
        data_file = next(p for p in path.iterdir() if p.suffix in (".mcap", ".db3"))
        size = data_file.stat().st_size
        with open(data_file, "r+b") as fh:
            fh.truncate(int(size * 0.6))
    return path


def write_labels(root: Path, capture: CaptureSpec, duration_s: int = 60) -> Path:
    """Per-capture annotations: distance to the parked cart at each lidar sweep, in mm."""
    labels = {}
    for n in range(duration_s * 10):
        stamp = lidar_stamp(capture, n)
        labels[str(stamp)] = range_mm((stamp - capture.t0_ns) // 1_000_000)
    path = root / "labels" / f"{capture.capture_id.replace('/', '_')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"capture_id": capture.capture_id, "unit": "mm", "range_to_cart": labels},
                               sort_keys=True, indent=0) + "\n")
    return path


def write_calibration(root: Path, vehicle_id: str) -> Path:
    k = int(sha256(vehicle_id.encode())[:4], 16) % 7
    cal = {
        "vehicle_id": vehicle_id,
        "camera_front": {"fx_px": 52.0 + k, "fy_px": 52.0 + k, "cx_px": 20.0, "cy_px": 15.0},
        "lidar_top_to_camera_front": {"xyz_m": [0.12, 0.0, -0.35], "rpy_rad": [0.0, 0.0, 0.0]},
    }
    path = root / "calibration" / f"{vehicle_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cal, sort_keys=True, indent=1) + "\n")
    return path


def generate(out: Path) -> dict[str, list[Path]]:
    """Write both upload batches and their side inputs. Returns bag paths per batch."""
    out = Path(out)
    result: dict[str, list[Path]] = {"batch1": [], "batch2": []}
    for batch, captures in (("batch1", BATCH1), ("batch2", BATCH2)):
        for cap in captures:
            for seg in cap.segments:
                result[batch].append(write_segment(out / batch, cap, seg))
    for cap in {c.capture_id: c for c in BATCH1 + BATCH2}.values():
        write_labels(out / "side", cap)
    for vehicle in sorted({c.vehicle_id for c in BATCH1 + BATCH2}):
        write_calibration(out / "side", vehicle)
    return result


def spec_for(bag_dir: Path) -> tuple[CaptureSpec, SegmentSpec]:
    """Oracle lookup by bag name (tests only)."""
    for cap in BATCH1 + BATCH2:
        for seg in cap.segments:
            if bag_name(cap, seg) == bag_dir.name and cap.vehicle_id == bag_dir.parent.name:
                return cap, seg
    raise KeyError(bag_dir)


if __name__ == "__main__":  # pragma: no cover
    import sys

    generate(Path(sys.argv[1] if len(sys.argv) > 1 else os.path.join("build", "corpus")))
