"""Stream and time rules. Each finding carries its evidence.

Time sources:
* ``header.stamp`` is the acquisition time for Image and PointCloud2
  (ROS message definitions); it is what pairing and skew use.
* ``log_time`` is when the recorder wrote the message; it is used to cut
  segments and to express intervals on one clock, never to measure skew.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import CAMERA, LIDAR
from .bagread import Segment


@dataclass
class Pair:
    lidar_idx: int
    camera_idx: int
    lidar_stamp: int
    camera_stamp: int


def _finding(rule: str, topic: str, interval, measure: str, threshold: str, action: str) -> dict:
    return {
        "rule": rule,
        "topic": topic,
        "interval_s": None if interval is None else [round(interval[0], 2), round(interval[1], 2)],
        "measure": measure,
        "threshold": threshold,
        "action": action,
    }


def _offset_ns(st) -> int:
    """Median (log_time - header.stamp) of a stream: its clock relation to the recorder."""
    return int(np.median(np.asarray(st.log_ns, dtype=np.int64) - np.asarray(st.stamp_ns, dtype=np.int64)))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        if not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def analyze(seg: Segment, recipe: dict) -> tuple[list[dict], dict, list[Pair]]:
    th = recipe["thresholds"]
    policy = recipe["policy"]
    required = recipe["required_topics"]
    findings: list[dict] = []
    info: dict = {"messages": {t: len(s.log_ns) for t, s in sorted(seg.streams.items())}}

    def rel(topic: str, stamp: int) -> float:
        st = seg.streams[topic]
        return (stamp + _offset_ns(st) - seg.start_log_ns) / 1e9

    present = {t for t, s in seg.streams.items() if s.log_ns}
    for topic in sorted(required):
        if topic not in present:
            findings.append(_finding("required_stream_absent", topic, None, "0 messages",
                                     "listed in required_topics", policy["required_stream_absent"]))

    for topic in sorted(present & set(required)):
        st = seg.streams[topic]
        stamps = np.sort(np.asarray(st.stamp_ns, dtype=np.int64))
        if len(stamps) < 2:
            continue
        d = np.diff(stamps)
        i = int(np.argmax(d))
        if d[i] / 1e9 > th["max_gap_s"]:
            findings.append(_finding("stream_gap", topic, (rel(topic, int(stamps[i])), rel(topic, int(stamps[i + 1]))),
                                     f"{d[i] / 1e9:.2f} s without message", f"max_gap_s {th['max_gap_s']}",
                                     policy["stream_gap"]))
        expected = required[topic]["expected_hz"]
        # rate over the segment, gaps already reported excluded from the span
        span = (stamps[-1] - stamps[0] - d[d / 1e9 > th["max_gap_s"]].sum()) / 1e9
        hz = (len(stamps) - 1 - int((d / 1e9 > th["max_gap_s"]).sum())) / span if span > 0 else 0.0
        if hz < th["min_rate_ratio"] * expected:
            findings.append(_finding("low_rate", topic, (rel(topic, int(stamps[0])), rel(topic, int(stamps[-1]))),
                                     f"{hz:.1f} Hz over the segment", f">= {th['min_rate_ratio']} x {expected} Hz",
                                     policy["low_rate"]))

    if CAMERA in present:
        st = seg.streams[CAMERA]
        order = np.argsort(np.asarray(st.stamp_ns))
        same = np.array([st.content[order[k]] == st.content[order[k - 1]] for k in range(1, len(order))])
        for a, b in _runs(same):
            frames = b - a + 2
            if frames >= th["frozen_frames"]:
                s0, s1 = st.stamp_ns[order[a]], st.stamp_ns[order[b + 1]]
                findings.append(_finding("frozen_content", CAMERA, (rel(CAMERA, s0), rel(CAMERA, s1)),
                                         f"{frames} identical frames in a row", f">= {th['frozen_frames']} frames",
                                         policy["frozen_content"]))

    if LIDAR in present:
        st = seg.streams[LIDAR]
        ratios = np.asarray(st.valid_ratio)
        for a, b in _runs(ratios < th["lidar_min_valid_ratio"]):
            if b - a + 1 >= th["lidar_degraded_scans"]:
                worst = float(ratios[a:b + 1].min())
                findings.append(_finding("lidar_degraded", LIDAR, (rel(LIDAR, st.stamp_ns[a]), rel(LIDAR, st.stamp_ns[b])),
                                         f"{b - a + 1} scans, valid points down to {worst:.0%}",
                                         f"valid < {th['lidar_min_valid_ratio']:.0%} on >= {th['lidar_degraded_scans']} scans",
                                         policy["lidar_degraded"]))

    pairs: list[Pair] = []
    if CAMERA in present and LIDAR in present:
        cam, lid = seg.streams[CAMERA], seg.streams[LIDAR]
        off_c, off_l = _offset_ns(cam), _offset_ns(lid)
        info["recording_offset_ms"] = {CAMERA: round(off_c / 1e6, 1), LIDAR: round(off_l / 1e6, 1)}
        if abs(off_c - off_l) > th["clock_domain_tolerance_ms"] * 1_000_000:
            info["acquisition_skew"] = "not evaluable"
            findings.append(_finding("clock_not_comparable", f"{CAMERA} vs {LIDAR}", None,
                                     f"clocks {abs(off_c - off_l) / 1e9:.0f} s apart vs recorder",
                                     f"<= {th['clock_domain_tolerance_ms']} ms", policy["clock_not_comparable"]))
        else:
            window = recipe["pairing"]["window_ms"] * 1_000_000
            cs = np.asarray(cam.stamp_ns, dtype=np.int64)
            corder = np.argsort(cs)
            css = cs[corder]
            deltas = []
            for li, ls in enumerate(lid.stamp_ns):
                j = int(np.searchsorted(css, ls))
                best = None
                for k in (j - 1, j):
                    if 0 <= k < len(css) and (best is None or abs(css[k] - ls) < abs(css[best] - ls)):
                        best = k
                if best is not None and abs(int(css[best]) - ls) <= window:
                    ci = int(corder[best])
                    pairs.append(Pair(li, ci, int(ls), int(cam.stamp_ns[ci])))
                    deltas.append(int(cam.stamp_ns[ci]) - int(ls))
            info["pairs"] = {"matched": len(pairs), "unmatched_lidar": len(lid.stamp_ns) - len(pairs)}
            if len(deltas) >= 10:
                skew = float(np.median(deltas)) / 1e6
                info["acquisition_skew_ms"] = round(skew, 1)
                if abs(skew) > th["max_acq_skew_ms"]:
                    findings.append(_finding("acquisition_skew", f"{CAMERA} vs {LIDAR}",
                                             (rel(LIDAR, lid.stamp_ns[0]), rel(LIDAR, lid.stamp_ns[-1])),
                                             f"median {skew:+.1f} ms over {len(deltas)} pairs",
                                             f"|skew| <= {th['max_acq_skew_ms']} ms", policy["acquisition_skew"]))
            arrival = (off_c - off_l) / 1e6
            if abs(arrival) > th["recording_delay_info_ms"]:
                findings.append(_finding("recording_delay", CAMERA, None,
                                         f"arrives {arrival:+.0f} ms vs lidar; stamps agree",
                                         f"info above {th['recording_delay_info_ms']} ms", policy["recording_delay"]))
    return findings, info, pairs


def decide(findings: list[dict]) -> str:
    return "excluded" if any(f["action"] == "exclude_segment" for f in findings) else "retained"
