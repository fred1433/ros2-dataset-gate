"""Stream and time rules. Each finding carries its evidence.

Time sources:
* ``header.stamp``: acquisition time of Image and PointCloud2 (ROS message
  definitions); pairing and residuals use it.
* ``log_time``: the recording (receive) timestamp the recorder put on the
  message. It bounds the observation window of the segment and places
  intervals on one clock; it is never used to measure sensor timing.

The observation window of a segment is [first, last] ``log_time`` over its
required streams. Each required stream must cover it: a silence at the start
or the end counts as a gap, like a silence in the middle, and every gap above
the threshold is its own finding.

The pairing statistic is a *nearest-pair timestamp residual*: for each point
cloud, the signed difference to the nearest camera ``header.stamp``. It is
measured modulo the camera period: an offset of a whole camera period (50 ms
at 20 Hz) pairs every sweep with the neighbouring frame and is invisible from
timestamps alone. Detecting it needs trigger or synchronisation metadata.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bagread import Segment
from .recipe import roles


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
    """Median (log_time - header.stamp) of a stream: the observed offset to the recorder clock."""
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


def analyze(seg: Segment, analysis: dict) -> tuple[list[dict], dict, list[Pair]]:
    th = analysis["thresholds"]
    policy = analysis["policy"]
    required = analysis["required_topics"]
    r = roles(analysis)
    cam_t, lid_t = r["camera"], r["lidar"]
    findings: list[dict] = []
    info: dict = {"messages": {t: len(s.log_ns) for t, s in sorted(seg.streams.items())},
                  "window_s": round((seg.end_log_ns - seg.start_log_ns) / 1e9, 2)}
    w_end = (seg.end_log_ns - seg.start_log_ns) / 1e9
    max_gap = th["max_gap_s"]

    def rel(topic: str, stamp: int) -> float:
        return (stamp + _offset_ns(seg.streams[topic]) - seg.start_log_ns) / 1e9

    present = {t for t, s in seg.streams.items() if s.log_ns}
    for topic in sorted(required):
        if topic not in present:
            findings.append(_finding("required_stream_absent", topic, None, "0 messages",
                                     "listed in required_topics", policy["required_stream_absent"]))

    usable = set()
    for topic in sorted(present & set(required)):
        st = seg.streams[topic]
        if len(st.log_ns) < th["min_messages"]:
            findings.append(_finding("stream_insufficient", topic, None, f"{len(st.log_ns)} message(s)",
                                     f">= {th['min_messages']} messages", policy["stream_insufficient"]))
            continue
        usable.add(topic)
        stamps = np.sort(np.asarray(st.stamp_ns, dtype=np.int64))
        points = [0.0] + [rel(topic, int(s)) for s in stamps] + [w_end]
        edges = np.diff(points)
        for i in np.nonzero(edges > max_gap)[0]:
            where = "start of window" if i == 0 else "end of window" if i == len(edges) - 1 else "between messages"
            findings.append(_finding("stream_gap", topic, (points[i], points[i + 1]),
                                     f"{edges[i]:.2f} s without message ({where})", f"max_gap_s {max_gap}",
                                     policy["stream_gap"]))
        d = np.diff(stamps) / 1e9
        active = (stamps[-1] - stamps[0]) / 1e9 - d[d > max_gap].sum()
        hz = (len(stamps) - 1 - int((d > max_gap).sum())) / active if active > 0 else 0.0
        expected = required[topic]["expected_hz"]
        if hz < th["min_rate_ratio"] * expected:
            findings.append(_finding("low_rate", topic, (rel(topic, int(stamps[0])), rel(topic, int(stamps[-1]))),
                                     f"{hz:.1f} Hz while active", f">= {th['min_rate_ratio']} x {expected} Hz",
                                     policy["low_rate"]))

    if cam_t in usable:
        st = seg.streams[cam_t]
        order = np.argsort(np.asarray(st.stamp_ns))
        same = np.array([st.content[order[k]] == st.content[order[k - 1]] for k in range(1, len(order))])
        for a, b in _runs(same):
            frames = b - a + 2
            if frames >= th["frozen_frames"]:
                s0, s1 = st.stamp_ns[order[a]], st.stamp_ns[order[b + 1]]
                findings.append(_finding("frozen_content", cam_t, (rel(cam_t, s0), rel(cam_t, s1)),
                                         f"{frames} identical frames in a row", f">= {th['frozen_frames']} frames",
                                         policy["frozen_content"]))

    if lid_t in usable:
        st = seg.streams[lid_t]
        ratios = np.asarray(st.valid_ratio)
        for a, b in _runs(ratios < th["lidar_min_valid_ratio"]):
            if b - a + 1 >= th["lidar_degraded_scans"]:
                worst = float(ratios[a:b + 1].min())
                findings.append(_finding("lidar_degraded", lid_t, (rel(lid_t, st.stamp_ns[a]), rel(lid_t, st.stamp_ns[b])),
                                         f"{b - a + 1} sweeps, valid points down to {worst:.0%}",
                                         f"valid < {th['lidar_min_valid_ratio']:.0%} on >= {th['lidar_degraded_scans']} sweeps",
                                         policy["lidar_degraded"]))

    pairs: list[Pair] = []
    if cam_t in usable and lid_t in usable:
        cam, lid = seg.streams[cam_t], seg.streams[lid_t]
        off_c, off_l = _offset_ns(cam), _offset_ns(lid)
        info["observed_recording_offset_ms"] = {cam_t: round(off_c / 1e6, 1), lid_t: round(off_l / 1e6, 1)}
        if abs(off_c - off_l) > th["clock_domain_tolerance_ms"] * 1_000_000:
            info["pair_residual"] = "not evaluable"
            findings.append(_finding("clock_not_comparable", f"{cam_t} vs {lid_t}", None,
                                     f"clocks {abs(off_c - off_l) / 1e9:.0f} s apart vs recorder",
                                     f"<= {th['clock_domain_tolerance_ms']} ms", policy["clock_not_comparable"]))
        else:
            window = analysis["pairing"]["window_ms"] * 1_000_000
            cs = np.asarray(cam.stamp_ns, dtype=np.int64)
            corder = np.argsort(cs)
            css = cs[corder]
            residuals = []
            for li, ls in enumerate(lid.stamp_ns):
                j = int(np.searchsorted(css, ls))
                best = None
                for k in (j - 1, j):
                    if 0 <= k < len(css) and (best is None or abs(css[k] - ls) < abs(css[best] - ls)):
                        best = k
                if best is not None and abs(int(css[best]) - ls) <= window:
                    ci = int(corder[best])
                    pairs.append(Pair(li, ci, int(ls), int(cam.stamp_ns[ci])))
                    residuals.append(int(cam.stamp_ns[ci]) - int(ls))
            ratio = len(pairs) / len(lid.stamp_ns)
            info["pairs"] = {"matched": len(pairs), "sweeps": len(lid.stamp_ns), "ratio": round(ratio, 3)}
            if ratio < th["min_pair_ratio"]:
                findings.append(_finding("low_pairing", f"{cam_t} vs {lid_t}", None,
                                         f"{len(pairs)}/{len(lid.stamp_ns)} sweeps with a frame within "
                                         f"{analysis['pairing']['window_ms']} ms",
                                         f"ratio >= {th['min_pair_ratio']}", policy["low_pairing"]))
            if residuals:
                res = np.asarray(residuals) / 1e6
                med = float(np.median(res))
                info["pair_residual_ms"] = {"median": round(med, 1),
                                            "abs_p50": round(float(np.percentile(np.abs(res), 50)), 1),
                                            "abs_p95": round(float(np.percentile(np.abs(res), 95)), 1)}
                if abs(med) > th["max_pair_residual_ms"]:
                    findings.append(_finding("pair_residual", f"{cam_t} vs {lid_t}",
                                             (rel(lid_t, lid.stamp_ns[0]), rel(lid_t, lid.stamp_ns[-1])),
                                             f"median {med:+.1f} ms over {len(residuals)} pairs",
                                             f"|median| <= {th['max_pair_residual_ms']} ms", policy["pair_residual"]))
            observed = (off_c - off_l) / 1e6
            if abs(observed) > th["recording_offset_info_ms"]:
                findings.append(_finding("recording_offset", f"{cam_t} vs {lid_t}", None,
                                         f"log_time - header.stamp, camera vs lidar {observed:+.0f} ms",
                                         f"info above {th['recording_offset_info_ms']} ms", policy["recording_offset"]))
    return findings, info, pairs


def decide(findings: list[dict]) -> str:
    return "excluded" if any(f["action"] == "exclude_segment" for f in findings) else "retained"
