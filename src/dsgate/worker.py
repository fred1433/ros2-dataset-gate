"""Batch worker entrypoint: one segment per process.

Exit codes:
  0  terminal receipt written (retained, excluded or quarantined), or already
     present with exactly the expected bytes
  3  source integrity failure (bytes differ from the frozen submission)
  4  conflict: an immutable key already holds different bytes
  5  this worker's code, lockfile or image differs from what the plan expects
  other non-zero: unexpected error (a bug or an infrastructure failure)
"""

from __future__ import annotations

import io
import json
import os
import signal
import sys
import tempfile
from pathlib import Path

import numpy as np

from .bagread import Unreadable, read_segment
from .batch import compute_inputs, compute_key, labels_ref, resolve_index
from .canon import cjson, sha256
from .checks import _finding, analyze, decide
from .provenance import build_provenance
from .recipe import roles
from .store import Conflict, Store


def _npy(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def process(store: Store, plan: dict, segment_id: str, log=print) -> int:
    mine = build_provenance()
    if mine != plan["worker"]:
        log(f"{segment_id}: worker provenance {mine} differs from the plan's {plan['worker']}")
        return 5
    submission = json.loads(store.get(f"submissions/{plan['submission_id']}.json"))
    analysis = json.loads(store.get(plan["analysis_key"]))
    entry = next(s for s in submission["segments"] if s["segment_id"] == segment_id)
    inputs = compute_inputs(submission, entry, plan["analysis_digest"], plan["worker"])
    # No shortcut on an existing receipt: the result is recomputed and an
    # existing receipt only counts if it holds exactly the expected bytes.
    rkey = compute_key(inputs)
    r = roles(analysis)

    with tempfile.TemporaryDirectory() as tmp:
        bag = Path(tmp) / entry["bag"].rsplit("/", 1)[1]
        bag.mkdir()
        for f in entry["files"]:
            body = store.get(f["key"], f["version_id"])
            if len(body) != f["size"] or sha256(body) != f["sha256"]:
                log(f"{segment_id}: {f['key']} differs from the frozen submission")
                return 3
            (bag / f["key"].rsplit("/", 1)[1]).write_bytes(body)

        receipt = {"format": "dsgate.receipt/2", "segment_id": segment_id, "bag": entry["bag"],
                   "compute": inputs, "worker": mine}
        outputs: list[dict] = []
        states: list[str] = []
        try:
            if not entry["capture_id"]:
                raise Unreadable("capture_identity_missing", "no identity declared at upload and no rosbag2 "
                                                             "custom_data capture_id/vehicle_id")
            seg = read_segment(bag, analysis["required_topics"], (entry["capture_id"], entry["vehicle_id"]))
        except Unreadable as exc:
            receipt.update({
                "status": "quarantined", "capture_id": entry["capture_id"], "vehicle_id": entry["vehicle_id"],
                "storage": None,
                "start_log_ns": None, "schemas": {},
                "quarantine": {"kind": exc.kind, "error": exc.detail,
                               "original": [{"key": f["key"], "version_id": f["version_id"]} for f in entry["files"]]},
                "findings": [], "info": {}, "samples": [], "outputs": [],
            })
        else:
            findings, info, pairs = analyze(seg, analysis)
            status = decide(findings)
            samples = []
            if status == "retained":
                ref = labels_ref(submission, seg.capture_id)
                labels = {}
                if ref is not None:
                    body = store.get(ref["key"], ref["version_id"])
                    if sha256(body) != ref["sha256"]:
                        log(f"{segment_id}: label file differs from the frozen submission")
                        return 3
                    labels = json.loads(body)["range_to_cart"]
                kept = [p for p in pairs if str(p.lidar_stamp) in labels]
                info["unlabeled_pairs"] = len(pairs) - len(kept)
                if not kept:
                    findings.append(_finding("no_labeled_samples", r["lidar"], None,
                                             f"{len(pairs)} pairs, 0 with a label",
                                             ">= 1 labeled pair", analysis["policy"]["no_labeled_samples"]))
                    status = decide(findings)
                cam, lid = seg.streams[r["camera"]], seg.streams[r["lidar"]]
                if kept and status == "retained":
                    clouds = [lid.payload[p.lidar_idx] for p in kept]
                    arrays = {
                        "camera": np.stack([cam.payload[p.camera_idx] for p in kept]).astype(np.uint8),
                        "lidar_points": np.concatenate(clouds).astype("<f4"),
                        "lidar_offsets": np.concatenate([[0], np.cumsum([len(c) for c in clouds])]).astype("<i8"),
                    }
                    for kind, arr in arrays.items():
                        body = _npy(arr)
                        key = f"cas/{sha256(body)}.npy"
                        states.append(store.put_immutable(key, body))
                        outputs.append({"kind": kind, "key": key, "sha256": sha256(body), "size": len(body),
                                        "shape": list(arr.shape), "dtype": str(arr.dtype)})
                    for row, p in enumerate(kept):
                        samples.append({
                            "capture_id": seg.capture_id,
                            "ref_stamp_ns": p.lidar_stamp,
                            "cam_stamp_ns": p.camera_stamp,
                            "cam_sha256": cam.content[p.camera_idx],
                            "lidar_sha256": lid.content[p.lidar_idx],
                            "label_mm": int(labels[str(p.lidar_stamp)]),
                            "row": row,
                        })
            receipt.update({
                "status": status, "capture_id": seg.capture_id, "vehicle_id": seg.vehicle_id,
                "storage": seg.storage, "start_log_ns": seg.start_log_ns, "schemas": seg.schemas,
                "findings": findings, "info": info, "samples": samples, "outputs": outputs,
            })

    if os.environ.get("DSGATE_KILL_BEFORE_RECEIPT") == segment_id:
        log(f"{segment_id}: outputs written, killing worker before the receipt (fault injection)")
        sys.stdout.flush()
        os.kill(os.getpid(), signal.SIGKILL)

    try:
        state = store.put_immutable(rkey, cjson(receipt))
    except Conflict as exc:
        log(f"{segment_id}: CONFLICT {exc}")
        return 4
    log(f"{segment_id}: {receipt['status']}; receipt {state}; outputs created {states.count('created')}, "
        f"already present and identical {states.count('exists_same')}")
    return 0


def main() -> int:
    store = Store(os.environ["DSGATE_BUCKET"])
    plan = json.loads(store.get(os.environ["DSGATE_PLAN_KEY"]))
    segment_id = resolve_index(plan, os.environ)
    return process(store, plan, segment_id)


if __name__ == "__main__":
    sys.exit(main())
