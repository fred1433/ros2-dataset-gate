"""Batch worker entrypoint: one segment per process.

Exit codes:
  0  terminal receipt written (retained, excluded or quarantined), or already
     present with exactly the expected bytes
  3  source integrity failure (bytes differ from the frozen submission)
  4  conflict: an immutable key already holds different bytes
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
from .batch import receipt_key, resolve_index
from .canon import cjson, sha256
from .checks import analyze, decide
from .provenance import build_provenance
from .store import Conflict, Store


def _npy(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def process(store: Store, plan: dict, segment_id: str, log=print) -> int:
    submission = json.loads(store.get(f"submissions/{plan['submission_id']}.json"))
    recipe = json.loads(store.get(plan["recipe_key"]))
    # No shortcut on an existing receipt: the result is recomputed and an
    # existing receipt only counts if it holds exactly the expected bytes.
    rkey = receipt_key(plan["analysis_digest"], segment_id)
    entry = next(s for s in submission["segments"] if s["segment_id"] == segment_id)

    with tempfile.TemporaryDirectory() as tmp:
        bag = Path(tmp) / entry["bag"].rsplit("/", 1)[1]
        bag.mkdir()
        for f in entry["files"]:
            body = store.get(f["key"], f["version_id"])
            if len(body) != f["size"] or sha256(body) != f["sha256"]:
                log(f"{segment_id}: {f['key']} differs from the frozen submission")
                return 3
            (bag / f["key"].rsplit("/", 1)[1]).write_bytes(body)

        receipt = {
            "format": "dsgate.receipt/1",
            "segment_id": segment_id,
            "bag": entry["bag"],
            "analysis_digest": plan["analysis_digest"],
            "source": {"source_digest": entry["source_digest"], "files_verified": len(entry["files"])},
            "worker": build_provenance(),
        }
        outputs: list[dict] = []
        states: list[str] = []
        try:
            seg = read_segment(bag)
        except Unreadable as exc:
            receipt.update({
                "status": "quarantined",
                "quarantine": {"kind": exc.kind, "error": exc.detail, "evidence": exc.evidence,
                               "original": [{"key": f["key"], "version_id": f["version_id"]} for f in entry["files"]]},
                "findings": [], "info": {}, "samples": [], "outputs": [],
            })
        else:
            findings, info, pairs = analyze(seg, recipe)
            status = decide(findings)
            samples = []
            if status == "retained":
                label_ref = submission["side_inputs"]["labels"][seg.capture_id.replace("/", "_")]
                label_body = store.get(label_ref["key"], label_ref["version_id"])
                if sha256(label_body) != label_ref["sha256"]:
                    log(f"{segment_id}: label file differs from the frozen submission")
                    return 3
                labels = json.loads(label_body)["range_to_cart"]
                kept = [p for p in pairs if str(p.lidar_stamp) in labels]
                info["unlabeled_pairs"] = len(pairs) - len(kept)
                cam = seg.streams["/camera/front/image_raw"]
                lid = seg.streams["/lidar/points"]
                if kept:
                    cam_arr = np.stack([cam.payload[p.camera_idx] for p in kept]).astype(np.uint8)
                    lid_arr = np.stack([lid.payload[p.lidar_idx] for p in kept]).astype("<f4")
                    for kind, arr in (("camera", cam_arr), ("lidar", lid_arr)):
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
        if receipt["status"] == "quarantined":
            meta_capture = _capture_from_metadata(bag)
            receipt.update({"capture_id": meta_capture[0], "vehicle_id": meta_capture[1], "storage": meta_capture[2]})

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


def _capture_from_metadata(bag: Path) -> tuple[str, str, str]:
    from .bagread import _metadata, capture_identity

    try:
        meta = _metadata(bag)
        cid, vid = capture_identity(meta)
        return cid, vid, meta.get("storage_identifier", "")
    except Exception:
        return "unknown", "unknown", ""


def main() -> int:
    store = Store(os.environ["DSGATE_BUCKET"])
    plan = json.loads(store.get(os.environ["DSGATE_PLAN_KEY"]))
    segment_id = resolve_index(plan, os.environ)
    return process(store, plan, segment_id)


if __name__ == "__main__":
    sys.exit(main())
