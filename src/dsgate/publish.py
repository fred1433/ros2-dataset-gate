"""Publish a dataset version from terminal receipts only.

Three identities, kept apart:
* source integrity: SHA-256 of raw bytes + S3 object version, per file
  (frozen in the submission, verified by the worker);
* dataset identity (``version_id``): canonical JSON of the published samples
  (capture, acquisition stamps, content hashes, label), the schemas, the
  calibration digests and the train/val assignment. Nothing else;
* build provenance: recipe, code, lockfile, image; git commit as metadata.

Re-uploads: two candidate samples with the same capture id and the same
acquisition stamp and the same content are one observation. Same capture and
stamp with different content is a *possible duplicate*: both are kept and
reported, never dropped silently. Identical pixels at different instants or in
different captures are distinct observations.
"""

from __future__ import annotations

import json
from collections import defaultdict

from .batch import receipt_key
from .canon import cjson, digest, sha256, short
from .provenance import build_provenance, git_commit
from .store import Store

ANALYSIS_KEYS = ("required_topics", "time_source", "pairing", "thresholds", "policy")


def analysis_digest(recipe: dict) -> str:
    return digest({k: recipe[k] for k in ANALYSIS_KEYS})


def recipe_digest(recipe: dict) -> str:
    return digest(recipe)


def assign_split(group: str, split_cfg: dict) -> str:
    h = int(sha256(f"{split_cfg['salt']}|{group}".encode())[:8], 16) / 2**32
    return "val" if h < split_cfg["val_fraction"] else "train"


def identity_body(samples: list[dict], schemas: dict, calibration: dict) -> dict:
    rows = sorted([s["capture_id"], s["ref_stamp_ns"], s["cam_stamp_ns"], s["cam_sha256"],
                   s["lidar_sha256"], s["label_mm"], s["split"]] for s in samples)
    return {"format": "dsgate.identity/1", "schemas": schemas, "calibration": calibration, "samples": rows}


def version_id(body: dict) -> str:
    return "ds-" + short(digest(body))


def dedup(candidates: list[dict]) -> tuple[list[dict], dict, list[dict]]:
    """Returns (kept samples, {segment_id: set(segment ids it duplicates)}, possible duplicates)."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for c in candidates:
        groups[(c["capture_id"], c["ref_stamp_ns"])].append(c)
    kept, dup_of, possible = [], defaultdict(set), []
    for key in sorted(groups):
        g = sorted(groups[key], key=lambda c: c["src"]["segment_id"])
        contents = {(c["cam_sha256"], c["lidar_sha256"], c["label_mm"]) for c in g}
        if len(contents) == 1:
            kept.append(g[0])
            for other in g[1:]:
                dup_of[other["src"]["segment_id"]].add(g[0]["src"]["segment_id"])
        else:
            kept.extend(g)
            possible.append({"capture_id": key[0], "ref_stamp_ns": key[1],
                             "segments": [c["src"]["segment_id"] for c in g]})
    return kept, dup_of, possible


def publish(store: Store, submission: dict, recipe: dict, name: str, previous: dict | None = None,
            write: bool = True) -> dict:
    ad = analysis_digest(recipe)
    receipts, missing = {}, []
    for s in submission["segments"]:
        key = receipt_key(ad, s["segment_id"])
        if store.exists(key):
            receipts[s["segment_id"]] = json.loads(store.get(key))
        else:
            missing.append(s["segment_id"])

    candidates, schemas = [], {}
    for sid in sorted(receipts):
        r = receipts[sid]
        if r["status"] != "retained":
            continue
        schemas.update(r["schemas"])
        out = {o["kind"]: o["key"] for o in r["outputs"]}
        for smp in r["samples"]:
            c = {k: smp[k] for k in ("capture_id", "ref_stamp_ns", "cam_stamp_ns", "cam_sha256", "lidar_sha256", "label_mm")}
            c["src"] = {"segment_id": sid, "camera_key": out["camera"], "lidar_key": out["lidar"], "row": smp["row"]}
            candidates.append(c)
    kept, dup_of, possible = dedup(candidates)

    # split: previous registry wins, new groups by hash
    split_cfg = recipe["split"]
    registry = dict((previous or {}).get("split_registry", {})) if split_cfg["unit"] == "capture" else {}
    for c in kept:
        group = c["capture_id"] if split_cfg["unit"] == "capture" else c["src"]["segment_id"]
        if group not in registry:
            registry[group] = assign_split(group, split_cfg)
        c["split"] = registry[group]
    sides = defaultdict(set)
    for c in kept:
        sides[c["capture_id"]].add(c["split"])
    leakage = sorted(cap for cap, s in sides.items() if len(s) > 1)

    vehicles = sorted({r["vehicle_id"] for r in receipts.values() if r["status"] == "retained"})
    calibration = {}
    for v in vehicles:
        ref = submission["side_inputs"]["calibration"][v]
        body = store.get(ref["key"], ref["version_id"])
        if sha256(body) != ref["sha256"]:
            raise ValueError(f"calibration {v} differs from the frozen submission")
        calibration[v] = ref["sha256"]

    body = identity_body(kept, schemas, calibration)
    vid = version_id(body)

    kept_by_seg = defaultdict(int)
    for c in kept:
        kept_by_seg[c["src"]["segment_id"]] += 1
    segments = []
    for s in submission["segments"]:
        sid = s["segment_id"]
        r = receipts.get(sid)
        if r is None:
            segments.append({"segment_id": sid, "bag": s["bag"], "status": "no_receipt"})
            continue
        status = r["status"]
        n_cand = len(r["samples"]) if status == "retained" else 0
        n_kept = kept_by_seg.get(sid, 0)
        if status == "retained" and n_cand and n_kept == 0:
            status = "deduplicated"
        segments.append({
            "segment_id": sid, "bag": s["bag"], "capture_id": r.get("capture_id"), "vehicle_id": r.get("vehicle_id"),
            "storage": r.get("storage"), "start_log_ns": r.get("start_log_ns"), "status": status,
            "findings": r["findings"], "info": r["info"], "quarantine": r.get("quarantine"),
            "source_digest": s["source_digest"],
            "samples_candidate": n_cand, "samples_published": n_kept, "samples_deduplicated": n_cand - n_kept,
            "duplicate_of": sorted(dup_of.get(sid, ())),
        })

    count = defaultdict(int)
    for s in segments:
        count[s["status"]] += 1
    examined = len(segments)
    processed = examined - count["quarantined"] - count["deduplicated"] - count["no_receipt"]
    retained_fraction = count["retained"] / processed if processed else 0.0
    per_split = {"train": sum(1 for c in kept if c["split"] == "train"),
                 "val": sum(1 for c in kept if c["split"] == "val")}
    adm = recipe["admission"]
    gates = [
        {"gate": "complete", "ok": not missing,
         "detail": f"{examined - len(missing)}/{examined} segments have a terminal receipt"},
        {"gate": "train_val_leakage", "ok": len(leakage) <= adm["max_leakage_groups"],
         "detail": f"{len(leakage)} capture(s) on both sides (max {adm['max_leakage_groups']})"},
        {"gate": "retained_fraction", "ok": retained_fraction >= adm["min_retained_fraction"],
         "detail": f"{count['retained']}/{processed} processed segments retained "
                   f"= {retained_fraction:.2f} (min {adm['min_retained_fraction']})"},
        {"gate": "samples_per_split", "ok": min(per_split.values()) >= adm["min_samples_per_split"],
         "detail": f"train {per_split['train']}, val {per_split['val']} (min {adm['min_samples_per_split']})"},
    ]
    manifest = {
        "format": "dsgate.version/1",
        "name": name,
        "version_id": vid,
        "status": "complete" if not missing else "incomplete",
        "admitted": all(g["ok"] for g in gates),
        "gates": gates,
        "submission_id": submission["submission_id"],
        "recipe": recipe,
        "recipe_digest": recipe_digest(recipe),
        "analysis_digest": ad,
        "previous": (previous or {}).get("version_id"),
        "provenance": build_provenance(),
        "metadata": {"git_commit": git_commit()},
        "identity": {"schemas": schemas, "calibration": calibration, "sample_count": len(kept)},
        "counts": {"examined": examined, **{k: count[k] for k in
                   ("retained", "excluded", "deduplicated", "quarantined", "no_receipt")},
                   "samples_candidate": len(candidates), "samples_published": len(kept),
                   "samples_deduplicated": len(candidates) - len(kept),
                   "train": per_split["train"], "val": per_split["val"]},
        "leakage": leakage,
        "possible_duplicates": possible,
        "missing_receipts": missing,
        "exact_reuploads": submission.get("exact_reuploads", []),
        "split_registry": dict(sorted(registry.items())) if split_cfg["unit"] == "capture" else {},
        "segments": segments,
        "samples": sorted(kept, key=lambda c: (c["capture_id"], c["ref_stamp_ns"], c["src"]["segment_id"])),
    }
    if write and manifest["status"] == "complete":
        store.put_immutable(f"versions/{vid}.json", cjson(_stable(manifest)))
    return manifest


def _stable(manifest: dict) -> dict:
    """The stored manifest without volatile metadata (git commit)."""
    return {k: v for k, v in manifest.items() if k != "metadata"}


def manifest_digest(manifest: dict) -> str:
    return digest(_stable(manifest))
