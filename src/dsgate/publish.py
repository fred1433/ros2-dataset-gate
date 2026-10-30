"""Publish a dataset version from terminal receipts only.

Three identities, kept apart, and stored apart:
* source integrity: SHA-256 of raw bytes + S3 object version, per file
  (frozen in the submission, verified by the worker);
* dataset content (``version_id``): canonical JSON of the published samples
  (capture, acquisition stamps, content hashes, label), the schemas, the
  calibration digests and the train/val assignment. Stored under
  ``datasets/<version_id>.json``; the same content under another name or
  policy is the same object;
* build and admission (``build_id``): submission, recipe, receipts' computation
  identities, worker and publisher provenance, gates and verdict, stored under
  ``builds/<build_id>.json``; the sample-to-shard mapping goes to
  ``layouts/<layout_id>.json``.

Duplicates. The observation key is (capture id, point-cloud acquisition
stamp). Candidates with that key are grouped into variants by (camera stamp,
camera content, point-cloud content, label). Each variant is one observation:
one candidate is kept, preferring a segment already published by the previous
version, then the earliest segment, and the others are recorded as duplicates
of it. Two or more variants for one key are a *possible duplicate*: every
variant is kept and reported, never dropped silently.
"""

from __future__ import annotations

import json
from collections import defaultdict

from .batch import compute_inputs, compute_key
from .canon import cjson, digest, sha256, short
from .provenance import build_provenance, git_commit
from .recipe import analysis_part, validate
from .store import Store


def analysis_digest(recipe: dict) -> str:
    return digest(analysis_part(recipe))


def recipe_digest(recipe: dict) -> str:
    return digest(recipe)


def assign_split(group: str, split_cfg: dict) -> str:
    h = int(sha256(f"{split_cfg['salt']}|{group}".encode())[:8], 16) / 2**32
    return "val" if h < split_cfg["val_fraction"] else "train"


OBS_FIELDS = ("capture_id", "ref_stamp_ns", "cam_stamp_ns", "cam_sha256", "lidar_sha256", "label_mm")


def identity_body(samples: list[dict], schemas: dict, calibration: dict) -> dict:
    rows = sorted([*(s[k] for k in OBS_FIELDS), s["split"]] for s in samples)
    return {"format": "dsgate.identity/1", "schemas": schemas, "calibration": calibration, "samples": rows}


def version_id(body: dict) -> str:
    return "ds-" + short(digest(body))


def dedup(candidates: list[dict], prefer: frozenset | set = frozenset()) -> tuple[list[dict], dict, list[dict]]:
    """Returns (kept samples, {segment_id: set(segment ids it duplicates)}, possible duplicates)."""
    groups: dict[tuple, dict[tuple, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for c in candidates:
        variant = tuple(c[k] for k in OBS_FIELDS[2:])
        groups[(c["capture_id"], c["ref_stamp_ns"])][variant].append(c)
    kept, dup_of, possible = [], defaultdict(set), []

    def priority(c):
        return (0 if c["src"]["segment_id"] in prefer else 1, c.get("start_log_ns") or 0, c["src"]["segment_id"])

    for key in sorted(groups):
        variants = groups[key]
        for variant in sorted(variants):
            g = sorted(variants[variant], key=priority)
            kept.append(g[0])
            for other in g[1:]:
                dup_of[other["src"]["segment_id"]].add(g[0]["src"]["segment_id"])
        if len(variants) > 1:
            possible.append({"capture_id": key[0], "ref_stamp_ns": key[1], "variants": len(variants),
                             "segments": sorted({c["src"]["segment_id"] for v in variants.values() for c in v})})
    return kept, dup_of, possible


def assign_splits(kept: list[dict], split_cfg: dict, previous: dict | None) -> dict:
    """Sets c["split"]; the previous version's registry wins, new groups go by salted hash."""
    registry = dict((previous or {}).get("split_registry", {})) if split_cfg["unit"] == "capture" else {}
    for c in kept:
        group = c["capture_id"] if split_cfg["unit"] == "capture" else c["src"]["segment_id"]
        if group not in registry:
            registry[group] = assign_split(group, split_cfg)
        c["split"] = registry[group]
    return registry


def leaking_captures(kept: list[dict]) -> list[str]:
    sides = defaultdict(set)
    for c in kept:
        sides[c["capture_id"]].add(c["split"])
    return sorted(cap for cap, s in sides.items() if len(s) > 1)


def admission_gates(count: dict, examined: int, missing: list, leakage: list, per_split: dict, adm: dict) -> list[dict]:
    processed = examined - count["quarantined"] - count["deduplicated"] - count["no_receipt"]
    retained_fraction = count["retained"] / processed if processed else 0.0
    return [
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


def publish(store: Store, submission: dict, recipe: dict, name: str, previous: dict | None = None,
            write: bool = True, worker: dict | None = None) -> dict:
    validate(recipe)
    ad = analysis_digest(recipe)
    worker = worker or build_provenance()
    receipts, missing = {}, []
    for s in submission["segments"]:
        inputs = compute_inputs(submission, s, ad, worker)
        key = compute_key(inputs)
        if store.exists(key):
            r = json.loads(store.get(key))
            if r["compute"] != inputs:
                raise ValueError(f"receipt {key} does not record the computation it is stored under")
            receipts[s["segment_id"]] = r
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
            c = {k: smp[k] for k in OBS_FIELDS}
            c["start_log_ns"] = r["start_log_ns"]
            c["src"] = {"segment_id": sid, "camera_key": out["camera"], "lidar_points_key": out["lidar_points"],
                        "lidar_offsets_key": out["lidar_offsets"], "row": smp["row"]}
            candidates.append(c)
    prefer = {smp["src"]["segment_id"] for smp in (previous or {}).get("samples", [])}
    kept, dup_of, possible = dedup(candidates, prefer)
    for c in kept:
        c.pop("start_log_ns", None)

    split_cfg = recipe["split"]
    registry = assign_splits(kept, split_cfg, previous)
    leakage = leaking_captures(kept)

    vehicles = sorted({r["vehicle_id"] for r in receipts.values() if r["status"] == "retained"})
    calibration = {}
    for v in vehicles:
        ref = submission["side_inputs"]["calibration"].get(v)
        if ref is None:
            raise ValueError(f"no calibration for vehicle {v} in the submission")
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
            "source_digest": s["source_digest"], "receipt": compute_key(r["compute"]),
            "samples_candidate": n_cand, "samples_published": n_kept, "samples_deduplicated": n_cand - n_kept,
            "duplicate_of": sorted(dup_of.get(sid, ())),
        })

    count = defaultdict(int)
    for s in segments:
        count[s["status"]] += 1
    examined = len(segments)
    per_split = {"train": sum(1 for c in kept if c["split"] == "train"),
                 "val": sum(1 for c in kept if c["split"] == "val")}
    gates = admission_gates(count, examined, missing, leakage, per_split, recipe["admission"])
    workers = {cjson(r["worker"]): r["worker"] for r in receipts.values()}
    manifest = {
        "format": "dsgate.version/2",
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
        "provenance": {"publisher": build_provenance(), "workers": [workers[k] for k in sorted(workers)]},
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
        "samples": sorted(kept, key=lambda c: (c["capture_id"], c["ref_stamp_ns"], c["cam_stamp_ns"],
                                               c["src"]["segment_id"])),
    }
    layout = [[*(s[k] for k in OBS_FIELDS[:3]), s["src"]] for s in manifest["samples"]]
    manifest["layout_id"] = "layout-" + short(digest(layout))
    build = {k: v for k, v in _stable(manifest).items() if k not in ("samples",)}
    manifest["build_id"] = "build-" + short(digest(build))
    if write and manifest["status"] == "complete":
        store.put_immutable(f"recipes/{manifest['recipe_digest'][:16]}.json", cjson(recipe))
        store.put_immutable(f"datasets/{vid}.json", cjson(body))
        store.put_immutable(f"layouts/{manifest['layout_id']}.json", cjson(layout))
        store.put_immutable(f"builds/{manifest['build_id']}.json", cjson(build))
    return manifest


def _stable(manifest: dict) -> dict:
    """The manifest without volatile metadata (git commit)."""
    return {k: v for k, v in manifest.items() if k != "metadata"}


def manifest_digest(manifest: dict) -> str:
    return digest(_stable(manifest))
