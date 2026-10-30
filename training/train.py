"""Training entry point (SageMaker training-container contract, numpy only).

Reads one exported dataset version from the ``train`` channel and refuses it,
before any optimisation step, unless:
* the manifest's SHA-256 is the one the launcher sealed after admission;
* it is complete and admitted, and the invariants recomputable here hold
  (every gate passed, no receipt missing, no capture on both sides);
* its content hashes to the version id the job asked for;
* every shard matches its content hash and every sample row matches the
  hashes in the manifest.
Then it runs a few gradient steps of a tiny linear model (distance to the cart
from a downsampled camera frame): a smoke test that the verified samples are
consumed, not a perception model. ``model.json`` carries the version id and
the manifest digest it read.
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

CANONICAL_NAN = np.frombuffer(np.uint32(0x7FC00000).tobytes(), dtype="<f4")[0]
OBS = ("capture_id", "ref_stamp_ns", "cam_stamp_ns", "cam_sha256", "lidar_sha256", "label_mm")


def cjson(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def identity(manifest):
    rows = sorted([*(s[k] for k in OBS), s["split"]] for s in manifest["samples"])
    body = {"format": "dsgate.identity/1", "schemas": manifest["identity"]["schemas"],
            "calibration": manifest["identity"]["calibration"], "samples": rows}
    return "ds-" + sha(cjson(body))[:12]


def fail(msg):
    path = os.environ.get("SM_OUTPUT_FAILURE", "/opt/ml/output/failure")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(msg + "\n")
    except OSError:
        pass
    print("REFUSED: " + msg, flush=True)
    sys.exit(2)


def load_verified(channel, expected_version, expected_sha):
    raw = open(os.path.join(channel, "manifest.json"), "rb").read()
    if sha(raw) != expected_sha:
        fail(f"manifest sha256 {sha(raw)[:12]} is not the one sealed at admission ({expected_sha[:12]})")
    m = json.loads(raw)
    if m.get("version_id") != expected_version:
        fail(f"channel holds {m.get('version_id')}, job asked for {expected_version}")
    if m.get("status") != "complete" or not m.get("admitted"):
        fail(f"{m.get('version_id')} is {m.get('status')}, admitted={m.get('admitted')}")
    if m.get("missing_receipts") or not all(g["ok"] for g in m.get("gates", [])):
        fail("admitted flag contradicts the recorded gates")
    sides = {}
    for s in m["samples"]:
        sides.setdefault(s["capture_id"], set()).add(s["split"])
    leaking = sorted(c for c, v in sides.items() if len(v) > 1)
    if leaking:
        fail(f"capture(s) on both sides of the split: {', '.join(leaking)}")
    if identity(m) != m["version_id"]:
        fail("manifest content does not hash to its version id")
    arrays = {}
    for s in m["samples"]:
        for k in ("camera_key", "lidar_points_key", "lidar_offsets_key"):
            key = s["src"][k]
            if key not in arrays:
                path = os.path.join(channel, "shards", key.rsplit("/", 1)[1])
                if not os.path.exists(path):
                    fail(f"missing shard {key}")
                body = open(path, "rb").read()
                if "cas/" + sha(body) + ".npy" != key:
                    fail(f"shard {key} does not match its content hash")
                arrays[key] = np.load(path, allow_pickle=False)
    cams, ys, split = [], [], []
    for s in m["samples"]:
        row = s["src"]["row"]
        img = arrays[s["src"]["camera_key"]][row]
        h, w = img.shape[:2]
        head = f"img/v1|rgb8|{h}|{w}|{w * 3}|".encode("ascii")
        if sha(head + img.tobytes()) != s["cam_sha256"]:
            fail(f"camera row {row} of {s['src']['camera_key']} does not match the manifest")
        off = arrays[s["src"]["lidar_offsets_key"]]
        pts = np.ascontiguousarray(arrays[s["src"]["lidar_points_key"]][off[row]:off[row + 1]], dtype="<f4").copy()
        pts[np.isnan(pts)] = CANONICAL_NAN
        if sha(f"pc/v1|xyzi-f32le|{pts.shape[0]}|".encode("ascii") + pts.tobytes()) != s["lidar_sha256"]:
            fail(f"point cloud row {row} of {s['src']['lidar_points_key']} does not match the manifest")
        cams.append(img)
        ys.append(s["label_mm"] / 1000.0)
        split.append(s["split"])
    return m, sha(raw), np.stack(cams), np.asarray(ys), np.asarray(split)


def features(cams, rows=6, cols=8):
    gray = cams.astype(np.float64).mean(axis=3)
    blocks = [[b.mean(axis=(1, 2)) for b in np.array_split(r, cols, axis=2)]
              for r in np.array_split(gray, rows, axis=1)]
    small = np.stack([c for r in blocks for c in r], axis=1) / 255.0
    return np.hstack([small, np.ones((len(small), 1))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_version", required=True)
    ap.add_argument("--manifest_sha256", required=True)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.05)
    args, _ = ap.parse_known_args()
    channel = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    m, manifest_sha, cams, y, split = load_verified(channel, args.dataset_version, args.manifest_sha256)
    x = features(cams)
    tr, va = split == "train", split == "val"
    w = np.zeros(x.shape[1])
    losses = []
    for _ in range(args.steps):
        err = x[tr] @ w - y[tr]
        losses.append(float(np.mean(err ** 2)))
        w -= args.lr * 2 * x[tr].T @ err / tr.sum()
    val_mse = float(np.mean((x[va] @ w - y[va]) ** 2))
    os.makedirs(model_dir, exist_ok=True)
    out = {
        "dataset_version_id": m["version_id"],
        "dataset_manifest_sha256": manifest_sha,
        "build_id": m.get("build_id"),
        "recipe_digest": m["recipe_digest"],
        "samples": {"train": int(tr.sum()), "val": int(va.sum())},
        "steps": args.steps,
        "train_mse_per_step": [round(v, 4) for v in losses],
        "val_mse": round(val_mse, 4),
        "note": "smoke test: a few steps prove the verified samples are consumed, not model quality",
        "weights": [round(float(v), 6) for v in w],
    }
    with open(os.path.join(model_dir, "model.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"trained {args.steps} steps on {m['version_id']}: train mse {losses[0]:.2f} -> {losses[-1]:.2f}", flush=True)


if __name__ == "__main__":
    main()
