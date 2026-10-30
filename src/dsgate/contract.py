"""Run the worker image the way AWS Batch would start it (CI job `batch-contract`).

An S3 emulator runs in its own container; each child is a separate
``docker run`` of the worker image with only the environment Batch would set:
the request's container overrides and, for array children only,
``AWS_BATCH_JOB_ARRAY_INDEX``. Child 1 of the first array is held back, then
resumed as a plain job (no array, no index variable).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import synth
from .batch import (build_submission, compute_inputs, compute_key, make_plan, remaining, submit_requests,
                    upload_bags, upload_side, validate_requests)
from .canon import cjson
from .provenance import build_provenance
from .publish import analysis_digest, publish
from .recipe import analysis_part
from .store import Store

CAP = synth.CaptureSpec("tractor-07", "2026-10-21T07:30:00Z")
SEGMENTS = (synth.SegmentSpec(0, 10, "mcap"), synth.SegmentSpec(10, 20, "sqlite3"),
            synth.SegmentSpec(20, 30, "mcap", damage="truncate"))
EXPECT = {"rosbag2_2026_10_21-07_30_00": "retained", "rosbag2_2026_10_21-07_30_10": "retained",
          "rosbag2_2026_10_21-07_30_20": "quarantined"}


def children(reqs, skip=()):
    for r in reqs:
        env = {e["name"]: e["value"] for e in r["containerOverrides"]["environment"]}
        size = r.get("arrayProperties", {}).get("size")
        for i in range(size or 1):
            if (r["jobName"], i) in skip:
                continue
            child = dict(env)
            if size:
                child["AWS_BATCH_JOB_ARRAY_INDEX"] = str(i)
            yield r["jobName"], (i if size else None), child


def image_id(image):
    return subprocess.run(["docker", "image", "inspect", "-f", "{{.Id}}", image], check=True,
                          capture_output=True, text=True).stdout.strip()


def run_container(image, network, endpoint, env):
    # provenance records the immutable image id, not the mutable tag
    cmd = ["docker", "run", "--rm", "--network", network, "-e", f"DSGATE_S3_ENDPOINT={endpoint}",
           "-e", f"DSGATE_IMAGE={image_id(image)}",
           "-e", "AWS_ACCESS_KEY_ID=testing", "-e", "AWS_SECRET_ACCESS_KEY=testing", "-e", "AWS_DEFAULT_REGION=us-east-1"]
    for k, v in env.items():
        cmd += ["-e", f"{k}={v}"]
    return subprocess.run(cmd + [image], capture_output=True, text=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dsgate.contract")
    ap.add_argument("--work", default="build/contract")
    ap.add_argument("--image", default="dsgate-worker:contract")
    ap.add_argument("--network", default="dsgate-net")
    ap.add_argument("--container-endpoint", default="http://moto:5000")
    args = ap.parse_args(argv)
    work = Path(args.work).resolve()
    recipe = json.loads((Path(__file__).resolve().parents[2] / "config" / "recipe_v2.json").read_text())

    bags = [synth.write_segment(work / "bags", CAP, s) for s in SEGMENTS]
    synth.write_labels(work / "side", CAP)
    synth.write_calibration(work / "side", CAP.vehicle_id)
    store = Store("dsgate-contract")
    store.create_bucket()
    sub = build_submission(store, upload_bags(store, bags), upload_side(store, work / "side"))
    ad = analysis_digest(recipe)
    akey = f"analyses/{ad[:16]}.json"
    store.put_immutable(akey, cjson(analysis_part(recipe)))
    worker = {**build_provenance(), "image": image_id(args.image)}
    print(f"expected worker image {worker['image'][:19]}...")

    plan = make_plan(store, sub, akey, ad, [s["segment_id"] for s in sub["segments"]], worker)
    reqs = submit_requests(plan, store.bucket)
    validate_requests(reqs)
    print(f"array request size {reqs[0]['arrayProperties']['size']}; child 1 held back")
    ok = True
    for job, idx, env in children(reqs, skip={(reqs[0]["jobName"], 1)}):
        p = run_container(args.image, args.network, args.container_endpoint, env)
        print(f"  container child {idx}: exit {p.returncode}  {p.stdout.strip().splitlines()[-1] if p.stdout.strip() else p.stderr[-300:]}")
        ok &= p.returncode == 0

    rem = remaining(store, sub, ad, worker)
    plan2 = make_plan(store, sub, akey, ad, rem, worker)
    reqs2 = submit_requests(plan2, store.bucket)
    validate_requests(reqs2)
    assert len(rem) == 1 and "arrayProperties" not in reqs2[0], "one remaining segment must be a plain job"
    for job, idx, env in children(reqs2):
        assert "AWS_BATCH_JOB_ARRAY_INDEX" not in env
        p = run_container(args.image, args.network, args.container_endpoint, env)
        print(f"  plain job (no AWS_BATCH_JOB_ARRAY_INDEX): exit {p.returncode}  {p.stdout.strip().splitlines()[-1] if p.stdout.strip() else p.stderr[-300:]}")
        ok &= p.returncode == 0

    m = publish(store, sub, recipe, "contract", write=False, worker=worker)
    for s in m["segments"]:
        want = EXPECT[s["bag"].split("/", 1)[1]]
        seg = next(x for x in sub["segments"] if x["segment_id"] == s["segment_id"])
        rec = json.loads(store.get(compute_key(compute_inputs(sub, seg, ad, worker))))
        print(f"  {s['bag']}: {s['status']} (expected {want}); worker image recorded: {rec['worker']['image'][:19]}...")
        ok &= s["status"] == want and rec["worker"]["image"] == worker["image"]
    ok &= m["status"] == "complete"
    print("batch contract in containers:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
