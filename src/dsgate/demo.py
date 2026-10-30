"""`make demo`: one story, end to end, on an S3 emulator (moto server).

1. generate a small synthetic corpus (two upload batches)
2. build v1 and its report
3. add batch 2 (an overlapping re-upload and known defects), build v2,
   killing two workers after their outputs and before their receipts
4. resume from the frozen submission with a compacted plan, publish v2
5. rebuild v2 in a fresh bucket without interruption: same version
6. diff v1 -> v2
7. training refuses a non-admitted version, then trains on v2
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

from . import synth
from .batch import (build_submission, make_plan, remaining, run_local, submit_requests, summarize,
                    upload_bags, upload_side, validate_requests)
from .canon import cjson
from .publish import analysis_digest, publish
from .report import render_diff, render_report, seg_label
from .store import Store
from .train_launch import NotAdmitted, build_image, export, gate, run_process, run_sagemaker_local

CONFIG = Path(__file__).resolve().parents[2] / "config"
KILL = ("tractor-03/rosbag2_2026_10_14-13_00_30", "tractor-04/rosbag2_2026_10_15-09_10_40")


def load_recipe(name: str) -> dict:
    return json.loads((CONFIG / f"{name}.json").read_text())


def start_emulator() -> tuple[object, str]:
    import logging

    from moto.server import ThreadedMotoServer

    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    srv = ThreadedMotoServer(ip_address="127.0.0.1", port=0, verbose=False)
    srv.start()
    host, port = srv.get_host_and_port()
    return srv, f"http://{host}:{port}"


def run_batch(store: Store, sub: dict, recipe: dict, segment_ids: list[str], say, *, parallel=4,
              reverse=False, kill=None) -> list[dict]:
    rkey = f"recipes/{analysis_digest(recipe)[:12]}.json"
    store.put_immutable(rkey, cjson(recipe))
    plan = make_plan(store, sub, rkey, analysis_digest(recipe), segment_ids)
    reqs = submit_requests(plan, store.bucket)
    validate_requests(reqs)
    if not reqs:
        say("     nothing to submit (0 segments remaining)")
        return []
    results = run_local(reqs, {}, parallel=parallel, reverse=reverse, kill_before_receipt=kill, plan=plan)
    for line in summarize(reqs, results):
        say("     " + line.split("  ", 1)[1] if line.startswith("plan-") else "     " + line)
    bad = [r for r in results if r["exit"] not in (0, -9, 137)]
    if bad:
        raise RuntimeError("unexpected worker failure:\n" + "\n".join(r["stderr"] for r in bad))
    return results


def run_demo(out: Path, train: str = "sagemaker-local", parallel: int = 4, quiet: bool = False) -> dict:
    out = Path(out).resolve()
    if out.exists():
        if not (out / ".dsgate-demo").exists() and any(out.iterdir()):
            raise SystemExit(f"{out} exists and is not a previous demo output")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / ".dsgate-demo").write_text("output directory of dsgate demo; safe to delete\n")
    transcript: list[str] = []

    def say(line: str = "") -> None:
        transcript.append(line)
        if not quiet:
            print(line, flush=True)

    srv, endpoint = start_emulator()
    env = {"DSGATE_S3_ENDPOINT": endpoint, "AWS_ACCESS_KEY_ID": "testing", "AWS_SECRET_ACCESS_KEY": "testing",
           "AWS_DEFAULT_REGION": "us-east-1"}
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        return _story(out, train, parallel, say, transcript)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        srv.stop()


def _story(out: Path, train: str, parallel: int, say, transcript: list[str]) -> dict:
    corpus = synth.generate(out / "corpus")
    r1, r2, rbad = load_recipe("recipe_v1"), load_recipe("recipe_v2"), load_recipe("recipe_v2_split_by_segment")
    n_caps = len({c.capture_id for c in synth.BATCH1 + synth.BATCH2})
    say(f"[1] synthetic corpus: {len(corpus['batch1'])} bags in batch 1, {len(corpus['batch2'])} in batch 2 "
        f"(sqlite3 and MCAP, {n_caps} captures), labels and calibration as side inputs")

    store = Store("dsgate-demo")
    store.create_bucket()
    side = upload_side(store, out / "corpus" / "side")
    e1 = upload_bags(store, corpus["batch1"], out / "corpus" / "side")
    sub1 = build_submission(store, e1, side)
    say(f"[2] v1: submission frozen ({len(sub1['segments'])} segments: key, object version, size, sha256); "
        "SubmitJob requests validated against the botocore Batch model")
    run_batch(store, sub1, r1, [s["segment_id"] for s in sub1["segments"]], say, parallel=parallel)
    v1 = publish(store, sub1, r1, "v1")
    say(f"     published v1 = {v1['version_id']}  {v1['status'].upper()}, "
        f"{'ADMITTED' if v1['admitted'] else 'NOT ADMITTED'}")

    e2 = upload_bags(store, corpus["batch2"], out / "corpus" / "side")
    sub2 = build_submission(store, e1 + e2, side)
    by_bag = {s["bag"]: s for s in sub2["segments"]}
    kill = {by_bag[b]["segment_id"] for b in KILL}
    say(f"[3] v2: batch 2 uploaded, submission frozen ({len(sub2['segments'])} segments), recipe v2; "
        f"fault injection: the workers of {', '.join(seg_label({'bag': b}) for b in KILL)} are killed "
        "after writing their outputs, before their receipt")
    run_batch(store, sub2, r2, [s["segment_id"] for s in sub2["segments"]], say, parallel=parallel, kill=kill)
    attempt = publish(store, sub2, r2, "v2")
    try:
        gate(attempt)
        raise AssertionError("an incomplete version passed the gate")
    except NotAdmitted as exc:
        say(f"     publish attempt: {attempt['status'].upper()}; training launcher refuses it: {exc}")

    rem = remaining(store, sub2, analysis_digest(r2))
    say(f"[4] resume: {len(rem)} segment(s) without a receipt; new plan maps "
        + ", ".join(f"index {i} -> {sid}" for i, sid in enumerate(rem)) + " (stable ids, not original indices)")
    results = run_batch(store, sub2, r2, rem, say, parallel=parallel)
    for r in results:
        say(f"     {r['stdout'].splitlines()[-1]}")
    v2 = publish(store, sub2, r2, "v2", previous=v1)
    say(f"     published v2 = {v2['version_id']}  {v2['status'].upper()}, "
        f"{'ADMITTED' if v2['admitted'] else 'NOT ADMITTED'}")

    ref = Store("dsgate-reference")
    ref.create_bucket()
    ref_side = upload_side(ref, out / "corpus" / "side")
    ref_entries = upload_bags(ref, list(reversed(corpus["batch1"] + corpus["batch2"])), out / "corpus" / "side")
    ref_sub = build_submission(ref, ref_entries, ref_side)
    say("[5] reference: fresh bucket, empty caches, arrival order reversed, 2 workers in reverse order, "
        "no interruption")
    run_batch(ref, ref_sub, r1, [s["segment_id"] for s in ref_sub["segments"] if s["bag"] in {e["bag"] for e in e1}],
              say, parallel=2, reverse=True)
    ref_v1 = publish(ref, ref_sub, r1, "v1-ref", write=False)
    run_batch(ref, ref_sub, r2, [s["segment_id"] for s in ref_sub["segments"]], say, parallel=2, reverse=True)
    ref_v2 = publish(ref, ref_sub, r2, "v2-ref", previous=v1)
    same_samples = ref_v2["samples"] == v2["samples"]
    same_decisions = sorted((s["segment_id"], s["status"]) for s in ref_v2["segments"]) == \
        sorted((s["segment_id"], s["status"]) for s in v2["segments"])
    identical = ref_v2["version_id"] == v2["version_id"] and same_samples and same_decisions
    say(f"     reference v1 = {ref_v1['version_id']} ({'same as' if ref_v1['version_id'] == v1['version_id'] else 'DIFFERENT from'} v1)")
    say(f"     reference v2 = {ref_v2['version_id']}; resumed v2 = {v2['version_id']}: "
        f"{'IDENTICAL' if identical else 'DIFFERENT'} (version id, {len(v2['samples'])} samples with their shard keys, "
        f"{len(v2['segments'])} segment decisions)")
    if not identical:
        raise AssertionError("resumed build differs from the uninterrupted build")
    say("[6] diff v1 -> v2: see below")

    bad = publish(store, sub2, rbad, "v2-split-by-segment", previous=v1, write=False)
    say(f"[7] training gate on a misconfigured split (split.unit = segment): {bad['version_id']} "
        f"{'ADMITTED' if bad['admitted'] else 'NOT ADMITTED'}")
    try:
        gate(bad)
        raise AssertionError("a leaking version passed the gate")
    except NotAdmitted as exc:
        say(f"     refused before export or container start: {exc}")
    gate(v2)
    channel = export(store, v2, out / "train" / "channel")
    model = None
    if train == "sagemaker-local":
        image_id = build_image()
        proc = run_sagemaker_local(channel, out / "train", v2["version_id"])
        (out / "train" / "sagemaker.log").write_text(proc.stdout + proc.stderr)
        if proc.returncode != 0:
            raise RuntimeError("SageMaker local training failed:\n" + (proc.stdout + proc.stderr)[-4000:])
        model = json.loads((out / "train" / "container" / "artifacts" / "model" / "model.json").read_text())
        say(f"     ModelTrainer, training_mode LOCAL_CONTAINER (sagemaker 3.28.0), image built locally from "
            f"training/Dockerfile; child process with no AWS credentials or profile, AWS endpoints -> closed port")
        (out / "train" / "image_id.txt").write_text(image_id + "\n")
    elif train == "process":
        proc = run_process(channel, out / "train" / "model", v2["version_id"])
        if proc.returncode != 0:
            raise RuntimeError(proc.stdout + proc.stderr)
        model = json.loads((out / "train" / "model" / "model.json").read_text())
        say("     training/train.py run as a local process (no container)")
    if model is not None:
        ok = model["dataset_version_id"] == v2["version_id"]
        say(f"     model.json: dataset_version_id {model['dataset_version_id']} "
            f"({'matches v2' if ok else 'MISMATCH'}), {model['steps']} steps on {model['samples']['train']} train "
            f"samples, after the container re-verified the manifest, every shard and every sample row")
        if not ok:
            raise AssertionError("model does not carry the version it was trained on")
    else:
        say("     training skipped (--train skip)")

    for m in (v1, v2):
        d = out / m["name"]
        d.mkdir(exist_ok=True)
        (d / "manifest.json").write_text(json.dumps(m, indent=1, sort_keys=True) + "\n")
        (d / "report.txt").write_text(render_report(m))
    diff = render_diff(v1, v2)
    (out / "diff.txt").write_text(diff)
    (out / "transcript.txt").write_text("\n".join(transcript) + "\n")
    (out / "README_block.md").write_text(readme_block(transcript, render_report(v2), diff))
    return {"v1": v1, "v2": v2, "ref_v2": ref_v2, "ref_v1": ref_v1, "bad": bad, "attempt": attempt,
            "model": model, "sub2": sub2, "store": store, "kill": kill, "resumed": rem}


def readme_block(transcript: list[str], report: str, diff: str) -> str:
    return ("```text\n$ make demo\n" + "\n".join(transcript) + "\n```\n\n"
            "```text\n" + report + "```\n\n" + "```text\n" + diff + "```\n")


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="dsgate demo")
    ap.add_argument("--out", default="build/demo")
    ap.add_argument("--train", choices=["sagemaker-local", "process", "skip"], default="sagemaker-local")
    ap.add_argument("--parallel", type=int, default=4)
    args = ap.parse_args(argv)
    run_demo(Path(args.out), train=args.train, parallel=args.parallel)
    print(f"\nreports: {args.out}/v1/report.txt, {args.out}/v2/report.txt, diff: {args.out}/diff.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
