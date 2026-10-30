"""`make demo`: one story, end to end, on an S3 emulator (moto server).

1. generate a small synthetic corpus (two upload batches)
2. build v1 and its report
3. add batch 2 (an overlapping re-upload and known defects), build v2,
   killing two workers after their outputs and before their receipts
4. resume from the frozen submission with a compacted plan, publish v2
5. rebuild v1 then v2 in a fresh bucket without interruption: same versions
6. republish v2 under another name and another admission policy: no conflict
7. diff v1 -> v2
8. training refuses a non-admitted version, then trains on v2
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import sys
from pathlib import Path

from . import synth
from .batch import (build_submission, make_plan, remaining, run_local, submit_requests, summarize,
                    upload_bags, upload_side, validate_requests)
from .canon import cjson
from .provenance import build_provenance
from .publish import analysis_digest, publish
from .recipe import analysis_part
from .report import render_diff, render_report, seg_label
from .store import Store
from .train_launch import NotAdmitted, gate, launch

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


def run_batch(store: Store, sub: dict, recipe: dict, segment_ids: list[str], say, env: dict, *, parallel=4,
              reverse=False, kill=None, worker=None) -> list[dict]:
    ad = analysis_digest(recipe)
    akey = f"analyses/{ad[:16]}.json"
    store.put_immutable(akey, cjson(analysis_part(recipe)))
    plan = make_plan(store, sub, akey, ad, segment_ids, worker or build_provenance())
    reqs = submit_requests(plan, store.bucket)
    validate_requests(reqs)
    if not reqs:
        say("     nothing to submit (0 segments remaining)")
        return []
    results = run_local(reqs, env, parallel=parallel, reverse=reverse, kill_before_receipt=kill, plan=plan)
    for line in summarize(reqs, results):
        say("     " + line)
    bad = [r for r in results if r["exit"] not in (0, -9, 137)]
    if bad:
        raise RuntimeError("unexpected worker failure:\n" + "\n".join(r["stdout"] + r["stderr"] for r in bad))
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
        return _story(out, train, parallel, say, transcript, env)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        srv.stop()


def _adm(m: dict) -> str:
    return f"{m['status'].upper()}, {'ADMITTED' if m['admitted'] else 'NOT ADMITTED'}"


def _story(out: Path, train: str, parallel: int, say, transcript: list[str], env: dict) -> dict:
    corpus = synth.generate(out / "corpus")
    declared = corpus["declared"]
    side_dir = out / "corpus" / "side"
    r1, r2, rbad = load_recipe("recipe_v1"), load_recipe("recipe_v2"), load_recipe("recipe_v2_split_by_segment")
    n_caps = len({c.capture_id for c in synth.BATCH1 + synth.BATCH2})
    say(f"[1] synthetic corpus: {len(corpus['batch1'])} bags in batch 1, {len(corpus['batch2'])} in batch 2 "
        f"(sqlite3 and MCAP, {n_caps} captures; {len(declared)} bags without custom_data, identity declared at "
        "upload), labels and calibration as side inputs")

    store = Store("dsgate-demo")
    store.create_bucket()
    side = upload_side(store, side_dir)
    e1 = upload_bags(store, corpus["batch1"], declared)
    sub1 = build_submission(store, e1, side)
    say(f"[2] v1: submission frozen ({len(sub1['segments'])} segments: key, object version, size, sha256); "
        "SubmitJob requests validated against the botocore Batch model")
    run_batch(store, sub1, r1, [s["segment_id"] for s in sub1["segments"]], say, env, parallel=parallel)
    v1 = publish(store, sub1, r1, "v1")
    say(f"     published v1 = {v1['version_id']}  {_adm(v1)}")

    e2 = upload_bags(store, corpus["batch2"], declared)
    sub2 = build_submission(store, e1 + e2, side)
    by_bag = {s["bag"]: s for s in sub2["segments"]}
    kill = {by_bag[b]["segment_id"] for b in KILL}
    say(f"[3] v2: batch 2 uploaded, submission frozen ({len(sub2['segments'])} segments), recipe v2; "
        f"fault injection: the workers of {', '.join(seg_label({'bag': b}) for b in KILL)} are killed "
        "after writing their outputs, before their receipt")
    run_batch(store, sub2, r2, [s["segment_id"] for s in sub2["segments"]], say, env, parallel=parallel, kill=kill)
    attempt = publish(store, sub2, r2, "v2")
    try:
        launch(store, attempt, out / "train-refused-incomplete", mode="process")
        raise AssertionError("an incomplete version passed the gate")
    except NotAdmitted as exc:
        nothing = not (out / "train-refused-incomplete").exists()
        say(f"     publish attempt: {attempt['status'].upper()}; launch() refuses it "
            f"({'nothing exported' if nothing else 'FILES WRITTEN'}): {exc}")

    rem = remaining(store, sub2, analysis_digest(r2), build_provenance())
    say(f"[4] resume: {len(rem)} segment(s) without a receipt for this computation; new plan maps "
        + ", ".join(f"index {i} -> {sid}" for i, sid in enumerate(rem)) + " (stable ids, not original indices)")
    results = run_batch(store, sub2, r2, rem, say, env, parallel=parallel)
    resume_logs = [r["stdout"].splitlines()[-1] for r in results]
    for line in resume_logs:
        say(f"     {line}")
    v2 = publish(store, sub2, r2, "v2", previous=v1)
    say(f"     published v2 = {v2['version_id']}  {_adm(v2)}")

    ref = Store("dsgate-reference")
    ref.create_bucket()
    ref_side = upload_side(ref, side_dir)
    batch1_rev = list(reversed(corpus["batch1"]))
    ref_e1 = upload_bags(ref, batch1_rev, declared)
    ref_sub1 = build_submission(ref, ref_e1, ref_side)
    say("[5] reference: fresh bucket, empty caches, arrival order reversed, 2 workers in reverse order, "
        "no interruption; v1 rebuilt on its own 7 segments, then used as the history of v2")
    run_batch(ref, ref_sub1, r1, [s["segment_id"] for s in ref_sub1["segments"]], say, env, parallel=2, reverse=True)
    ref_v1 = publish(ref, ref_sub1, r1, "v1-ref")
    same_v1 = ref_v1["version_id"] == v1["version_id"] and ref_v1["samples"] == v1["samples"]
    say(f"     reference v1 = {ref_v1['version_id']}  {_adm(ref_v1)}: {'IDENTICAL to' if same_v1 else 'DIFFERENT from'} v1")
    ref_e2 = upload_bags(ref, list(reversed(corpus["batch2"])), declared)
    ref_sub2 = build_submission(ref, ref_e1 + ref_e2, ref_side)
    run_batch(ref, ref_sub2, r2, [s["segment_id"] for s in ref_sub2["segments"]], say, env, parallel=2, reverse=True)
    ref_v2 = publish(ref, ref_sub2, r2, "v2-ref", previous=ref_v1)
    same_samples = ref_v2["samples"] == v2["samples"]
    same_decisions = sorted((s["segment_id"], s["status"]) for s in ref_v2["segments"]) == \
        sorted((s["segment_id"], s["status"]) for s in v2["segments"])
    identical = ref_v2["version_id"] == v2["version_id"] and same_samples and same_decisions
    say(f"     reference v2 = {ref_v2['version_id']}; resumed v2 = {v2['version_id']}: "
        f"{'IDENTICAL' if identical else 'DIFFERENT'} (version id, {len(v2['samples'])} samples with their shard keys, "
        f"{len(v2['segments'])} segment decisions)")
    if not (identical and same_v1):
        raise AssertionError("a reference build differs from the demo build")

    again = publish(store, sub2, r2, "v2-again", previous=v1)
    looser = copy.deepcopy(r2)
    looser["admission"]["min_retained_fraction"] = 0.25
    loose = publish(store, sub2, looser, "v2-looser-policy", previous=v1)
    say(f"[6] same bucket: v2 republished under another name, then with min_retained_fraction 0.25: content "
        f"{again['version_id']} and {loose['version_id']} (datasets/ object reused), builds "
        f"{'distinct' if len({v2['build_id'], again['build_id'], loose['build_id']}) == 3 else 'NOT distinct'}, no conflict")
    say("[7] diff v1 -> v2: see below")

    bad = publish(store, sub2, rbad, "v2-split-by-segment", previous=v1, write=False)
    say(f"[8] training: misconfigured split (split.unit = segment) gives {bad['version_id']} {_adm(bad)}")
    try:
        launch(store, bad, out / "train-refused-leak", mode="process")
        raise AssertionError("a leaking version passed the gate")
    except NotAdmitted as exc:
        nothing = not (out / "train-refused-leak").exists()
        say(f"     launch() refuses it ({'nothing exported, no container' if nothing else 'FILES WRITTEN'}): {exc}")
    model = None
    if train in ("sagemaker-local", "process"):
        result = launch(store, v2, out / "train", mode=train)
        model = result["model"]
        if train == "sagemaker-local":
            say("     launch(v2): ModelTrainer, training_mode LOCAL_CONTAINER (sagemaker 3.28.0), image built locally "
                "from training/Dockerfile; child process with no AWS credentials or profile, AWS endpoints -> closed port")
            (out / "train" / "image_id.txt").write_text(result["image_id"] + "\n")
        else:
            say("     launch(v2): training/train.py run as a local process (no container)")
        ok = model["dataset_version_id"] == v2["version_id"] and \
            model["dataset_manifest_sha256"] == result["manifest_sha256"]
        say(f"     model.json: dataset_version_id {model['dataset_version_id']} "
            f"({'matches v2' if ok else 'MISMATCH'}), {model['steps']} smoke-test steps on {model['samples']['train']} "
            "train samples, after the container checked the sealed manifest digest, the admission invariants, "
            "every shard and every sample row")
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
    blocks = readme_blocks(transcript, render_report(v2), diff)
    (out / "README_excerpt.md").write_text(blocks["excerpt"])
    (out / "README_block.md").write_text(blocks["full"])
    return {"v1": v1, "v2": v2, "ref_v2": ref_v2, "ref_v1": ref_v1, "bad": bad, "attempt": attempt,
            "again": again, "loose": loose, "model": model, "sub2": sub2, "store": store, "kill": kill, "resumed": rem,
            "resume_logs": resume_logs, "transcript": transcript}


EXCERPT_MARKS = ("[3] v2:", "parent FAILED", "launch() refuses it (nothing exported)", "[4] resume:",
                 "already present and identical", "published v2 =", "reference v2 =", "launch() refuses it (nothing exported, no container)",
                 "model.json:")


def readme_blocks(transcript: list[str], report: str, diff: str) -> dict:
    excerpt = [line for line in transcript if any(m in line for m in EXCERPT_MARKS)]
    return {
        "excerpt": "```text\n" + "\n".join(excerpt) + "\n```\n",
        "full": ("```text\n$ make demo\n" + "\n".join(transcript) + "\n```\n\n"
                 "```text\n" + report + "```\n\n" + "```text\n" + diff + "```\n"),
    }


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
