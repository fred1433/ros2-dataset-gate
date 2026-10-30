"""AWS Batch contract: frozen submission, compacted plans, SubmitJob requests.

Rules taken from the AWS Batch documentation (array jobs):
* an array job has between 2 and 10,000 children, indexed from 0 through
  ``AWS_BATCH_JOB_ARRAY_INDEX``; one remaining segment is a plain job
  (no ``arrayProperties``, no index variable), more than 10,000 go in waves;
* a failed child does not stop its siblings, but the parent ends FAILED once
  every child has finished, and a job depending on a failed job fails too.

So a data defect is a successful child that writes a ``quarantined`` receipt
(exit 0); a worker failure exits non-zero and is resumed. A resume never
re-indexes the original list: it writes a new plan whose index ``i`` maps to a
stable segment id, so a sibling finishing meanwhile cannot shift the mapping.
The scheduler itself is not run here: ``run_local`` executes the same worker
with the same environment the scheduler would set.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .canon import cjson, digest, sha256, short
from .store import Store

MAX_ARRAY = 10_000


# ----------------------------------------------------------------- submission

def upload_bags(store: Store, bags: list[Path], side: Path) -> list[dict]:
    """Upload raw bag directories into the versioned raw prefix; return entries."""
    entries = []
    for bag in bags:
        files = []
        for f in sorted(p for p in bag.iterdir() if p.is_file()):
            body = f.read_bytes()
            key = f"raw/{bag.parent.name}/{bag.name}/{f.name}"
            files.append({"key": key, "version_id": store.put(key, body), "size": len(body), "sha256": sha256(body)})
        entries.append({"bag": f"{bag.parent.name}/{bag.name}", "files": files})
    return entries


def upload_side(store: Store, side: Path) -> dict:
    out: dict[str, dict] = {"labels": {}, "calibration": {}}
    for kind in ("labels", "calibration"):
        for f in sorted((side / kind).glob("*.json")):
            body = f.read_bytes()
            key = f"side/{kind}/{f.name}"
            out[kind][f.stem] = {"key": key, "version_id": store.put(key, body), "size": len(body), "sha256": sha256(body)}
    return out


def build_submission(store: Store, entries: list[dict], side: dict) -> dict:
    """Freeze content: id, object version, size, digest per file. Identical uploads collapse."""
    segments: dict[str, dict] = {}
    exact_reuploads = []
    for e in entries:
        source_digest = digest(sorted(f["sha256"] for f in e["files"]))
        seg_id = "seg-" + short(source_digest)
        if seg_id in segments:
            exact_reuploads.append({"segment_id": seg_id, "bag": e["bag"], "same_bytes_as": segments[seg_id]["bag"]})
            continue
        segments[seg_id] = {"segment_id": seg_id, "bag": e["bag"], "source_digest": source_digest, "files": e["files"]}
    body = {"format": "dsgate.submission/1", "segments": [segments[k] for k in sorted(segments)],
            "side_inputs": side, "exact_reuploads": exact_reuploads}
    body["submission_id"] = "sub-" + short(digest(body))
    store.put_immutable(f"submissions/{body['submission_id']}.json", cjson(body))
    return body


# ----------------------------------------------------------------- plans/jobs

def make_plan(store: Store, submission: dict, recipe_key: str, analysis_digest: str,
              segment_ids: list[str]) -> dict:
    plan = {"format": "dsgate.plan/1", "submission_id": submission["submission_id"],
            "recipe_key": recipe_key, "analysis_digest": analysis_digest, "segment_ids": list(segment_ids)}
    plan["plan_id"] = "plan-" + short(digest(plan))
    plan["key"] = f"plans/{plan['plan_id']}.json"
    store.put_immutable(plan["key"], cjson(plan))
    return plan


def submit_requests(plan: dict, bucket: str, job_queue: str = "dsgate-curation",
                    job_definition: str = "dsgate-worker") -> list[dict]:
    """SubmitJob requests for a plan: 0 -> none, 1 -> plain job, 2..10000 -> one array, more -> waves."""
    ids = plan["segment_ids"]
    reqs = []
    for wave, offset in enumerate(range(0, len(ids), MAX_ARRAY)):
        size = min(MAX_ARRAY, len(ids) - offset)
        req = {
            "jobName": f"{plan['plan_id']}-w{wave:03d}",
            "jobQueue": job_queue,
            "jobDefinition": job_definition,
            "containerOverrides": {"environment": [
                {"name": "DSGATE_BUCKET", "value": bucket},
                {"name": "DSGATE_PLAN_KEY", "value": plan["key"]},
                {"name": "DSGATE_PLAN_OFFSET", "value": str(offset)},
            ]},
        }
        if size >= 2:
            req["arrayProperties"] = {"size": size}
        reqs.append(req)
    return reqs


def validate_requests(reqs: list[dict]) -> None:
    """Validate against the botocore model of the Batch API, plus the array-size rule."""
    import botocore.session
    from botocore.validate import ParamValidator

    shape = botocore.session.get_session().get_service_model("batch").operation_model("SubmitJob").input_shape
    for r in reqs:
        report = ParamValidator().validate(r, shape)
        if report.has_errors():
            raise ValueError(report.generate_report())
        size = r.get("arrayProperties", {}).get("size")
        if size is not None and not 2 <= size <= MAX_ARRAY:
            raise ValueError(f"array size {size} outside 2..{MAX_ARRAY}")


def resolve_index(plan: dict, env: dict) -> str:
    """What a worker does: offset + array index (absent for a plain job) -> stable segment id."""
    idx = int(env.get("DSGATE_PLAN_OFFSET", "0")) + int(env.get("AWS_BATCH_JOB_ARRAY_INDEX", "0"))
    return plan["segment_ids"][idx]


def receipt_key(analysis_digest: str, segment_id: str) -> str:
    return f"receipts/{short(analysis_digest)}/{segment_id}.json"


def remaining(store: Store, submission: dict, analysis_digest: str) -> list[str]:
    done = {k.rsplit("/", 1)[1][:-5] for k in store.keys(f"receipts/{short(analysis_digest)}/")}
    return [s["segment_id"] for s in submission["segments"] if s["segment_id"] not in done]


# --------------------------------------------------------------- local runner

def run_local(reqs: list[dict], base_env: dict, parallel: int = 4, reverse: bool = False,
              kill_before_receipt: set[str] | None = None, plan: dict | None = None,
              log=print) -> list[dict]:
    """Run every child of every request as a separate worker process.

    The environment matches what Batch sets: the overrides of the request and,
    for array jobs only, AWS_BATCH_JOB_ARRAY_INDEX.
    """
    children = []
    for r in reqs:
        env = {e["name"]: e["value"] for e in r["containerOverrides"]["environment"]}
        size = r.get("arrayProperties", {}).get("size")
        for i in range(size or 1):
            child_env = dict(env)
            if size:
                child_env["AWS_BATCH_JOB_ARRAY_INDEX"] = str(i)
            children.append((r["jobName"], i if size else None, child_env))
    if reverse:
        children.reverse()

    def run(child):
        job, idx, env = child
        full = {**os.environ, **base_env, **env}
        seg = resolve_index(plan, env) if plan else None
        if kill_before_receipt and seg in kill_before_receipt:
            full["DSGATE_KILL_BEFORE_RECEIPT"] = seg
        proc = subprocess.run([sys.executable, "-m", "dsgate.worker"], env=full, capture_output=True, text=True)
        out = {"job": job, "index": idx, "segment_id": seg, "exit": proc.returncode,
               "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()[-2000:]}
        return out

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        results = list(pool.map(run, children))
    results.sort(key=lambda r: (r["job"], -1 if r["index"] is None else r["index"]))
    return results


def summarize(reqs: list[dict], results: list[dict]) -> list[str]:
    lines = []
    for r in reqs:
        mine = [x for x in results if x["job"] == r["jobName"]]
        failed = [x for x in mine if x["exit"] != 0]
        kind = f"array size {r['arrayProperties']['size']}" if "arrayProperties" in r else "plain job (1 segment)"
        status = "FAILED" if failed else "SUCCEEDED"
        lines.append(f"{r['jobName']}  {kind}: {len(mine) - len(failed)} succeeded, {len(failed)} failed -> parent {status}")
        for x in failed:
            why = "killed (SIGKILL)" if x["exit"] in (-9, 137) else f"exit {x['exit']}"
            lines.append(f"    child {x['index']} -> {x['segment_id']}: {why}")
    return lines


def load_json(store: Store, key: str) -> dict:
    return json.loads(store.get(key))
