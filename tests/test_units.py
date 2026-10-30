"""Unit and contract tests: S3 emulator semantics, Batch rules, time, identity, workers."""

import json
import random
import sqlite3
from pathlib import Path

import pytest

from dsgate import CAMERA, LIDAR, synth
from dsgate.batch import (MAX_ARRAY, build_submission, make_plan, receipt_key, resolve_index, submit_requests,
                          upload_bags, upload_side, validate_requests)
from dsgate.bagread import read_segment
from dsgate.canon import cjson, digest
from dsgate.checks import analyze, decide
from dsgate.publish import analysis_digest, dedup, identity_body, version_id
from dsgate.provenance import code_digest
from dsgate.store import Conflict, Store
from dsgate.worker import process

RECIPE = json.loads((Path(__file__).parents[1] / "config" / "recipe_v2.json").read_text())
CAP = synth.CaptureSpec("tractor-09", "2026-10-20T10:00:00Z")


def bag(tmp_path, *faults, storage="mcap", name=None, start=0):
    seg = synth.SegmentSpec(start, start + 10, storage, tuple(faults))
    return synth.write_segment(tmp_path, CAP, seg, name=name)


# ---------------------------------------------------------------- S3 emulator

def test_emulator_rejects_a_second_conditional_put(s3):
    import boto3
    from botocore.exceptions import ClientError

    c = boto3.client("s3", endpoint_url=s3["DSGATE_S3_ENDPOINT"], region_name="us-east-1")
    c.create_bucket(Bucket="cond-raw")
    c.put_object(Bucket="cond-raw", Key="k", Body=b"first", IfNoneMatch="*")
    with pytest.raises(ClientError) as err:
        c.put_object(Bucket="cond-raw", Key="k", Body=b"second", IfNoneMatch="*")
    assert err.value.response["ResponseMetadata"]["HTTPStatusCode"] == 412
    assert c.get_object(Bucket="cond-raw", Key="k")["Body"].read() == b"first"


def test_put_immutable_same_bytes_ok_different_bytes_conflict(s3):
    st = Store("cond-store")
    st.create_bucket()
    assert st.put_immutable("a", b"x") == "created"
    assert st.put_immutable("a", b"x") == "exists_same"
    with pytest.raises(Conflict):
        st.put_immutable("a", b"y")


# ----------------------------------------------------------------- Batch rules

def plan_of(n):
    return {"plan_id": "plan-test", "key": "plans/plan-test.json", "segment_ids": [f"seg-{i:06d}" for i in range(n)]}


@pytest.mark.parametrize("n,expected", [(0, []), (1, [None]), (2, [2]), (MAX_ARRAY, [MAX_ARRAY]),
                                        (MAX_ARRAY + 1, [MAX_ARRAY, None]), (25_000, [MAX_ARRAY, MAX_ARRAY, 5000])])
def test_submit_requests_respect_array_limits(n, expected):
    reqs = submit_requests(plan_of(n), "bucket")
    assert [r.get("arrayProperties", {}).get("size") for r in reqs] == expected
    validate_requests(reqs)


def test_validation_rejects_an_array_of_one():
    req = submit_requests(plan_of(2), "bucket")[0]
    req["arrayProperties"]["size"] = 1
    with pytest.raises(ValueError, match="outside 2..10000"):
        validate_requests([req])


def test_every_child_resolves_to_a_distinct_stable_id():
    plan = plan_of(MAX_ARRAY + 1)
    seen = []
    for r in submit_requests(plan, "bucket"):
        env = {e["name"]: e["value"] for e in r["containerOverrides"]["environment"]}
        size = r.get("arrayProperties", {}).get("size")
        for i in range(size or 1):
            child = dict(env)
            if size:
                child["AWS_BATCH_JOB_ARRAY_INDEX"] = str(i)
            seen.append(resolve_index(plan, child))
    assert seen == plan["segment_ids"]


def test_recomputing_missing_per_child_loses_segments_a_frozen_plan_does_not():
    """6 segments, s1 and s4 done. Children that each recompute `missing[index]` while siblings finish
    skip segments; a frozen, compacted plan maps every index to one stable id."""
    done = {"s1", "s4"}
    all_ids = [f"s{i}" for i in range(6)]
    built = set()
    for index in range(4):  # naive: each child re-lists what is missing when it starts
        missing = [s for s in all_ids if s not in done | built]
        if index < len(missing):
            built.add(missing[index])
    assert built != set(all_ids) - done
    frozen = [s for s in all_ids if s not in done]
    plan = {"segment_ids": frozen}
    assert {resolve_index(plan, {"AWS_BATCH_JOB_ARRAY_INDEX": str(i)}) for i in range(4)} == set(frozen)


# --------------------------------------------------------------------- time

def test_arrival_delay_alone_is_not_an_acquisition_skew(tmp_path):
    seg = read_segment(bag(tmp_path, synth.Fault("rec_delay", CAMERA, value=60)))
    findings, info, _ = analyze(seg, RECIPE)
    assert "acquisition_skew" not in {f["rule"] for f in findings}
    assert abs(info["acquisition_skew_ms"]) < 2
    assert {f["rule"] for f in findings} == {"recording_delay"} and decide(findings) == "retained"


def test_acquisition_skew_with_aligned_arrival_is_detected(tmp_path):
    seg = read_segment(bag(tmp_path, synth.Fault("acq_skew", CAMERA, value=18)))
    findings, info, _ = analyze(seg, RECIPE)
    assert {f["rule"] for f in findings} == {"acquisition_skew"}
    assert 17 <= info["acquisition_skew_ms"] <= 19


def test_incomparable_clocks_are_not_evaluable(tmp_path):
    seg = read_segment(bag(tmp_path, synth.Fault("boot_clock", CAMERA, value=3600)))
    findings, info, pairs = analyze(seg, RECIPE)
    assert {f["rule"] for f in findings} == {"clock_not_comparable"}
    assert info["acquisition_skew"] == "not evaluable" and pairs == []


# ----------------------------------------------------------------- identity

def _records(seg):
    _, _, pairs = analyze(seg, RECIPE)
    cam, lid = seg.streams[CAMERA], seg.streams[LIDAR]
    return [(p.lidar_stamp, p.camera_stamp, cam.content[p.camera_idx], lid.content[p.lidar_idx]) for p in pairs]


def test_same_recording_in_sqlite3_and_mcap_has_the_same_logical_content(tmp_path):
    a = read_segment(bag(tmp_path / "a", storage="sqlite3"))
    b = read_segment(bag(tmp_path / "b", storage="mcap"))
    assert a.storage == "sqlite3" and b.storage == "mcap"
    assert _records(a) == _records(b) and a.schemas == b.schemas
    raw_a = next(p for p in (tmp_path / "a").rglob("*.db3")).read_bytes()
    raw_b = next(p for p in (tmp_path / "b").rglob("*.mcap")).read_bytes()
    assert raw_a != raw_b  # different source integrity, same logical content


def _sample(cap="c/1", t=1, cam="a", lid="b", label=10, split="train", seg="seg-1"):
    return {"capture_id": cap, "ref_stamp_ns": t, "cam_stamp_ns": t, "cam_sha256": cam, "lidar_sha256": lid,
            "label_mm": label, "split": split, "src": {"segment_id": seg}}


def test_identity_ignores_order_and_changes_on_meaningful_mutation():
    base = [_sample(t=i, cam=f"c{i}") for i in range(20)]
    vid = version_id(identity_body(base, {}, {}))
    shuffled = base[:]
    random.Random(1).shuffle(shuffled)
    assert version_id(identity_body(shuffled, {}, {})) == vid
    for field, value in (("label_mm", 11), ("cam_sha256", "zz"), ("split", "val")):
        mutated = [dict(s) for s in base]
        mutated[3][field] = value
        assert version_id(identity_body(mutated, {}, {})) != vid
    assert version_id(identity_body(base, {}, {"tractor-01": "cal2"})) != vid
    assert "git" not in json.dumps(identity_body(base, {}, {}))


def test_dedup_rules():
    same = [_sample(seg="seg-b"), _sample(seg="seg-a")]
    kept, dup_of, possible = dedup(same)
    assert len(kept) == 1 and kept[0]["src"]["segment_id"] == "seg-a" and dup_of["seg-b"] == {"seg-a"}
    differ = [_sample(seg="seg-a"), _sample(cam="other", seg="seg-b")]
    kept, _, possible = dedup(differ)
    assert len(kept) == 2 and possible[0]["segments"] == ["seg-a", "seg-b"]  # flagged, never dropped
    elsewhere = [_sample(cap="c/1"), _sample(cap="c/2"), _sample(cap="c/1", t=2)]
    assert len(dedup(elsewhere)[0]) == 3  # identical content, other capture or instant: kept


def test_code_digest_covers_module_sources_only(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "a.py").write_text("x = 1\n")
    d1 = code_digest(pkg)
    (tmp_path / "README.md").write_text("edited")
    (pkg / "notes.md").write_text("not code")
    assert code_digest(pkg) == d1
    (pkg / "a.py").write_text("x = 2\n")
    assert code_digest(pkg) != d1


# ------------------------------------------------------- independent readers

def test_message_counts_agree_with_independent_readers(tmp_path):
    from mcap.reader import make_reader

    for storage in ("mcap", "sqlite3"):
        b = bag(tmp_path / storage, synth.Fault("gap", LIDAR, at_s=3.05, dur_s=0.6), storage=storage)
        seg = read_segment(b)
        ours = {t: len(s.log_ns) for t, s in seg.streams.items()}
        data = next(p for p in b.iterdir() if p.suffix in (".mcap", ".db3"))
        if storage == "mcap":
            theirs = {}
            with open(data, "rb") as fh:
                for _, channel, _ in make_reader(fh).iter_messages():
                    theirs[channel.topic] = theirs.get(channel.topic, 0) + 1
        else:
            con = sqlite3.connect(data)
            theirs = dict(con.execute("SELECT t.name, count(*) FROM messages m JOIN topics t ON m.topic_id = t.id "
                                      "GROUP BY t.name").fetchall())
        assert ours == theirs
        # expected counts computed from the timeline, not from any reader
        assert ours[LIDAR] == 100 - 6 and ours[CAMERA] == 200


# -------------------------------------------------------------------- worker

def _one_segment_setup(tmp_path, s3, bucket, *faults, damage=None):
    seg = synth.SegmentSpec(0, 10, "mcap", tuple(faults), damage=damage)
    path = synth.write_segment(tmp_path / "bags", CAP, seg)
    synth.write_labels(tmp_path / "side", CAP)
    synth.write_calibration(tmp_path / "side", CAP.vehicle_id)
    st = Store(bucket)
    st.create_bucket()
    side = upload_side(st, tmp_path / "side")
    sub = build_submission(st, upload_bags(st, [path], tmp_path / "side"), side)
    rkey = "recipes/test.json"
    st.put_immutable(rkey, cjson(RECIPE))
    plan = make_plan(st, sub, rkey, analysis_digest(RECIPE), [s["segment_id"] for s in sub["segments"]])
    return st, sub, plan, plan["segment_ids"][0]


def test_damaged_bag_is_quarantined_with_exit_0(tmp_path, s3):
    st, sub, plan, sid = _one_segment_setup(tmp_path, s3, "w-quar", damage="truncate")
    assert process(st, plan, sid, log=lambda *_: None) == 0
    r = json.loads(st.get(receipt_key(plan["analysis_digest"], sid)))
    assert r["status"] == "quarantined" and r["quarantine"]["kind"] == "mcap_truncated"
    assert r["quarantine"]["original"][0]["version_id"]


def test_overwritten_source_is_read_at_its_frozen_version(tmp_path, s3):
    st, sub, plan, sid = _one_segment_setup(tmp_path, s3, "w-frozen")
    data_key = next(f["key"] for f in sub["segments"][0]["files"] if f["key"].endswith(".mcap"))
    st.put(data_key, b"replaced after submission")  # new object version under the same key
    assert process(st, plan, sid, log=lambda *_: None) == 0
    assert json.loads(st.get(receipt_key(plan["analysis_digest"], sid)))["status"] == "retained"


def test_source_not_matching_submission_fails_the_worker(tmp_path, s3):
    st, sub, plan, sid = _one_segment_setup(tmp_path, s3, "w-integrity")
    bad = json.loads(json.dumps(sub))
    bad["segments"][0]["files"][0]["sha256"] = "0" * 64
    bad["submission_id"] = "sub-tampered"
    st.put_immutable("submissions/sub-tampered.json", cjson(bad))
    plan2 = dict(plan, submission_id="sub-tampered")
    assert process(st, plan2, sid, log=lambda *_: None) == 3


def test_existing_receipt_with_other_content_is_a_conflict(tmp_path, s3):
    st, sub, plan, sid = _one_segment_setup(tmp_path, s3, "w-conflict")
    st.put_immutable(receipt_key(plan["analysis_digest"], sid), b'{"status": "retained", "forged": true}')
    assert process(st, plan, sid, log=lambda *_: None) == 4


def test_rerun_after_success_is_byte_identical(tmp_path, s3):
    st, sub, plan, sid = _one_segment_setup(tmp_path, s3, "w-rerun")
    logs = []
    assert process(st, plan, sid, log=logs.append) == 0
    assert process(st, plan, sid, log=logs.append) == 0
    assert "receipt exists_same" in logs[-1] and "created 0" in logs[-1]


def test_byte_identical_upload_collapses_at_submission(tmp_path, s3):
    path = synth.write_segment(tmp_path / "bags", CAP, synth.SegmentSpec(0, 10, "mcap"))
    st = Store("w-exact")
    st.create_bucket()
    first = upload_bags(st, [path], tmp_path)
    copy = tmp_path / "copy" / CAP.vehicle_id / path.name
    copy.parent.mkdir(parents=True)
    import shutil

    shutil.copytree(path, copy)
    second = upload_bags(st, [copy], tmp_path)
    second[0]["bag"] = "tractor-09/uploaded-again"
    sub = build_submission(st, first + second, {"labels": {}, "calibration": {}})
    assert len(sub["segments"]) == 1
    assert sub["exact_reuploads"] == [{"segment_id": sub["segments"][0]["segment_id"], "bag": "tractor-09/uploaded-again",
                                       "same_bytes_as": first[0]["bag"]}]
