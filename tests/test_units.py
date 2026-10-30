"""Unit and contract tests: S3 emulator semantics, Batch rules, streams and time, decoding, identity, workers."""

import copy
import json
import random
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from dsgate import CAMERA, LIDAR, ODOM, synth
from dsgate.batch import (MAX_ARRAY, build_submission, make_plan, receipt_key, remaining, resolve_index,
                          submit_requests, upload_bags, upload_side, validate_requests)
from dsgate.bagread import TS, Unreadable, decode_cloud, decode_image, read_segment
from dsgate.canon import CANONICAL_NAN, cjson, digest, points_hash
from dsgate.checks import analyze, decide
from dsgate.provenance import build_provenance, code_digest
from dsgate.publish import (admission_gates, analysis_digest, assign_splits, dedup, identity_body, publish,
                            version_id)
from dsgate.recipe import RecipeError, analysis_part, validate
from dsgate.store import Conflict, Store
from dsgate.worker import process

RECIPE = json.loads((Path(__file__).parents[1] / "config" / "recipe_v2.json").read_text())
REQ = RECIPE["required_topics"]
CAP = synth.CaptureSpec("tractor-09", "2026-10-20T10:00:00Z")
T = TS.types


def bag(tmp_path, *faults, storage="mcap", name=None, start=0):
    seg = synth.SegmentSpec(start, start + 10, storage, tuple(faults))
    return synth.write_segment(tmp_path, CAP, seg, name=name)


def rules(findings):
    return sorted(f["rule"] for f in findings)


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


def test_existing_object_is_compared_by_sha256_not_etag(s3):
    """A multipart object's ETag is not the MD5 of its bytes: comparing ETags would call it a conflict."""
    import boto3

    c = boto3.client("s3", endpoint_url=s3["DSGATE_S3_ENDPOINT"], region_name="us-east-1")
    c.create_bucket(Bucket="cond-mpu")
    body = b"a" * (5 * 1024 * 1024) + b"tail"
    up = c.create_multipart_upload(Bucket="cond-mpu", Key="big")
    parts = []
    for n, chunk in enumerate((body[:5 * 1024 * 1024], body[5 * 1024 * 1024:]), start=1):
        etag = c.upload_part(Bucket="cond-mpu", Key="big", UploadId=up["UploadId"], PartNumber=n, Body=chunk)["ETag"]
        parts.append({"ETag": etag, "PartNumber": n})
    c.complete_multipart_upload(Bucket="cond-mpu", Key="big", UploadId=up["UploadId"],
                                MultipartUpload={"Parts": parts})
    assert "-" in c.head_object(Bucket="cond-mpu", Key="big")["ETag"]
    assert Store("cond-mpu").put_immutable("big", body) == "exists_same"


def test_409_is_retried_once():
    from botocore.exceptions import ClientError

    class Flaky:
        calls = 0

        def put_object(self, **kw):
            Flaky.calls += 1
            if Flaky.calls == 1:
                raise ClientError({"Error": {"Code": "ConditionalRequestConflict"},
                                   "ResponseMetadata": {"HTTPStatusCode": 409}}, "PutObject")
            return {}

    assert Store("any", s3=Flaky()).put_immutable("k", b"x") == "created" and Flaky.calls == 2


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
    for index in range(4):
        missing = [s for s in all_ids if s not in done | built]
        if index < len(missing):
            built.add(missing[index])
    assert built != set(all_ids) - done
    frozen = [s for s in all_ids if s not in done]
    plan = {"segment_ids": frozen}
    assert {resolve_index(plan, {"AWS_BATCH_JOB_ARRAY_INDEX": str(i)}) for i in range(4)} == set(frozen)


# ------------------------------------------------------------------ recipe

@pytest.mark.parametrize("edit,match", [
    (lambda r: r.update(extra=1), "unknown"),
    (lambda r: r["time_source"].update(acquisition="log_time"), "not implemented"),
    (lambda r: r["pairing"].update(match="/odom"), "pairing.match"),
    (lambda r: r["required_topics"][CAMERA].update(msgtype="sensor_msgs/msg/CompressedImage"), "not supported"),
    (lambda r: r["thresholds"].update(max_acq_skew_ms=10), "unknown"),
    (lambda r: r["policy"].update(stream_gap="drop"), "not in"),
])
def test_recipe_refuses_what_the_code_would_ignore(edit, match):
    r = copy.deepcopy(RECIPE)
    edit(r)
    with pytest.raises(RecipeError, match=match):
        validate(r)


def test_topics_come_from_the_recipe(tmp_path):
    b = bag(tmp_path)
    r = copy.deepcopy(RECIPE)
    r["required_topics"]["/sensing/front/image"] = r["required_topics"].pop(CAMERA)
    r["pairing"]["match"] = "/sensing/front/image"
    findings, _, pairs = analyze(read_segment(b, r["required_topics"]), r)
    assert rules(findings) == ["required_stream_absent"] and pairs == []


# --------------------------------------------------------- streams coverage

@pytest.mark.parametrize("faults,expected", [
    ((synth.Fault("gap", LIDAR, at_s=5.0, dur_s=10),), {"stream_gap"}),       # lidar stops at 5 s
    ((synth.Fault("gap", LIDAR, at_s=0.0, dur_s=6.0),), {"stream_gap"}),      # lidar starts at 6 s
    ((synth.Fault("gap", LIDAR, at_s=0.05, dur_s=10),), {"stream_insufficient"}),  # one sweep only
    ((synth.Fault("gap", CAMERA, at_s=1.0, dur_s=10),), {"stream_gap", "low_pairing"}),  # camera for 1 s
    ((synth.Fault("gap", ODOM, at_s=0.02, dur_s=10),), {"stream_insufficient"}),  # one odometry message
])
def test_a_stream_must_cover_the_segment_window(tmp_path, faults, expected):
    findings, _, _ = analyze(read_segment(bag(tmp_path, *faults), REQ), RECIPE)
    assert set(rules(findings)) == expected and decide(findings) == "excluded"


def test_every_gap_is_its_own_finding(tmp_path):
    b = bag(tmp_path, synth.Fault("gap", LIDAR, at_s=2.05, dur_s=0.6), synth.Fault("gap", LIDAR, at_s=6.05, dur_s=0.5))
    findings, _, _ = analyze(read_segment(b, REQ), RECIPE)
    gaps = [f for f in findings if f["rule"] == "stream_gap"]
    assert len(gaps) == 2 and [g["measure"][:4] for g in gaps] == ["0.70", "0.60"]
    assert "low_rate" not in rules(findings)  # gaps are left out of the active rate


# --------------------------------------------------------------------- time

def test_recording_delay_alone_is_not_a_pair_residual(tmp_path):
    seg = read_segment(bag(tmp_path, synth.Fault("rec_delay", CAMERA, value=60)), REQ)
    findings, info, _ = analyze(seg, RECIPE)
    assert rules(findings) == ["recording_offset"] and decide(findings) == "retained"
    assert abs(info["pair_residual_ms"]["median"]) < 2
    assert "agree" not in findings[0]["measure"]


@pytest.mark.parametrize("offset,measured,flagged", [
    (18, 18, True), (20, 20, True),
    (30, -20, True),    # beyond half a camera period: reads as -20 ms
    (-40, 10, False),   # reads as +10 ms, not above the 10 ms threshold
    (50, 0, False),     # a whole camera period: invisible from timestamps
])
def test_pair_residual_is_read_modulo_the_camera_period(tmp_path, offset, measured, flagged):
    seg = read_segment(bag(tmp_path, synth.Fault("acq_skew", CAMERA, value=offset)), REQ)
    findings, info, _ = analyze(seg, RECIPE)
    assert abs(info["pair_residual_ms"]["median"] - measured) <= 1.5
    assert ("pair_residual" in rules(findings)) == flagged


def test_incomparable_clocks_are_not_evaluable(tmp_path):
    seg = read_segment(bag(tmp_path, synth.Fault("boot_clock", CAMERA, value=3600)), REQ)
    findings, info, pairs = analyze(seg, RECIPE)
    assert rules(findings) == ["clock_not_comparable"]
    assert info["pair_residual"] == "not evaluable" and pairs == []


# ----------------------------------------------------------------- decoding

def _roundtrip(msg, msgtype):
    return TS.deserialize_cdr(TS.serialize_cdr(msg, msgtype), msgtype)


def _header():
    return T["std_msgs/msg/Header"](stamp=T["builtin_interfaces/msg/Time"](sec=1, nanosec=0), frame_id="f")


def test_image_rows_are_read_with_their_step():
    data = np.array([1, 2, 3, 4, 5, 6, 0, 0, 7, 8, 9, 10, 11, 12, 0, 0], dtype=np.uint8)
    msg = _roundtrip(T["sensor_msgs/msg/Image"](header=_header(), height=2, width=2, encoding="rgb8",
                                                is_bigendian=0, step=8, data=data), "sensor_msgs/msg/Image")
    assert decode_image(msg).reshape(-1).tolist() == list(range(1, 13))


def test_organized_cloud_with_row_padding_and_extra_fields():
    pf = T["sensor_msgs/msg/PointField"]
    fields = [pf(name="ring", offset=0, datatype=4, count=1), pf(name="x", offset=4, datatype=7, count=1),
              pf(name="y", offset=8, datatype=7, count=1), pf(name="z", offset=12, datatype=7, count=1),
              pf(name="intensity", offset=16, datatype=7, count=1)]
    point = lambda v: np.array([7], dtype="<u2").tobytes() + b"\0\0" + np.array([v] * 4, dtype="<f4").tobytes()
    rows = [point(np.nan) + b"\0" * 12, point(np.nan) + b"\0" * 12]  # point_step 20, row_step 32
    msg = _roundtrip(T["sensor_msgs/msg/PointCloud2"](
        header=_header(), height=2, width=1, fields=fields, is_bigendian=False, point_step=20, row_step=32,
        data=np.frombuffer(b"".join(rows), dtype=np.uint8).copy(), is_dense=False), "sensor_msgs/msg/PointCloud2")
    pts = decode_cloud(msg)
    assert pts.shape == (2, 4) and not np.isfinite(pts[:, :3]).any()  # 0 % valid, padding not read as points


def test_cloud_without_float32_xyz_is_refused():
    pf = T["sensor_msgs/msg/PointField"]
    msg = T["sensor_msgs/msg/PointCloud2"](header=_header(), height=1, width=1,
                                           fields=[pf(name="x", offset=0, datatype=8, count=1)], is_bigendian=False,
                                           point_step=16, row_step=16, data=np.zeros(16, np.uint8), is_dense=True)
    with pytest.raises(Unreadable, match="schema_unsupported"):
        decode_cloud(msg)


def test_point_count_varies_between_sweeps(tmp_path):
    seg = read_segment(bag(tmp_path), REQ)
    assert len({len(p) for p in seg.streams[LIDAR].payload}) > 1


@pytest.mark.parametrize("damage,kind", [("missing", "metadata_unreadable"), ("empty", "metadata_unreadable"),
                                         ("half", "metadata_unreadable")])
def test_damaged_metadata_is_quarantined_not_a_crash(tmp_path, damage, kind):
    b = bag(tmp_path, storage="sqlite3")
    meta = b / "metadata.yaml"
    if damage == "missing":
        meta.unlink()
    elif damage == "empty":
        meta.write_text("")
    else:
        meta.write_text(meta.read_text()[: len(meta.read_text()) // 2] + "\n  - [unclosed")
    with pytest.raises(Unreadable) as err:
        read_segment(b, REQ)
    assert err.value.kind == kind and str(tmp_path.resolve()) not in err.value.detail and str(tmp_path) not in err.value.detail


@pytest.mark.parametrize("storage", ["mcap", "sqlite3"])
def test_one_truncated_message_payload_is_quarantined(tmp_path, storage):
    from rosbags.rosbag2 import StoragePlugin, Writer

    p = tmp_path / storage / "tractor-09" / "bag_corrupt_msg"
    p.parent.mkdir(parents=True)
    msgs = synth.segment_messages(CAP, synth.SegmentSpec(0, 10, storage))
    plugin = StoragePlugin.MCAP if storage == "mcap" else StoragePlugin.SQLITE3
    with Writer(p, version=9, storage_plugin=plugin) as w:
        w.set_custom_data("capture_id", CAP.capture_id)
        w.set_custom_data("vehicle_id", CAP.vehicle_id)
        conns = {t: w.add_connection(t, synth.TYPES[t], typestore=synth.TS) for t in (CAMERA, LIDAR, ODOM)}
        cut = False
        for i, (log, t, data) in enumerate(msgs):
            if t == CAMERA and not cut and i > 100:
                data, cut = data[:40], True
            w.write(conns[t], log, data)
    with pytest.raises(Unreadable, match="message_malformed"):
        read_segment(p, REQ)


def test_divergent_message_definition_is_refused(tmp_path):
    from rosbags.rosbag2 import StoragePlugin, Writer
    from rosbags.typesys import Stores, get_types_from_msg, get_typestore

    defs = dict(TS.fielddefs)
    defs.pop("sensor_msgs/msg/Image")
    ts2 = get_typestore(Stores.EMPTY)
    ts2.register(defs)
    ts2.register(get_types_from_msg("std_msgs/Header header\nuint64 height\nuint32 width\nstring encoding\n"
                                    "uint8 is_bigendian\nuint32 step\nuint8[] data\n", "sensor_msgs/msg/Image"))
    p = tmp_path / "tractor-09" / "bag_custom_image"
    p.parent.mkdir(parents=True)
    with Writer(p, version=9, storage_plugin=StoragePlugin.MCAP) as w:
        w.set_custom_data("capture_id", CAP.capture_id)
        w.set_custom_data("vehicle_id", CAP.vehicle_id)
        c = w.add_connection(CAMERA, "sensor_msgs/msg/Image", typestore=ts2)
        t2 = ts2.types
        m = t2["sensor_msgs/msg/Image"](header=t2["std_msgs/msg/Header"](
            stamp=t2["builtin_interfaces/msg/Time"](sec=1, nanosec=0), frame_id="c"), height=2, width=2,
            encoding="rgb8", is_bigendian=0, step=6, data=np.zeros(12, np.uint8))
        w.write(c, 1_000_000_000, ts2.serialize_cdr(m, "sensor_msgs/msg/Image"))
    with pytest.raises(Unreadable, match="schema_mismatch"):
        read_segment(p, REQ)


def test_truncated_mcap_error_carries_no_temporary_path(tmp_path, s3):
    st, sub, plan, sid = _setup(tmp_path, "w-path", damage="truncate")
    assert process(st, plan, sid, log=lambda *_: None) == 0
    q = json.loads(st.get(receipt_key(sub, sid, plan["analysis_digest"], plan["worker"])))["quarantine"]
    assert q["kind"] == "mcap_truncated" and "/" not in q["error"] and "tmp" not in q["error"].lower()


# ----------------------------------------------------------------- identity

def _records(seg):
    _, _, pairs = analyze(seg, RECIPE)
    cam, lid = seg.streams[CAMERA], seg.streams[LIDAR]
    return [(p.lidar_stamp, p.camera_stamp, cam.content[p.camera_idx], lid.content[p.lidar_idx]) for p in pairs]


def test_same_recording_in_sqlite3_and_mcap_has_the_same_logical_content(tmp_path):
    b_sql = bag(tmp_path / "a", synth.Fault("nan", LIDAR, at_s=3.0, dur_s=0.3, value=0.2), storage="sqlite3")
    b_mc = bag(tmp_path / "b", synth.Fault("nan", LIDAR, at_s=3.0, dur_s=0.3, value=0.2), storage="mcap")
    a, b = read_segment(b_sql, REQ), read_segment(b_mc, REQ)
    assert a.storage == "sqlite3" and b.storage == "mcap"
    assert _records(a) == _records(b) and a.schemas == b.schemas
    assert next(b_sql.glob("*.db3")).read_bytes() != next(b_mc.glob("*.mcap")).read_bytes()


def test_nan_payloads_hash_canonically():
    pts = np.zeros((3, 4), dtype="<f4")
    other_nan = np.frombuffer(np.uint32(0x7FC00001).tobytes(), dtype="<f4")[0]
    a, b = pts.copy(), pts.copy()
    a[1, 0], b[1, 0] = CANONICAL_NAN, other_nan
    assert a.tobytes() != b.tobytes() and points_hash(a) == points_hash(b)


def _sample(cap="c/1", t=1, cam="a", lid="b", label=10, split="train", seg="seg-1", cam_t=None, start=0):
    return {"capture_id": cap, "ref_stamp_ns": t, "cam_stamp_ns": t if cam_t is None else cam_t, "cam_sha256": cam,
            "lidar_sha256": lid, "label_mm": label, "split": split, "src": {"segment_id": seg}, "start_log_ns": start}


def test_identity_ignores_order_and_changes_on_meaningful_mutation():
    base = [_sample(t=i, cam=f"c{i}") for i in range(20)]
    vid = version_id(identity_body(base, {}, {}))
    shuffled = base[:]
    random.Random(1).shuffle(shuffled)
    assert version_id(identity_body(shuffled, {}, {})) == vid
    for field, value in (("label_mm", 11), ("cam_sha256", "zz"), ("split", "val"), ("cam_stamp_ns", 99)):
        mutated = [dict(s) for s in base]
        mutated[3][field] = value
        assert version_id(identity_body(mutated, {}, {})) != vid
    assert version_id(identity_body(base, {}, {"tractor-01": "cal2"})) != vid
    assert version_id(identity_body(base, {CAMERA: {"msgtype": "x", "rihs01": "y"}}, {})) != vid
    assert "git" not in json.dumps(identity_body(base, {}, {}))


def test_dedup_keeps_one_per_variant_and_flags_contradictions():
    kept, dup_of, possible = dedup([_sample(seg="seg-b"), _sample(seg="seg-a")])
    assert len(kept) == 1 and dup_of["seg-b"] == {"seg-a"} and not possible
    kept, dup_of, possible = dedup([_sample(seg="seg-a"), _sample(seg="seg-c"), _sample(cam="other", seg="seg-b")])
    assert len(kept) == 2 and possible[0]["variants"] == 2  # A, A, B -> two observations, flagged
    for field in ({"cam_t": 5}, {"label": 11}):
        kept, _, possible = dedup([_sample(seg="seg-a"), _sample(seg="seg-b", **field)])
        assert len(kept) == 2 and possible, field
    assert len(dedup([_sample(cap="c/1"), _sample(cap="c/2"), _sample(cap="c/1", t=2)])[0]) == 3


def test_dedup_prefers_what_is_already_published_then_the_earliest_segment():
    original, reupload = _sample(seg="seg-ffff", start=10), _sample(seg="seg-0000", start=14)
    kept, dup_of, _ = dedup([original, reupload], prefer={"seg-ffff"})
    assert kept[0]["src"]["segment_id"] == "seg-ffff" and dup_of["seg-0000"] == {"seg-ffff"}
    kept, _, _ = dedup([_sample(seg="seg-ffff", start=10), _sample(seg="seg-0000", start=14)])
    assert kept[0]["src"]["segment_id"] == "seg-ffff"


def test_previous_split_registry_wins_over_a_new_salt():
    kept = [_sample(cap=f"cap-{i}") for i in range(8)]
    first = assign_splits(kept, {"unit": "capture", "val_fraction": 0.5, "salt": "s1"}, None)
    again = [_sample(cap=f"cap-{i}") for i in range(8)]
    second = assign_splits(again, {"unit": "capture", "val_fraction": 0.5, "salt": "other"},
                           {"split_registry": first})
    assert second == first and [s["split"] for s in again] == [first[s["capture_id"]] for s in again]


def test_retained_fraction_gate_fails_below_its_minimum():
    count = {"retained": 2, "excluded": 8, "quarantined": 0, "deduplicated": 0, "no_receipt": 0}
    gates = admission_gates(count, 10, [], [], {"train": 100, "val": 100}, RECIPE["admission"])
    assert {g["gate"]: g["ok"] for g in gates}["retained_fraction"] is False


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
        seg = read_segment(b, REQ)
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
        assert ours[LIDAR] == 100 - 6 and ours[CAMERA] == 200


# -------------------------------------------------------------------- worker

def _setup(tmp_path, bucket, *faults, damage=None, declare=None, custom=True):
    cap = CAP if custom else synth.CaptureSpec(CAP.vehicle_id, CAP.start_utc, declared_identity=True)
    seg = synth.SegmentSpec(0, 10, "mcap", tuple(faults), damage=damage)
    path = synth.write_segment(tmp_path / "bags", cap, seg)
    synth.write_labels(tmp_path / "side", CAP)
    synth.write_calibration(tmp_path / "side", CAP.vehicle_id)
    st = Store(bucket)
    st.create_bucket()
    side = upload_side(st, tmp_path / "side")
    decl = {f"{path.parent.name}/{path.name}": {"capture_id": CAP.capture_id, "vehicle_id": CAP.vehicle_id}} \
        if declare else None
    sub = build_submission(st, upload_bags(st, [path], decl), side)
    ad = analysis_digest(RECIPE)
    akey = "analyses/test.json"
    st.put_immutable(akey, cjson(analysis_part(RECIPE)))
    plan = make_plan(st, sub, akey, ad, [s["segment_id"] for s in sub["segments"]], build_provenance())
    return st, sub, plan, plan["segment_ids"][0]


def _receipt(st, sub, plan, sid):
    return json.loads(st.get(receipt_key(sub, sid, plan["analysis_digest"], plan["worker"])))


def test_damaged_bag_is_quarantined_with_exit_0(tmp_path, s3):
    st, sub, plan, sid = _setup(tmp_path, "w-quar", damage="truncate")
    assert process(st, plan, sid, log=lambda *_: None) == 0
    r = _receipt(st, sub, plan, sid)
    assert r["status"] == "quarantined" and r["quarantine"]["kind"] == "mcap_truncated"
    assert r["quarantine"]["original"][0]["version_id"]


def test_capture_identity_declared_at_upload_or_quarantined(tmp_path, s3):
    st, sub, plan, sid = _setup(tmp_path / "a", "w-noident", custom=False)
    assert sub["segments"][0]["capture_id"] is None
    assert process(st, plan, sid, log=lambda *_: None) == 0
    assert _receipt(st, sub, plan, sid)["quarantine"]["kind"] == "capture_identity_missing"
    st, sub, plan, sid = _setup(tmp_path / "b", "w-declared", custom=False, declare=True)
    assert sub["segments"][0]["identity_source"] == "declared at upload"
    assert process(st, plan, sid, log=lambda *_: None) == 0
    assert _receipt(st, sub, plan, sid)["status"] == "retained"


def test_overwritten_source_is_read_at_its_frozen_version(tmp_path, s3):
    st, sub, plan, sid = _setup(tmp_path, "w-frozen")
    data_key = next(f["key"] for f in sub["segments"][0]["files"] if f["key"].endswith(".mcap"))
    st.put(data_key, b"replaced after submission")
    assert process(st, plan, sid, log=lambda *_: None) == 0
    assert _receipt(st, sub, plan, sid)["status"] == "retained"


def test_source_not_matching_submission_fails_the_worker(tmp_path, s3):
    st, sub, plan, sid = _setup(tmp_path, "w-integrity")
    bad = json.loads(json.dumps(sub))
    bad["segments"][0]["files"][0]["sha256"] = "0" * 64
    bad["submission_id"] = "sub-tampered"
    st.put_immutable("submissions/sub-tampered.json", cjson(bad))
    assert process(st, dict(plan, submission_id="sub-tampered"), sid, log=lambda *_: None) == 3


def test_worker_with_other_code_or_image_than_the_plan_refuses(tmp_path, s3):
    st, sub, plan, sid = _setup(tmp_path, "w-prov")
    other = dict(plan, worker={**plan["worker"], "image": "sha256:" + "1" * 64})
    assert process(st, other, sid, log=lambda *_: None) == 5


def test_existing_receipt_with_other_content_is_a_conflict(tmp_path, s3):
    st, sub, plan, sid = _setup(tmp_path, "w-conflict")
    st.put_immutable(receipt_key(sub, sid, plan["analysis_digest"], plan["worker"]), b'{"forged": true}')
    assert process(st, plan, sid, log=lambda *_: None) == 4


def test_rerun_after_success_is_byte_identical(tmp_path, s3):
    st, sub, plan, sid = _setup(tmp_path, "w-rerun")
    logs = []
    assert process(st, plan, sid, log=logs.append) == 0
    assert process(st, plan, sid, log=logs.append) == 0
    assert "receipt exists_same" in logs[-1] and "created 0" in logs[-1]


def test_corrected_annotations_are_recomputed_and_published(tmp_path, s3):
    """Labels edited after a first build: the old receipt is not reused, the new labels are published."""
    st, sub, plan, sid = _setup(tmp_path, "w-labels")
    assert process(st, plan, sid, log=lambda *_: None) == 0
    recipe = copy.deepcopy(RECIPE)
    recipe["admission"].update(min_samples_per_split=0, min_retained_fraction=0.0)
    first = publish(st, sub, recipe, "first")
    lab = tmp_path / "side" / "labels" / f"{CAP.capture_id.replace('/', '_')}.json"
    doc = json.loads(lab.read_text())
    doc["range_to_cart"] = {k: v + 1000 for k, v in doc["range_to_cart"].items()}
    lab.write_text(json.dumps(doc))
    sub2 = build_submission(st, upload_bags(st, [next((tmp_path / "bags").rglob("rosbag2_*"))]),
                            upload_side(st, tmp_path / "side"))
    worker = build_provenance()
    assert remaining(st, sub2, plan["analysis_digest"], worker) == [sid]
    plan2 = make_plan(st, sub2, plan["analysis_key"], plan["analysis_digest"], [sid], worker)
    assert process(st, plan2, sid, log=lambda *_: None) == 0
    second = publish(st, sub2, recipe, "second")
    assert second["version_id"] != first["version_id"]
    assert [s["label_mm"] for s in second["samples"]] == [s["label_mm"] + 1000 for s in first["samples"]]


def test_byte_identical_upload_collapses_at_submission(tmp_path, s3):
    import shutil

    path = synth.write_segment(tmp_path / "bags", CAP, synth.SegmentSpec(0, 10, "mcap"))
    st = Store("w-exact")
    st.create_bucket()
    first = upload_bags(st, [path])
    copy_dir = tmp_path / "copy" / CAP.vehicle_id / path.name
    copy_dir.parent.mkdir(parents=True)
    shutil.copytree(path, copy_dir)
    second = upload_bags(st, [copy_dir])
    second[0]["bag"] = "tractor-09/uploaded-again"
    sub = build_submission(st, second + first, {"labels": {}, "calibration": {}})
    assert len(sub["segments"]) == 1
    assert sub["exact_reuploads"] == [{"segment_id": sub["segments"][0]["segment_id"], "bag": "tractor-09/uploaded-again",
                                       "same_bytes_as": first[0]["bag"]}]
