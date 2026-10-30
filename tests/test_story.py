"""Assertions on the end-to-end story (one run shared by the session)."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from dsgate import synth
from dsgate.canon import sha256
from dsgate.train_launch import NotAdmitted, gate, run_process

STATUSES = {"retained", "excluded", "deduplicated", "quarantined"}


def expected(seg_spec, version):
    status, rules = None, set()
    for tok in seg_spec.expect:
        name, _, ver = tok.partition("@")
        if ver and ver != version:
            continue
        if name in STATUSES:
            status = name
        else:
            rules.add(name)
    return status, rules


def specs_by_bag():
    out = {}
    for cap in synth.BATCH1 + synth.BATCH2:
        for seg in cap.segments:
            out[f"{cap.vehicle_id}/{synth.bag_name(cap, seg)}"] = seg
    return out


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_findings_match_injected_faults(demo, version):
    """The oracle is the generator's ledger; the analyzer never reads it."""
    specs = specs_by_bag()
    m = demo[version]
    assert m["segments"], "no segments"
    for s in m["segments"]:
        status, rules = expected(specs[s["bag"]], version)
        found = {f["rule"] for f in s["findings"]}
        if s["status"] == "quarantined":
            found = {s["quarantine"]["kind"]}
        assert s["status"] == status, (s["bag"], s["status"], status)
        assert found == rules, (s["bag"], found, rules)


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_counters_reconcile(demo, version):
    c = demo[version]["counts"]
    assert c["examined"] == c["retained"] + c["excluded"] + c["deduplicated"] + c["quarantined"] + c["no_receipt"]
    assert c["samples_candidate"] == c["samples_published"] + c["samples_deduplicated"]
    assert c["samples_published"] == c["train"] + c["val"] == len(demo[version]["samples"])
    per_seg = sum(s.get("samples_published", 0) for s in demo[version]["segments"])
    assert per_seg == c["samples_published"]


def test_every_finding_carries_its_evidence(demo):
    for s in demo["v2"]["segments"]:
        for f in s["findings"]:
            assert f["rule"] and f["topic"] and f["measure"] and f["threshold"] and f["action"]
            assert s["segment_id"].startswith("seg-")


def test_interruption_then_resume_equals_uninterrupted_build(demo):
    attempt, v2, ref = demo["attempt"], demo["v2"], demo["ref_v2"]
    assert attempt["status"] == "incomplete"
    assert set(attempt["missing_receipts"]) == demo["kill"]
    assert sorted(demo["resumed"]) == sorted(demo["kill"])
    # the kill happened after the outputs: the resumed workers found all 3 of them, identical
    assert all("already present and identical 3" in line for line in demo["resume_logs"])
    assert v2["status"] == "complete" and v2["admitted"]
    assert ref["version_id"] == v2["version_id"]
    assert ref["samples"] == v2["samples"]


def test_reference_v1_is_complete_and_is_the_history_of_reference_v2(demo):
    ref1 = demo["ref_v1"]
    assert ref1["status"] == "complete" and ref1["admitted"] and ref1["counts"]["examined"] == 7
    assert ref1["samples"] == demo["v1"]["samples"]
    assert demo["ref_v2"]["previous"] == ref1["version_id"]


def test_same_content_under_another_name_or_policy_reuses_the_dataset_object(demo):
    v2, again, loose = demo["v2"], demo["again"], demo["loose"]
    assert v2["version_id"] == again["version_id"] == loose["version_id"]
    assert len({v2["build_id"], again["build_id"], loose["build_id"]}) == 3


def test_workers_provenance_comes_from_the_receipts(demo):
    workers = demo["v2"]["provenance"]["workers"]
    assert len(workers) == 1 and set(workers[0]) == {"code_digest", "lock_digest", "image"}


def test_incomplete_and_leaking_versions_are_refused_before_training(demo):
    with pytest.raises(NotAdmitted, match="incomplete"):
        gate(demo["attempt"])
    with pytest.raises(NotAdmitted, match="train_val_leakage"):
        gate(demo["bad"])
    gate(demo["v2"])
    assert not (demo["out"] / "train-refused-incomplete").exists()
    assert not (demo["out"] / "train-refused-leak").exists()


def test_reupload_is_one_contribution(demo):
    segs = {s["bag"]: s for s in demo["v2"]["segments"]}
    re = segs["tractor-01/rosbag2_2026_10_12-08_14_14"]  # the re-cut upload
    assert re["status"] == "deduplicated" and re["samples_published"] == 0 and re["samples_deduplicated"] == 100
    originals = {segs["tractor-01/rosbag2_2026_10_12-08_14_10"]["segment_id"],
                 segs["tractor-01/rosbag2_2026_10_12-08_14_20"]["segment_id"]}
    assert set(re["duplicate_of"]) == originals
    keys = [(s["capture_id"], s["ref_stamp_ns"]) for s in demo["v2"]["samples"]]
    assert len(keys) == len(set(keys))


def test_identical_pixels_in_distinct_captures_are_kept(demo):
    flat = synth.np.full((synth.HEIGHT, synth.WIDTH, 3), 128, dtype=synth.np.uint8)
    from dsgate.canon import image_hash

    h = image_hash("rgb8", synth.HEIGHT, synth.WIDTH, synth.WIDTH * 3, flat.tobytes())
    hits = [s for s in demo["v2"]["samples"] if s["cam_sha256"] == h]
    assert {s["capture_id"] for s in hits} == {"tractor-01/20261012T081400Z", "tractor-03/20261012T111500Z"}


def test_split_is_stable_and_has_no_leakage(demo):
    v1, v2 = demo["v1"], demo["v2"]
    for cap, side in v1["split_registry"].items():
        assert v2["split_registry"][cap] == side
    assert v2["leakage"] == [] and demo["bad"]["leakage"]


def test_recipe_change_is_reported_as_recipe_not_data(demo):
    diff = (demo["out"] / "diff.txt").read_text()
    assert "thresholds.max_gap_s         1.0 -> 0.5" in diff
    line = next(l for l in diff.splitlines() if "retained -> excluded" in l)
    assert line.strip().startswith("tractor-02 10-12 09:30:10") and "stream_gap" in line


def test_model_carries_the_version_it_read(demo):
    assert demo["model"]["dataset_version_id"] == demo["v2"]["version_id"]
    assert demo["model"]["train_mse_per_step"][-1] < demo["model"]["train_mse_per_step"][0]


def _channel(demo, tmp_path):
    ch = tmp_path / "channel"
    shutil.copytree(demo["out"] / "train" / "channel", ch)
    return ch


def _seal(ch, m=None):
    if m is not None:
        (ch / "manifest.json").write_text(json.dumps(m, sort_keys=True, separators=(",", ":")))
    return sha256((ch / "manifest.json").read_bytes())


def _train(demo, ch, tmp_path, digest, version=None):
    return run_process(ch, tmp_path / "model", version or demo["v2"]["version_id"], digest)


def test_training_container_refuses_a_tampered_shard(demo, tmp_path):
    ch = _channel(demo, tmp_path)
    shard = sorted((ch / "shards").iterdir())[0]
    data = bytearray(shard.read_bytes())
    data[-1] ^= 0xFF
    shard.write_bytes(bytes(data))
    proc = _train(demo, ch, tmp_path, _seal(ch))
    assert proc.returncode == 2 and "REFUSED" in proc.stdout and "content hash" in proc.stdout
    assert not (tmp_path / "model" / "model.json").exists()


def test_training_container_refuses_an_unsealed_manifest(demo, tmp_path):
    ch = _channel(demo, tmp_path)
    m = json.loads((ch / "manifest.json").read_text())
    m["name"] = "edited"
    sealed = demo["model"]["dataset_manifest_sha256"]
    _seal(ch, m)
    proc = _train(demo, ch, tmp_path, sealed)
    assert proc.returncode == 2 and "not the one sealed at admission" in proc.stdout


def test_training_container_rechecks_admission_invariants(demo, tmp_path):
    """A leaking manifest whose admitted flag was flipped, sealed by a careless launcher: still refused."""
    ch = _channel(demo, tmp_path)
    m = json.loads((ch / "manifest.json").read_text())
    s0 = next(s for s in m["samples"] if s["split"] == "train")
    s0["split"] = "val"
    proc = _train(demo, ch, tmp_path, _seal(ch, m))
    assert proc.returncode == 2 and "both sides of the split" in proc.stdout
    m = json.loads((ch / "manifest.json").read_text())
    m["gates"][1]["ok"] = False
    proc = _train(demo, ch, tmp_path, _seal(ch, m))
    assert proc.returncode == 2 and "contradicts the recorded gates" in proc.stdout


def test_training_container_refuses_a_row_moved_to_another_valid_row(demo, tmp_path):
    ch = _channel(demo, tmp_path)
    m = json.loads((ch / "manifest.json").read_text())
    s0 = m["samples"][0]
    s0["src"]["row"] = 1  # another valid row of the same shards
    proc = _train(demo, ch, tmp_path, _seal(ch, m))
    assert proc.returncode == 2 and "row 1" in proc.stdout and "does not match the manifest" in proc.stdout


def test_training_container_refuses_a_camera_row_from_another_shard(demo, tmp_path):
    ch = _channel(demo, tmp_path)
    m = json.loads((ch / "manifest.json").read_text())
    s0 = m["samples"][0]
    other = next(s for s in m["samples"] if s["src"]["camera_key"] != s0["src"]["camera_key"])
    s0["src"]["camera_key"] = other["src"]["camera_key"]  # valid shard, valid row, wrong content
    proc = _train(demo, ch, tmp_path, _seal(ch, m))
    assert proc.returncode == 2 and "camera row 0" in proc.stdout


def test_training_container_refuses_another_version_or_an_edited_label(demo, tmp_path):
    ch = _channel(demo, tmp_path)
    proc = _train(demo, ch, tmp_path, _seal(ch), version=demo["v1"]["version_id"])
    assert proc.returncode == 2 and "job asked for" in proc.stdout
    m = json.loads((ch / "manifest.json").read_text())
    m["samples"][0]["label_mm"] += 1
    proc = _train(demo, ch, tmp_path, _seal(ch, m))
    assert proc.returncode == 2 and "does not hash to its version id" in proc.stdout


def test_readme_blocks_are_written(demo):
    block = (demo["out"] / "README_block.md").read_text()
    assert demo["v2"]["version_id"] in block and "IDENTICAL" in block
    excerpt = (demo["out"] / "README_excerpt.md").read_text().splitlines()
    assert 8 <= len(excerpt) - 2 <= 11 and any("parent FAILED" in l for l in excerpt)
