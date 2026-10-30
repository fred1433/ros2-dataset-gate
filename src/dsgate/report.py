"""Plain-text report of one version and diff between two versions.

Everything printed comes from the version manifests; nothing is recomputed
here, so the report and the manifest cannot disagree.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from . import CAMERA, LIDAR, ODOM

SHORT = {CAMERA: "camera", LIDAR: "lidar", ODOM: "odom"}


def topic_name(t: str) -> str:
    for full, s in SHORT.items():
        t = t.replace(full, s)
    return t


def seg_label(s: dict) -> str:
    if s.get("start_log_ns") and s.get("vehicle_id") and s["vehicle_id"] != "unknown":
        t = datetime.fromtimestamp(s["start_log_ns"] // 1_000_000_000, tz=timezone.utc)
        return f"{s['vehicle_id']} {t:%m-%d %H:%M:%S}"
    vehicle, bag = s["bag"].split("/", 1)
    return f"{vehicle} {bag[13:15]}-{bag[16:18]} {bag[19:21]}:{bag[22:24]}:{bag[25:27]}"


def _interval(f: dict) -> str:
    return "-" if f["interval_s"] is None else f"{f['interval_s'][0]:.2f}-{f['interval_s'][1]:.2f}"


def render_report(m: dict) -> str:
    c = m["counts"]
    status = m["status"].upper() + (", ADMITTED" if m["admitted"] else ", NOT ADMITTED")
    pv = m["provenance"]
    lines = [
        f"dataset version {m['name']} = {m['version_id']}    {status}",
        f"recipe {m['recipe_digest'][:12]}  analysis {m['analysis_digest'][:12]}"
        + (f"  previous {m['previous']}" if m.get("previous") else ""),
        f"provenance: code {pv['code_digest'][:12]}  lock {pv['lock_digest'][:12]}  image {pv['image']}",
        "time: header.stamp = acquisition (Image, PointCloud2), used for pairing and skew;",
        "      log_time = recorder clock, used to cut segments and to place intervals",
        "",
        f"segments                        samples (lidar sweep + camera frame within {m['recipe']['pairing']['window_ms']} ms)",
        f"  examined          {c['examined']:>4}          candidate in retained segments {c['samples_candidate']:>6}",
        f"  retained          {c['retained']:>4}          duplicate of a published one   {c['samples_deduplicated']:>6}",
        f"  excluded by rule  {c['excluded']:>4}          published                      {c['samples_published']:>6}",
        f"  deduplicated      {c['deduplicated']:>4}            train {c['train']}, val {c['val']}",
        f"  quarantined       {c['quarantined']:>4}",
        f"  no receipt yet    {c['no_receipt']:>4}",
        f"  check: {c['examined']} = {c['retained']} + {c['excluded']} + {c['deduplicated']} + "
        f"{c['quarantined']} + {c['no_receipt']}  (retained + excluded + deduplicated + quarantined + no receipt)",
        "",
        "findings (one line per occurrence; camera = /camera/front/image_raw, lidar = /lidar/points)",
        f"  {'segment':<25} {'rule':<22} {'topic':<16} {'interval s':<11} {'measure':<38} {'threshold':<26} action",
    ]
    n_find = 0
    for s in sorted(m["segments"], key=seg_label):
        for f in s.get("findings", []):
            n_find += 1
            lines.append(f"  {seg_label(s):<25} {f['rule']:<22} {topic_name(f['topic']):<16} {_interval(f):<11} "
                         f"{topic_name(f['measure']):<38} {f['threshold']:<26} {f['action']}")
    if not n_find:
        lines.append("  (none)")
    quar = [s for s in m["segments"] if s["status"] == "quarantined"]
    if quar:
        lines += ["", "quarantined (original kept with its object version; the rest of the batch continued)"]
        for s in quar:
            q = s["quarantine"]
            extra = ""
            if q["kind"] == "mcap_truncated":
                n = q["evidence"].get("complete_records_before_cut", 0)
                extra = f"; {n} complete message records before the cut; recovery (mcap recover) not attempted"
            lines.append(f"  {seg_label(s):<25} {q['kind']:<16} \"{q['error']}\"{extra}")
    dups = [s for s in m["segments"] if s.get("duplicate_of")]
    by_id = {s["segment_id"]: s for s in m["segments"]}
    if dups or m.get("exact_reuploads") or m.get("possible_duplicates"):
        lines += ["", "re-uploads (same capture id, same acquisition stamp, same content)"]
        for s in dups:
            others = ", ".join(seg_label(by_id[o])[-8:] for o in s["duplicate_of"])
            lines.append(f"  {seg_label(s):<25} {s['samples_deduplicated']} of {s['samples_candidate']} samples "
                         f"already published from {others}; {s['samples_published']} new")
        for e in m.get("exact_reuploads", []):
            lines.append(f"  {e['bag']}: same bytes as {e['same_bytes_as']}, collapsed at submission")
        for p in m.get("possible_duplicates", []):
            lines.append(f"  possible duplicate, kept: {p['capture_id']} @ {p['ref_stamp_ns']} in {', '.join(p['segments'])}")
    lines += ["", "kept per capture",
              f"  {'capture':<30} {'segments kept':>14} {'samples':>8}  split"]
    per = defaultdict(lambda: [0, 0, 0])
    for s in m["segments"]:
        if s.get("capture_id") in (None, "unknown"):
            continue
        p = per[s["capture_id"]]
        p[1] += 1
        if s["status"] == "retained":
            p[0] += 1
        p[2] += s.get("samples_published", 0)
    reg = m.get("split_registry", {})
    for cap in sorted(per):
        k, n, smp = per[cap]
        lines.append(f"  {cap:<30} {f'{k}/{n}':>14} {smp:>8}  {reg.get(cap, '-')}")
    lines += ["", "admission"]
    for g in m["gates"]:
        lines.append(f"  [{'PASS' if g['ok'] else 'FAIL'}] {g['gate']:<19} {g['detail']}")
    return "\n".join(lines) + "\n"


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def render_diff(a: dict, b: dict) -> str:
    fa, fb = _flatten(a["recipe"]), _flatten(b["recipe"])
    changes = [(k, fa.get(k), fb.get(k)) for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)]
    sa = {s["segment_id"]: s for s in a["segments"]}
    sb = {s["segment_id"]: s for s in b["segments"]}
    lines = [f"diff {a['name']} {a['version_id']} -> {b['name']} {b['version_id']}", "",
             "recipe changes" + ("" if changes else ": none")]
    for k, x, y in changes:
        lines.append(f"  {k:<28} {x} -> {y}")
    added = [sb[k] for k in sb if k not in sa]
    removed = [sa[k] for k in sa if k not in sb]
    lines += ["", f"inputs: {len(sa)} -> {len(sb)} segments (+{len(added)} new, -{len(removed)} removed)", "",
              "decisions changed on unchanged inputs (same source bytes: the cause is the recipe, not the data)"]
    n = 0
    for k in sorted(set(sa) & set(sb), key=lambda k: seg_label(sb[k])):
        x, y = sa[k], sb[k]
        if x["status"] != y["status"]:
            n += 1
            why = "; ".join(f"{f['rule']} {topic_name(f['topic'])} {f['measure']} ({f['threshold']})"
                            for f in y["findings"] if f["action"] == "exclude_segment") or "-"
            lines.append(f"  {seg_label(y):<25} {x['status']} -> {y['status']}: {why}")
    if not n:
        lines.append("  (none)")
    tally = defaultdict(int)
    for s in added:
        tally[s["status"]] += 1
    lines += ["", "new inputs: " + ", ".join(f"{k} {v}" for k, v in sorted(tally.items()))]
    for s in sorted(added, key=seg_label):
        rules = ", ".join(sorted({f["rule"] for f in s.get("findings", [])})) or "-"
        if s["status"] == "quarantined":
            rules = s["quarantine"]["kind"]
        if s["status"] == "deduplicated":
            rules = f"re-upload of {', '.join(seg_label(sb[o])[-8:] for o in s['duplicate_of'])}"
        lines.append(f"  {seg_label(s):<25} {s['status']:<13} {rules}")
    ra, rb = a.get("split_registry", {}), b.get("split_registry", {})
    moved = [c for c in ra if c in rb and ra[c] != rb[c]]
    new = {c: rb[c] for c in rb if c not in ra}
    lines += ["", f"train/val: {len(moved)} capture(s) changed side; new captures: "
              + (", ".join(f"{c} -> {s}" for c, s in sorted(new.items())) or "none")]
    ca, cb = a["counts"], b["counts"]
    lines.append(f"published samples: {ca['samples_published']} -> {cb['samples_published']} "
                 f"(train {ca['train']} -> {cb['train']}, val {ca['val']} -> {cb['val']})")
    keys_a = {smp["src"][k] for smp in a["samples"] for k in ("camera_key", "lidar_key")}
    keys_b = {smp["src"][k] for smp in b["samples"] for k in ("camera_key", "lidar_key")}
    lines.append(f"storage: {len(keys_b & keys_a)} of {len(keys_b)} shard objects used by {b['name']} were already "
                 f"written for {a['name']} (content-addressed keys, nothing copied)")
    per_a, per_b = defaultdict(int), defaultdict(int)
    for smp in a["samples"]:
        per_a[smp["capture_id"]] += 1
    for smp in b["samples"]:
        per_b[smp["capture_id"]] += 1
    lines += ["", "published samples per capture"]
    for cap in sorted(set(per_a) | set(per_b)):
        lines.append(f"  {cap:<30} {per_a.get(cap, 0):>5} -> {per_b.get(cap, 0):>5}")
    adm = lambda m: "ADMITTED" if m["admitted"] else "NOT ADMITTED"
    lines += ["", f"admission: {adm(a)} -> {adm(b)}"]
    return "\n".join(lines) + "\n"
