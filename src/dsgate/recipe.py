"""Recipe validation: every accepted parameter changes behaviour, the rest is refused.

A parameter that is accepted but ignored would change the provenance digest
without changing what the code does, so unknown keys and unimplemented values
raise ``RecipeError`` instead.
"""

from __future__ import annotations

IMAGE = "sensor_msgs/msg/Image"
CLOUD = "sensor_msgs/msg/PointCloud2"
ODOMETRY = "nav_msgs/msg/Odometry"
SUPPORTED_TYPES = {IMAGE, CLOUD, ODOMETRY}

RULES = ("required_stream_absent", "stream_insufficient", "stream_gap", "low_rate", "frozen_content",
         "lidar_degraded", "clock_not_comparable", "low_pairing", "pair_residual", "recording_offset",
         "no_labeled_samples")
THRESHOLDS = ("max_gap_s", "min_messages", "min_rate_ratio", "frozen_frames", "lidar_min_valid_ratio",
              "lidar_degraded_scans", "clock_domain_tolerance_ms", "max_pair_residual_ms", "min_pair_ratio",
              "recording_offset_info_ms")
ACTIONS = {"exclude_segment", "warn", "none"}
TOP = {"recipe", "required_topics", "time_source", "pairing", "thresholds", "policy", "split", "admission"}
IMPLEMENTED_TIME = {"acquisition": "header.stamp", "segmentation": "log_time"}


class RecipeError(ValueError):
    pass


def _keys(obj: dict, expected, where: str) -> None:
    got, want = set(obj), set(expected)
    if got != want:
        raise RecipeError(f"{where}: unknown {sorted(got - want)}, missing {sorted(want - got)}")


def validate(recipe: dict) -> dict:
    """Returns the roles {"camera": topic, "lidar": topic}."""
    _keys(recipe, TOP, "recipe")
    if recipe["time_source"] != IMPLEMENTED_TIME:
        raise RecipeError(f"time_source {recipe['time_source']} not implemented; only {IMPLEMENTED_TIME} "
                          "(segments are the bags as uploaded; the pipeline does not re-cut them)")
    req = recipe["required_topics"]
    for topic, spec in req.items():
        _keys(spec, ("msgtype", "expected_hz"), f"required_topics[{topic}]")
        if spec["msgtype"] not in SUPPORTED_TYPES:
            raise RecipeError(f"{topic}: msgtype {spec['msgtype']} not supported ({sorted(SUPPORTED_TYPES)})")
    pairing = recipe["pairing"]
    _keys(pairing, ("reference", "match", "window_ms"), "pairing")
    if req.get(pairing["reference"], {}).get("msgtype") != CLOUD:
        raise RecipeError("pairing.reference must be a required PointCloud2 topic")
    if req.get(pairing["match"], {}).get("msgtype") != IMAGE:
        raise RecipeError("pairing.match must be a required Image topic")
    _keys(recipe["thresholds"], THRESHOLDS, "thresholds")
    _keys(recipe["policy"], RULES, "policy")
    for rule, action in recipe["policy"].items():
        if action not in ACTIONS:
            raise RecipeError(f"policy.{rule}: {action} not in {sorted(ACTIONS)}")
    _keys(recipe["split"], ("unit", "val_fraction", "salt"), "split")
    if recipe["split"]["unit"] not in ("capture", "segment"):
        raise RecipeError("split.unit must be capture or segment")
    _keys(recipe["admission"], ("min_retained_fraction", "max_leakage_groups", "min_samples_per_split"), "admission")
    return {"camera": pairing["match"], "lidar": pairing["reference"]}


ANALYSIS_KEYS = ("required_topics", "time_source", "pairing", "thresholds", "policy")


def analysis_part(recipe: dict) -> dict:
    validate(recipe)
    return {k: recipe[k] for k in ANALYSIS_KEYS}


def roles(analysis: dict) -> dict:
    return {"camera": analysis["pairing"]["match"], "lidar": analysis["pairing"]["reference"]}
