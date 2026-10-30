"""The one way to train: ``launch()`` = gate, then export, then train.

The gate runs before anything that costs: for a version that is incomplete or
not admitted, nothing is exported, no image is built, no container starts.
The launcher then seals what it exports: the SHA-256 of the exported manifest
is passed to the training job, and ``training/train.py`` refuses a manifest
with another digest, re-checks the admission invariants it can recompute, and
verifies every shard and every sample row.

``sagemaker-local`` runs SageMaker Python SDK 3 ``ModelTrainer`` with
``training_mode=Mode.LOCAL_CONTAINER`` on an image built locally, in a child
process whose environment carries no AWS credentials or profile and points
every AWS endpoint at a closed local port.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from .canon import cjson, sha256
from .publish import _stable
from .store import Store

ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = ROOT / "training"
IMAGE = "dsgate-train:local"
SHARD_KEYS = ("camera_key", "lidar_points_key", "lidar_offsets_key")


class NotAdmitted(Exception):
    pass


def gate(manifest: dict) -> None:
    name = f"{manifest['name']} ({manifest['version_id']})"
    if manifest["status"] != "complete":
        missing = manifest["missing_receipts"]
        raise NotAdmitted(f"{name} is incomplete: {len(missing)} segment(s) without a terminal receipt "
                          f"({', '.join(missing)})")
    if not manifest["admitted"]:
        failed = [f"{g['gate']}: {g['detail']}" for g in manifest["gates"] if not g["ok"]]
        raise NotAdmitted(f"{name} is not admitted: {'; '.join(failed)}")


def export(store: Store, manifest: dict, out: Path) -> tuple[Path, str]:
    """Write the channel; returns (channel, SHA-256 of the exported manifest)."""
    out = Path(out)
    (out / "shards").mkdir(parents=True, exist_ok=True)
    body = cjson(_stable(manifest))
    (out / "manifest.json").write_bytes(body)
    for key in sorted({s["src"][k] for s in manifest["samples"] for k in SHARD_KEYS}):
        data = store.get(key)
        if f"cas/{sha256(data)}.npy" != key:
            raise ValueError(f"{key} does not match its content hash")
        (out / "shards" / key.rsplit("/", 1)[1]).write_bytes(data)
    return out, sha256(body)


def run_process(channel: Path, model_dir: Path, version_id: str, manifest_sha256: str,
                steps: int = 5) -> subprocess.CompletedProcess:
    env = {**os.environ, "SM_CHANNEL_TRAIN": str(channel), "SM_MODEL_DIR": str(model_dir),
           "SM_OUTPUT_FAILURE": str(model_dir.parent / "failure")}
    return subprocess.run([sys.executable, str(TRAINING_DIR / "train.py"), "--dataset_version", version_id,
                           "--manifest_sha256", manifest_sha256, "--steps", str(steps)],
                          env=env, capture_output=True, text=True)


def build_image() -> str:
    subprocess.run(["docker", "build", "-q", "-t", IMAGE, str(TRAINING_DIR)], check=True, capture_output=True)
    return subprocess.run(["docker", "image", "inspect", "-f", "{{.Id}}", IMAGE], check=True,
                          capture_output=True, text=True).stdout.strip()


def sandbox_env(workdir: Path) -> dict:
    cfg = workdir / "sagemaker-config.yaml"
    cfg.write_text("SchemaVersion: '1.0'\nSageMaker:\n  PythonSDK:\n    Modules:\n      TelemetryOptOut: true\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("AWS_", "DSGATE_S3"))}
    env.update({
        "AWS_CONFIG_FILE": "/nonexistent/aws-config",
        "AWS_SHARED_CREDENTIALS_FILE": "/nonexistent/aws-credentials",
        "AWS_EC2_METADATA_DISABLED": "true",
        "AWS_ENDPOINT_URL": "http://127.0.0.1:9",
        "SAGEMAKER_USER_CONFIG_OVERRIDE": str(cfg),
        "PYTHONWARNINGS": "ignore",
    })
    return env


def run_sagemaker_local(channel: Path, workdir: Path, version_id: str, manifest_sha256: str,
                        steps: int = 5) -> subprocess.CompletedProcess:
    workdir.mkdir(parents=True, exist_ok=True)
    return subprocess.run([sys.executable, "-m", "dsgate.train_launch", str(channel), str(workdir), version_id,
                           manifest_sha256, str(steps)], env=sandbox_env(workdir), capture_output=True, text=True)


def launch(store: Store, manifest: dict, workdir: Path, mode: str = "sagemaker-local", steps: int = 5) -> dict:
    """Gate, export, train. Returns {"model": model.json, "manifest_sha256": ..., "image_id": ...}."""
    gate(manifest)  # raises before any file, image or container exists
    workdir = Path(workdir).resolve()
    channel, digest = export(store, manifest, workdir / "channel")
    if mode == "sagemaker-local":
        image_id = build_image()
        proc = run_sagemaker_local(channel, workdir, manifest["version_id"], digest, steps)
        (workdir / "sagemaker.log").write_text(proc.stdout + proc.stderr)
        model_path = workdir / "container" / "artifacts" / "model" / "model.json"
    elif mode == "process":
        image_id = None
        proc = run_process(channel, workdir / "model", manifest["version_id"], digest, steps)
        model_path = workdir / "model" / "model.json"
    else:
        raise ValueError(f"unknown mode {mode}")
    if proc.returncode != 0 or not model_path.exists():
        raise RuntimeError(f"training failed ({mode}):\n" + (proc.stdout + proc.stderr)[-4000:])
    return {"model": json.loads(model_path.read_text()), "manifest_sha256": digest, "image_id": image_id}


def _child(channel: str, workdir: str, version_id: str, manifest_sha256: str, steps: str) -> int:
    import boto3

    assert boto3.Session().get_credentials() is None, "AWS credentials leaked into the training process"
    print("no AWS credentials resolvable in this process", flush=True)
    from importlib.metadata import version

    from sagemaker.train import ModelTrainer
    from sagemaker.train.configs import Compute, InputData, SourceCode
    from sagemaker.train.model_trainer import Mode

    root = Path(workdir).resolve() / "container"
    root.mkdir(parents=True, exist_ok=True)
    trainer = ModelTrainer(
        training_image=IMAGE,
        training_mode=Mode.LOCAL_CONTAINER,
        source_code=SourceCode(source_dir=str(TRAINING_DIR), entry_script="train.py"),
        compute=Compute(instance_type="local_cpu", instance_count=1),
        hyperparameters={"dataset_version": version_id, "manifest_sha256": manifest_sha256, "steps": int(steps)},
        # the container runs as root: no bytecode files left in the mounted code dirs,
        # which the SDK removes as the host user afterwards
        environment={"PYTHONDONTWRITEBYTECODE": "1"},
        local_container_root=str(root),
        base_job_name="dsgate-train",
    )
    trainer.train(input_data_config=[InputData(channel_name="train", data_source=str(channel))], wait=True)
    model = root / "artifacts" / "model" / "model.json"
    print(f"sagemaker {version('sagemaker')}; artifact {model}", flush=True)
    return 0 if model.exists() else 1


if __name__ == "__main__":
    sys.exit(_child(*sys.argv[1:6]))
