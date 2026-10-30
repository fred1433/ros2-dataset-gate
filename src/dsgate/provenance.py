"""Build provenance, kept apart from dataset identity.

* code digest: SHA-256 over the module's own source files (relative path and
  bytes, sorted). Editing the README, Terraform or tests does not change it.
* lock digest: SHA-256 of ``uv.lock``.
* image: the container image the step ran in, when there is one.
* git commit: recorded as metadata only, never part of any digest
  (``.git`` is usually absent from a container build context).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .canon import digest, sha256_file

PKG = Path(__file__).resolve().parent


def code_digest(pkg: Path = PKG) -> str:
    files = sorted(p for p in pkg.rglob("*.py") if "__pycache__" not in p.parts)
    return digest([[p.relative_to(pkg).as_posix(), sha256_file(p)] for p in files])


def _find_lock() -> Path | None:
    env = os.environ.get("DSGATE_LOCKFILE")
    if env:
        return Path(env)
    for parent in [PKG, *PKG.parents]:
        if (parent / "uv.lock").exists():
            return parent / "uv.lock"
    return None


def lock_digest() -> str:
    lock = _find_lock()
    return sha256_file(lock) if lock else "no-lockfile"


def git_commit() -> str:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=PKG).stdout.strip()
    except Exception:
        return "unknown"


def build_provenance() -> dict:
    return {
        "code_digest": code_digest(),
        "lock_digest": lock_digest(),
        "image": os.environ.get("DSGATE_IMAGE", "none (local process)"),
    }
