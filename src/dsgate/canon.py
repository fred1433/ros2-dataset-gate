"""Canonical encodings and digests.

Every digest in this repository is SHA-256 over an explicit byte encoding:

* objects: canonical JSON (UTF-8, keys sorted, no whitespace, ASCII only,
  NaN/Infinity rejected). Identity-bearing objects carry integers only
  (nanoseconds, millimetres), never floats.
* camera frames: ``b"img/v1|<encoding>|<height>|<width>|<step>|" + data``.
* point clouds: ``b"pc/v1|xyzi-f32le|<n>|" + float32 little-endian (n, 4)``
  with every NaN rewritten to the quiet NaN 0x7FC00000.

The S3 ETag is never used as a content hash.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

CANONICAL_NAN = np.frombuffer(np.uint32(0x7FC00000).tobytes(), dtype="<f4")[0]


def cjson(obj: Any) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(obj: Any) -> str:
    return sha256(cjson(obj))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def image_hash(encoding: str, height: int, width: int, step: int, data: bytes) -> str:
    head = f"img/v1|{encoding}|{height}|{width}|{step}|".encode("ascii")
    return sha256(head + bytes(data))


def canonical_points(points: np.ndarray) -> np.ndarray:
    pts = np.ascontiguousarray(points, dtype="<f4").copy()
    pts[np.isnan(pts)] = CANONICAL_NAN
    return pts


def points_hash(points: np.ndarray) -> str:
    pts = canonical_points(points)
    head = f"pc/v1|xyzi-f32le|{pts.shape[0]}|".encode("ascii")
    return sha256(head + pts.tobytes())


def short(hexdigest: str, n: int = 12) -> str:
    return hexdigest[:n]
