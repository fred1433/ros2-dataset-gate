"""S3 adapter: immutable writes, verified reads.

Protocol (per segment): write immutable outputs, then write the receipt last.
Every immutable write is a conditional ``PutObject`` with ``If-None-Match: *``.
"Already present" is only a success when the existing bytes are the expected
bytes (SHA-256 of the body, never the ETag); otherwise it is a conflict.
A conditional write does not make computation happen once; it makes a second
contribution impossible.
"""

from __future__ import annotations

import os

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from .canon import sha256


class Conflict(Exception):
    """An object exists under an immutable key with different bytes."""


def client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("DSGATE_S3_ENDPOINT") or None,
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        config=Config(retries={"max_attempts": 3, "mode": "standard"}),
    )


class Store:
    def __init__(self, bucket: str, s3=None):
        self.bucket = bucket
        self.s3 = s3 or client()

    def put_immutable(self, key: str, body: bytes) -> str:
        """Returns "created" or "exists_same"; raises Conflict."""
        for _ in range(2):
            try:
                self.s3.put_object(Bucket=self.bucket, Key=key, Body=body, IfNoneMatch="*")
                return "created"
            except ClientError as exc:
                code = exc.response["ResponseMetadata"]["HTTPStatusCode"]
                if code == 412:
                    existing = self.get(key)
                    if sha256(existing) == sha256(body):
                        return "exists_same"
                    raise Conflict(f"s3://{self.bucket}/{key} exists with different content") from None
                if code == 409:  # concurrent delete; PutObject may be retried
                    continue
                raise
        raise Conflict(f"s3://{self.bucket}/{key}: repeated 409 ConditionalRequestConflict")

    def put(self, key: str, body: bytes) -> str | None:
        """Plain upload (raw logs into a versioned bucket). Returns the VersionId."""
        return self.s3.put_object(Bucket=self.bucket, Key=key, Body=body).get("VersionId")

    def get(self, key: str, version_id: str | None = None) -> bytes:
        kw = {"VersionId": version_id} if version_id else {}
        return self.s3.get_object(Bucket=self.bucket, Key=key, **kw)["Body"].read()

    def exists(self, key: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response["ResponseMetadata"]["HTTPStatusCode"] == 404:
                return False
            raise

    def keys(self, prefix: str) -> list[str]:
        out = []
        for page in self.s3.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix):
            out.extend(o["Key"] for o in page.get("Contents", []))
        return sorted(out)

    def create_bucket(self, versioned: bool = True) -> None:
        self.s3.create_bucket(Bucket=self.bucket)
        if versioned:
            self.s3.put_bucket_versioning(Bucket=self.bucket, VersioningConfiguration={"Status": "Enabled"})
