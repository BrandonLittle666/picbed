from __future__ import annotations

import mimetypes
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterable

from boto3.exceptions import Boto3Error
from boto3.session import Session
from botocore.client import Config
from botocore.exceptions import ClientError
from loguru import logger

from picbed.utils import format_size


@dataclass
class S3Config:
    endpoint_url: str | None
    region_name: str | None
    bucket: str
    access_key: str | None
    secret_key: str | None
    use_path_style: bool = True
    prefix: str = ""
    signature_version: str | None = None
    unsigned_payload: bool = True
    remove_sha256_header: bool = False
    upload_path_template: str = ""
    output_url_pattern: str = ""
    over_write_existing: bool = False
    max_upload_size: int = 10 * 1024 * 1024    # local server max upload size, default to 100MB


class S3Client:
    def __init__(self, config: S3Config) -> None:
        if not config.bucket:
            raise ValueError("S3_BUCKET is required")
        logger.debug("Initializing S3 client for bucket '{}' at '{}'", config.bucket, config.endpoint_url)
        session = Session(
            aws_access_key_id=config.access_key,
            aws_secret_access_key=config.secret_key,
            region_name=config.region_name or "us-east-1",
        )
        self._session = session
        self._config = config
        # Track whether we are using unsigned payloads (UNSIGNED-PAYLOAD) or fully signed payloads
        self._unsigned_payload = bool(config.unsigned_payload)
        self._s3 = self._build_client(unsigned_payload=self._unsigned_payload)
        self.bucket = config.bucket
        self.prefix = config.prefix.strip("/")
        # Expose templates for consumers
        self.upload_path_template = config.upload_path_template
        self.output_url_pattern = config.output_url_pattern

        # Mark if it using in task
        self._is_in_task: bool = False

    def _build_client(self, *, unsigned_payload: bool):
        return self._session.client(
            "s3",
            endpoint_url=self._config.endpoint_url,
            config=Config(
                signature_version=self._config.signature_version or "s3v4",
                s3={
                    "addressing_style": "path" if self._config.use_path_style else "virtual",
                    # When unsigned_payload is True, payload_signing_enabled must be False
                    "payload_signing_enabled": not unsigned_payload,
                },
            ),
        )

    def _key(self, name: str) -> str:
        name = name.lstrip("/")
        return f"{self.prefix}/{name}" if self.prefix else name

    def list_objects(self, prefix: str = "") -> list[dict]:
        list_prefix = self._key(prefix)
        paginator = self._s3.get_paginator("list_objects_v2")
        results: list[dict] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=list_prefix):
            for obj in page.get("Contents", []) or []:
                results.append(obj)
        logger.debug("Listed {} objects with prefix '{}'", len(results), list_prefix)
        return results

    def exist(self, url: str) -> bool:
        try:
            return self._s3.head_object(Bucket=self.bucket, Key=url) is not None
        except Exception:
            return False

    def upload_file(self, local_path: str, key: str, content_type: str | None = None, public: bool = True) -> str:
        key = self._key(key)
        if not self._config.over_write_existing and self.exist(key):
            logger.info(f'Object {key} already exists, skipping upload')
            return key
        if content_type is None:
            guessed, _ = mimetypes.guess_type(local_path)
            content_type = guessed or "application/octet-stream"
        extra_args = {"ContentType": content_type}
        if public:
            extra_args["ACL"] = "public-read"
        logger.debug(f"Uploading file '{local_path}' as '{key}' (type: {content_type}, size: {format_size(os.path.getsize(local_path))})")
        message_on_success = f"Uploaded file '{local_path}' as '{key}' (type: {content_type}, size: {format_size(os.path.getsize(local_path))})"
        # If we are configured to use UNSIGNED-PAYLOAD, avoid the x-amz-content-sha256 header entirely
        # by using a presigned PUT upfront. This is the most compatible path with strict providers.
        if self._unsigned_payload:
            with open(local_path, "rb") as f:
                data = f.read()
            self._put_object_via_presigned(key, data, content_type, public, message=message_on_success)
            return key
        try:
            self._s3.upload_file(local_path, self.bucket, key, ExtraArgs=extra_args)
        except (ClientError, Boto3Error) as e:
            if self._should_retry_signed_payload(e):
                logger.warning(f"Provider rejected UNSIGNED-PAYLOAD; retrying '{key}' with signed payloads")
                self._unsigned_payload = False
                self._s3 = self._build_client(unsigned_payload=False)
                try:
                    self._s3.upload_file(local_path, self.bucket, key, ExtraArgs=extra_args)
                except (ClientError, Boto3Error) as e2:
                    if self._is_sha256_header_issue(e2):
                        logger.warning(f"Header still rejected on signed upload; falling back to presigned PUT for '{key}'")
                        with open(local_path, "rb") as f:
                            data = f.read()
                        self._put_object_via_presigned(key, data, content_type, public, message=message_on_success)
                        return key
                    else:
                        raise
            elif self._is_sha256_header_issue(e):
                logger.warning(f"Header rejected; falling back to presigned PUT for '{key}'")
                with open(local_path, "rb") as f:
                    data = f.read()
                self._put_object_via_presigned(key, data, content_type, public, message=message_on_success)
                return key
            else:
                raise
        logger.info(message_on_success)
        return key

    def upload_bytes(self, data: bytes, key: str, content_type: str, public: bool = True) -> str:
        key = self._key(key)
        if not self._config.over_write_existing and self.exist(key):
            logger.info(f'Object {key} already exists, skipping upload')
            return key
        extra_args = {"ContentType": content_type}
        if public:
            extra_args["ACL"] = "public-read"
        logger.debug(f'Uploading bytes as {key} (type: {content_type}, size: {format_size(len(data))})')
        message_on_success = f'Uploaded bytes as {key} (type: {content_type}, size: {format_size(len(data))})'
        if self._unsigned_payload:
            self._put_object_via_presigned(key, data, content_type, public, message=message_on_success)
            return key
        try:
            self._s3.put_object(Bucket=self.bucket, Key=key, Body=data, **extra_args)
        except (ClientError, Boto3Error) as e:
            if self._should_retry_signed_payload(e):
                logger.warning(f"Provider rejected UNSIGNED-PAYLOAD; retrying '{key}' with signed payloads")
                self._unsigned_payload = False
                self._s3 = self._build_client(unsigned_payload=False)
                try:
                    self._s3.put_object(Bucket=self.bucket, Key=key, Body=data, **extra_args)
                except (ClientError, Boto3Error) as e2:
                    if self._is_sha256_header_issue(e2):
                        logger.warning(f"Header still rejected on signed upload; falling back to presigned PUT for '{key}'")
                        self._put_object_via_presigned(key, data, content_type, public, message=message_on_success)
                        return key
                    else:
                        raise
            elif self._is_sha256_header_issue(e):
                logger.warning(f"Header rejected; falling back to presigned PUT for '{key}'")
                self._put_object_via_presigned(key, data, content_type, public, message=message_on_success)
                return key
            else:
                raise
        logger.info(message_on_success)
        return key

    def _should_retry_signed_payload(self, e: ClientError | Boto3Error) -> bool:
        if not self._unsigned_payload:
            return False
        return self._is_sha256_header_issue(e)

    @staticmethod
    def _is_sha256_header_issue(e: ClientError | Boto3Error) -> bool:
        """ 针对 RustFS 的兼容性问题 https://github.com/Nugine/s3s/issues/14 """
        try:
            if isinstance(e, Boto3Error):
                return 'XAmzContentSHA256Mismatch' in str(e)
            code = (e.response or {}).get("Error", {}).get("Code") if hasattr(e, "response") else None
            message = (e.response or {}).get("Error", {}).get("Message", str(e)) if hasattr(e, "response") else str(e)
            if code == "XAmzContentSHA256Mismatch":
                return True
            if message and "x-amz-content-sha256" in str(message).lower():
                return True
        except Exception:  # noqa: BLE001
            pass
        return False

    def _put_object_via_presigned(self, key: str, data: bytes, content_type: str, public: bool, message: str | None = None) -> None:
        ''' 有的S3服务提供商不兼容标准AWS SDK, 需要使用Presigned URL进行上传 '''
        if not self._config.over_write_existing and self.exist(key):
            logger.info(f'Object {key} already exists, skipping upload')
            return
        params: dict = {"Bucket": self.bucket, "Key": key, "ContentType": content_type}
        if public:
            params["ACL"] = "public-read"
        url = self._s3.generate_presigned_url(
            "put_object",
            Params=params,
            ExpiresIn=600,
        )
        headers = {"Content-Type": content_type}
        if public:
            headers["x-amz-acl"] = "public-read"
        req = urllib.request.Request(url=url, data=data, method="PUT", headers=headers)
        try:
            with urllib.request.urlopen(req) as resp:  # nosec - URL is generated by AWS SDK
                status = getattr(resp, "status", None) or getattr(resp, "code", None) or 200
                if status >= 400:
                    raise RuntimeError(f"Presigned PUT failed with status {status}")
            message = message or f"Uploaded bytes as {key} (type: {content_type}, size: {format_size(len(data))})"
            logger.info(message)
        except urllib.error.HTTPError as he:  # noqa: PERF203
            raise RuntimeError(f"Presigned PUT failed: {he.code} {he.reason}") from he

    def download_file(self, key: str, local_path: str) -> None:
        key = self._key(key)
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        logger.info(f"Downloading '{key}' to '{local_path}'")
        self._s3.download_file(self.bucket, key, local_path)

    def delete_objects(self, keys: Iterable[str]) -> None:
        objects = [{"Key": self._key(k)} for k in keys]
        if not objects:
            return
        logger.warning(f"Deleting {len(objects)} object(s)")
        self._s3.delete_objects(Bucket=self.bucket, Delete={"Objects": objects})

    def object_url(self, key: str) -> str:
        # Always use presigned URL for fetching content (thumbnails/preview/download)
        key2 = self._key(key)
        url = self._s3.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": key2}, ExpiresIn=7 * 24 * 3600
        )
        logger.trace(f"Generated presigned URL for '{key2}'")
        return url

    def public_url_for_output(self, key: str, *, original_name: str | None = None) -> str:
        """Build a public URL for displaying/copying.

        If OUTPUT_URL_PATTERN is provided, render it; otherwise fall back to presigned URL.
        This must not be used for fetching thumbnails/preview.
        """
        pattern = (self.output_url_pattern or "").strip()
        if pattern:
            full_key = self._key(key)
            from picbed.utils import render_placeholders
            url = render_placeholders(
                pattern,
                original_name=original_name or os.path.basename(key),
                extra={
                    "key": key.lstrip("/"),
                    "fullKey": full_key.lstrip("/"),
                    "bucket": self.bucket,
                    "endpoint": (self._config.endpoint_url or "").rstrip("/"),
                    "prefix": self.prefix,
                },
            )
            logger.trace(f"Built output URL for '{key}' via pattern")
            return url
        return self.object_url(key)

