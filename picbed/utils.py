from __future__ import annotations

import datetime
import hashlib
import json
import os
import random
import re
import string
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from threading import Lock, RLock
from typing import Any, Iterator

from loguru import logger
from PySide6.QtCore import QBuffer, QByteArray
from PySide6.QtGui import QGuiApplication, QImage
from PySide6.QtWidgets import QApplication

_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9._-]+")
_TPL_RE = re.compile(r"\{([A-Za-z0-9_]+)(?::(\d+))?\}")

SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".svg", ".ico")
SUPPORT_PLACEHOLDERS = (
    'year',
    'month',
    'day',
    'hour',
    'minute',
    'second',
    'millisecond',
    'timestamp',
    'timestampMS',
    'fullName',
    'fileName',
    'extName',
    'md5',
    'sha1',
    'sha256',
    'randstr:n',
    'randnum:n',
    'fullName',
    'fileName',
    'extName',
)
OUTPUT_PLACEHOLDERS = (
    "key",
    "fullKey",
    "bucket",
    "endpoint",
    "prefix",
)



def is_image_key(key: str) -> bool:
    lower = key.lower()
    return any(lower.endswith(ext) for ext in SUPPORTED_IMAGE_EXTENSIONS)


def slugify_filename(filename: str) -> str:
    base, ext = os.path.splitext(filename)
    base = _SAFE_NAME_RE.sub("-", base).strip("-._") or "file"
    ext = _SAFE_NAME_RE.sub("", ext)
    return f"{base}{ext}" if ext else base


def auto_rename(filename: str, content_bytes: bytes | None = None) -> str:
    base, ext = os.path.splitext(slugify_filename(filename))
    unique = uuid.uuid4().hex[:8]
    if content_bytes:
        content_hash = hashlib.sha1(content_bytes).hexdigest()[:8]
        unique = f"{unique}-{content_hash}"
    ts = time.strftime("%Y%m%d-%H%M%S")
    new_name = f"{base}-{ts}-{unique}{ext}"
    logger.debug(f"Auto-renamed '{filename}' -> '{new_name}'")
    return new_name

def _now_parts(now: datetime.datetime | None = None) -> dict[str, str]:
    dt = now or datetime.datetime.now()
    ts = int(dt.timestamp())
    ts_ms = int(dt.timestamp() * 1000)
    return {
        "year": f"{dt.year:04d}",
        "month": f"{dt.month:02d}",
        "day": f"{dt.day:02d}",
        "hour": f"{dt.hour:02d}",
        "minute": f"{dt.minute:02d}",
        "second": f"{dt.second:02d}",
        "millisecond": f"{dt.microsecond // 1000:03d}",
        "timestamp": str(ts),
        "timestampMS": str(ts_ms),
    }

def _random_string(length: int) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(random.choice(alphabet) for _ in range(max(0, int(length))))

def _random_digits(length: int) -> str:
    return "".join(random.choice(string.digits) for _ in range(max(0, int(length))))

def _compute_hashes(content_bytes: bytes | None) -> dict[str, str]:
    if not content_bytes:
        return {"md5": "", "sha1": "", "sha256": ""}
    return {
        "md5": hashlib.md5(content_bytes).hexdigest(),  # noqa: S324 - accepted here for fingerprinting
        "sha1": hashlib.sha1(content_bytes).hexdigest(),
        "sha256": hashlib.sha256(content_bytes).hexdigest(),
    }

def normalize_s3_key_path(path: str) -> str:
    # Use forward slashes and collapse duplicates
    path = path.replace("\\", "/")
    path = re.sub(r"(?<!:)/+", "/", path)
    return path.strip("/")

def render_placeholders(
        template: str,
        *,
        original_name: str,
        content_bytes: bytes | None = None,
        file_path: str | None = None,
        now: datetime.datetime | None = None,
        extra: dict[str, str] | None = None,
    ) -> str:
    """Render a template with placeholders.

    Supported placeholders (general):
      {year} {month} {day} {hour} {minute} {second} {millisecond} {timestamp} {timestampMS}

    Upload path extras:
      {fullName} {fileName} {extName} {md5} {sha1} {sha256} {randstr:n} {randnum:n}

    Additional variables can be provided via `extra`.
    """
    # Prepare name parts
    base, ext = os.path.splitext(slugify_filename(original_name))
    full = f"{base}{ext}" if ext else base
    ext_name = ext[1:] if ext.startswith(".") else ext

    # Load content if needed for hashes
    needs_hash = any(x in template for x in ("{md5}", "{sha1}", "{sha256}"))
    if needs_hash and content_bytes is None and file_path:
        try:
            with open(file_path, "rb") as f:
                content_bytes = f.read()
        except Exception:
            content_bytes = None

    context: dict[str, str] = {}
    context.update(_now_parts(now))
    context.update({
        "fullName": full,
        "fileName": base,
        "extName": ext_name,
    })
    context.update(_compute_hashes(content_bytes))
    if extra:
        context.update({k: str(v) for k, v in extra.items()})

    def _replace(m: re.Match[str]) -> str:
        key = m.group(1)
        count = m.group(2)
        if key == "randstr":
            n = int(count) if count else 6
            return _random_string(n)
        if key == "randnum":
            n = int(count) if count else 6
            return _random_digits(n)
        return context.get(key, m.group(0))

    rendered = _TPL_RE.sub(_replace, template)
    return normalize_s3_key_path(rendered)

def read_clipboard_image_bytes() -> tuple[bytes, str] | None:
    app = QApplication.instance() or QGuiApplication.instance()
    if app is None:
        _ = QGuiApplication([])
    clipboard = QGuiApplication.clipboard()
    mime = clipboard.mimeData()
    if mime is None:
        return None
    if mime.hasImage():
        image = clipboard.image()
        if image.isNull():
            return None
        qimg: QImage = image.convertToFormat(QImage.Format.Format_RGBA8888)
        # Prefer PNG lossless
        byte_array = QByteArray()
        buffer = QBuffer(byte_array)
        buffer.open(QBuffer.OpenModeFlag.WriteOnly)
        qimg.save(buffer, "PNG")
        buffer.close()
        logger.info(f"Captured image from clipboard ({format_size(len(byte_array))} bytes)")
        return bytes(byte_array), "image/png"
    if mime.hasUrls():
        # If user copied a file
        urls = mime.urls()
        if urls:
            local_path = urls[0].toLocalFile()
            if local_path and os.path.isfile(local_path):
                with open(local_path, "rb") as f:
                    data = f.read()
                from mimetypes import guess_type

                ctype = guess_type(local_path)[0] or "application/octet-stream"
                logger.info(f"Read file from clipboard: '{local_path}' ({format_size(len(data))} bytes, {ctype})")
                return data, ctype
    return None

def format_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.2f} KB"
    if size < 1024 * 1024 * 1024:
        return f"{size / 1024 / 1024:.2f} MB"
    return f"{size / 1024 / 1024 / 1024:.2f} GB"

def setup_stdout() -> None:
    if sys.stdout is None:
        # 重定向到空设备
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        # 重定向到空设备
        sys.stderr = open(os.devnull, "w")

def json_escape(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


@contextmanager
def acquire_lock(lock: Lock|RLock, block: bool = True, timeout: float = 1.5, already_acquired: bool = False) -> Iterator[bool]:
    """Context manager to acquire a lock with timeout.

    Yields True if the lock was acquired within the timeout, otherwise False.

    """
    if already_acquired:
        yield True
        return
    acquired = False
    try:
        acquired = bool(lock.acquire(blocking=block, timeout=timeout))
        yield acquired
    finally:
        if acquired:
            try:
                lock.release()
            except Exception:
                # Keep silent to avoid masking original exceptions; optionally log
                logger.exception("Failed to release lock in acquire_timeout")


def is_nuitka():
    try:
        import __main__
        return hasattr(__main__, '__compiled__')
    except ImportError:
        return False


def appdir():
    """获取应用目录
    
    对于 Nuitka 打包的应用，返回打包后的目录；
    对于普通 Python 应用，返回当前工作目录。
    """
    if is_nuitka():
        return Path(sys.executable).parent
    else:
        return Path('.')
