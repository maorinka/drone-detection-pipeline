"""Small shared helpers: atomic JSON, content hashes, locks, the spare-disk rule, logging."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

DEFAULT_MIN_FREE_FRACTION = 0.30
MIN_FREE_BYTES = 10 * 1024 ** 3


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S ") + message, flush=True)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, obj) -> None:
    """Write JSON atomically (temp file + rename), so readers never see half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(obj, f, indent=2, sort_keys=False, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def sha(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 ** 2), b""):
            h.update(block)
    return h.hexdigest()


def identity(obj) -> str:
    """Stable hash of a JSON-serialisable value (key order does not matter)."""
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


class LockError(RuntimeError):
    pass


@contextlib.contextmanager
def lock(path):
    """Non-blocking exclusive lock: a second holder fails at once (LockError) instead of waiting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as f:
        if not path.stat().st_size:
            f.write(b"0")
            f.flush()
        f.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                raise LockError(f"another gpumake process holds {path}") from None
        else:
            import fcntl
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise LockError(f"another gpumake process holds {path}") from None
        try:
            yield
        finally:
            f.seek(0)
            if os.name == "nt":
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def reserve_bytes(total: int, fraction: float) -> int:
    return max(MIN_FREE_BYTES if total > 4 * MIN_FREE_BYTES else 0, int(fraction * total))


def space_ok(path, fraction: float = DEFAULT_MIN_FREE_FRACTION, needed: int = 0) -> bool:
    """True if writing ``needed`` more bytes keeps the disk above the spare-space rule."""
    if fraction <= 0:
        return True
    usage = shutil.disk_usage(path)
    return usage.free - needed >= reserve_bytes(usage.total, fraction)


def python_version_ok() -> None:
    if sys.version_info < (3, 8):
        raise RuntimeError("gpumake needs Python 3.8 or newer")
