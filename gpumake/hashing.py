"""File hashing with a cache keyed by (size, mtime): big data files are hashed once."""
from __future__ import annotations

from pathlib import Path

from .util import log, read, save, sha


class HashCache:
    """sha256 of files, re-hashing a file only when its size or modification time changes."""

    def __init__(self, path):
        self.path = Path(path)
        try:
            self.values = read(self.path) if self.path.exists() else {}
        except ValueError:
            self.values = {}
        self.used = {}
        self.hashed_bytes = 0

    def __call__(self, path) -> str:
        path = Path(path)
        st = path.stat()
        key = str(path.resolve())
        signature = [st.st_size, st.st_mtime_ns]
        cached = self.values.get(key)
        if cached and cached[:2] == signature:
            value = cached[2]
        else:
            if st.st_size > 64 * 1024 ** 2:
                log(f"hashing {path} ({st.st_size / 1024 ** 2:.0f} MiB)")
            value = sha(path)
            self.hashed_bytes += st.st_size
        self.used[key] = signature + [value]
        return value

    def save(self) -> None:
        """Keep only the entries used in this scan, so deleted files drop out."""
        save(self.path, self.used)
