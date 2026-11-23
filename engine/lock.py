"""One learner per folder: a lock file held for as long as the engine runs.

The OS releases the lock when the process dies, even on `kill -9`, so a stale lock file never blocks a restart.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


class AlreadyRunning(RuntimeError):
    pass


class FolderLock:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fh = None

    def __enter__(self) -> "FolderLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a+")
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._fh.close()
            self._fh = None
            raise AlreadyRunning(f"another learner is already running on {self.path.parent}") from exc
        self._fh.seek(0)
        self._fh.truncate()
        self._fh.write(str(os.getpid()))
        self._fh.flush()
        return self

    def __exit__(self, *exc) -> None:
        if self._fh is not None:
            self._fh.close()  # closing releases the lock on every platform
            self._fh = None
