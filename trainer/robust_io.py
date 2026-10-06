"""Crash-safe artifact I/O for long-running training jobs.

The 2026-10-05 OOM kill landed inside `torch.save(...)`, leaving a 0-byte
cache that made every later run die with `EOFError: Ran out of input`.
Two rules fix that class of failure for good:

  1. WRITES ARE ATOMIC — serialize to `<path>.tmp-<pid>`, fsync, then
     os.replace(). A crash leaves either the old file or nothing, never a
     half-written one that looks valid.
  2. LOADS ARE VERIFIED — `load_artifact` retries once after unlinking a
     corrupt file, so a run heals instead of dying.
"""
from __future__ import annotations

import os

import torch


def save_atomic(obj, path: str) -> str:
    """torch.save to a temp file, fsync, then atomically rename into place."""
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "wb") as fh:
        torch.save(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def _is_truncation(exc: BaseException) -> bool:
    """True only for the signatures of a half-written artifact.

    Deliberately narrow: an earlier version healed on ANY exception and would
    delete a perfectly good checkpoint when, say, `weights_only=True` was
    passed for a numpy array. Destroying an 18 h run's output to retry a load
    is far worse than surfacing the error.
    """
    if isinstance(exc, EOFError):
        return True
    msg = str(exc).lower()
    return ("ran out of input" in msg
            or "unexpected end of file" in msg
            or "pytorchstreamreader failed" in msg
            or "truncated" in msg)


def load_artifact(path: str, *, heal: bool = True, weights_only: bool = False):
    """torch.load that survives a truncated/corrupt file: unlink and retry once.

    Only truncation signatures trigger the unlink (see `_is_truncation`); every
    other error is re-raised with the file untouched. Raises the retry's error
    if the retry also fails, so real problems are not masked.
    """
    try:
        return torch.load(path, weights_only=weights_only)
    except Exception as exc:  # noqa: BLE001 - re-raised unless truncation
        if not heal or not os.path.exists(path) or not _is_truncation(exc):
            raise
        size = os.path.getsize(path)
        print(f"[io] truncated artifact {path} ({size} B): {type(exc).__name__}: "
              f"{exc} -- unlinking and rebuilding once", flush=True)
        os.unlink(path)
        return torch.load(path, weights_only=weights_only)
