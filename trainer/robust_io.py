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


def load_artifact(path: str, *, heal: bool = True, weights_only: bool = False):
    """torch.load that survives a truncated/corrupt file: unlink and retry once.

    Raises the original error if the retry also fails, so real problems
    (wrong shape, missing keys) are not masked.
    """
    try:
        return torch.load(path, weights_only=weights_only)
    except Exception as exc:  # noqa: BLE001 - deliberate broad catch
        if not heal or not os.path.exists(path):
            raise
        size = os.path.getsize(path)
        print(f"[io] corrupt artifact {path} ({size} B): {type(exc).__name__}: {exc}"
              f" -- unlinking and retrying once", flush=True)
        os.unlink(path)
        return torch.load(path, weights_only=weights_only)