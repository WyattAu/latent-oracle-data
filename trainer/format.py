"""Shard format v1 reader (see ../README.md for the field table).

64-byte fixed records; decode with numpy structured dtype for speed.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

MAGIC = 0x48534F4C  # "LOSH" LE
VERSION = 1
HEADER_SIZE = 16
RECORD_SIZE = 64

REC_DTYPE = np.dtype([
    ("side", "<u1"),
    ("castling", "<u1"),
    ("ep", "<u1"),
    ("halfmove", "<u1"),
    ("fullmove", "<u2"),
    ("n_targets", "<u1"),
    ("pad", "<u1"),
    ("board", "<u4", (8,)),      # 32 bytes as 8 u32; nibble order a1 = low nibble of byte 0
    ("eval_cp", "<i2"),
    ("targets", "<u1", (9,)),    # 3 x (from, to, promo)
    ("wdl", "<f4", (3,)),
    ("pad2", "<u1"),
])
assert REC_DTYPE.itemsize == RECORD_SIZE, REC_DTYPE.itemsize

PROMO_TO_UCI = {0: "n", 1: "b", 2: "r", 3: "q"}
PIECE_CHARS = {1: "P", 2: "N", 3: "B", 4: "R", 5: "Q", 6: "K",
               9: "p", 10: "n", 11: "b", 12: "r", 13: "q", 14: "k"}


@dataclass
class Sample:
    board_codes: np.ndarray   # (64,) u8 piece codes
    side: int
    targets: list            # [(from, to, promo)] trimmed to n_targets
    eval_cp: int
    wdl: np.ndarray          # (3,) float32, white POV
    castling: int = 0        # rights bitmask (shard convention)
    ep: int = 255            # ep square, 255 = none


def read_header(path: str) -> tuple[int, int]:
    with open(path, "rb") as f:
        magic, version, hdr = struct.unpack("<IHH", f.read(8))
    if magic != MAGIC:
        raise ValueError(f"bad shard magic {magic:#x} in {path}")
    if version != VERSION:
        raise ValueError(f"unsupported shard version {version}")
    return version, hdr


def iter_records(path: str):
    """Yield Sample objects. Streaming: reads in 8 MiB blocks."""
    read_header(path)
    with open(path, "rb") as f:
        f.seek(HEADER_SIZE)
        while True:
            block = f.read(RECORD_SIZE * 65536)
            if not block:
                break
            block = block[: len(block) // RECORD_SIZE * RECORD_SIZE]
            arr = np.frombuffer(block, dtype=REC_DTYPE)
            for rec in arr:
                n = int(rec["n_targets"])
                t = rec["targets"]
                targets = [(int(t[3 * i]), int(t[3 * i + 1]), int(t[3 * i + 2])) for i in range(min(n, 3))]
                targets = [x for x in targets if x[0] != 255]
                board_codes = np.empty(64, dtype=np.uint8)
                raw = rec["board"].tobytes()
                for sq in range(64):
                    board_codes[sq] = (raw[sq >> 1] >> ((sq & 1) * 4)) & 0x0F
                yield Sample(
                    board_codes=board_codes,
                    side=int(rec["side"]),
                    targets=targets,
                    eval_cp=int(rec["eval_cp"]),
                    wdl=rec["wdl"].astype(np.float32),
                    castling=int(rec["castling"]),
                    ep=int(rec["ep"]),
                )


def count_records(path: str) -> int:
    import os
    return (os.path.getsize(path) - HEADER_SIZE) // RECORD_SIZE


def square_name(sq: int) -> str:
    return chr(ord("a") + sq % 8) + chr(ord("1") + sq // 8)


def move_to_uci(t: tuple[int, int, int]) -> str:
    s = square_name(t[0]) + square_name(t[1])
    if t[2] != 255:
        s += PROMO_TO_UCI.get(t[2], "")
    return s


class MaskSidecar:
    """mmap reader for legal-move sidecars ("LOMS" v1, see src/masks.rs).

    mask_indices(i) -> np.ndarray of flat legal indices (from*64+to) for
    record i. Training uses these to build the CE mask without python-chess.
    """

    def __init__(self, path: str):
        import mmap
        self._f = open(path, "rb")
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        magic, version, _res, count = struct.unpack("<IHHQ", self._mm[:16])
        if magic != 0x534D4F4C or version != 1:
            raise ValueError(f"bad mask sidecar {path}")
        self.count = count
        self.offsets = np.frombuffer(self._mm, dtype="<u8", count=count + 1, offset=16)
        # payload begins after the header and the (count+1)-entry offset table
        self._base = 16 + (count + 1) * 8

    def mask_indices(self, i: int) -> np.ndarray:
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        return np.frombuffer(self._mm, dtype="<u2", count=(b - a - 1) // 2,
                             offset=self._base + a + 1)  # +1 skips the n_moves byte

    def close(self):
        self._mm.close()
        self._f.close()
