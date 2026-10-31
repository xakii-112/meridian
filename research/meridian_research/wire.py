"""Meridian Wire Protocol (MWP) tape writer - byte-for-byte compatible with engine/include/meridian/wire.hpp.

A tape is a 40-byte header followed by fixed 48-byte little-endian inbound frames. The research
simulator records every order it sends so the C++ engine can replay the exact same flow and
confirm it produces the identical trade tape (see ``meridian_replay``).
"""

from __future__ import annotations

import struct
from pathlib import Path

#  type side ordtype tif flags stp symbol owner reserved cl_ord_id price qty timestamp
FRAME = struct.Struct("<BBBBBBHiIqqqq")
HEADER = struct.Struct("<4sHHIIQqq")  # magic version symbol frame_size reserved count min max
assert FRAME.size == 48 and HEADER.size == 40

ENTER, CANCEL, REPLACE = ord("O"), ord("X"), ord("U")
LIMIT, MARKET = 0, 1
GTC, IOC, FOK = 0, 1, 2
POST_ONLY = 1


class TapeWriter:
    """Accumulates inbound frames in memory; ``save`` writes header + frames."""

    def __init__(self, symbol: int = 0, min_price: int = 1, max_price: int = 1 << 20) -> None:
        self.symbol = symbol
        self.min_price = min_price
        self.max_price = max_price
        self._frames: list[bytes] = []
        self._pack = FRAME.pack

    def __len__(self) -> int:
        return len(self._frames)

    def enter(self, owner: int, cl: int, side: int, price: int, qty: int, ioc: bool, market: bool,
              post_only: bool, ts_ns: int) -> None:
        self._frames.append(self._pack(ENTER, side, MARKET if market else LIMIT, IOC if ioc else GTC,
                                       POST_ONLY if post_only else 0, 0, self.symbol, owner, 0, cl,
                                       0 if market else price, qty, ts_ns))

    def cancel(self, owner: int, cl: int, ts_ns: int) -> None:
        self._frames.append(self._pack(CANCEL, 0, 0, 0, 0, 0, self.symbol, owner, 0, cl, 0, 0, ts_ns))

    def to_bytes(self) -> bytes:
        header = HEADER.pack(b"MWP1", 1, self.symbol, FRAME.size, 0, len(self._frames), self.min_price,
                             self.max_price)
        return header + b"".join(self._frames)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_bytes(self.to_bytes())
        return path


def read_tape(data: bytes) -> tuple[dict, list[tuple]]:
    """Parse a tape back into (header, frames) - used by tests to check the layout."""
    magic, version, symbol, frame_size, _, count, lo, hi = HEADER.unpack_from(data, 0)
    if magic != b"MWP1" or version != 1 or frame_size != FRAME.size:
        raise ValueError("not an MWP1 tape")
    frames = [FRAME.unpack_from(data, HEADER.size + i * FRAME.size) for i in range(count)]
    return {"symbol": symbol, "count": count, "min_price": lo, "max_price": hi}, frames
