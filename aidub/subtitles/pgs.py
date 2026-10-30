"""Minimal reader for PGS (Blu-ray .sup) bitmap subtitles, for OCR.

Yields (start, end, RGBA image) for each displayed caption.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def _rle_decode(data: bytes, width: int, height: int) -> np.ndarray:
    out = np.zeros((height, width), dtype=np.uint8)
    x = y = 0
    i = 0
    n = len(data)
    try:
        while i < n and y < height:
            b = data[i]; i += 1
            if b != 0:
                if x < width:
                    out[y, x] = b
                x += 1
                continue
            b2 = data[i]; i += 1
            if b2 == 0:
                x, y = 0, y + 1
                continue
            flag = b2 >> 6
            if flag == 0:
                run, color = b2 & 0x3F, 0
            elif flag == 1:
                run, color = ((b2 & 0x3F) << 8) | data[i], 0; i += 1
            elif flag == 2:
                run, color = b2 & 0x3F, data[i]; i += 1
            else:
                run, color = ((b2 & 0x3F) << 8) | data[i], data[i + 1]; i += 2
            end = min(width, x + run)
            if x < width:
                out[y, x:end] = color
            x += run
    except IndexError:  # truncated object data: keep what was decoded
        pass
    return out


def _ycbcr_to_rgba(palette: dict[int, tuple[int, int, int, int]]) -> np.ndarray:
    lut = np.zeros((256, 4), dtype=np.uint8)
    for idx, (yy, cr, cb, a) in palette.items():
        r = yy + 1.402 * (cr - 128)
        g = yy - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
        bl = yy + 1.772 * (cb - 128)
        lut[idx] = [int(np.clip(r, 0, 255)), int(np.clip(g, 0, 255)), int(np.clip(bl, 0, 255)), a]
    return lut


LAST_CAPTION_S = 3.0


def read(path: str | Path):
    buf = Path(path).read_bytes()
    pos = 0
    palette: dict[int, tuple[int, int, int, int]] = {}
    objects: dict[int, dict] = {}
    composition: list[tuple[int, int, int]] = []
    shown: tuple[float, np.ndarray] | None = None
    while pos + 13 <= len(buf):
        if buf[pos:pos + 2] != b"PG":
            pos += 1
            continue
        pts = struct.unpack(">I", buf[pos + 2:pos + 6])[0] / 90000.0
        seg_type = buf[pos + 10]
        size = struct.unpack(">H", buf[pos + 11:pos + 13])[0]
        seg = buf[pos + 13:pos + 13 + size]
        pos += 13 + size
        if seg_type == 0x14:  # palette
            for k in range(2, len(seg) - 4, 5):
                palette[seg[k]] = (seg[k + 1], seg[k + 2], seg[k + 3], seg[k + 4])
        elif seg_type == 0x15:  # object
            oid = struct.unpack(">H", seg[0:2])[0]
            flags = seg[3]
            if flags & 0x80:
                w, h = struct.unpack(">HH", seg[7:11])
                objects[oid] = {"w": w, "h": h, "data": bytearray(seg[11:])}
            elif oid in objects:
                objects[oid]["data"] += seg[4:]
        elif seg_type == 0x16:  # presentation composition
            count = seg[10]
            composition = []
            off = 11
            for _ in range(count):
                oid = struct.unpack(">H", seg[off:off + 2])[0]
                cx, cy = struct.unpack(">HH", seg[off + 4:off + 8])
                cropped = seg[off + 3] & 0x80
                composition.append((oid, cx, cy))
                off += 16 if cropped else 8
            if count == 0 and shown is not None:
                yield shown[0], pts, shown[1]
                shown = None
        elif seg_type == 0x80:  # end of display set
            if composition and objects:
                lut = _ycbcr_to_rgba(palette)
                imgs = []
                for oid, cx, cy in composition:
                    o = objects.get(oid)
                    if not o:
                        continue
                    idx = _rle_decode(bytes(o["data"]), o["w"], o["h"])
                    imgs.append((cy, lut[idx]))
                if imgs:
                    imgs.sort(key=lambda t: t[0])
                    width = max(i.shape[1] for _, i in imgs)
                    stacked = np.concatenate([np.pad(i, ((0, 0), (0, width - i.shape[1]), (0, 0))) for _, i in imgs])
                    if shown is not None:
                        yield shown[0], pts, shown[1]
                    shown = (pts, stacked)
                composition = []
    if shown is not None:  # the stream ended without clearing the last caption
        yield shown[0], shown[0] + LAST_CAPTION_S, shown[1]
