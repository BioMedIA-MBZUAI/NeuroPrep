from __future__ import annotations

import base64
import binascii
import html
import struct
import zlib
from pathlib import Path

import nibabel as nib
import numpy as np


def write_image_qc_report(
    image_path: Path,
    mask_path: Path | None,
    report_path: Path,
    title: str,
) -> Path:
    """Write a compact, dependency-free three-plane PNG embedded in an HTML report."""

    image = nib.load(str(image_path))
    data = np.asanyarray(
        image.dataobj[..., 0] if len(image.shape) == 4 else image.dataobj,
        dtype=np.float32,
    )
    if data.ndim != 3:
        raise ValueError(f"Visual QC expects a 3D/4D NIfTI: {image_path}")
    if mask_path is not None:
        mask = np.asanyarray(nib.load(str(mask_path)).dataobj, dtype=np.float32) > 0.5
    else:
        mask = np.isfinite(data) & (data != 0)
    canvas = _orthogonal_canvas(data, mask)
    png = _encode_png(canvas)
    encoded = base64.b64encode(png).decode("ascii")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    document = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<style>body{background:#111;color:#eee;font:14px sans-serif;margin:24px}"
        "img{max-width:100%;image-rendering:auto;border:1px solid #555}</style></head><body>"
        f"<h1>{html.escape(title)}</h1>"
        f"<p>{html.escape(str(image_path))}</p>"
        f"<img alt='orthogonal central slices with red brain-mask boundary' "
        f"src='data:image/png;base64,{encoded}'></body></html>"
    )
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(report_path)
    return report_path


def _orthogonal_canvas(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = np.isfinite(data) & mask
    values = data[valid]
    if values.size:
        low, high = np.percentile(values, (1.0, 99.0))
    else:
        finite = data[np.isfinite(data)]
        low, high = np.percentile(finite, (1.0, 99.0)) if finite.size else (0.0, 1.0)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = 0.0, 1.0
    scaled = np.clip((np.nan_to_num(data, nan=low) - low) / (high - low), 0, 1)
    grayscale = np.asarray(np.round(scaled * 255), dtype=np.uint8)
    boundary = mask & ~(
        np.roll(mask, 1, 0)
        & np.roll(mask, -1, 0)
        & np.roll(mask, 1, 1)
        & np.roll(mask, -1, 1)
        & np.roll(mask, 1, 2)
        & np.roll(mask, -1, 2)
    )
    indices = np.argwhere(mask)
    center = (
        tuple(int(round(value)) for value in indices.mean(axis=0))
        if indices.size
        else tuple(int(value // 2) for value in data.shape)
    )
    views = (
        (np.rot90(grayscale[center[0], :, :]), np.rot90(boundary[center[0], :, :])),
        (np.rot90(grayscale[:, center[1], :]), np.rot90(boundary[:, center[1], :])),
        (np.rot90(grayscale[:, :, center[2]]), np.rot90(boundary[:, :, center[2]])),
    )
    rendered: list[np.ndarray] = []
    for gray, edge in views:
        rgb = np.repeat(gray[..., None], 3, axis=2)
        rgb[edge] = (255, 32, 32)
        rendered.append(_resize_nearest(rgb, 192))
    gap = np.zeros((192, 6, 3), dtype=np.uint8)
    return np.concatenate((rendered[0], gap, rendered[1], gap, rendered[2]), axis=1)


def _resize_nearest(image: np.ndarray, height: int) -> np.ndarray:
    source_height, source_width = image.shape[:2]
    width = max(1, int(round(source_width * height / source_height)))
    rows = np.minimum((np.arange(height) * source_height / height).astype(int), source_height - 1)
    columns = np.minimum((np.arange(width) * source_width / width).astype(int), source_width - 1)
    return image[rows[:, None], columns[None, :]]


def _encode_png(rgb: np.ndarray) -> bytes:
    height, width, channels = rgb.shape
    if channels != 3 or rgb.dtype != np.uint8:
        raise ValueError("PNG encoder expects uint8 RGB data")
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        checksum = binascii.crc32(kind + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(
        b"IDAT", zlib.compress(raw, level=6)
    ) + chunk(b"IEND", b"")
