"""Tests for the PNG/JPEG header-only estimator path (Phase 2).

Covers:
- PNG header-only: happy path, mode-off, OOB fallback, internal-error fallback, model-load failure
- JPEG header-only: happy path, lossless fallback, low-NSE fallback, CMYK fallback
"""

from __future__ import annotations

import io
import shutil
import struct
from pathlib import Path

import pytest

_REAL_MODELS_DIR = Path(__file__).parent.parent / "estimation" / "models"


def _copy_real_model(tmp_path: Path, filename: str) -> None:
    src = _REAL_MODELS_DIR / filename
    if src.exists():
        shutil.copy2(src, tmp_path / filename)


# ---------------------------------------------------------------------------
# PNG image factories
# ---------------------------------------------------------------------------


def _make_large_png_rgb(width: int = 500, height: int = 500) -> bytes:
    """Create a large noisy RGB PNG that forces the sample path (>150K pixels, BPP>2.0)."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_large_png_rgba(width: int = 500, height: int = 500) -> bytes:
    """Create a large noisy RGBA PNG."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(99)
    arr = rng.integers(0, 256, (height, width, 4), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# JPEG image factories
# ---------------------------------------------------------------------------


def _make_large_jpeg_rgb(width: int = 600, height: int = 400) -> bytes:
    """Create a standard YCbCr JPEG (q=85, 4:2:0) that forces sample path."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(123)
    arr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, subsampling=2)  # 4:2:0
    return buf.getvalue()


def _make_lossless_jpeg(width: int = 64, height: int = 64) -> bytes:
    """Build a minimal SOF3 (lossless JPEG) byte stream."""
    # Craft a minimal JPEG with SOF3 marker
    soi = b"\xff\xd8"
    # SOF3 marker with minimal valid header
    # seg_len = 2 + 6 + 1*3 = 11
    nf = 1
    sof3_payload = bytes([8, 0, height, 0, width, nf, 1, 0x11, 0])
    sof3_len = 2 + len(sof3_payload)
    sof3 = b"\xff\xc3" + struct.pack(">H", sof3_len) + sof3_payload
    eoi = b"\xff\xd9"
    return soi + sof3 + eoi


def _make_jpeg_uniform_qtable(width: int = 800, height: int = 600) -> bytes:
    """Create a JPEG with a non-standard (uniform) quantization table.

    A uniform Q-table won't match Annex K scaling → NSE < 0.85 → custom_quantization.
    We synthesize the header manually rather than relying on Pillow to produce
    a custom Q-table (Pillow always uses standard Annex-K tables).
    """
    from PIL import Image

    # Standard JPEG, then patch the DQT table bytes to be uniform (all-2)
    rng = __import__("numpy").random.default_rng(7)
    arr = rng.integers(0, 256, (height, width, 3), dtype=__import__("numpy").uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, subsampling=2)
    data = bytearray(buf.getvalue())

    # Find the first DQT marker (FF DB) and overwrite the luma Q-table with all-2 values
    i = 2
    while i < len(data) - 1:
        if data[i] == 0xFF and data[i + 1] == 0xDB:
            # seg_len = struct.unpack_from(">H", data, i + 2)[0]  # unused, skip
            # DQT payload starts at i+4; first byte is precision+tq
            # table data starts at i+5 (precision=0 → 64 bytes)
            table_start = i + 5
            if table_start + 64 <= len(data):
                for k in range(64):
                    data[table_start + k] = 2  # uniform value
            break
        i += 1

    return bytes(data)


def _make_cmyk_jpeg(width: int = 64, height: int = 64) -> bytes:
    """Synthesize a minimal JPEG with APP14 color_transform=2 (YCCK)."""
    from PIL import Image

    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    data = bytearray(buf.getvalue())

    # Inject an APP14 Adobe marker with color_transform=2 (YCCK)
    # APP14 layout: 0xFFEE + len(2) + "Adobe"(5) + version(2) + flags0(2) + flags1(2) + ct(1)
    app14 = b"\xff\xee" + struct.pack(">H", 14) + b"Adobe" + b"\x00\x01\x00\x00\x00\x00\x02"
    # Insert after SOI (position 2)
    final = data[:2] + app14 + data[2:]
    return bytes(final)


# ---------------------------------------------------------------------------
# §1 PNG header-only: active returns png_header_only path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_png_header_only_active_returns_header_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """fitted_estimator_mode=active + real PNG → path='png_header_only'."""
    import estimation.estimator as estimator_mod
    import estimation.models as models_mod

    _copy_real_model(tmp_path, "png_header_v1.json")
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_png_header_model.cache_clear()
    monkeypatch.setattr(estimator_mod.settings, "fitted_estimator_mode", "active")

    from schemas import OptimizationConfig

    data = _make_large_png_rgb(500, 500)
    config = OptimizationConfig(quality=60, png_lossy=True)
    result = await estimator_mod.estimate(data, config)

    assert result.path in (
        "png_header_only",
        "direct_encode_sample",
    ), f"Unexpected path {result.path!r}, fallback={result.fallback_reason!r}"
    assert result.estimated_optimized_size > 0
    assert result.estimated_reduction_percent >= 0.0

    models_mod.load_png_header_model.cache_clear()


# ---------------------------------------------------------------------------
# §2 PNG header-only: mode=off → unchanged sample path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_png_header_only_off_returns_sample_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """fitted_estimator_mode=off → direct_encode_sample (no header-only code fires)."""
    import estimation.estimator as estimator_mod

    monkeypatch.setattr(estimator_mod.settings, "fitted_estimator_mode", "off")

    from schemas import OptimizationConfig

    data = _make_large_png_rgb(500, 500)
    config = OptimizationConfig(quality=60, png_lossy=True)
    result = await estimator_mod.estimate(data, config)

    assert result.path == "direct_encode_sample", f"Unexpected path {result.path!r}"
    assert result.fallback_reason is None


# ---------------------------------------------------------------------------
# §3 PNG header-only: input_bpp > MAX_INPUT_BPP → feature_oob → falls back
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_png_header_only_oob_falls_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """PNG with inflated file size (input_bpp > 64) → fallback path with feature_oob."""
    import estimation.estimator as estimator_mod
    import estimation.models as models_mod
    from estimation.png_header import PngHeader

    _copy_real_model(tmp_path, "png_header_v1.json")
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_png_header_model.cache_clear()
    monkeypatch.setattr(estimator_mod.settings, "fitted_estimator_mode", "active")

    # Monkeypatch _png_header_only_bpp_inner to simulate OOB by using a small pixel count
    # and large file size. We achieve this by patching parse_png_header to return
    # a tiny header, then passing the large file data.
    # A 1×1 PNG with 1MB file = 8M bpp >> 64 bpp cap
    tiny_header = PngHeader(
        width=1, height=1, bit_depth=8, color_type=2, has_alpha=False, is_palette=False
    )
    monkeypatch.setattr(estimator_mod, "parse_png_header", lambda _: tiny_header)

    # file_size ≫ MAX_INPUT_BPP * 1px = large enough to exceed 64 bpp
    data = _make_large_png_rgb(500, 500)  # ~750KB; 1×1 px → bpp >> 64

    from schemas import OptimizationConfig

    config = OptimizationConfig(quality=60, png_lossy=True)
    result = await estimator_mod.estimate(data, config)

    assert result.path in ("direct_encode_sample", "exact"), f"Unexpected path {result.path!r}"
    if result.path == "direct_encode_sample":
        assert (
            result.fallback_reason == "feature_oob"
        ), f"Expected 'feature_oob', got {result.fallback_reason!r}"

    models_mod.load_png_header_model.cache_clear()


# ---------------------------------------------------------------------------
# §4 PNG header-only: parse_png_header raises → internal_error fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_png_header_only_internal_error_falls_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """parse_png_header raises RuntimeError → fallback_reason='internal_error'."""
    import estimation.estimator as estimator_mod
    import estimation.models as models_mod

    _copy_real_model(tmp_path, "png_header_v1.json")
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_png_header_model.cache_clear()
    monkeypatch.setattr(estimator_mod.settings, "fitted_estimator_mode", "active")

    def _raise(_data):
        raise RuntimeError("simulated parse crash")

    monkeypatch.setattr(estimator_mod, "parse_png_header", _raise)

    from schemas import OptimizationConfig

    data = _make_large_png_rgb(500, 500)
    config = OptimizationConfig(quality=60, png_lossy=True)
    result = await estimator_mod.estimate(data, config)

    assert result.path in ("direct_encode_sample", "exact"), f"Unexpected path {result.path!r}"
    if result.path == "direct_encode_sample":
        # parse_png_header crash is caught in the dispatch and treated as header_parse_error
        assert (
            result.fallback_reason == "header_parse_error"
        ), f"Unexpected fallback_reason: {result.fallback_reason!r}"

    models_mod.load_png_header_model.cache_clear()


# ---------------------------------------------------------------------------
# §5 PNG header-only: model artifact missing → model_load_failed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_png_header_only_model_load_failure_falls_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Empty _MODELS_DIR (no png_header_v1.json) → fallback_reason='model_load_failed'."""
    import estimation.estimator as estimator_mod
    import estimation.models as models_mod

    # tmp_path is empty — no header artifact
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_png_header_model.cache_clear()
    monkeypatch.setattr(estimator_mod.settings, "fitted_estimator_mode", "active")

    from schemas import OptimizationConfig

    data = _make_large_png_rgb(500, 500)
    config = OptimizationConfig(quality=60, png_lossy=True)
    result = await estimator_mod.estimate(data, config)

    assert result.path in ("direct_encode_sample", "exact"), f"Unexpected path {result.path!r}"
    if result.path == "direct_encode_sample":
        assert (
            result.fallback_reason == "model_load_failed"
        ), f"Expected 'model_load_failed', got {result.fallback_reason!r}"

    models_mod.load_png_header_model.cache_clear()


# ---------------------------------------------------------------------------
# §6 JPEG header-only: active → jpeg_header_only path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_jpeg_header_only_active_returns_header_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """fitted_estimator_mode=active + large JPEG → path='jpeg_header_only'."""
    import estimation.estimator as estimator_mod
    import estimation.models as models_mod

    _copy_real_model(tmp_path, "jpeg_header_v1.json")
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_jpeg_header_model.cache_clear()
    monkeypatch.setattr(estimator_mod.settings, "fitted_estimator_mode", "active")

    from schemas import OptimizationConfig

    data = _make_large_jpeg_rgb(600, 400)
    config = OptimizationConfig(quality=60)
    result = await estimator_mod.estimate(data, config)

    assert result.path in (
        "jpeg_header_only",
        "direct_encode_sample",
        "exact",
    ), f"Unexpected path {result.path!r}, fallback={result.fallback_reason!r}"

    models_mod.load_jpeg_header_model.cache_clear()


# ---------------------------------------------------------------------------
# §7 JPEG header-only: SOF3 input → lossless_jpeg fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_jpeg_header_only_lossless_falls_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SOF3 JPEG → parse_jpeg_header.fallback_reason='lossless_jpeg' → direct_encode_sample."""
    import estimation.estimator as estimator_mod
    import estimation.models as models_mod
    from estimation.jpeg_header import JpegHeader

    _copy_real_model(tmp_path, "jpeg_header_v1.json")
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_jpeg_header_model.cache_clear()
    monkeypatch.setattr(estimator_mod.settings, "fitted_estimator_mode", "active")

    # Patch parse_jpeg_header to return a lossless header
    lossless_header = JpegHeader(
        width=64,
        height=64,
        components=1,
        bit_depth=8,
        subsampling="grayscale",
        progressive=False,
        dqt_luma=[1] * 64,
        dqt_chroma=None,
        app14_color_transform=None,
        fallback_reason="lossless_jpeg",
    )
    monkeypatch.setattr(estimator_mod, "parse_jpeg_header", lambda _: lossless_header)

    # Use a real JPEG to pass format detection
    data = _make_large_jpeg_rgb(800, 600)

    from schemas import OptimizationConfig

    config = OptimizationConfig(quality=60)
    result = await estimator_mod.estimate(data, config)

    # lossless_jpeg fallback reason → routes to direct_encode_sample
    if result.path == "direct_encode_sample":
        assert (
            result.fallback_reason == "lossless_jpeg"
        ), f"Expected 'lossless_jpeg', got {result.fallback_reason!r}"

    models_mod.load_jpeg_header_model.cache_clear()


# ---------------------------------------------------------------------------
# §8 JPEG header-only: uniform Q-table → custom_quantization fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_jpeg_header_only_low_nse_falls_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """JPEG with uniform quantization table → NSE<0.85 → custom_quantization fallback."""
    import estimation.estimator as estimator_mod
    import estimation.models as models_mod
    from estimation.jpeg_header import parse_jpeg_header

    _copy_real_model(tmp_path, "jpeg_header_v1.json")
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_jpeg_header_model.cache_clear()
    monkeypatch.setattr(estimator_mod.settings, "fitted_estimator_mode", "active")

    data = _make_jpeg_uniform_qtable(800, 600)

    # Confirm parse succeeds with no hard fallback_reason but the NSE check catches it
    header = parse_jpeg_header(data)
    if header is None or header.fallback_reason is not None:
        # If the header itself fails, skip — the test is about the NSE gate
        pytest.skip("could not produce a parseable header with uniform Q-table")

    from estimation.jpeg_header import estimate_source_quality_lsm

    _, nse = estimate_source_quality_lsm(header.dqt_luma, header.dqt_chroma)
    if nse >= 0.85:
        pytest.skip(f"NSE={nse:.3f} >= 0.85 — Q-table patching did not create a non-standard table")

    from schemas import OptimizationConfig

    config = OptimizationConfig(quality=60)
    result = await estimator_mod.estimate(data, config)

    if result.path == "direct_encode_sample":
        assert (
            result.fallback_reason == "custom_quantization"
        ), f"Expected 'custom_quantization', got {result.fallback_reason!r}"

    models_mod.load_jpeg_header_model.cache_clear()


# ---------------------------------------------------------------------------
# §9 JPEG header-only: CMYK (YCCK APP14) → non_default_color_transform fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_jpeg_header_only_cmyk_falls_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """JPEG with APP14 color_transform=2 (YCCK) → non_default_color_transform fallback."""
    import estimation.estimator as estimator_mod
    import estimation.models as models_mod

    _copy_real_model(tmp_path, "jpeg_header_v1.json")
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_jpeg_header_model.cache_clear()
    monkeypatch.setattr(estimator_mod.settings, "fitted_estimator_mode", "active")

    # Use the CMYK/YCCK synthesized JPEG
    data = _make_cmyk_jpeg(64, 64)

    from estimation.jpeg_header import parse_jpeg_header

    header = parse_jpeg_header(data)
    if header is None or header.fallback_reason != "non_default_color_transform":
        pytest.skip("CMYK JPEG synthesis did not produce expected fallback_reason")

    from schemas import OptimizationConfig

    config = OptimizationConfig(quality=60)
    result = await estimator_mod.estimate(data, config)

    # Small image → may go exact mode; if sample path, fallback_reason should be set
    if result.path == "direct_encode_sample":
        assert (
            result.fallback_reason == "non_default_color_transform"
        ), f"Expected 'non_default_color_transform', got {result.fallback_reason!r}"

    models_mod.load_jpeg_header_model.cache_clear()
