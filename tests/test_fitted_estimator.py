"""Tests for the PNG fitted BPP estimator path.

Covers consensus items #10 (read settings at call time) and #11 (patch
consumer's binding in tests).

All tests that exercise the full ``estimate()`` pipeline use ``@pytest.mark.asyncio``
per the project convention (strict mode).
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_large_png(mode: str = "RGB", width: int = 500, height: int = 500) -> bytes:
    """Create a large noisy PNG (>150K pixels) that forces the sample path.

    The image must be noisy enough to have high BPP (> 2.0) so the estimator
    does not trigger the low-BPP exact-mode shortcut.
    """
    import numpy as np

    rng = np.random.default_rng(42)
    if mode == "RGB":
        arr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
    elif mode == "RGBA":
        arr = rng.integers(0, 256, (height, width, 4), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGBA")
    else:
        arr = rng.integers(0, 256, (height, width), dtype=np.uint8)
        img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_large_png_i16(width: int = 500, height: int = 500) -> bytes:
    """Create a large I;16 PNG (unsupported mode for fitted estimator)."""
    # Create as I mode (32-bit), then save — PIL will encode as 16-bit PNG internally
    img = Image.new("I", (width, height), color=32768)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _valid_model_json() -> dict:
    """Minimal valid png_v1.json that passes PngModel.from_json (model_version=2)."""
    return {
        "model_version": 2,
        "format": "png",
        "features": [
            "has_alpha",
            "log10_unique_colors",
            "mean_sobel",
            "edge_density",
            "quality",
            "log10_orig_pixels",
            "input_bpp",
        ],
        "supported_modes": ["RGB", "RGBA", "L", "LA", "P"],
        "scaler": {
            "mean": [0.0, 3.0, 50.0, 0.3, 60.0, 5.5, 8.0],
            "scale": [1.0, 0.5, 30.0, 0.2, 15.0, 1.0, 4.0],
        },
        "coefficients": {
            # intercept=12.0 → predicted_bpp ≈ 12.0 for a noisy 500×500 PNG
            # (input_bpp ≈ 24 → ratio ≈ 0.5, well above min_ratio=0.10 for q=60)
            "intercept": 12.0,
            "betas": [0.0, 0.5, 0.3, -0.5, 0.1, -0.2, -0.3],
            "knot_beta": 0.5,
            "knot_q50_beta": -0.02,
            "knot_q70_beta": 0.03,
        },
        "knot_log10_unique_colors": 3.3,
        "knot_q50": 50.0,
        "knot_q70": 70.0,
        "training_envelope": {
            "has_alpha": [0.0, 1.0],
            "log10_unique_colors": [1.0, 5.0],
            "mean_sobel": [5.0, 200.0],
            "edge_density": [0.0, 1.0],
            "quality": [40.0, 85.0],
            "log10_orig_pixels": [3.0, 8.0],
            "input_bpp": [1.0, 32.0],
        },
        "training_corpus_sha256": "abc123",
        "git_sha": "deadbeef",
        "fit_environment": {"numpy_version": "2.0.0", "scipy_version": "1.13.0"},
        "created_at": "2026-05-07T00:00:00+00:00",
    }


# ---------------------------------------------------------------------------
# Test: fitted mode active → returns png_fitted_curve path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_png_fitted_active_returns_fitted_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With mode=active and a valid model, estimate() returns path='png_fitted_curve'."""
    import estimation.estimator as estimator_mod
    from estimation.models import _artifact as artifact_mod

    # Write a valid model artifact to a temp dir
    model_path = tmp_path / "png_v1.json"
    model_path.write_text(json.dumps(_valid_model_json()))

    # Patch _MODELS_DIR so load_png_model() loads from tmp_path
    monkeypatch.setattr(artifact_mod, "_SUPPORTED_MODEL_VERSION", 2)
    import estimation.models as models_mod

    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    # Clear the lru_cache so the patched path is used
    models_mod.load_png_model.cache_clear()

    # Activate fitted mode via the consumer's binding (consensus #11)
    monkeypatch.setattr(estimator_mod.settings, "fitted_estimator_mode", "active")

    data = _make_large_png("RGB", 500, 500)

    from schemas import OptimizationConfig

    config = OptimizationConfig(quality=60, png_lossy=True)

    # Run estimate — this should hit the fitted path
    result = await estimator_mod.estimate(data, config)

    assert result.path == "png_fitted_curve", (
        f"Expected path='png_fitted_curve', got {result.path!r}. "
        f"fallback_reason={result.fallback_reason!r}"
    )
    assert result.estimated_optimized_size > 0
    assert result.estimated_reduction_percent >= 0.0
    assert result.fallback_reason is None

    # Cleanup cache so subsequent tests aren't affected
    models_mod.load_png_model.cache_clear()


# ---------------------------------------------------------------------------
# Test: fitted mode off → returns direct_encode_sample path (existing behavior)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_png_fitted_off_returns_sample_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """With mode=off (default), estimate() uses the existing direct_encode_sample path."""
    import estimation.estimator as estimator_mod

    # Ensure mode is off (default)
    monkeypatch.setattr(estimator_mod.settings, "fitted_estimator_mode", "off")

    data = _make_large_png("RGB", 500, 500)

    from schemas import OptimizationConfig

    config = OptimizationConfig(quality=60, png_lossy=True)

    result = await estimator_mod.estimate(data, config)

    # With mode=off, the fitted path is never attempted
    assert (
        result.path == "direct_encode_sample"
    ), f"Expected 'direct_encode_sample', got {result.path!r}"
    assert result.fallback_reason is None, (
        f"Expected fallback_reason=None (fitted never attempted), "
        f"got {result.fallback_reason!r}"
    )


# ---------------------------------------------------------------------------
# Test: unsupported mode (I) → fallback with fallback_reason set
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_png_fitted_unsupported_mode_falls_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Mode=I PNG (not in supported_modes) → path='direct_encode_sample', fallback_reason set."""
    import estimation.estimator as estimator_mod
    import estimation.models as models_mod

    # Write valid model
    model_path = tmp_path / "png_v1.json"
    model_path.write_text(json.dumps(_valid_model_json()))
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_png_model.cache_clear()

    # Activate fitted mode
    monkeypatch.setattr(estimator_mod.settings, "fitted_estimator_mode", "active")

    # I;16 PNG → extract_png_features returns None (unsupported mode)
    data = _make_large_png_i16(500, 500)

    from schemas import OptimizationConfig

    config = OptimizationConfig(quality=60, png_lossy=True)

    result = await estimator_mod.estimate(data, config)

    # Either exact mode (small file) or fallback sample — either way, not fitted
    # The key assertion is that fallback_reason is set if path==direct_encode_sample
    # (If the file is small enough to go exact mode, fallback_reason would be None — that's OK)
    if result.path == "direct_encode_sample":
        assert result.fallback_reason == "mode_unsupported_or_oob", (
            f"Expected fallback_reason='mode_unsupported_or_oob', "
            f"got {result.fallback_reason!r}"
        )
    # If exact mode was chosen (file < EXACT_PIXEL_THRESHOLD), path='exact' is also acceptable
    assert result.path in ("direct_encode_sample", "exact"), f"Unexpected path {result.path!r}"

    models_mod.load_png_model.cache_clear()


# ---------------------------------------------------------------------------
# Test: model load failure → fallback with fallback_reason='model_load_failed'
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_png_fitted_model_load_failure_falls_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Empty _MODELS_DIR (no png_v1.json) → fallback_reason='model_load_failed'."""
    import estimation.estimator as estimator_mod
    import estimation.models as models_mod

    # Point to an empty dir — no png_v1.json exists
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_png_model.cache_clear()

    # Activate fitted mode
    monkeypatch.setattr(estimator_mod.settings, "fitted_estimator_mode", "active")

    data = _make_large_png("RGB", 500, 500)

    from schemas import OptimizationConfig

    config = OptimizationConfig(quality=60, png_lossy=True)

    result = await estimator_mod.estimate(data, config)

    # Should fall back to direct_encode_sample with model_load_failed reason
    assert result.path in ("direct_encode_sample", "exact"), f"Unexpected path {result.path!r}"
    if result.path == "direct_encode_sample":
        assert (
            result.fallback_reason == "model_load_failed"
        ), f"Expected 'model_load_failed', got {result.fallback_reason!r}"

    models_mod.load_png_model.cache_clear()


# ---------------------------------------------------------------------------
# Test: _resolve_estimate_strategy reads settings at call time
# ---------------------------------------------------------------------------


def test_resolve_strategy_reads_settings_at_call_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_resolve_estimate_strategy returns 'fitted' only when settings.fitted_estimator_mode='active'."""
    import estimation.estimator as estimator_mod
    from utils.format_detect import ImageFormat

    monkeypatch.setattr(estimator_mod.settings, "fitted_estimator_mode", "off")
    assert estimator_mod._resolve_estimate_strategy(ImageFormat.PNG) == "sample"

    monkeypatch.setattr(estimator_mod.settings, "fitted_estimator_mode", "active")
    assert estimator_mod._resolve_estimate_strategy(ImageFormat.PNG) == "fitted"

    # Non-PNG formats always return 'sample' even in active mode
    monkeypatch.setattr(estimator_mod.settings, "fitted_estimator_mode", "active")
    assert estimator_mod._resolve_estimate_strategy(ImageFormat.JPEG) == "sample"
    assert estimator_mod._resolve_estimate_strategy(ImageFormat.WEBP) == "sample"


# ---------------------------------------------------------------------------
# Test: prediction_disagreement fires when ratio is implausible
# ---------------------------------------------------------------------------


def test_prediction_disagreement_fires_on_implausible_ratio(tmp_path: Path) -> None:
    """_png_fitted_bpp returns FittedFallback(reason='prediction_disagreement') when
    the model predicts a compression ratio that is outside the content-aware bounds.

    We construct a model that will always predict a BPP that is implausibly high
    relative to the input_bpp (ratio > MAX_RATIO=1.10), so the ratio gate fires.
    """
    import json

    from PIL import Image

    import estimation.models as models_mod
    from estimation.estimator import FittedFallback, _png_fitted_bpp

    # Write a model that predicts a very high BPP (intercept=30, all betas=0)
    # so that predicted_bpp ≈ 30 >> input_bpp * 1.10 → ratio > MAX_RATIO
    high_bpp_model = _valid_model_json()
    high_bpp_model["coefficients"]["intercept"] = 30.0
    high_bpp_model["coefficients"]["betas"] = [0.0] * 7
    high_bpp_model["coefficients"]["knot_beta"] = 0.0
    high_bpp_model["coefficients"]["knot_q50_beta"] = 0.0
    high_bpp_model["coefficients"]["knot_q70_beta"] = 0.0

    model_path = tmp_path / "png_v1.json"
    model_path.write_text(json.dumps(high_bpp_model))
    models_mod._MODELS_DIR = tmp_path
    models_mod.load_png_model.cache_clear()

    # Create a small photographic-like RGB image; provide a realistic orig_size
    # (input_bpp ≈ 8 bpp = 8 bits/pixel for lossless PNG of 32×32 = 1024px at 1024 bytes)
    img = Image.new("RGB", (32, 32), color=(100, 150, 200))
    orig_size = 1024  # 1024 bytes × 8 bits = 8192 bits / 1024 pixels = 8.0 bpp

    result = _png_fitted_bpp(img, 32, 32, quality=60, orig_size=orig_size)

    assert isinstance(result, FittedFallback), f"Expected FittedFallback, got {result!r}"
    assert (
        result.reason == "prediction_disagreement"
    ), f"Expected 'prediction_disagreement', got {result.reason!r}"

    models_mod.load_png_model.cache_clear()
    models_mod._MODELS_DIR = Path(__file__).parent.parent / "estimation" / "models"


# ---------------------------------------------------------------------------
# Test: internal_error fires when extract_png_features raises unexpectedly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_png_fitted_internal_error_falls_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When extract_png_features raises, _png_fitted_bpp returns FittedFallback('internal_error')
    and estimate() falls back to path='direct_encode_sample' with fallback_reason='internal_error'.
    """
    import estimation.estimator as estimator_mod
    import estimation.models as models_mod

    # Write a valid model so model load succeeds
    model_path = tmp_path / "png_v1.json"
    model_path.write_text(json.dumps(_valid_model_json()))
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_png_model.cache_clear()

    # Activate fitted mode
    monkeypatch.setattr(estimator_mod.settings, "fitted_estimator_mode", "active")

    # Monkeypatch extract_png_features to raise (patch the name bound in estimator_mod)
    def _raise(*args, **kwargs):
        raise RuntimeError("simulated internal failure")

    monkeypatch.setattr(estimator_mod, "extract_png_features", _raise)

    data = _make_large_png("RGB", 500, 500)

    from schemas import OptimizationConfig

    config = OptimizationConfig(quality=60, png_lossy=True)

    result = await estimator_mod.estimate(data, config)

    # Should fall back to direct_encode_sample with internal_error reason
    assert result.path in ("direct_encode_sample", "exact"), f"Unexpected path {result.path!r}"
    if result.path == "direct_encode_sample":
        assert result.fallback_reason == "internal_error", (
            f"Expected fallback_reason='internal_error', got {result.fallback_reason!r}"
        )

    models_mod.load_png_model.cache_clear()
