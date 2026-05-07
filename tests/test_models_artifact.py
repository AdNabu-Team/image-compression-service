"""Tests for estimation/models/__init__.py and estimation/models/_artifact.py.

Covers:
- Missing file → LoadFailed("missing")
- Corrupt JSON → LoadFailed("parse_error")
- model_version mismatch → LoadFailed("version_mismatch")
- Happy path → Loaded(model) with correct PngModel fields
- load_png_model() never raises (including on all failure modes)
"""

import json
from pathlib import Path

import pytest

from estimation.models import Loaded, LoadFailed, load_png_model
from estimation.models._artifact import PngModel as PngModelDirect

# ---------------------------------------------------------------------------
# Minimal valid artifact payload
# ---------------------------------------------------------------------------

_VALID_ARTIFACT: dict = {
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
        "mean": [0.0, 2.5, 15.0, 0.1, 70.0, 5.0, 8.0],
        "scale": [1.0, 1.2, 10.0, 0.05, 20.0, 0.8, 4.0],
    },
    "coefficients": {
        "intercept": 0.5,
        "betas": [0.1, -0.2, 0.05, 0.3, -0.01, 0.15, -0.05],
        "knot_beta": 0.3,
        "knot_q50_beta": -0.01,
        "knot_q70_beta": 0.02,
    },
    "knot_log10_unique_colors": 3.3,
    "knot_q50": 50.0,
    "knot_q70": 70.0,
    "training_envelope": {"log10_unique_colors": [0.0, 5.7], "mean_sobel": [0.0, 80.0]},
    "training_corpus_sha256": "abc123def456",
    "git_sha": "cafebabe",
    "fit_environment": {
        "sklearn_version": "1.5.0",
        "numpy_version": "2.0.2",
        "openblas_threads": 1,
    },
    "created_at": "2026-05-07T00:00:00Z",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_artifact(tmp_path: Path, payload: dict | None = None, raw: str | None = None) -> Path:
    """Write a JSON artifact file and return its path."""
    p = tmp_path / "png_v1.json"
    if raw is not None:
        p.write_text(raw)
    else:
        p.write_text(json.dumps(payload or _VALID_ARTIFACT))
    return p


# ---------------------------------------------------------------------------
# PngModel.from_json tests (direct, via _artifact classmethod)
# ---------------------------------------------------------------------------


class TestPngModelFromJson:
    def test_missing_file_returns_load_failed(self, tmp_path: Path):
        missing = tmp_path / "does_not_exist.json"
        result = PngModelDirect.from_json(missing)
        assert isinstance(result, LoadFailed)
        assert result.reason == "missing"

    def test_corrupt_json_returns_load_failed(self, tmp_path: Path):
        p = _write_artifact(tmp_path, raw="{not valid json")
        result = PngModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"

    def test_version_mismatch_returns_load_failed(self, tmp_path: Path):
        bad = dict(_VALID_ARTIFACT, model_version=99)
        p = _write_artifact(tmp_path, payload=bad)
        result = PngModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "version_mismatch"

    def test_missing_required_field_returns_parse_error(self, tmp_path: Path):
        # Drop a required field — should produce parse_error, not crash.
        bad = {k: v for k, v in _VALID_ARTIFACT.items() if k != "training_corpus_sha256"}
        p = _write_artifact(tmp_path, payload=bad)
        result = PngModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"

    def test_happy_path_returns_loaded(self, tmp_path: Path):
        p = _write_artifact(tmp_path)
        result = PngModelDirect.from_json(p)
        assert isinstance(result, Loaded)
        model = result.model
        assert isinstance(model, PngModelDirect)
        assert model.model_version == 2
        assert model.format == "png"
        assert model.knot_log10_unique_colors == pytest.approx(3.3)
        assert model.knot_q50 == pytest.approx(50.0)
        assert model.knot_q70 == pytest.approx(70.0)
        assert model.git_sha == "cafebabe"
        assert "sklearn_version" in model.fit_environment

    def test_happy_path_features_preserved(self, tmp_path: Path):
        p = _write_artifact(tmp_path)
        result = PngModelDirect.from_json(p)
        assert isinstance(result, Loaded)
        assert result.model.features == _VALID_ARTIFACT["features"]
        assert result.model.supported_modes == _VALID_ARTIFACT["supported_modes"]

    def test_happy_path_training_envelope_forensic(self, tmp_path: Path):
        p = _write_artifact(tmp_path)
        result = PngModelDirect.from_json(p)
        assert isinstance(result, Loaded)
        # training_envelope is present but is forensic only
        assert "log10_unique_colors" in result.model.training_envelope

    def test_model_is_frozen(self, tmp_path: Path):
        p = _write_artifact(tmp_path)
        result = PngModelDirect.from_json(p)
        assert isinstance(result, Loaded)
        with pytest.raises((AttributeError, TypeError)):
            result.model.format = "jpeg"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Schema validation negative tests (#3 — coefficients shape / bounds)
# ---------------------------------------------------------------------------


class TestPngModelSchemaValidation:
    """Negative tests for _validate_schema — structural and numeric bounds."""

    def test_from_json_rejects_unknown_coefficient_key(self, tmp_path: Path):
        """coefficients shape with 'coef' instead of 'betas' must return LoadFailed('parse_error')."""
        bad = dict(_VALID_ARTIFACT)
        bad["coefficients"] = {
            "intercept": 0.5,
            "coef": [0.1, -0.2, 0.05, 0.3, -0.01, 0.15, -0.05],  # wrong key
            "knot_beta": 0.3,
            "knot_q50_beta": -0.01,
            "knot_q70_beta": 0.02,
        }
        p = _write_artifact(tmp_path, payload=bad)
        result = PngModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"

    def test_from_json_rejects_missing_intercept(self, tmp_path: Path):
        """coefficients dict without 'intercept' must return LoadFailed('parse_error')."""
        bad = dict(_VALID_ARTIFACT)
        bad["coefficients"] = {
            "betas": [0.1, -0.2, 0.05, 0.3, -0.01, 0.15, -0.05],
            # 'intercept' intentionally missing
            "knot_beta": 0.3,
            "knot_q50_beta": -0.01,
            "knot_q70_beta": 0.02,
        }
        p = _write_artifact(tmp_path, payload=bad)
        result = PngModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"

    def test_from_json_rejects_mismatched_betas_length(self, tmp_path: Path):
        """betas list with wrong length must return LoadFailed('parse_error')."""
        bad = dict(_VALID_ARTIFACT)
        bad["coefficients"] = {
            "intercept": 0.5,
            "betas": [0.1, -0.2],  # too short — features has 7 entries
            "knot_beta": 0.3,
            "knot_q50_beta": -0.01,
            "knot_q70_beta": 0.02,
        }
        p = _write_artifact(tmp_path, payload=bad)
        result = PngModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"

    def test_from_json_rejects_non_finite_coefficients(self, tmp_path: Path):
        """Non-finite betas (inf/nan) must return LoadFailed('parse_error')."""
        bad = dict(_VALID_ARTIFACT)
        bad["coefficients"] = {
            "intercept": 0.5,
            "betas": [float("inf"), -0.2, 0.05, 0.3, -0.01, 0.15, -0.05],
            "knot_beta": 0.3,
            "knot_q50_beta": -0.01,
            "knot_q70_beta": 0.02,
        }
        p = _write_artifact(tmp_path, payload=bad)
        result = PngModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"

    def test_from_json_rejects_beta_out_of_bounds(self, tmp_path: Path):
        """Beta value exceeding ±100 must return LoadFailed('parse_error')."""
        bad = dict(_VALID_ARTIFACT)
        bad["coefficients"] = {
            "intercept": 0.5,
            "betas": [999.0, -0.2, 0.05, 0.3, -0.01, 0.15, -0.05],  # 999 > 100
            "knot_beta": 0.3,
            "knot_q50_beta": -0.01,
            "knot_q70_beta": 0.02,
        }
        p = _write_artifact(tmp_path, payload=bad)
        result = PngModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"

    def test_from_json_rejects_missing_knot_beta(self, tmp_path: Path):
        """coefficients dict without 'knot_beta' must return LoadFailed('parse_error')."""
        bad = dict(_VALID_ARTIFACT)
        bad["coefficients"] = {
            "intercept": 0.5,
            "betas": [0.1, -0.2, 0.05, 0.3, -0.01, 0.15, -0.05],
            # 'knot_beta' intentionally missing
            "knot_q50_beta": -0.01,
            "knot_q70_beta": 0.02,
        }
        p = _write_artifact(tmp_path, payload=bad)
        result = PngModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"

    def test_from_json_rejects_oob_knot_beta(self, tmp_path: Path):
        """knot_q50_beta value exceeding ±100 must return LoadFailed('parse_error')."""
        bad = dict(_VALID_ARTIFACT)
        bad["coefficients"] = {
            "intercept": 0.5,
            "betas": [0.1, -0.2, 0.05, 0.3, -0.01, 0.15, -0.05],
            "knot_beta": 0.3,
            "knot_q50_beta": 1e10,  # way out of bounds
            "knot_q70_beta": 0.02,
        }
        p = _write_artifact(tmp_path, payload=bad)
        result = PngModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"

    def test_from_json_rejects_non_finite_knot_beta(self, tmp_path: Path):
        """Non-finite knot_q70_beta (inf) must return LoadFailed('parse_error')."""
        bad = dict(_VALID_ARTIFACT)
        bad["coefficients"] = {
            "intercept": 0.5,
            "betas": [0.1, -0.2, 0.05, 0.3, -0.01, 0.15, -0.05],
            "knot_beta": 0.3,
            "knot_q50_beta": -0.01,
            "knot_q70_beta": float("inf"),
        }
        p = _write_artifact(tmp_path, payload=bad)
        result = PngModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"


# ---------------------------------------------------------------------------
# load_png_model() wrapper tests (goes through estimation/models/__init__.py)
# ---------------------------------------------------------------------------


class TestLoadPngModel:
    def _clear_cache(self):
        """Clear the lru_cache between tests to ensure isolation."""
        load_png_model.cache_clear()

    def test_does_not_raise_on_missing_artifact(self):
        """load_png_model() should return LoadFailed, not raise, when artifact is absent."""
        self._clear_cache()
        # The real png_v1.json is committed in this branch; this test loads it via the same
        # code path production uses.  Whether the file is present (Loaded) or absent
        # (LoadFailed), the key contract is: it never raises.
        result = load_png_model()
        assert isinstance(result, (Loaded, LoadFailed))

    def test_does_not_raise_when_called_multiple_times(self):
        """Calling load_png_model() multiple times must not raise."""
        self._clear_cache()
        for _ in range(5):
            result = load_png_model()
            assert isinstance(result, (Loaded, LoadFailed))

    def test_returns_load_failed_when_file_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Patch the models dir to a tmp_path without png_v1.json → LoadFailed."""
        self._clear_cache()
        import estimation.models as models_mod

        monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
        # Also need to clear cache again after the patch since the lru_cache already ran.
        load_png_model.cache_clear()

        result = load_png_model()
        assert isinstance(result, LoadFailed)
        assert result.reason == "missing"

    def test_returns_loaded_on_valid_artifact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """With a valid artifact in tmp_path → Loaded."""
        _write_artifact(tmp_path)
        self._clear_cache()
        import estimation.models as models_mod

        monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
        load_png_model.cache_clear()

        result = load_png_model()
        assert isinstance(result, Loaded)
        assert result.model.format == "png"

    def test_returns_load_failed_on_corrupt_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Corrupt JSON → LoadFailed("parse_error") without raising."""
        (tmp_path / "png_v1.json").write_text("{bad json")
        self._clear_cache()
        import estimation.models as models_mod

        monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
        load_png_model.cache_clear()

        result = load_png_model()
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"

    def test_returns_load_failed_on_version_mismatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """model_version=99 → LoadFailed("version_mismatch") without raising."""
        bad = dict(_VALID_ARTIFACT, model_version=99)
        (tmp_path / "png_v1.json").write_text(json.dumps(bad))
        self._clear_cache()
        import estimation.models as models_mod

        monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
        load_png_model.cache_clear()

        result = load_png_model()
        assert isinstance(result, LoadFailed)
        assert result.reason == "version_mismatch"
