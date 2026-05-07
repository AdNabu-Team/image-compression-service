"""Shared utilities for fitted BPP regression model training.

Uses ``numpy.linalg.lstsq`` (no sklearn dependency) with an explicit piecewise-
linear knot.  The ``StandardScaler`` parameters (mean + std per feature) are
baked into the output so the consumer can apply them without numpy at inference.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def train_one(
    features_df: pd.DataFrame,
    targets: np.ndarray,
    knot: float = 3.3,
    knot_q50: float = 50.0,
    knot_q70: float = 70.0,
) -> dict[str, Any]:
    """Fit a piecewise-linear OLS regression with three knots.

    Parameters
    ----------
    features_df :
        DataFrame where each column is a feature (in the order that matches the
        ``features`` field in the model JSON).  Must include columns named
        ``log10_unique_colors`` and ``quality`` for the knot terms.
    targets :
        1-D float array of actual BPP values (one per row).
    knot :
        Knot position on the ``log10_unique_colors`` axis.  Defaults to 3.3
        (~2000 colours — the pngquant reduction-profile flip point).
    knot_q50 :
        Knot position on the ``quality`` axis at q=50.  Captures the pngquant
        max_colors=64→256 transition (aggressive lossy → moderate lossy).
    knot_q70 :
        Knot position on the ``quality`` axis at q=70.  Captures the lossy →
        lossless cliff where pngquant stops applying lossy quantization.

    Returns
    -------
    dict with keys:
        ``features``        — ordered feature names (list[str])
        ``scaler``          — {``mean``: list[float], ``scale``: list[float]}
        ``coefficients``    — {``intercept``: float, ``betas``: list[float],
                               ``knot_beta``: float, ``knot_q50_beta``: float,
                               ``knot_q70_beta``: float}
        ``knot_log10_unique_colors`` — float (knot value used)
        ``knot_q50``        — float (quality knot at 50)
        ``knot_q70``        — float (quality knot at 70)
        ``train_residuals`` — {``median_rel_err``: float, ``p95_rel_err``: float,
                               ``max_rel_err``: float} on the training set
    """
    feature_names: list[str] = list(features_df.columns)
    X_raw = features_df.values.astype(np.float64)
    y = targets.astype(np.float64)

    n, p = X_raw.shape
    assert n == len(y), f"Row count mismatch: X has {n} rows, y has {len(y)}"

    # --- StandardScaler ---
    mean_ = X_raw.mean(axis=0)
    std_ = X_raw.std(axis=0, ddof=0)
    # Avoid division by zero for constant features (e.g. has_alpha all-True corpus)
    scale_ = np.where(std_ > 1e-9, std_, 1.0)
    X_scaled = (X_raw - mean_) / scale_

    # --- Piecewise-linear knot column: log10_unique_colors ---
    try:
        knot_col_idx = feature_names.index("log10_unique_colors")
    except ValueError as exc:
        raise ValueError(
            f"features_df must contain 'log10_unique_colors' column for knot term; "
            f"got columns: {feature_names}"
        ) from exc

    # log10_unique_colors is already scaled; we need the *original* values for the knot
    log10_uc_raw = X_raw[:, knot_col_idx]
    knot_lc_col = np.clip(log10_uc_raw - knot, 0.0, None)  # (x - knot)+

    # --- Piecewise-linear knot columns: quality at q=50 and q=70 ---
    try:
        quality_col_idx = feature_names.index("quality")
    except ValueError as exc:
        raise ValueError(
            f"features_df must contain 'quality' column for quality knot terms; "
            f"got columns: {feature_names}"
        ) from exc

    quality_raw = X_raw[:, quality_col_idx]
    knot_q50_col = np.clip(quality_raw - knot_q50, 0.0, None)  # (q - 50)+
    knot_q70_col = np.clip(quality_raw - knot_q70, 0.0, None)  # (q - 70)+

    # Design matrix: [1 | X_scaled | knot_lc_col | knot_q50_col | knot_q70_col]
    ones = np.ones((n, 1), dtype=np.float64)
    A = np.hstack(
        [
            ones,
            X_scaled,
            knot_lc_col.reshape(-1, 1),
            knot_q50_col.reshape(-1, 1),
            knot_q70_col.reshape(-1, 1),
        ]
    )

    # --- OLS via lstsq ---
    coeffs, _residuals, _rank, _sv = np.linalg.lstsq(A, y, rcond=None)
    intercept = float(coeffs[0])
    betas = [float(c) for c in coeffs[1 : p + 1]]
    knot_beta = float(coeffs[p + 1])
    knot_q50_beta = float(coeffs[p + 2])
    knot_q70_beta = float(coeffs[p + 3])

    # --- Training residuals ---
    y_pred = A @ coeffs
    rel_err = np.abs((y_pred - y) / np.clip(y, 1e-9, None))
    median_rel_err = float(np.median(rel_err))
    p95_rel_err = float(np.percentile(rel_err, 95))
    max_rel_err = float(rel_err.max())

    return {
        "features": feature_names,
        "scaler": {
            "mean": mean_.tolist(),
            "scale": scale_.tolist(),
        },
        "coefficients": {
            "intercept": intercept,
            "betas": betas,
            "knot_beta": knot_beta,
            "knot_q50_beta": knot_q50_beta,
            "knot_q70_beta": knot_q70_beta,
        },
        "knot_log10_unique_colors": knot,
        "knot_q50": knot_q50,
        "knot_q70": knot_q70,
        "train_residuals": {
            "median_rel_err": round(median_rel_err, 6),
            "p95_rel_err": round(p95_rel_err, 6),
            "max_rel_err": round(max_rel_err, 6),
        },
    }
