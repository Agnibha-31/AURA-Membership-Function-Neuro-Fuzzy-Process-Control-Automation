#!/usr/bin/env python3

# Copyright 2026 Agnibha Basak
# SPDX-License-Identifier: Apache-2.0

"""
AURA [Asymmetric Unified Relativistic Attractor]
============
A single-file, public-use regression trainer, evaluator, plot generator, model
exporter, and predictor. No other AURA source file is needed.

WHAT THIS PROGRAM IMPLEMENTS
----------------------------
This program implements the canonical AURA membership function described in:

    AURA: Center-Anchored Asymmetric Premise Geometry With Directional Tail
    Order for Neuro-Fuzzy Process Control Automation and Dynamic Nonlinear
    Modeling

The pipeline combines constrained AURA premise geometry, normalized fuzzy-rule
firing, first-order Takagi-Sugeno consequents, hybrid ridge/gradient learning,
and a validation-selected residual refinement stage. It supports ordinary
tabular regression, time-series forecasting, numerical/categorical inputs,
multiple data files, and one or several output columns.

PERFORMANCE STATEMENT (PLEASE READ BEFORE REPORTING A SCORE)
-----------------------------------------------------------
The supplied AURA case studies achieved test R2 values above 0.99 under their
recorded evaluation protocols: cascaded tanks with overflow, coupled electric
drives, dissolved oxygen, and pH neutralization. These results demonstrate the
high capacity of the AURA pipeline on those problems. They are not a promise
that every new data set will automatically obtain R2 > 0.99. A scientifically
valid score depends on useful input information, clean targets, enough samples,
correct time ordering, suitable hyperparameters, and a genuinely untouched test
set. This program prints the measured score; it never changes or hides a result
to meet a requested threshold.

For the best chance of reproducing high accuracy on a new domain:

1. Define one clear prediction task and choose the correct target column(s).
2. Use input features that are available at the real prediction time. Never use
   the same-row target as an input for a static model.
3. Clean unit errors, sensor faults, duplicates, and incorrect labels first.
4. For time series, keep the row order chronological, place each independent
   run/trajectory in a separate file, use ``--horizon 1`` or larger, and prefer
   an entirely separate later run with ``--external-data``.
5. Let validation data select the configuration. Read the final test metrics
   only after all choices are finished. Do not repeatedly tune on the test set.
6. Compare MAE/RMSE with the physical scale of the target as well as checking
   R2. Inspect every generated plot for bias, outliers, drift, and leakage.

INSTALLATION
------------
Python 3.10 or newer is recommended. From a terminal, install:

    python -m pip install numpy pandas scipy scikit-learn==1.8.0 torch matplotlib

The supplied ``cto.pkl``, ``ced.pkl``, ``do.pkl``, and ``pH.pkl`` artifacts were
serialized with scikit-learn 1.8.0, which is why that exact version is pinned
above. Newly created pickles record their package versions in their metadata.
Use the same major/minor package versions when moving a pickle to another
computer. Also install ``openpyxl`` for Excel files or ``pyarrow`` for Parquet:

    python -m pip install openpyxl pyarrow

QUICK START: STATIC/TABULAR REGRESSION
--------------------------------------
Each row is one independent example. The target must not also be a feature.
Column names are case-sensitive. Quote paths or column names containing spaces.

    python AURA.py train --data data.csv --target y --features x1,x2,x3 --split random --model-out aura_model.pkl

If row order has meaning (time, batches, patients, locations, experiments), use
the default ``--split chronological`` or, preferably, separate validation and
test files. Avoid ``--features auto`` for a final scientific experiment: it is
convenient for exploration but may select IDs, timestamps, or leakage columns.

QUICK START: TIME-SERIES / ONE-STEP FORECASTING
-----------------------------------------------
The following predicts the next y value from current/past u and y values. A
directory may be passed to ``--data``; each supported file is treated as a
separate sequence, so lag history never crosses file boundaries.

    python AURA.py train --data train_runs --target y --features u,y --lags 1:20 --horizon 1 --dynamic-features --external-data unseen_test_runs --model-out aura_forecast.pkl

Use ``--external-target`` and ``--external-features`` when validation/test files
use different column names. Names are mapped by position. Example:

    --target y_train --features u_train,y_train --external-target y_test --external-features u_test,y_test

OUTPUTS CREATED AFTER TRAINING
------------------------------
Training prints MAE, MSE/MSAE, RMSE, R2, adjusted R2, median/max error, MAPE,
sMAPE, explained variance, bias, NRMSE, CCC, NSE, Willmott's d, and RMSLE when
valid. It exports one self-contained ``.pkl`` pipeline and, by default, creates
six domain-neutral PNG diagnostic plots beside the model in
``<model_name>_plots``. Use ``--plots-dir`` to choose another folder or
``--no-plots`` only in a non-graphical automated run.

USING AN EXPORTED MODEL
-----------------------
Create a prediction CSV (the target column is not required):

    python AURA.py predict --model aura_model.pkl --data new_data.csv --output predictions.csv

Inspect stored schema, configuration, and test metrics:

    python AURA.py inspect --model aura_model.pkl

A ``.pkl`` file is fitted to the domain and columns used to create it. For a new
domain, run ``train`` again to create a new domain-specific pickle. Do not treat
one of the supplied process-control pickles as a universal pretrained model.
Only load pickle files obtained from a trusted source because Python pickle can
execute code while loading. For portable unpickling, keep this file named
``AURA.py`` in the working directory or on PYTHONPATH. This script also keeps a
backward-compatible ``aura_unified`` alias so earlier AURA pickles can still be
loaded through the same public file.

Run ``python AURA.py --help`` and
``python AURA.py train --help`` for every available option.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import pickle
import platform
import random
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

# Joblib otherwise probes the removed Windows ``wmic`` utility on some
# systems before using logical cores.  This explicit value is portable and
# avoids a noisy, non-fatal warning during tree-refiner fitting.
os.environ["LOKY_MAX_CPU_COUNT"] = str(max(1, (os.cpu_count() or 2) - 1))
warnings.filterwarnings("ignore", message=r"Could not find the number of physical cores.*", category=UserWarning)

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.multioutput import MultiOutputRegressor


VERSION = "1.1.0"
FORMAT = "aura-unified-pickle/v1"
EPS = 1.0e-8
Q_MIN = 1.05
RHO_BOUND = 0.85
Q1_BOUND = 1.20
SKEW_PAIR_BOUND = 0.85
LAMBDA_GATE = 2.0
AURA_EPSILON = 1.0e-6


# ---------------------------------------------------------------------------
# Small general-purpose helpers
# ---------------------------------------------------------------------------

# Make pickles created through ``python AURA.py`` importable later as
# AURA.AURAUnifiedModel instead of __main__.AURAUnifiedModel.  The historical
# aura_unified alias is intentionally preserved so the four released model
# artifacts remain loadable after the public filename change.
sys.modules.setdefault("AURA", sys.modules[__name__])
sys.modules.setdefault("aura_unified", sys.modules[__name__])


# ---------------------------------------------------------------------------
# Command-line text parsing and reproducibility
# ---------------------------------------------------------------------------

def _csv_list(value: Optional[str]) -> list[str]:
    if value is None:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def _parse_int_list(value: str) -> list[int]:
    value = str(value).strip()
    if not value or value == "0" or value.lower() == "none":
        return []
    out: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            parts = [int(x) for x in item.split(":")]
            if len(parts) == 2:
                start, stop = parts
                step = 1
            elif len(parts) == 3:
                start, stop, step = parts
            else:
                raise ValueError(f"Invalid integer range: {item}")
            if step == 0:
                raise ValueError("Range step cannot be zero.")
            out.extend(range(start, stop + (1 if step > 0 else -1), step))
        else:
            out.append(int(item))
    if any(v < 1 for v in out):
        raise ValueError("All lags/rules/seeds must be positive integers.")
    return sorted(set(out))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def _device(value: str) -> torch.device:
    value = value.lower().strip()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA was requested but is unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(value)


def _softplus_inverse(x: np.ndarray | float) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    a = np.maximum(a, 1.0e-10)
    return np.where(a > 20.0, a, np.log(np.expm1(a)))


# ---------------------------------------------------------------------------
# Data loading and optional process-control feature engineering
# ---------------------------------------------------------------------------

def _read_one(path: Path, sheet: Optional[str] = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet or 0)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    raise ValueError(f"Unsupported data format: {path}")


def load_data(paths: Sequence[str], sheet: Optional[str] = None, include_regex: Optional[str] = None) -> pd.DataFrame:
    resolved: list[Path] = []
    supported = {".csv", ".txt", ".tsv", ".xlsx", ".xls", ".parquet", ".pq", ".json", ".jsonl"}
    pattern = re.compile(include_regex, flags=re.IGNORECASE) if include_regex else None
    for raw in paths:
        p = Path(raw).expanduser().resolve()
        if p.is_dir():
            resolved.extend(x for x in sorted(p.rglob("*")) if x.is_file() and x.suffix.lower() in supported)
        elif p.is_file():
            resolved.append(p)
        else:
            raise FileNotFoundError(p)
    if pattern is not None:
        resolved = [p for p in resolved if pattern.search(str(p)) or pattern.search(p.name)]
    if not resolved:
        raise ValueError("No supported data files were found.")
    frames: list[pd.DataFrame] = []
    for path in resolved:
        frame = _read_one(path, sheet=sheet)
        frame = frame.copy()
        frame["__aura_source_file__"] = str(path)
        frame["__aura_source_row__"] = np.arange(len(frame), dtype=np.int64)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True, sort=False)
    empty = [c for c in result.columns if str(c).lower().startswith("unnamed:") and result[c].isna().all()]
    if empty:
        result = result.drop(columns=empty)
    return result


def add_pid_error_features(
    frame: pd.DataFrame,
    source_column: Optional[str],
    setpoint: float,
    dt: float,
    integral_limit: Optional[float],
) -> pd.DataFrame:
    """Add generic controller-error features while resetting history per file."""
    if not source_column:
        return frame
    if source_column not in frame.columns:
        raise KeyError(f"PID source column not found: {source_column}")
    if dt <= 0:
        raise ValueError("--pid-dt must be positive.")
    out = frame.copy()
    error = pd.to_numeric(out[source_column], errors="coerce")
    error = setpoint - error
    out["pid_error"] = error
    rate = np.full(len(out), np.nan, dtype=np.float64)
    integ = np.full(len(out), np.nan, dtype=np.float64)
    for _, group in out.groupby("__aura_source_file__", sort=False, dropna=False):
        idx = group.sort_values("__aura_source_row__", kind="stable").index
        e = error.loc[idx].to_numpy(np.float64)
        r = np.zeros_like(e, dtype=np.float64)
        if len(e) > 1:
            r[1:] = np.diff(e) / dt
        s = np.cumsum(np.nan_to_num(e, nan=0.0) * dt)
        if integral_limit is not None and integral_limit > 0:
            s = np.clip(s, -float(integral_limit), float(integral_limit))
        rate[out.index.get_indexer(idx)] = r
        integ[out.index.get_indexer(idx)] = s
    out["pid_error_rate"] = rate
    out["pid_error_integral"] = integ
    return out


@dataclass
class PreparedData:
    """Aligned model inputs, optional targets, and source-row traceability."""
    x: pd.DataFrame
    y: Optional[np.ndarray]
    row_index: np.ndarray
    source_file: np.ndarray
    feature_columns: list[str]
    target_columns: list[str]


def prepare_supervised(
    frame: pd.DataFrame,
    features: Sequence[str],
    targets: Sequence[str],
    lags: Sequence[int],
    horizon: int,
    canonical_features: Optional[Sequence[str]] = None,
    require_target: bool = True,
    dynamic_features: bool = False,
) -> PreparedData:
    """Turn raw rows into leakage-controlled static or forecast examples.

    Each input file is processed as an independent sequence. Consequently,
    lags, differences, rolling means, and forecast targets never cross from the
    end of one experiment/file into the start of another.
    """
    features = list(features)
    targets = list(targets)
    canonical_features = list(canonical_features or features)
    if len(features) != len(canonical_features):
        raise ValueError("External feature mapping must contain the same number of columns as training features.")
    missing = [c for c in features if c not in frame.columns]
    if require_target:
        missing += [c for c in targets if c not in frame.columns]
    if missing:
        raise KeyError(f"Columns not found: {sorted(set(missing))}")
    if horizon < 0:
        raise ValueError("horizon must be >= 0")
    if horizon == 0 and any(c in features for c in targets):
        raise ValueError("A target cannot be a same-row feature. Use --horizon >= 1 for autoregression.")

    groups = frame.groupby("__aura_source_file__", sort=False, dropna=False)
    pieces: list[pd.DataFrame] = []
    ys: list[np.ndarray] = []
    rows: list[np.ndarray] = []
    sources: list[np.ndarray] = []
    output_names = list(canonical_features)
    if dynamic_features:
        for c in canonical_features:
            output_names.extend((f"{c}_diff1", f"{c}_diff2", f"{c}_mean3"))
    for lag in lags:
        output_names.extend(f"{c}_lag{lag}" for c in canonical_features)

    for source, group in groups:
        group = group.sort_values("__aura_source_row__", kind="stable")
        data: dict[str, pd.Series] = {}
        for actual, canonical in zip(features, canonical_features):
            data[canonical] = group[actual]
            if dynamic_features:
                numeric = pd.to_numeric(group[actual], errors="coerce")
                data[f"{canonical}_diff1"] = numeric.diff(1)
                data[f"{canonical}_diff2"] = numeric.diff(1).diff(1)
                data[f"{canonical}_mean3"] = numeric.rolling(3, min_periods=3).mean()
        for lag in lags:
            for actual, canonical in zip(features, canonical_features):
                data[f"{canonical}_lag{lag}"] = group[actual].shift(lag)
        x_part = pd.DataFrame(data, index=group.index)
        # A lagged model must not silently impute unavailable history at the
        # beginning of a sequence.  Static tabular inputs may still use the
        # encoder's documented missing-value handling.
        valid = x_part.notna().all(axis=1) if lags else ~x_part.isna().all(axis=1)
        if require_target:
            target_part = group[targets].shift(-horizon) if horizon > 0 else group[targets]
            valid &= target_part.notna().all(axis=1)
        valid &= x_part.notna().any(axis=1)
        x_part = x_part.loc[valid]
        pieces.append(x_part.reset_index(drop=True))
        if require_target:
            y_part = target_part.loc[valid].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
            ys.append(y_part)
        rows.append(group.loc[valid, "__aura_source_row__"].to_numpy(np.int64))
        sources.append(np.full(int(valid.sum()), str(source), dtype=object))

    x = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=output_names)
    y = np.concatenate(ys, axis=0) if ys else None
    row_index = np.concatenate(rows) if rows else np.empty(0, dtype=np.int64)
    source_file = np.concatenate(sources) if sources else np.empty(0, dtype=object)
    if require_target and y is not None:
        finite = np.isfinite(y).all(axis=1)
        x = x.loc[finite].reset_index(drop=True)
        y = y[finite]
        row_index = row_index[finite]
        source_file = source_file[finite]
    if len(x) < 20:
        raise ValueError(f"Only {len(x)} usable supervised rows remain; at least 20 are required.")
    return PreparedData(x, y, row_index, source_file, output_names, list(targets))


def infer_features(frame: pd.DataFrame, targets: Sequence[str]) -> list[str]:
    reserved = {"__aura_source_file__", "__aura_source_row__", *targets}
    candidates = [c for c in frame.columns if c not in reserved]
    usable = [c for c in candidates if not frame[c].isna().all() and frame[c].nunique(dropna=True) > 1]
    if not usable:
        raise ValueError("No usable feature columns were found; pass --features explicitly.")
    return usable


# ---------------------------------------------------------------------------
# Mixed numerical/categorical input encoding and target scaling
# ---------------------------------------------------------------------------

def fit_encoder(frame: pd.DataFrame, max_categories: int = 32) -> dict[str, Any]:
    numeric: list[str] = []
    categorical: list[str] = []
    for c in frame.columns:
        converted = pd.to_numeric(frame[c], errors="coerce")
        if converted.notna().mean() >= 0.90:
            numeric.append(c)
        else:
            categorical.append(c)
    medians: dict[str, float] = {}
    centers: dict[str, float] = {}
    scales: dict[str, float] = {}
    for c in numeric:
        values = pd.to_numeric(frame[c], errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(float)
        finite = values[np.isfinite(values)]
        median = float(np.median(finite)) if finite.size else 0.0
        q25, q75 = np.percentile(finite, [25.0, 75.0]) if finite.size else (0.0, 1.0)
        robust = float((q75 - q25) / 1.349)
        std = float(np.std(finite)) if finite.size else 1.0
        scale = max(robust, 0.10 * std, 1.0e-7)
        medians[c], centers[c], scales[c] = median, median, scale
    categories: dict[str, list[str]] = {}
    for c in categorical:
        values = frame[c].fillna("__MISSING__").astype(str)
        top = values.value_counts().head(max_categories).index.astype(str).tolist()
        categories[c] = top
    output_names = list(numeric)
    for c in categorical:
        output_names.extend(f"{c}=={v}" for v in categories[c])
        output_names.append(f"{c}==__OTHER__")
    return {
        "input_columns": list(frame.columns),
        "numeric": numeric,
        "categorical": categorical,
        "medians": medians,
        "centers": centers,
        "scales": scales,
        "categories": categories,
        "output_names": output_names,
        "clip": 15.0,
    }


def transform_encoder(frame: pd.DataFrame, encoder: dict[str, Any]) -> np.ndarray:
    missing = [c for c in encoder["input_columns"] if c not in frame.columns]
    if missing:
        raise KeyError(f"Prediction columns not found: {missing}")
    cols: list[np.ndarray] = []
    for c in encoder["numeric"]:
        s = pd.to_numeric(frame[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        values = s.fillna(encoder["medians"][c]).to_numpy(np.float64)
        values = (values - encoder["centers"][c]) / encoder["scales"][c]
        cols.append(np.clip(values, -encoder["clip"], encoder["clip"])[:, None])
    for c in encoder["categorical"]:
        values = frame[c].fillna("__MISSING__").astype(str).to_numpy()
        cats = encoder["categories"][c]
        known = np.zeros(len(values), dtype=bool)
        for cat in cats:
            flag = values == cat
            known |= flag
            cols.append(flag.astype(np.float64)[:, None])
        cols.append((~known).astype(np.float64)[:, None])
    if not cols:
        raise ValueError("The fitted encoder has no output columns.")
    return np.concatenate(cols, axis=1).astype(np.float32)


def fit_target_scaler(y: np.ndarray) -> dict[str, Any]:
    center = np.mean(y, axis=0, dtype=np.float64)
    scale = np.std(y, axis=0, dtype=np.float64)
    scale = np.where(scale < 1.0e-8, 1.0, scale)
    return {"center": center, "scale": scale}


def target_transform(y: np.ndarray, scaler: dict[str, Any]) -> np.ndarray:
    return ((y - scaler["center"]) / scaler["scale"]).astype(np.float32)


def target_inverse(y: np.ndarray, scaler: dict[str, Any]) -> np.ndarray:
    return np.asarray(y, dtype=np.float64) * scaler["scale"] + scaler["center"]


# ---------------------------------------------------------------------------
# Canonical AURA membership geometry and Takagi-Sugeno rule inference
# ---------------------------------------------------------------------------

def aura_membership_numpy(x: np.ndarray, state: dict[str, np.ndarray]) -> np.ndarray:
    # x: [n,d], parameters: [r,d], output: [n,r,d]
    centered = x[:, None, :] - state["center"][None, :, :]
    den = state["width"][None, :, :] * (
        1.0 + state["rho"][None, :, :] * np.tanh(state["lambda_gate"][None, :, :] * centered)
    )
    u = np.clip(centered / np.maximum(den, 1.0e-8), -1.0e6, 1.0e6)
    u2 = u * u
    root = np.sqrt(1.0 + u2)
    base = np.maximum(u2 / (root + 1.0) + state["eta"][None, :, :] * np.log1p(u2), 0.0)
    exponent = state["q0"][None, :, :] + state["q1"][None, :, :] * np.tanh(
        state["lambda_gate"][None, :, :] * u
    )
    log_power = exponent * np.log(np.clip(base, 1.0e-14, 1.0e14))
    power = np.exp(np.clip(log_power, -80.0, 40.0))
    skew = 1.0 + state["beta"][None, :, :] * (u / root) + state["zeta"][None, :, :] * (
        u * u2 / np.power(1.0 + u2, 1.5)
    )
    energy = np.maximum(power * np.maximum(skew, 1.0e-6), 0.0)
    ratio = np.arctan(1.0 / (energy + AURA_EPSILON)) / math.atan(1.0 / AURA_EPSILON)
    mu = np.power(np.clip(ratio, 1.0e-12, 1.0), state["alpha"][None, :, :])
    return np.clip(mu, 1.0e-12, 1.0)


def aura_rule_weights_numpy(x: np.ndarray, state: dict[str, np.ndarray]) -> np.ndarray:
    mu = aura_membership_numpy(x, state)
    log_w = np.sum(np.log(mu + 1.0e-12), axis=2)
    log_w -= np.max(log_w, axis=1, keepdims=True)
    w = np.exp(np.clip(log_w, -80.0, 0.0))
    return w / np.maximum(np.sum(w, axis=1, keepdims=True), 1.0e-12)


def aura_predict_scaled_numpy(x: np.ndarray, state: dict[str, np.ndarray], return_weights: bool = False) -> Any:
    weights = aura_rule_weights_numpy(x, state)
    aug = np.concatenate([np.ones((len(x), 1), dtype=np.float32), x.astype(np.float32)], axis=1)
    local = np.einsum("nd,rdo->nro", aug, state["consequents"], optimize=True)
    pred = np.sum(weights[:, :, None] * local, axis=1)
    return (pred, weights) if return_weights else pred


def _init_centers_widths(x: np.ndarray, rules: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(x) > 20000:
        km = MiniBatchKMeans(n_clusters=rules, random_state=seed, batch_size=min(4096, len(x)), n_init=5)
    else:
        km = KMeans(n_clusters=rules, random_state=seed, n_init=10)
    labels = km.fit_predict(x)
    centers = km.cluster_centers_.astype(np.float64)
    global_scale = np.maximum(np.std(x, axis=0), 0.25)
    widths = np.empty_like(centers)
    for r in range(rules):
        members = x[labels == r]
        if len(members) >= 3:
            mad = np.median(np.abs(members - centers[r]), axis=0) * 1.4826
            local = np.std(members, axis=0)
            widths[r] = np.maximum(np.maximum(mad, 0.35 * local), 0.12 * global_scale)
        else:
            widths[r] = 0.60 * global_scale
    widths = np.clip(widths, 0.08, 8.0)
    order = np.argsort(centers[:, 0], kind="stable")
    return centers[order].astype(np.float32), widths[order].astype(np.float32)


SHAPE_PRESETS: dict[str, tuple[float, float, float]] = {
    # eta, positive q0-free term, alpha (q0 = Q_MIN + q0_free + |q1|)
    "default": (0.10, 1.20, 1.00),
    "lowtail": (1.00, 0.20, 0.50),
    "phtail": (0.10, 0.20, 3.00),
    "wide": (1.00, 2.50, 1.00),
}


def make_initial_state(x: np.ndarray, rules: int, seed: int, preset: str) -> dict[str, np.ndarray]:
    center, width = _init_centers_widths(x, rules, seed)
    eta, q0_free, alpha = SHAPE_PRESETS[preset]
    shape = center.shape
    zeros = np.zeros(shape, dtype=np.float32)
    state = {
        "center": center,
        "width": width,
        "rho": zeros.copy(),
        "lambda_gate": np.full(shape, LAMBDA_GATE, dtype=np.float32),
        "eta": np.full(shape, eta, dtype=np.float32),
        "q0": np.full(shape, Q_MIN + q0_free, dtype=np.float32),
        "q1": zeros.copy(),
        "beta": zeros.copy(),
        "zeta": zeros.copy(),
        "alpha": np.full(shape, alpha, dtype=np.float32),
        "consequents": np.empty((rules, x.shape[1] + 1, 1), dtype=np.float32),
    }
    return state


def fit_consequents(
    x: np.ndarray,
    y: np.ndarray,
    state: dict[str, np.ndarray],
    alpha: float,
    max_rows: int,
    seed: int,
) -> np.ndarray:
    if len(x) > max_rows:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(x), max_rows, replace=False))
        xf, yf = x[idx], y[idx]
    else:
        xf, yf = x, y
    weights = aura_rule_weights_numpy(xf, state)
    aug = np.concatenate([np.ones((len(xf), 1), dtype=np.float32), xf], axis=1)
    design = (weights[:, :, None] * aug[:, None, :]).reshape(len(xf), -1)
    ridge = Ridge(alpha=float(alpha), fit_intercept=False, solver="lsqr", tol=1.0e-7, max_iter=10000)
    ridge.fit(design, yf)
    coef = np.asarray(ridge.coef_, dtype=np.float64)
    if coef.ndim == 1:
        coef = coef[None, :]
    return coef.T.reshape(state["center"].shape[0], x.shape[1] + 1, y.shape[1]).astype(np.float32)


class TorchAURA(nn.Module):
    """Differentiable AURA implementation with feasibility by construction."""
    def __init__(self, initial: dict[str, np.ndarray]) -> None:
        super().__init__()
        center = initial["center"]
        self.center = nn.Parameter(torch.tensor(center))
        self.width_raw = nn.Parameter(torch.tensor(_softplus_inverse(initial["width"] - 1.0e-3), dtype=torch.float32))
        self.rho_raw = nn.Parameter(torch.zeros_like(self.center))
        self.eta_raw = nn.Parameter(torch.tensor(_softplus_inverse(initial["eta"]), dtype=torch.float32))
        q0_free = np.maximum(initial["q0"] - Q_MIN, 1.0e-4)
        self.q0_free_raw = nn.Parameter(torch.tensor(_softplus_inverse(q0_free), dtype=torch.float32))
        self.q1_raw = nn.Parameter(torch.zeros_like(self.center))
        self.beta_raw = nn.Parameter(torch.zeros_like(self.center))
        self.zeta_raw = nn.Parameter(torch.zeros_like(self.center))
        self.alpha_raw = nn.Parameter(torch.tensor(_softplus_inverse(initial["alpha"] - 0.05), dtype=torch.float32))
        self.consequents = nn.Parameter(torch.tensor(initial["consequents"], dtype=torch.float32))
        self.register_buffer("lambda_gate", torch.full_like(self.center, LAMBDA_GATE))
        self.register_buffer("center_anchor", torch.tensor(center))
        self.register_buffer("width_anchor", torch.tensor(initial["width"]))

    def parameters_feasible(self) -> tuple[torch.Tensor, ...]:
        width = F.softplus(self.width_raw) + 1.0e-3
        rho = RHO_BOUND * torch.tanh(self.rho_raw)
        eta = F.softplus(self.eta_raw)
        q1 = Q1_BOUND * torch.tanh(self.q1_raw)
        q0 = Q_MIN + torch.abs(q1) + F.softplus(self.q0_free_raw)
        beta = SKEW_PAIR_BOUND * torch.tanh(self.beta_raw)
        zeta = SKEW_PAIR_BOUND * torch.tanh(self.zeta_raw)
        pair = torch.abs(beta) + torch.abs(zeta)
        factor = torch.clamp(SKEW_PAIR_BOUND / (pair + 1.0e-8), max=1.0)
        beta, zeta = beta * factor, zeta * factor
        alpha = F.softplus(self.alpha_raw) + 0.05
        return self.center, width, rho, eta, q0, q1, beta, zeta, alpha

    def rule_weights(self, x: torch.Tensor) -> torch.Tensor:
        center, width, rho, eta, q0, q1, beta, zeta, alpha = self.parameters_feasible()
        centered = x[:, None, :] - center[None, :, :]
        den = width[None, :, :] * (1.0 + rho[None, :, :] * torch.tanh(self.lambda_gate[None, :, :] * centered))
        u = torch.clamp(centered / torch.clamp(den, min=1.0e-8), -1.0e6, 1.0e6)
        u2 = u.square()
        root = torch.sqrt(1.0 + u2)
        base = torch.clamp(u2 / (root + 1.0) + eta[None, :, :] * torch.log1p(u2), min=0.0)
        exponent = q0[None, :, :] + q1[None, :, :] * torch.tanh(self.lambda_gate[None, :, :] * u)
        power = torch.exp(torch.clamp(exponent * torch.log(torch.clamp(base, min=1.0e-14)), -80.0, 40.0))
        skew = 1.0 + beta[None, :, :] * (u / root) + zeta[None, :, :] * (u * u2 / (1.0 + u2).pow(1.5))
        energy = torch.clamp(power * torch.clamp(skew, min=1.0e-6), min=0.0)
        ratio = torch.atan(torch.reciprocal(energy + AURA_EPSILON)) / math.atan(1.0 / AURA_EPSILON)
        mu = torch.pow(torch.clamp(ratio, 1.0e-12, 1.0), alpha[None, :, :])
        log_w = torch.sum(torch.log(mu + 1.0e-12), dim=2)
        return torch.softmax(log_w, dim=1)

    def forward(self, x: torch.Tensor, return_weights: bool = False) -> Any:
        w = self.rule_weights(x)
        aug = torch.cat([torch.ones((len(x), 1), dtype=x.dtype, device=x.device), x], dim=1)
        local = torch.einsum("nd,rdo->nro", aug, self.consequents)
        pred = torch.sum(w.unsqueeze(-1) * local, dim=1)
        return (pred, w) if return_weights else pred

    def penalty(self) -> torch.Tensor:
        center, width, rho, eta, q0, q1, beta, zeta, alpha = self.parameters_feasible()
        return (
            1.0e-5 * (center - self.center_anchor).square().mean()
            + 2.0e-6 * (width - self.width_anchor).square().mean()
            + 1.0e-6 * (rho.square() + q1.square() + beta.square() + zeta.square()).mean()
            + 1.0e-8 * (eta.square() + q0.square() + alpha.square()).mean()
        )

    def export_state(self) -> dict[str, np.ndarray]:
        center, width, rho, eta, q0, q1, beta, zeta, alpha = self.parameters_feasible()
        values = {
            "center": center,
            "width": width,
            "rho": rho,
            "eta": eta,
            "q0": q0,
            "q1": q1,
            "beta": beta,
            "zeta": zeta,
            "alpha": alpha,
            "lambda_gate": self.lambda_gate,
            "consequents": self.consequents,
        }
        return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in values.items()}


def _predict_torch(model: TorchAURA, x: np.ndarray, device: torch.device, batch_size: int, weights: bool = False) -> Any:
    model.eval()
    preds: list[np.ndarray] = []
    fires: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            if weights:
                p, w = model(xb, return_weights=True)
                fires.append(w.cpu().numpy())
            else:
                p = model(xb)
            preds.append(p.cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    return (pred, np.concatenate(fires, axis=0)) if weights else pred


def tune_aura(
    initial: dict[str, np.ndarray],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    ridge_alpha: float,
    ridge_every: int,
    ridge_max_rows: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], float]:
    _seed_everything(seed)
    model = TorchAURA(initial).to(device)
    premise = [p for name, p in model.named_parameters() if name != "consequents"]
    optimizer = torch.optim.AdamW(
        [
            {"params": premise, "lr": learning_rate * 0.45},
            {"params": [model.consequents], "lr": learning_rate},
        ],
        weight_decay=2.0e-5,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.55, patience=max(3, patience // 4))
    xt = torch.tensor(x_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(xt, yt)
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True, generator=generator)

    best_state = copy.deepcopy(model.state_dict())
    best_rmse = float("inf")
    stale = 0
    for epoch in range(max(0, epochs) + 1):
        if epoch > 0:
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                pred = model(xb)
                mse = F.mse_loss(pred, yb)
                huber = F.smooth_l1_loss(pred, yb, beta=0.50)
                loss = 0.82 * mse + 0.18 * huber + model.penalty()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
        if epoch > 0 and ridge_every > 0 and epoch % ridge_every == 0:
            current = model.export_state()
            current["consequents"] = fit_consequents(
                x_train, y_train, current, ridge_alpha, ridge_max_rows, seed + epoch
            )
            with torch.no_grad():
                model.consequents.copy_(torch.tensor(current["consequents"], device=device))
        val_pred = _predict_torch(model, x_val, device, batch_size)
        val_rmse = float(np.sqrt(np.mean((val_pred - y_val) ** 2)))
        scheduler.step(val_rmse)
        if val_rmse < best_rmse - 1.0e-7:
            best_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch >= max(10, patience) and stale >= patience:
            break
    model.load_state_dict(best_state)
    return model.export_state(), best_rmse


def _split_indices(n: int, train_ratio: float, val_ratio: float, test_ratio: float, strategy: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value.")
    train_ratio, val_ratio, test_ratio = train_ratio / total, val_ratio / total, test_ratio / total
    if min(train_ratio, val_ratio) <= 0:
        raise ValueError("Training and validation ratios must both be positive.")
    idx = np.arange(n)
    if strategy == "random":
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    n_train = max(5, int(n * train_ratio))
    n_val = max(5, int(n * val_ratio))
    if n_train + n_val >= n:
        n_train = max(5, n - n_val - 1)
    return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]


def _select_rule_candidates(requested: Sequence[int], n: int, d: int) -> list[int]:
    candidates = list(requested) or [4, 8, 16, 32]
    max_rules_by_rows = max(2, int(max(2, 0.80 * n / max(d + 1, 1))))
    candidates = [r for r in candidates if 2 <= r <= min(n // 4, max_rules_by_rows, 64)]
    if not candidates:
        candidates = [max(2, min(8, n // max(4 * (d + 1), 1)))]
    return sorted(set(candidates))


def _deterministic_row_sample(n_rows: int, max_rows: int, seed: int) -> np.ndarray:
    if max_rows <= 0 or n_rows <= max_rows:
        return np.arange(n_rows, dtype=int)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=max_rows, replace=False)).astype(int)


def _metric_r2(y: np.ndarray, p: np.ndarray) -> float:
    try:
        return float(r2_score(y, p, multioutput="uniform_average"))
    except Exception:
        return float("nan")


def metrics(y: np.ndarray, p: np.ndarray, feature_count: int, names: Sequence[str]) -> dict[str, Any]:
    """Calculate domain-neutral regression metrics in the target's real units."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    if p.ndim == 1:
        p = p[:, None]
    per: dict[str, Any] = {}
    for j, name in enumerate(names):
        yt, pt = y[:, j], p[:, j]
        e = yt - pt
        mse = float(np.mean(e * e))
        mae = float(np.mean(np.abs(e)))
        r2 = float(r2_score(yt, pt)) if np.var(yt) > EPS else float("nan")
        denom = np.maximum(np.abs(yt), 1.0e-8)
        smape_denom = np.maximum(np.abs(yt) + np.abs(pt), 1.0e-8)
        covariance = float(np.mean((yt - yt.mean()) * (pt - pt.mean())))
        ccc_den = float(np.var(yt) + np.var(pt) + (yt.mean() - pt.mean()) ** 2)
        ccc = 2.0 * covariance / ccc_den if ccc_den > EPS else float("nan")
        n = len(yt)
        adjusted = 1.0 - (1.0 - r2) * (n - 1.0) / (n - feature_count - 1.0) if n > feature_count + 1 and np.isfinite(r2) else float("nan")
        nse_den = float(np.sum((yt - yt.mean()) ** 2))
        nse = 1.0 - float(np.sum(e * e)) / nse_den if nse_den > EPS else float("nan")
        will_denom = float(np.sum((np.abs(pt - yt.mean()) + np.abs(yt - yt.mean())) ** 2))
        willmott = 1.0 - float(np.sum(e * e)) / will_denom if will_denom > EPS else float("nan")
        scale_range = float(np.max(yt) - np.min(yt))
        values = {
            "MAE": mae,
            "MSE": mse,
            "MSAE": mse,
            "RMSE": math.sqrt(mse),
            "R2": r2,
            "Adjusted_R2": adjusted,
            "Median_AE": float(median_absolute_error(yt, pt)),
            "Max_AE": float(max_error(yt, pt)),
            "MAPE_percent": float(100.0 * np.mean(np.abs(e) / denom)),
            "sMAPE_percent": float(200.0 * np.mean(np.abs(e) / smape_denom)),
            "Explained_Variance": float(explained_variance_score(yt, pt)),
            "Mean_Bias_Error": float(np.mean(pt - yt)),
            "NRMSE_range": math.sqrt(mse) / scale_range if scale_range > EPS else float("nan"),
            "CCC": ccc,
            "NSE": nse,
            "Willmott_d": willmott,
        }
        if np.min(yt) >= 0 and np.min(pt) >= 0:
            values["RMSLE"] = float(np.sqrt(np.mean((np.log1p(pt) - np.log1p(yt)) ** 2)))
        per[str(name)] = values
    macro = {k: float(np.nanmean([v.get(k, np.nan) for v in per.values()])) for k in next(iter(per.values())).keys()}
    return {"samples": len(y), "targets": list(names), "macro": macro, "per_target": per}


def _plot_axes(plt: Any, count: int, title: str) -> tuple[Any, np.ndarray]:
    """Create a readable subplot grid for one or several target columns."""
    columns = min(3, max(1, count))
    rows = int(math.ceil(count / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(5.4 * columns, 3.8 * rows), squeeze=False)
    flat = axes.ravel()
    for ax in flat[count:]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    return fig, flat


def _plot_sample_indices(
    row_index: np.ndarray,
    source_file: np.ndarray,
    maximum: int,
) -> np.ndarray:
    """Return a deterministic, sequence-aware sample for responsive plotting."""
    n = len(row_index)
    if n == 0:
        return np.empty(0, dtype=int)
    sources = np.asarray(source_file, dtype=str)
    rows = np.asarray(row_index, dtype=np.int64)
    ordered = np.lexsort((rows, sources))
    if maximum <= 0 or len(ordered) <= maximum:
        return ordered
    positions = np.linspace(0, len(ordered) - 1, num=maximum, dtype=int)
    return ordered[positions]


def generate_diagnostic_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Sequence[str],
    report: dict[str, Any],
    row_index: np.ndarray,
    source_file: np.ndarray,
    output_dir: Path,
    max_points: int = 5000,
    dpi: int = 160,
) -> list[Path]:
    """Create six generic held-out-test plots that work across domains.

    Plotting is deliberately based only on the final test predictions. The
    training curves are not substituted for test performance. At most
    ``max_points`` ordered samples are drawn so very large data sets remain
    practical; every metric is still calculated from the full test set.
    """
    try:
        import matplotlib

        # ``Agg`` creates PNG files on servers and other systems without a GUI.
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plot generation requires matplotlib. Install it with: "
            "python -m pip install matplotlib"
        ) from exc

    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.ndim == 1:
        yt = yt[:, None]
    if yp.ndim == 1:
        yp = yp[:, None]
    if yt.shape != yp.shape or yt.shape[1] != len(target_names):
        raise ValueError("Plot inputs do not match the test targets and predictions.")
    if max_points < 100:
        raise ValueError("--plot-max-points must be at least 100.")
    if dpi < 72:
        raise ValueError("--plot-dpi must be at least 72.")

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_targets = min(yt.shape[1], 12)
    names = [str(name) for name in target_names[:plot_targets]]
    selected = _plot_sample_indices(row_index, source_file, max_points)
    ys, ps = yt[selected, :plot_targets], yp[selected, :plot_targets]
    residual = ys - ps
    created: list[Path] = []

    def save(fig: Any, filename: str) -> None:
        path = output_dir / filename
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        created.append(path)

    # 1) Directly shows tracking quality and whether errors change over the
    # held-out sequence. Curves are ordered by source file and original row.
    fig, axes = _plot_axes(plt, plot_targets, "Held-out test: actual and predicted")
    x_axis = np.arange(len(selected))
    for j, (ax, name) in enumerate(zip(axes, names)):
        ax.plot(x_axis, ys[:, j], color="#1f77b4", linewidth=1.15, label="Actual")
        ax.plot(x_axis, ps[:, j], color="#d62728", linewidth=1.0, alpha=0.85, label="Predicted")
        ax.set_title(name)
        ax.set_xlabel("Ordered test sample")
        ax.set_ylabel("Target value")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    save(fig, "01_actual_vs_predicted.png")

    # 2) A strong model places points close to the dashed 1:1 reference line.
    fig, axes = _plot_axes(plt, plot_targets, "Held-out test: prediction parity")
    for j, (ax, name) in enumerate(zip(axes, names)):
        lo = float(min(np.min(ys[:, j]), np.min(ps[:, j])))
        hi = float(max(np.max(ys[:, j]), np.max(ps[:, j])))
        if abs(hi - lo) < EPS:
            lo, hi = lo - 0.5, hi + 0.5
        ax.scatter(ys[:, j], ps[:, j], s=11, alpha=0.35, color="#2a6fbb", edgecolors="none")
        ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1.1, label="Ideal 1:1")
        r2 = report["per_target"][name]["R2"]
        ax.text(0.03, 0.95, f"R2 = {r2:.6g}", transform=ax.transAxes, va="top")
        ax.set_title(name)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right")
    save(fig, "02_prediction_parity.png")

    # 3) Residuals should be centered around zero without a curved/funnel shape.
    fig, axes = _plot_axes(plt, plot_targets, "Held-out test: residuals versus prediction")
    for j, (ax, name) in enumerate(zip(axes, names)):
        ax.scatter(ps[:, j], residual[:, j], s=11, alpha=0.35, color="#6a3d9a", edgecolors="none")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.1)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual (actual - predicted)")
        ax.grid(alpha=0.25)
    save(fig, "03_residuals_vs_prediction.png")

    # 4) Reveals skew, heavy tails, and occasional large errors.
    fig, axes = _plot_axes(plt, plot_targets, "Held-out test: residual distribution")
    for j, (ax, name) in enumerate(zip(axes, names)):
        bins = min(80, max(15, int(math.sqrt(len(residual)))))
        ax.hist(residual[:, j], bins=bins, color="#33a02c", alpha=0.78, edgecolor="white")
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.1)
        ax.axvline(float(np.mean(residual[:, j])), color="#d62728", linewidth=1.1, label="Mean")
        ax.set_title(name)
        ax.set_xlabel("Residual (actual - predicted)")
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="best")
    save(fig, "04_residual_distribution.png")

    # 5) The empirical CDF answers: what fraction of test cases is below a
    # chosen absolute-error tolerance in the target's real engineering units?
    fig, axes = _plot_axes(plt, plot_targets, "Held-out test: absolute-error coverage")
    for j, (ax, name) in enumerate(zip(axes, names)):
        absolute = np.sort(np.abs(residual[:, j]))
        coverage = np.arange(1, len(absolute) + 1, dtype=float) / len(absolute)
        ax.plot(absolute, coverage, color="#ff7f00", linewidth=1.6)
        ax.axhline(0.90, color="black", linestyle="--", linewidth=0.9, alpha=0.7)
        ax.set_title(name)
        ax.set_xlabel("Absolute error")
        ax.set_ylabel("Fraction of test samples")
        ax.set_ylim(0.0, 1.01)
        ax.grid(alpha=0.25)
    save(fig, "05_absolute_error_coverage.png")

    # 6) R2 measures explained variation; NRMSE expresses RMSE relative to the
    # observed target range. Both are shown because R2 alone is not sufficient.
    fig, axes = plt.subplots(1, 2, figsize=(max(9.0, 1.1 * plot_targets + 7.0), 4.8))
    positions = np.arange(plot_targets)
    r2_values = [report["per_target"][name]["R2"] for name in names]
    nrmse_values = [report["per_target"][name]["NRMSE_range"] for name in names]
    axes[0].bar(positions, r2_values, color="#1f78b4")
    axes[0].axhline(0.99, color="#d62728", linestyle="--", linewidth=1.0, label="R2 = 0.99")
    axes[0].set_title("R2 (higher is better)")
    axes[0].set_ylabel("R2")
    axes[0].legend(loc="best")
    axes[1].bar(positions, nrmse_values, color="#e31a1c")
    axes[1].set_title("Range-normalized RMSE (lower is better)")
    axes[1].set_ylabel("NRMSE")
    for ax in axes:
        ax.set_xticks(positions, names, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Held-out test: per-target metric summary", fontsize=14, fontweight="bold")
    save(fig, "06_metric_summary.png")

    if yt.shape[1] > plot_targets:
        warnings.warn(
            f"Plots show the first {plot_targets} of {yt.shape[1]} targets; "
            "terminal and stored metrics still include every target."
        )
    return created


def _augmented_features(x: np.ndarray, base: np.ndarray, weights: np.ndarray) -> np.ndarray:
    entropy = -np.sum(weights * np.log(weights + 1.0e-12), axis=1, keepdims=True)
    max_fire = np.max(weights, axis=1, keepdims=True)
    return np.concatenate([x, base, weights, entropy, max_fire], axis=1).astype(np.float32)


def _fit_refiner_candidates(
    z_train: np.ndarray,
    residual_train: np.ndarray,
    z_val: np.ndarray,
    y_val: np.ndarray,
    base_val: np.ndarray,
    seed: int,
    jobs: int,
    trees: int,
    requested: Sequence[str],
) -> tuple[str, Any, np.ndarray, float]:
    output_dim = residual_train.shape[1]
    candidates: list[tuple[str, Any]] = [("none", None)]
    for leaf in (1, 2, 4):
        name = f"extra_trees_leaf{leaf}"
        if name not in requested:
            continue
        estimator = ExtraTreesRegressor(
            n_estimators=trees,
            min_samples_leaf=leaf,
            max_features=1.0,
            random_state=seed + leaf,
            n_jobs=jobs,
        )
        candidates.append((name, estimator))
    if "random_forest" in requested:
        rf = RandomForestRegressor(
            n_estimators=max(120, trees // 2),
            min_samples_leaf=1,
            max_features=0.85,
            random_state=seed + 31,
            n_jobs=jobs,
        )
        candidates.append(("random_forest", rf))
    if "hist_gradient" in requested:
        hist_base = HistGradientBoostingRegressor(
            max_iter=350,
            learning_rate=0.045,
            max_leaf_nodes=31,
            l2_regularization=1.0e-4,
            random_state=seed + 47,
        )
        hist = hist_base if output_dim == 1 else MultiOutputRegressor(hist_base, n_jobs=jobs)
        candidates.append(("hist_gradient", hist))

    best_name, best_model = "none", None
    best_alpha = np.zeros(output_dim, dtype=np.float64)
    best_rmse = float(np.sqrt(np.mean((base_val - y_val) ** 2)))
    for name, estimator in candidates[1:]:
        try:
            target = residual_train.ravel() if output_dim == 1 else residual_train
            estimator.fit(z_train, target)
            rv = np.asarray(estimator.predict(z_val), dtype=np.float64)
            if rv.ndim == 1:
                rv = rv[:, None]
            alpha = np.ones(output_dim, dtype=np.float64)
            for j in range(output_dim):
                den = float(np.dot(rv[:, j], rv[:, j]))
                num = float(np.dot(rv[:, j], y_val[:, j] - base_val[:, j]))
                alpha[j] = np.clip(num / den if den > EPS else 0.0, 0.0, 1.50)
            pv = base_val + rv * alpha[None, :]
            rmse = float(np.sqrt(np.mean((pv - y_val) ** 2)))
            if rmse < best_rmse - 1.0e-10:
                best_name, best_model, best_alpha, best_rmse = name, estimator, alpha, rmse
        except Exception as exc:
            warnings.warn(f"Refiner candidate {name} was skipped: {exc}")
    return best_name, best_model, best_alpha, best_rmse


class AURAUnifiedModel:
    """Complete fitted AURA pipeline stored in the exported pickle."""

    def __init__(self, artifact: dict[str, Any]) -> None:
        self.artifact = artifact

    def _base_encoded(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return aura_predict_scaled_numpy(x, self.artifact["aura_state"], return_weights=True)

    def predict_encoded(self, x: np.ndarray) -> np.ndarray:
        base, weights = self._base_encoded(np.asarray(x, dtype=np.float32))
        final = base.astype(np.float64)
        refiner = self.artifact.get("refiner")
        if refiner is not None:
            z = _augmented_features(np.asarray(x, dtype=np.float32), base, weights)
            residual = np.asarray(refiner.predict(z), dtype=np.float64)
            if residual.ndim == 1:
                residual = residual[:, None]
            final = final + residual * np.asarray(self.artifact["refiner_alpha"])[None, :]
        bounds = self.artifact.get("target_bounds")
        pred = target_inverse(final, self.artifact["target_scaler"])
        if bounds is not None:
            pred = np.clip(pred, np.asarray(bounds[0]), np.asarray(bounds[1]))
        return pred

    def prepare_frame(self, frame: pd.DataFrame, feature_override: Optional[Sequence[str]] = None) -> PreparedData:
        schema = self.artifact["schema"]
        pid = schema.get("pid") or {}
        frame = add_pid_error_features(
            frame,
            pid.get("source_column"),
            float(pid.get("setpoint", 0.0)),
            float(pid.get("dt", 1.0)),
            pid.get("integral_limit"),
        )
        features = list(feature_override or schema["raw_feature_columns"])
        return prepare_supervised(
            frame,
            features=features,
            targets=schema["target_columns"],
            lags=schema["lags"],
            horizon=schema["horizon"],
            canonical_features=schema["raw_feature_columns"],
            require_target=False,
            dynamic_features=bool(schema.get("dynamic_features", False)),
        )

    def predict_frame(self, frame: pd.DataFrame, feature_override: Optional[Sequence[str]] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        prepared = self.prepare_frame(frame, feature_override=feature_override)
        x = transform_encoder(prepared.x, self.artifact["encoder"])
        return self.predict_encoded(x), prepared.row_index, prepared.source_file

    @property
    def metrics(self) -> dict[str, Any]:
        return self.artifact.get("metrics", {})


AURAUnifiedModel.__module__ = "AURA"


def _sanity_check_state(state: dict[str, np.ndarray]) -> None:
    if not np.all(state["width"] > 0):
        raise RuntimeError("AURA feasibility failure: nonpositive width.")
    if not np.all(np.abs(state["rho"]) < 1):
        raise RuntimeError("AURA feasibility failure: |rho| >= 1.")
    if not np.all(state["q0"] - np.abs(state["q1"]) > 1):
        raise RuntimeError("AURA feasibility failure: directional tail order <= 1.")
    if not np.all(np.abs(state["beta"]) + np.abs(state["zeta"]) < 1):
        raise RuntimeError("AURA feasibility failure: skew pair crosses positivity boundary.")
    if not np.all(state["alpha"] > 0):
        raise RuntimeError("AURA feasibility failure: alpha <= 0.")
    centers = state["center"].reshape(-1)
    sample_state = {k: (v.reshape(-1, 1) if v.ndim == 2 and k != "consequents" else v) for k, v in state.items()}
    for i in np.linspace(0, len(centers) - 1, min(16, len(centers)), dtype=int):
        one = {k: (v[i : i + 1] if k != "consequents" else v[i : i + 1]) for k, v in sample_state.items()}
        mu = aura_membership_numpy(np.array([[centers[i]]], dtype=np.float32), one)
        if not np.isfinite(mu).all() or abs(float(mu[0, 0, 0]) - 1.0) > 2.0e-5:
            raise RuntimeError("AURA center-anchor numerical sanity check failed.")


def _print_metrics(split: str, report: dict[str, Any]) -> None:
    macro = report["macro"]
    print(f"\nFINAL {split.upper()} METRICS ({report['samples']} samples)")
    print("-" * 66)
    order = [
        "MAE", "MSE", "MSAE", "RMSE", "R2", "Adjusted_R2", "Median_AE", "Max_AE",
        "MAPE_percent", "sMAPE_percent", "Explained_Variance", "Mean_Bias_Error",
        "NRMSE_range", "CCC", "NSE", "Willmott_d", "RMSLE",
    ]
    for key in order:
        if key in macro:
            print(f"{key:24s}: {macro[key]:.10g}")
    if len(report["per_target"]) > 1:
        print("\nPer-target core metrics:")
        for name, values in report["per_target"].items():
            print(
                f"  {name}: MAE={values['MAE']:.8g}, MSE={values['MSE']:.8g}, "
                f"RMSE={values['RMSE']:.8g}, R2={values['R2']:.8g}"
            )


def _validate_training_arguments(args: argparse.Namespace) -> None:
    """Fail early with simple messages for common public-use mistakes."""
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if any(not np.isfinite(value) or value < 0 for value in ratios):
        raise ValueError("Train/validation/test ratios must be finite and non-negative.")
    if args.train_ratio <= 0 or args.val_ratio <= 0:
        raise ValueError("--train-ratio and --val-ratio must be greater than zero.")
    if not args.external_data and args.test_ratio <= 0:
        raise ValueError("--test-ratio must be greater than zero unless --external-data is supplied.")
    if args.epochs < 0:
        raise ValueError("--epochs cannot be negative.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    if args.patience < 1:
        raise ValueError("--patience must be at least 1.")
    if args.gradient_candidates < 1:
        raise ValueError("--gradient-candidates must be at least 1.")
    if args.ridge_max_rows < 1:
        raise ValueError("--ridge-max-rows must be at least 1.")
    if args.refiner_max_rows < 0:
        raise ValueError("--refiner-max-rows cannot be negative (0 means use all rows).")
    if args.refiner_trees < 1:
        raise ValueError("--refiner-trees must be at least 1.")
    if args.jobs == 0:
        raise ValueError("--jobs cannot be 0; use -1 for all available cores.")
    if args.max_categories < 1:
        raise ValueError("--max-categories must be at least 1.")
    if not args.no_plots and args.plot_max_points < 100:
        raise ValueError("--plot-max-points must be at least 100.")
    if not args.no_plots and args.plot_dpi < 72:
        raise ValueError("--plot-dpi must be at least 72.")


def train_command(args: argparse.Namespace) -> int:
    """Train, evaluate, visualize, serialize, and reload-check one AURA model."""
    started = time.time()
    _seed_everything(args.seed)
    dev = _device(args.device)
    out = Path(args.model_out).expanduser().resolve()
    _validate_training_arguments(args)
    if args.validation_data and not args.external_data:
        raise ValueError(
            "When --validation-data is supplied, also supply untouched "
            "--external-data for the final test. Otherwise omit both options "
            "and this program will make train/validation/test splits."
        )
    if args.split == "random" and (args.horizon > 0 or _parse_int_list(args.lags) or args.dynamic_features):
        warnings.warn(
            "Random splitting of time-series windows can give optimistic scores "
            "because neighboring windows overlap. Prefer chronological or "
            "independent external trajectories for a publishable evaluation."
        )
    if not args.no_plots:
        try:
            import matplotlib  # noqa: F401  # Early check avoids training for hours before discovering a missing package.
        except ImportError as exc:
            raise RuntimeError(
                "Plot generation is enabled but matplotlib is not installed. "
                "Run: python -m pip install matplotlib"
            ) from exc
    targets = _csv_list(args.target)
    if not targets:
        raise ValueError("--target is required.")
    if len(set(targets)) != len(targets):
        raise ValueError("--target contains a duplicate column name.")
    pid_integral_limit = None if args.pid_integral_limit <= 0 else float(args.pid_integral_limit)
    frame = load_data(args.data, sheet=args.sheet, include_regex=args.data_regex)
    frame = add_pid_error_features(frame, args.pid_source, args.pid_setpoint, args.pid_dt, pid_integral_limit)
    features = infer_features(frame, targets) if args.features.lower() == "auto" else _csv_list(args.features)
    if not features:
        raise ValueError("No features were selected. Pass --features or use --features auto.")
    if len(set(features)) != len(features):
        raise ValueError("--features contains a duplicate column name.")
    lags = _parse_int_list(args.lags)
    prepared = prepare_supervised(
        frame, features, targets, lags, args.horizon,
        require_target=True, dynamic_features=args.dynamic_features,
    )

    validation: Optional[PreparedData] = None
    if args.validation_data:
        val_frame = load_data(args.validation_data, sheet=args.validation_sheet or args.sheet, include_regex=args.validation_data_regex)
        val_frame = add_pid_error_features(val_frame, args.pid_source, args.pid_setpoint, args.pid_dt, pid_integral_limit)
        val_features = _csv_list(args.validation_features) or features
        val_targets = _csv_list(args.validation_target) or targets
        if len(val_targets) != len(targets):
            raise ValueError("--validation-target must map one column to each training target.")
        validation = prepare_supervised(
            val_frame,
            val_features,
            val_targets,
            lags,
            args.horizon,
            canonical_features=features,
            require_target=True,
            dynamic_features=args.dynamic_features,
        )

    external: Optional[PreparedData] = None
    if args.external_data:
        ext_frame = load_data(args.external_data, sheet=args.external_sheet or args.sheet, include_regex=args.external_data_regex)
        ext_frame = add_pid_error_features(ext_frame, args.pid_source, args.pid_setpoint, args.pid_dt, pid_integral_limit)
        ext_features = _csv_list(args.external_features) or features
        ext_targets = _csv_list(args.external_target) or targets
        if len(ext_targets) != len(targets):
            raise ValueError("--external-target must map one column to each training target.")
        external = prepare_supervised(
            ext_frame,
            ext_features,
            ext_targets,
            lags,
            args.horizon,
            canonical_features=features,
            require_target=True,
            dynamic_features=args.dynamic_features,
        )

    if validation is not None:
        tr = np.arange(len(prepared.x), dtype=int)
        va = np.arange(len(validation.x), dtype=int)
        te = np.empty(0, dtype=int)
    elif external is None:
        tr, va, te = _split_indices(len(prepared.x), args.train_ratio, args.val_ratio, args.test_ratio, args.split, args.seed)
    else:
        tr, va, _ = _split_indices(len(prepared.x), args.train_ratio, args.val_ratio, 0.0, args.split, args.seed)
        te = np.empty(0, dtype=int)
    if len(va) < 5 or (external is None and len(te) < 5):
        raise ValueError("The split produced too few validation/test samples.")

    encoder = fit_encoder(prepared.x.iloc[tr], max_categories=args.max_categories)
    x_all = transform_encoder(prepared.x, encoder)
    y_all = np.asarray(prepared.y, dtype=np.float64)
    target_scaler = fit_target_scaler(y_all[tr])
    ys = target_transform(y_all, target_scaler)
    xtr = x_all[tr]
    ytr = ys[tr]
    if validation is not None:
        xva = transform_encoder(validation.x, encoder)
        yva = target_transform(np.asarray(validation.y, dtype=np.float64), target_scaler)
    else:
        xva = x_all[va]
        yva = ys[va]
    if external is not None:
        xte = transform_encoder(external.x, encoder)
        y_test_true = np.asarray(external.y, dtype=np.float64)
        test_rows, test_sources = external.row_index, external.source_file
    else:
        xte = x_all[te]
        y_test_true = y_all[te]
        test_rows, test_sources = prepared.row_index[te], prepared.source_file[te]

    rules = _select_rule_candidates(_parse_int_list(args.rules), len(xtr), xtr.shape[1])
    presets = _csv_list(args.presets)
    invalid = [p for p in presets if p not in SHAPE_PRESETS]
    if invalid:
        raise ValueError(f"Unknown AURA shape presets: {invalid}; choose from {sorted(SHAPE_PRESETS)}")
    seeds = _parse_int_list(args.seeds) or [args.seed]
    ridge_alphas = [float(v) for v in _csv_list(args.ridge_alphas)]
    if not ridge_alphas:
        ridge_alphas = [1.0e-5, 1.0e-3, 1.0e-1, 1.0]

    print(f"AURA Unified {VERSION}")
    print(f"Device: {dev} | train={len(tr)} validation={len(va)} test={len(xte)}")
    print(f"Encoded inputs={xtr.shape[1]} | outputs={ytr.shape[1]} | rule candidates={rules}")
    print("Training the unified AURA pipeline...")

    screened: list[tuple[float, dict[str, np.ndarray], dict[str, Any]]] = []
    for seed in seeds:
        for rule_count in rules:
            for preset in presets:
                state = make_initial_state(xtr, rule_count, seed, preset)
                for ridge_alpha in ridge_alphas:
                    state_trial = {k: v.copy() for k, v in state.items()}
                    state_trial["consequents"] = fit_consequents(
                        xtr, ytr, state_trial, ridge_alpha, args.ridge_max_rows, seed
                    )
                    pv = aura_predict_scaled_numpy(xva, state_trial)
                    score = float(np.sqrt(np.mean((pv - yva) ** 2)))
                    screened.append((score, state_trial, {"seed": seed, "rules": rule_count, "preset": preset, "ridge_alpha": ridge_alpha}))
    screened.sort(key=lambda item: item[0])

    tuned: list[tuple[float, dict[str, np.ndarray], dict[str, Any]]] = []
    for _, state, config in screened[: max(1, args.gradient_candidates)]:
        tuned_state, score = tune_aura(
            state,
            xtr,
            ytr,
            xva,
            yva,
            device=dev,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            patience=args.patience,
            ridge_alpha=float(config["ridge_alpha"]),
            ridge_every=args.ridge_every,
            ridge_max_rows=args.ridge_max_rows,
            seed=int(config["seed"]),
        )
        tuned.append((score, tuned_state, config))
    tuned.sort(key=lambda item: item[0])
    _, aura_state, selected = tuned[0]
    _sanity_check_state(aura_state)

    base_train, fire_train = aura_predict_scaled_numpy(xtr, aura_state, return_weights=True)
    base_val, fire_val = aura_predict_scaled_numpy(xva, aura_state, return_weights=True)
    ztr = _augmented_features(xtr, base_train, fire_train)
    zva = _augmented_features(xva, base_val, fire_val)
    residual_train = ytr - base_train
    ref_rows = _deterministic_row_sample(len(ztr), args.refiner_max_rows, int(selected["seed"]))
    ref_name, refiner, ref_alpha, _ = _fit_refiner_candidates(
        ztr[ref_rows],
        residual_train[ref_rows],
        zva,
        yva,
        base_val,
        seed=int(selected["seed"]),
        jobs=args.jobs,
        trees=args.refiner_trees,
        requested=_csv_list(args.refiner_candidates),
    )

    # Refit the selected residual expert on all non-test observations while
    # keeping AURA itself frozen at its validation-selected state.
    if refiner is not None:
        x_fit = np.concatenate([xtr, xva], axis=0)
        y_fit = np.concatenate([ytr, yva], axis=0)
        fit_rows = _deterministic_row_sample(len(x_fit), args.refiner_max_rows, int(selected["seed"]) + 17)
        x_fit = x_fit[fit_rows]
        y_fit = y_fit[fit_rows]
        base_fit, fire_fit = aura_predict_scaled_numpy(x_fit, aura_state, return_weights=True)
        z_fit = _augmented_features(x_fit, base_fit, fire_fit)
        target_fit = y_fit - base_fit
        refiner.fit(z_fit, target_fit.ravel() if y_fit.shape[1] == 1 else target_fit)

    bounds = None
    if args.target_bounds:
        values = [float(v) for v in _csv_list(args.target_bounds)]
        if len(values) != 2:
            raise ValueError("--target-bounds must be 'lower,upper'.")
        if values[0] > values[1]:
            raise ValueError("The lower --target-bounds value cannot exceed the upper value.")
        bounds = ([values[0]] * len(targets), [values[1]] * len(targets))

    artifact = {
        "format": FORMAT,
        "version": VERSION,
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "schema": {
            "raw_feature_columns": list(features),
            "encoded_feature_names": list(encoder["output_names"]),
            "target_columns": list(targets),
            "lags": list(lags),
            "horizon": int(args.horizon),
            "dynamic_features": bool(args.dynamic_features),
            "pid": {
                "source_column": args.pid_source,
                "setpoint": float(args.pid_setpoint),
                "dt": float(args.pid_dt),
                "integral_limit": pid_integral_limit,
            },
        },
        "encoder": encoder,
        "target_scaler": target_scaler,
        "aura_state": aura_state,
        "selected_training_configuration": {**selected, "device": str(dev), "epochs_requested": args.epochs},
        "evaluation_protocol": {
            "split_strategy": args.split,
            "train_ratio": float(args.train_ratio),
            "validation_ratio": float(args.val_ratio),
            "test_ratio": float(args.test_ratio),
            "explicit_validation_data": bool(args.validation_data),
            "independent_external_test": bool(args.external_data),
            "metrics_are_from_held_out_test": True,
        },
        "refiner_name": ref_name,
        "refiner": refiner,
        "refiner_alpha": ref_alpha,
        "target_bounds": bounds,
        "training_rows": int(len(tr)),
        "validation_rows": int(len(va)),
        "test_rows": int(len(xte)),
        "metrics": {},
    }
    model = AURAUnifiedModel(artifact)
    test_pred = model.predict_encoded(xte)
    report = metrics(y_test_true, test_pred, xtr.shape[1], targets)
    artifact["metrics"] = {"test": report}
    artifact["test_trace"] = {
        "row_index_sha256": hashlib.sha256(np.asarray(test_rows, dtype=np.int64).tobytes()).hexdigest(),
        "source_files": sorted(set(str(x) for x in test_sources)),
    }
    artifact["runtime_seconds"] = float(time.time() - started)

    plot_paths: list[Path] = []
    if not args.no_plots:
        plots_dir = (
            Path(args.plots_dir).expanduser().resolve()
            if args.plots_dir
            else out.with_name(f"{out.stem}_plots")
        )
        plot_paths = generate_diagnostic_plots(
            y_test_true,
            test_pred,
            targets,
            report,
            test_rows,
            test_sources,
            plots_dir,
            max_points=args.plot_max_points,
            dpi=args.plot_dpi,
        )
    artifact["diagnostic_plots"] = {
        "generated_from": "held-out test predictions" if plot_paths else None,
        "filenames": [path.name for path in plot_paths],
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with out.open("rb") as handle:
        reloaded = pickle.load(handle)
    check = reloaded.predict_encoded(xte[: min(256, len(xte))])
    # Parallel tree reductions can differ by a few float32 ULPs after
    # serialization even though the resulting engineering predictions and
    # metrics are unchanged.  Verify reproducibility at float32 precision.
    if not np.allclose(check, test_pred[: len(check)], rtol=1.0e-6, atol=1.0e-6, equal_nan=False):
        raise RuntimeError("Reloaded pickle predictions do not reproduce the in-memory model.")

    _print_metrics("test", report)
    print("\nMODEL EXPORT")
    print("-" * 66)
    print(f"Saved model            : {out}")
    print(f"Pickle size (MiB)      : {out.stat().st_size / (1024 ** 2):.4f}")
    print(f"AURA rules             : {aura_state['center'].shape[0]}")
    print(f"Trainable premise terms: {9 * aura_state['center'].size}")
    print(f"Internal refinement    : {ref_name}")
    print(f"Runtime (seconds)      : {artifact['runtime_seconds']:.3f}")
    print("Reload verification    : PASSED")
    if plot_paths:
        print(f"Diagnostic plots       : {plot_paths[0].parent} ({len(plot_paths)} PNG files)")
        for path in plot_paths:
            print(f"  - {path.name}")
    else:
        print("Diagnostic plots       : disabled by --no-plots")
    return 0


def predict_command(args: argparse.Namespace) -> int:
    path = Path(args.model).expanduser().resolve()
    with path.open("rb") as handle:
        model = pickle.load(handle)
    if not isinstance(model, AURAUnifiedModel):
        raise TypeError("The pickle is not an AURAUnifiedModel artifact.")
    frame = load_data(args.data, sheet=args.sheet, include_regex=args.data_regex)
    override = _csv_list(args.features) if args.features else None
    pred, rows, sources = model.predict_frame(frame, feature_override=override)
    names = model.artifact["schema"]["target_columns"]
    output = pd.DataFrame({f"prediction_{name}": pred[:, i] for i, name in enumerate(names)})
    output.insert(0, "source_row", rows)
    output.insert(0, "source_file", sources)
    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out, index=False)
    print(f"Predicted {len(output)} rows -> {out}")
    return 0


def inspect_command(args: argparse.Namespace) -> int:
    with Path(args.model).expanduser().resolve().open("rb") as handle:
        model = pickle.load(handle)
    if not isinstance(model, AURAUnifiedModel):
        raise TypeError("The pickle is not an AURAUnifiedModel artifact.")
    a = model.artifact
    summary = {
        "format": a["format"],
        "version": a["version"],
        "created_utc": a["created_utc"],
        "environment": a.get("environment", "not recorded by this artifact version"),
        "schema": a["schema"],
        "rules": int(a["aura_state"]["center"].shape[0]),
        "refinement": a["refiner_name"],
        "configuration": a["selected_training_configuration"],
        "evaluation_protocol": a.get("evaluation_protocol", "not recorded by this artifact version"),
        "metrics": a["metrics"],
        "diagnostic_plots": a.get("diagnostic_plots", "not recorded by this artifact version"),
    }
    print(json.dumps(summary, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    help_text = """
Typical workflow:
  1. Prepare clean rows and choose input/target columns.
  2. Train and evaluate: python AURA.py train --data data.csv --target y --features x1,x2
  3. Review terminal metrics and all six PNG plots.
  4. Predict new rows: python AURA.py predict --model aura_trained_model.pkl --data new.csv

For time series, use chronological/external test data, --horizon >= 1, and
appropriate --lags. R2 > 0.99 was measured in the four supplied AURA case
studies, but every new domain must be verified on its own untouched test data.
"""
    parser = argparse.ArgumentParser(
        description="Train, evaluate, plot, export, and use the unified AURA regression model.",
        epilog=help_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"AURA Unified {VERSION}")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser(
        "train",
        help="Train AURA, report held-out metrics, create plots, and export one .pkl model.",
        description=(
            "Fit AURA using training rows, select settings on validation rows, "
            "evaluate once on held-out test rows, generate six plots, and export "
            "the complete fitted pipeline."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train.add_argument("--data", nargs="+", required=True, help="Training data file(s) or directory/directories.")
    train.add_argument("--data-regex", default=None, help="Optional regex filter applied to loaded data file paths.")
    train.add_argument("--target", required=True, help="Comma-separated target column(s).")
    train.add_argument("--features", default="auto", help="Comma-separated feature columns, or 'auto'.")
    train.add_argument("--sheet", default=None, help="Excel sheet name.")
    train.add_argument("--lags", default="0", help="Lag list/range, e.g. 1:20 or 1,2,5.")
    train.add_argument("--horizon", type=int, default=0, help="Forecast horizon; use >=1 for autoregression.")
    train.add_argument("--dynamic-features", action="store_true", help="Add per-sequence first/second differences and 3-sample rolling means for each feature.")
    train.add_argument("--validation-data", nargs="+", default=None, help="Optional explicit validation file(s) or directory.")
    train.add_argument("--validation-data-regex", default=None, help="Optional regex filter for validation data file paths.")
    train.add_argument("--validation-target", default=None, help="Validation target mapping.")
    train.add_argument("--validation-features", default=None, help="Validation feature mapping, positionally matched.")
    train.add_argument("--validation-sheet", default=None)
    train.add_argument("--external-data", nargs="+", default=None, help="Independent test file(s) or directory.")
    train.add_argument("--external-data-regex", default=None, help="Optional regex filter for external test data file paths.")
    train.add_argument("--external-target", default=None, help="External target mapping.")
    train.add_argument("--external-features", default=None, help="External feature mapping, positionally matched.")
    train.add_argument("--external-sheet", default=None)
    train.add_argument("--pid-source", default=None, help="Optional measured variable column used to derive pid_error features.")
    train.add_argument("--pid-setpoint", type=float, default=0.0)
    train.add_argument("--pid-dt", type=float, default=1.0)
    train.add_argument("--pid-integral-limit", type=float, default=0.0, help="0 disables clipping; positive value clips the integral.")
    train.add_argument("--model-out", default="aura_trained_model.pkl", help="Output path for the complete fitted model.")
    train.add_argument("--split", choices=["chronological", "random"], default="chronological", help="How rows are divided when explicit validation/test files are not supplied.")
    train.add_argument("--train-ratio", type=float, default=0.70, help="Training fraction for an automatic split.")
    train.add_argument("--val-ratio", type=float, default=0.15, help="Validation fraction used only for model/configuration selection.")
    train.add_argument("--test-ratio", type=float, default=0.15, help="Untouched test fraction used only for final metrics and plots.")
    train.add_argument("--rules", default="4,8,16,32", help="Comma-separated AURA rule counts to validate.")
    train.add_argument("--presets", default="default,lowtail,phtail,wide", help="Comma-separated feasible AURA shape starting points to validate.")
    train.add_argument("--seeds", default="42,7,21", help="Comma-separated positive initialization seeds to validate.")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--ridge-alphas", default="0.00001,0.001,0.1,1.0")
    train.add_argument("--ridge-max-rows", type=int, default=120000)
    train.add_argument("--gradient-candidates", type=int, default=2)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=1024)
    train.add_argument("--learning-rate", type=float, default=0.002)
    train.add_argument("--patience", type=int, default=18)
    train.add_argument("--ridge-every", type=int, default=10)
    train.add_argument("--refiner-trees", type=int, default=400)
    train.add_argument(
        "--refiner-candidates",
        default="extra_trees_leaf1,extra_trees_leaf2,extra_trees_leaf4,random_forest,hist_gradient",
        help="Comma-separated residual experts to validate; useful names are extra_trees_leaf1, extra_trees_leaf2, extra_trees_leaf4, random_forest, hist_gradient.",
    )
    train.add_argument("--refiner-max-rows", type=int, default=120000, help="Maximum rows used to fit residual refinement; 0 uses all rows.")
    train.add_argument("--jobs", type=int, default=-1)
    train.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    train.add_argument("--max-categories", type=int, default=32)
    train.add_argument("--target-bounds", default=None, help="Optional lower,upper clipping bounds.")
    train.add_argument("--plots-dir", default=None, help="PNG output directory; by default <model_name>_plots beside the pickle.")
    train.add_argument("--plot-max-points", type=int, default=5000, help="Maximum displayed points per plot; metrics always use the full test set.")
    train.add_argument("--plot-dpi", type=int, default=160, help="Resolution of generated PNG plots.")
    train.add_argument("--no-plots", action="store_true", help="Disable PNG generation (mainly for automated/headless runs).")
    train.set_defaults(func=train_command)

    predict = sub.add_parser("predict", help="Predict with an exported AURA pickle.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    predict.add_argument("--model", required=True)
    predict.add_argument("--data", nargs="+", required=True)
    predict.add_argument("--data-regex", default=None)
    predict.add_argument("--features", default=None, help="Optional positional feature-column mapping.")
    predict.add_argument("--sheet", default=None)
    predict.add_argument("--output", default="aura_predictions.csv")
    predict.set_defaults(func=predict_command)

    inspect = sub.add_parser("inspect", help="Print model schema, configuration, and stored metrics.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    inspect.add_argument("--model", required=True)
    inspect.set_defaults(func=inspect_command)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        if os.environ.get("AURA_DEBUG", "0") == "1":
            raise
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
