"""
ML prediction layer for OlympiFlow.

Model 1 — Surge Predictor: Random Forest Regressor
  Predicts venue congestion intensity (0–1) from time/event features.

Model 2 — Condition Classifier: Gradient Boosting Classifier
  Classifies traffic into NORMAL / ELEVATED / PEAK / CRITICAL.

Both models are trained once on startup using synthetic BPR-derived data.
"""
from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_SURGE_MODEL_PATH = _HERE / "surge_model.pkl"
_COND_MODEL_PATH = _HERE / "condition_model.pkl"

# ---------------------------------------------------------------------------
# BPR helper (mirrors simulation_engine.py)
# ---------------------------------------------------------------------------

def _bpr(vc: float, alpha: float = 0.15, beta: float = 4) -> float:
    return 1.0 + alpha * (vc ** beta)


def _time_mult(hour: float) -> float:
    if hour < 6:
        return 0.2
    if hour < 9:
        return 0.5 + (hour - 6) / 3 * 0.5
    if hour < 15:
        return 0.6
    if hour < 19:
        return 0.7 + (hour - 15) / 4 * 0.3
    if hour < 22:
        return 0.55
    return 0.25


# ---------------------------------------------------------------------------
# Synthetic training data generators
# ---------------------------------------------------------------------------

def _generate_surge_data(n: int = 10_000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Features: [hour, day_of_week, event_type, capacity_util, hist_avg_congestion]
    Target:   surge intensity in [0, 1]
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    X, y = [], []
    for _ in range(n):
        hour = rng.uniform(0, 24)
        dow = rng.randint(0, 6)
        event_type = rng.choices([0, 1, 2], weights=[0.5, 0.4, 0.1])[0]
        cap_util = rng.uniform(0.0, 1.0)
        hist_avg = rng.uniform(0.1, 0.8)

        tm = _time_mult(hour)
        event_surge = {0: 0.0, 1: 0.5, 2: 1.0}[event_type] * cap_util
        vc = min(1.5, 0.4 * tm + event_surge * 0.45 + hist_avg * 0.2)

        bpr_ratio = _bpr(vc)
        intensity = min(1.0, (bpr_ratio - 1.0) / (1.0 + 0.15 * 1.5**4 - 1.0))
        intensity = max(0.0, intensity + np.random.normal(0, 0.03))

        X.append([hour, dow, event_type, cap_util, hist_avg])
        y.append(round(intensity, 4))

    return np.array(X, dtype=float), np.array(y, dtype=float)


def _generate_condition_data(n: int = 10_000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Features: [surge_intensity, vc_ratio, time_to_event_min, concurrent_events, transit_score]
    Target:   0=NORMAL, 1=ELEVATED, 2=PEAK, 3=CRITICAL
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    LABELS = ["NORMAL", "ELEVATED", "PEAK", "CRITICAL"]
    X, y = [], []

    for _ in range(n):
        surge = rng.uniform(0.0, 1.0)
        vc = rng.uniform(0.2, 1.5)
        tte = rng.uniform(-30, 180)   # minutes to event start (neg = already started)
        concurrent = rng.randint(0, 8)
        transit = rng.uniform(0.0, 1.0)

        score = (
            surge * 0.35
            + min(1.0, vc / 1.5) * 0.30
            + (1.0 if tte < 0 else max(0.0, 1.0 - tte / 120)) * 0.15
            + min(1.0, concurrent / 8) * 0.12
            + (1.0 - transit) * 0.08
        )
        score = max(0.0, min(1.0, score + np.random.normal(0, 0.04)))

        if score < 0.30:
            label = 0
        elif score < 0.55:
            label = 1
        elif score < 0.75:
            label = 2
        else:
            label = 3

        X.append([surge, vc, tte, concurrent, transit])
        y.append(label)

    return np.array(X, dtype=float), np.array(y, dtype=int)


# ---------------------------------------------------------------------------
# Train or load models
# ---------------------------------------------------------------------------

def _train_surge_model():
    from sklearn.ensemble import RandomForestRegressor
    X, y = _generate_surge_data()
    model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X, y)
    joblib.dump(model, _SURGE_MODEL_PATH)
    return model


def _train_condition_model():
    from sklearn.ensemble import GradientBoostingClassifier
    X, y = _generate_condition_data()
    model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)
    joblib.dump(model, _COND_MODEL_PATH)
    return model


def _load_or_train(path: Path, trainer):
    if path.exists():
        try:
            return joblib.load(path)
        except Exception:
            pass
    return trainer()


# Module-level singletons — loaded once on first import
_surge_model = None
_cond_model = None

CONDITION_LABELS = ["NORMAL", "ELEVATED", "PEAK", "CRITICAL"]


def _get_models():
    global _surge_model, _cond_model
    if _surge_model is None:
        _surge_model = _load_or_train(_SURGE_MODEL_PATH, _train_surge_model)
    if _cond_model is None:
        _cond_model = _load_or_train(_COND_MODEL_PATH, _train_condition_model)
    return _surge_model, _cond_model


# ---------------------------------------------------------------------------
# Public prediction functions
# ---------------------------------------------------------------------------

def predict_surge(
    venue_id: str,
    hour: float,
    day_of_week: int,
    event_type: int,
    capacity_util: float,
) -> dict[str, Any]:
    """
    Returns predicted surge intensity [0–1] plus a confidence score.
    Confidence is derived from the forest's per-tree variance.
    """
    model, _ = _get_models()

    # Historical average congestion: heuristic per venue
    venue_hist: dict[str, float] = {
        "sofi": 0.72, "intuit-dome": 0.55, "crypto-arena": 0.60,
        "la-coliseum": 0.68, "rose-bowl": 0.65, "pauley": 0.40,
        "bmo-stadium": 0.45, "long-beach-arena": 0.38,
        "sepulveda-basin": 0.30, "el-dorado": 0.25,
        "dignity-health": 0.50, "ucla-olympic": 0.42,
    }
    hist_avg = venue_hist.get(venue_id, 0.45)

    features = np.array([[hour, day_of_week, event_type, capacity_util, hist_avg]])

    # Collect per-tree predictions to compute variance-based confidence
    tree_preds = np.array([t.predict(features)[0] for t in model.estimators_])
    intensity = float(np.clip(tree_preds.mean(), 0.0, 1.0))
    std = float(tree_preds.std())
    # Map std → confidence: low variance = high confidence
    confidence = float(np.clip(1.0 - std * 4.0, 0.05, 0.99))

    return {
        "intensity": round(intensity, 4),
        "confidence": round(confidence, 4),
        "venue_id": venue_id,
        "hour": hour,
    }


def classify_conditions(
    surge_intensity: float,
    vc_ratio: float,
    time_to_event_min: float,
    concurrent_events: int,
    transit_availability_score: float,
) -> dict[str, Any]:
    """
    Returns condition class label + probability dict for all four classes.
    """
    _, model = _get_models()

    features = np.array([[
        surge_intensity, vc_ratio, time_to_event_min,
        concurrent_events, transit_availability_score,
    ]])

    pred = int(model.predict(features)[0])
    proba = model.predict_proba(features)[0]

    return {
        "condition": CONDITION_LABELS[pred],
        "condition_index": pred,
        "probabilities": {
            label: round(float(p), 4)
            for label, p in zip(CONDITION_LABELS, proba)
        },
    }
