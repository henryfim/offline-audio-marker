"""Apply the locally trained genre classifier alongside the existing audio evidence."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from style_features import FEATURE_VERSION, extract_style_features

MODEL_PATH = Path(__file__).resolve().parent / "models" / "local_style.json"


def should_try_local_style(previous: dict, _labels: list[str]) -> bool:
    evidence = previous.get("style_evidence", {})
    return (
        evidence.get("music_windows", 0) >= 2 and evidence.get("music_score", 0) >= 0.45
        and previous.get("style") == "Undetermined"
    )


@lru_cache(maxsize=4)
def _load(path: Path, modified: int, size: int) -> tuple | None:
    if size > 2_000_000:
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        labels = data["labels"]
        mean = np.asarray(data["scaler_mean"], dtype=float)
        scale = np.asarray(data["scaler_scale"], dtype=float)
        coef = np.asarray(data["coef"], dtype=float)
        intercept = np.asarray(data["intercept"], dtype=float)
        threshold = float(data.get("fallback_threshold", data["threshold"]))
        temperature = float(data.get("temperature", 1.0))
        valid = (
            data["feature_version"] == FEATURE_VERSION
            and isinstance(labels, list) and len(labels) >= 2
            and all(isinstance(label, str) and label for label in labels)
            and len(set(labels)) == len(labels)
            and mean.ndim == 1 and scale.shape == mean.shape
            and coef.shape == (len(labels), len(mean)) and intercept.shape == (len(labels),)
            and all(np.isfinite(value).all() for value in (mean, scale, coef, intercept))
            and np.all(scale > 0) and 0 <= threshold <= 1 and 0 < temperature < 100
        )
        if valid:
            return labels, mean, scale, coef, intercept, threshold, temperature
    except (OSError, KeyError, TypeError, ValueError, OverflowError):
        pass
    return None


def combine_local_style(audio: np.ndarray, sample_rate: int, previous: dict, *, model_path: Path = MODEL_PATH) -> dict:
    try:
        info = model_path.stat()
    except OSError:
        return previous
    model = _load(model_path, info.st_mtime_ns, info.st_size)
    old_evidence = previous.get("style_evidence", {})
    # The local model only fills gaps in the existing prediction.
    if model is None:
        return previous
    labels, mean, scale, coef, intercept, threshold, temperature = model
    if not should_try_local_style(previous, labels):
        return {**previous, "style_source": "yamnet"}
    features = extract_style_features(audio, sample_rate)
    if features is None or features.shape != mean.shape:
        return previous
    logits = (coef @ ((features - mean) / scale) + intercept) / temperature
    probability = np.exp(logits - logits.max())
    probability /= probability.sum()
    order = np.argsort(probability)[::-1]
    best = order[0]
    confidence = float(probability[best])
    usable = confidence >= threshold
    use_local = usable
    evidence = {
        **old_evidence,
        "local_classifier": {
            "feature_version": FEATURE_VERSION,
            "accepted": bool(use_local),
            "threshold": threshold,
            "candidates": [{"style": labels[index], "probability": round(float(probability[index]), 4)}
                           for index in order[:3]],
        },
    }
    result = {**previous, "style_evidence": evidence, "style_source": "yamnet"}
    if use_local:
        result.update({
            "style": labels[best],
            "style_confidence": "Medium" if confidence >= max(0.72, threshold) else "Low",
            "style_note": "Estimated by the locally trained music genre classifier.",
            "style_source": "local_classifier",
        })
    return result
