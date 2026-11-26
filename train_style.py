"""Train and evaluate the bundled offline music genre classifier."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy
import sklearn
import soundfile
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from audio_analysis import _load_audio
from style_features import FEATURE_VERSION, extract_style_features


PROJECT_DIR = Path(__file__).resolve().parent
APP_SAMPLE_RATE = 22050
SPLITS = ("train", "validation", "test")
REGULARIZATION_VALUES = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
REJECTED = "Undetermined"
BASELINE_ALIASES = {
    "Indie / Pop": "Pop",
    "Dubstep": "Electronic",
    "House": "Electronic",
    "Techno": "Electronic",
    "Drum and Bass": "Electronic",
    "Trance": "Electronic",
    "Piano Instrumental": "Instrumental",
    "Instrumental Music": "Instrumental",
    "Country / Bluegrass": "Country",
    "Funk / Disco": "Disco",
    "Jazz / Swing": "Jazz",
    "Reggae / Ska": "Reggae",
}


@dataclass(frozen=True)
class Track:
    path: Path
    label: str
    split: str
    track_id: str


def read_manifest(path: Path) -> list[Track]:
    tracks: list[Track] = []
    path_rows: set[Path] = set()
    track_ids: set[str] = set()
    group_splits: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fields = set(reader.fieldnames or ())
        path_column = next((column for column in ("path", "audio_path") if column in fields), None)
        label_column = next((column for column in ("label", "genre") if column in fields), None)
        if path_column is None or label_column is None or "split" not in fields:
            raise ValueError("Manifest must contain path (or audio_path), label (or genre), and split columns.")
        for row in reader:
            relative_path, label, split = ((row[name] or "").strip() for name in (path_column, label_column, "split"))
            if split == "training":
                split = "train"
            if not relative_path or not label or label == REJECTED or split not in SPLITS:
                raise ValueError(f"Invalid manifest row {reader.line_num}.")
            audio_path = (path.parent / relative_path).resolve()
            if audio_path in path_rows:
                raise ValueError(f"Duplicate audio path in manifest row {reader.line_num}.")
            path_rows.add(audio_path)
            track_id = (row.get("track_id") or "").strip()
            if track_id:
                if track_id in track_ids:
                    raise ValueError(f"Duplicate track ID {track_id} in the manifest.")
                track_ids.add(track_id)
            artist_id = (row.get("artist_id") or "").strip()
            if artist_id and artist_id not in {"-1", "0"}:
                previous_split = group_splits.setdefault(artist_id, split)
                if previous_split != split:
                    raise ValueError(f"Artist {artist_id} appears in more than one split.")
            tracks.append(Track(audio_path, label, split, track_id or relative_path))
    if not tracks:
        raise ValueError("The manifest is empty.")
    return tracks


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    # Match the prepared waveform passed to the classifier by analyze_audio.
    return _load_audio(path, APP_SAMPLE_RATE), APP_SAMPLE_RATE


def extract_file(path: Path) -> tuple[np.ndarray | None, str | None]:
    try:
        audio, rate = read_audio(path)
        features = extract_style_features(audio, rate)
        if features is None or features.ndim != 1 or not np.isfinite(features).all():
            return None, "No usable acoustic features."
        return np.asarray(features, dtype="<f4"), None
    except (OSError, ValueError, RuntimeError) as error:
        return None, str(error)


def extract_dataset(
    tracks: list[Track],
    cache_path: Path,
    workers: int,
    allow_missing: bool,
) -> tuple[list[Track], np.ndarray, list[dict]]:
    ready: dict[Path, np.ndarray] = {}
    skipped: list[dict] = []
    pending: list[tuple[Track, os.stat_result]] = []
    missing: list[Track] = []
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(cache_path) as cache:
        cache.execute(
            "CREATE TABLE IF NOT EXISTS style_features ("
            "path TEXT PRIMARY KEY, byte_size INTEGER, modified_ns INTEGER, "
            "feature_version INTEGER, features BLOB, failure TEXT)"
        )
        for track in tracks:
            try:
                stamp = track.path.stat()
            except FileNotFoundError:
                missing.append(track)
                continue
            cached = cache.execute(
                "SELECT features, failure FROM style_features "
                "WHERE path = ? AND byte_size = ? AND modified_ns = ? AND feature_version = ?",
                (str(track.path), stamp.st_size, stamp.st_mtime_ns, FEATURE_VERSION),
            ).fetchone()
            if cached is None:
                pending.append((track, stamp))
            elif cached[1] is None:
                ready[track.path] = np.frombuffer(cached[0], dtype="<f4")
            else:
                skipped.append({"track_id": track.track_id, "split": track.split, "error": cached[1]})
        if missing and not allow_missing:
            raise ValueError(f"{len(missing)} manifest audio files are missing. First missing file is {missing[0].path}.")
        print(f"{len(ready)} cached, {len(skipped)} cached decode failures, {len(pending)} to extract, {len(missing)} missing", flush=True)
        if pending:
            with ProcessPoolExecutor(max_workers=workers, initializer=threadpool_limits, initargs=(1,)) as pool:
                done = 0
                for offset in range(0, len(pending), 128):
                    batch = pending[offset:offset + 128]
                    results = pool.map(extract_file, (track.path for track, _ in batch), chunksize=4)
                    cache_rows = []
                    for (track, stamp), (features, error) in zip(batch, results, strict=True):
                        cache_rows.append((
                            str(track.path), stamp.st_size, stamp.st_mtime_ns, FEATURE_VERSION,
                            features.tobytes() if features is not None else None, error,
                        ))
                        if features is None:
                            skipped.append({"track_id": track.track_id, "split": track.split, "error": error})
                        else:
                            ready[track.path] = features
                    cache.executemany("INSERT OR REPLACE INTO style_features VALUES (?, ?, ?, ?, ?, ?)", cache_rows)
                    cache.commit()
                    done += len(batch)
                    print(f"Extracted {done}/{len(pending)}. Usable tracks: {len(ready)}", flush=True)
    available = [track for track in tracks if track.path in ready]
    features = np.stack([ready[track.path] for track in available]) if available else np.empty((0, 0))
    return available, features, skipped


def classification_metrics(truth: np.ndarray, predictions: np.ndarray, labels: list[str]) -> dict:
    accepted = predictions != REJECTED
    correct = predictions == truth
    columns = [*labels, REJECTED, "Other"]
    counts = {}
    by_class = {}
    for label in labels:
        actual = truth == label
        predicted = predictions == label
        hits = int(np.count_nonzero(actual & predicted))
        false_positive = int(np.count_nonzero(~actual & predicted))
        false_negative = int(np.count_nonzero(actual & ~predicted))
        precision_denominator = hits + false_positive
        recall_denominator = hits + false_negative
        f1_denominator = 2 * hits + false_positive + false_negative
        by_class[label] = {
            "precision": hits / precision_denominator if precision_denominator else 0.0,
            "recall": hits / recall_denominator if recall_denominator else 0.0,
            "f1": 2 * hits / f1_denominator if f1_denominator else 0.0,
            "support": recall_denominator,
        }
        counts[label] = [int(np.count_nonzero(actual & (predictions == value))) for value in columns[:-1]]
        counts[label].append(recall_denominator - sum(counts[label]))
    return {
        "tracks": len(truth),
        "coverage": float(accepted.mean()),
        "accuracy": float(correct.mean()),
        "accepted_accuracy": float(correct[accepted].mean()) if accepted.any() else None,
        "macro_f1": float(np.mean([values["f1"] for values in by_class.values()])),
        "per_class": by_class,
        "confusion_columns": columns,
        "confusion_rows": counts,
    }


def fit_model(
    features: np.ndarray,
    tracks: list[Track],
    target_coverage: float,
) -> tuple[dict, dict[str, np.ndarray], dict[str, np.ndarray]]:
    split = np.array([track.split for track in tracks])
    outcomes = np.array([track.label for track in tracks])
    labels = sorted(set(outcomes[split == "train"]))
    if len(labels) < 2 or any(set(outcomes[split == name]) != set(labels) for name in SPLITS):
        raise ValueError("Each split must have examples of the same two or more genres.")
    scaler = StandardScaler().fit(features[split == "train"])
    scaled = {name: scaler.transform(features[split == name]) for name in SPLITS}
    truth = {name: outcomes[split == name] for name in SPLITS}
    trials = []
    best = None
    best_key = (-1.0, float("-inf"))
    with threadpool_limits(limits=2):
        for regularization in REGULARIZATION_VALUES:
            estimator = LogisticRegression(C=regularization, max_iter=1000)
            estimator.fit(scaled["train"], truth["train"])
            probabilities = estimator.predict_proba(scaled["validation"])
            predictions = np.asarray(labels)[probabilities.argmax(axis=1)]
            macro_f1 = classification_metrics(truth["validation"], predictions, labels)["macro_f1"]
            loss = log_loss(truth["validation"], probabilities, labels=labels)
            trials.append({"C": regularization, "macro_f1": macro_f1, "log_loss": loss})
            key = (macro_f1, -loss)
            if key > best_key:
                best_key = key
                best = estimator
    assert best is not None
    coefficients = best.coef_
    intercept = best.intercept_
    if len(labels) == 2:
        # sklearn stores a single log-odds row for a binary classifier.
        coefficients = np.stack((-coefficients[0], coefficients[0])) / 2
        intercept = np.array([-intercept[0], intercept[0]]) / 2

    validation_logits = scaled["validation"] @ coefficients.T + intercept

    def calibration_loss(temperature: float) -> float:
        return log_loss(truth["validation"], softmax(validation_logits / temperature, axis=1), labels=labels)

    calibration = minimize_scalar(calibration_loss, bounds=(0.25, 5.0), method="bounded")
    temperature = float(calibration.x) if calibration.success else 1.0
    probabilities = {name: softmax((scaled[name] @ coefficients.T + intercept) / temperature, axis=1) for name in SPLITS}
    validation_probs = probabilities["validation"]
    confidence = validation_probs.max(axis=1)
    predicted = np.asarray(labels)[validation_probs.argmax(axis=1)]
    threshold = 0.0
    best_threshold_key = (-1.0, 0.0)
    for candidate in np.arange(0, 1.0, 0.01):
        accepted = confidence >= candidate
        coverage = float(accepted.mean())
        if coverage < target_coverage:
            break
        key = (float((predicted[accepted] == truth["validation"][accepted]).mean()), coverage)
        if key > best_threshold_key:
            threshold = float(candidate)
            best_threshold_key = key

    model = {
        "feature_version": FEATURE_VERSION,
        "labels": labels,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coef": coefficients.tolist(),
        "intercept": intercept.tolist(),
        "temperature": temperature,
        "threshold": round(threshold, 4),
        "training": {
            "estimator": "Multinomial logistic regression",
            "audio_input_sample_rate_hz": APP_SAMPLE_RATE,
            "regularization_C": float(best.C),
            "validation_target_coverage": target_coverage,
            "calibration": "Temperature scaling on validation",
            "validation_regularization_search": trials,
            "libraries": {"scikit-learn": sklearn.__version__, "numpy": np.__version__, "scipy": scipy.__version__, "soundfile": soundfile.__version__},
        },
    }
    return model, truth, probabilities


def classify(probabilities: np.ndarray, labels: list[str], threshold: float) -> np.ndarray:
    return np.where(probabilities.max(axis=1) >= threshold, np.asarray(labels)[probabilities.argmax(axis=1)], REJECTED)


def compare_baseline(
    tracks: list[Track],
    probabilities: np.ndarray,
    model: dict,
    per_genre: int,
    model_path: Path,
) -> dict:
    from instrument_analysis import _load_model, _model_input, _predict
    from local_style import should_try_local_style
    from style_analysis import identify_style, undetermined_style

    # Evaluate the existing YAMNet rules directly. The app may already route
    # successful predictions through the newly trained classifier.
    yamnet = _load_model(model_path)
    if yamnet is None:
        raise ValueError(f"Cannot load the YAMNet baseline from {model_path}.")
    labels = model["labels"]
    truth = np.array([track.label for track in tracks])
    generator = np.random.default_rng(20261125)
    selected = np.sort(np.concatenate([
        generator.choice(indices, min(len(indices), per_genre), replace=False)
        for genre in labels
        if (indices := np.flatnonzero(truth == genre)).size
    ]))
    selected_probabilities = probabilities[selected]
    local_predictions = classify(selected_probabilities, labels, model["threshold"])
    fallback_threshold = model.get("fallback_threshold", model["threshold"])
    fallback_predictions = classify(selected_probabilities, labels, fallback_threshold)
    baseline = []
    hybrid = []
    local_used = 0
    for count, index in enumerate(selected, start=1):
        audio, sample_rate = read_audio(tracks[index].path)
        mono, _ = _model_input(audio, sample_rate)
        floor = max(8e-5, float(np.max(np.abs(mono), initial=0.0)) * 0.0015)
        scores = _predict(yamnet, mono, floor)
        previous = identify_style(scores, yamnet[3]) if scores is not None else undetermined_style("No usable evidence.")
        estimate = previous["style"]
        baseline.append(estimate)
        local = fallback_predictions[count - 1]
        use_local = local != REJECTED and should_try_local_style(previous, labels)
        hybrid.append(local if use_local else estimate)
        local_used += int(use_local)
        if count % 16 == 0 or count == len(selected):
            print(f"Evaluated YAMNet baseline {count}/{len(selected)}", flush=True)
    aliases = {label: target for label, target in BASELINE_ALIASES.items() if target in labels}
    mapped = np.array([aliases.get(label, label) for label in baseline])
    mapped_hybrid = np.array([aliases.get(label, label) for label in hybrid])
    return {
        "method": "Seeded stratified subset of test clips. YAMNet scores whole clips. Labels outside the dataset count as Other and are incorrect for accuracy and F1.",
        "seed": 20261125,
        "track_ids": [tracks[index].track_id for index in selected],
        "baseline_label_mapping": aliases,
        "baseline_original_predictions": dict(Counter(baseline)),
        "baseline": classification_metrics(truth[selected], mapped, labels),
        "trained_model": classification_metrics(truth[selected], local_predictions, labels),
        "hybrid_model": classification_metrics(truth[selected], mapped_hybrid, labels),
        "hybrid_local_predictions": local_used,
        "hybrid_fallback_threshold": fallback_threshold,
        "trained_model_without_rejection": classification_metrics(
            truth[selected], classify(selected_probabilities, labels, 0.0), labels,
        ),
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="CSV containing path, label, split and optional group IDs")
    parser.add_argument("--dataset", default="FMA Small")
    parser.add_argument("--source-url", default="https://github.com/mdeff/fma")
    parser.add_argument("--feature-cache", type=Path, help="SQLite file for resuming acoustic feature extraction")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "models" / "local_style.json")
    parser.add_argument("--report", type=Path, help="Output metrics JSON")
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 8))
    parser.add_argument("--extract-only", action="store_true", help="Cache available audio without fitting a model")
    parser.add_argument("--validation-target-coverage", type=float, default=0.85)
    parser.add_argument("--fallback-threshold", type=float, help="Override the confidence floor for app fallback in hybrid evaluation")
    parser.add_argument("--baseline-per-genre", type=int, default=0, help="Evaluate old YAMNet rules on this many test clips per genre")
    parser.add_argument("--yamnet", type=Path, default=PROJECT_DIR / "models" / "yamnet.tflite")
    args = parser.parse_args()
    if args.workers < 1 or args.baseline_per_genre < 0 or not 0 < args.validation_target_coverage <= 1:
        parser.error("Use a positive worker count, a nonnegative baseline size and coverage in (0, 1].")
    if args.fallback_threshold is not None and not 0 <= args.fallback_threshold <= 1:
        parser.error("Fallback threshold must be between 0 and 1.")
    report_path = args.report or args.output.with_name("style_training_report.json")
    if args.output.resolve() == report_path.resolve():
        parser.error("The model and report must use different output paths.")
    tracks = read_manifest(args.manifest)
    cache = args.feature_cache or args.manifest.with_name("style_features_22050.sqlite")
    available, features, failures = extract_dataset(tracks, cache, args.workers, args.extract_only)
    if args.extract_only:
        return
    if len(failures) > max(1, len(tracks) * 0.02):
        raise ValueError(f"{len(failures)} tracks could not be decoded. Check the audio files and rerun.")
    model, truth, probabilities = fit_model(features, available, args.validation_target_coverage)
    labels = model["labels"]
    counts = {name: len(truth[name]) for name in SPLITS}
    dataset = {
        "name": args.dataset,
        "source_url": args.source_url,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "manifest_tracks": len(tracks),
        "used_tracks": counts,
        "decode_failures": len(failures),
    }
    model["training"]["dataset"] = dataset
    report = {
        "dataset": dataset,
        "training": model["training"],
        "feature_version": FEATURE_VERSION,
        "feature_count": features.shape[1],
        "labels": labels,
        "threshold": model["threshold"],
        "temperature": model["temperature"],
        "decode_failures": failures,
    }
    if args.fallback_threshold is not None:
        model["fallback_threshold"] = args.fallback_threshold
        report["fallback_threshold"] = args.fallback_threshold
    for name in ("validation", "test"):
        report[name] = classification_metrics(truth[name], classify(probabilities[name], labels, model["threshold"]), labels)
        report[f"{name}_without_rejection"] = classification_metrics(truth[name], classify(probabilities[name], labels, 0.0), labels)
    if args.baseline_per_genre:
        test_tracks = [track for track in available if track.split == "test"]
        report["baseline_comparison"] = compare_baseline(test_tracks, probabilities["test"], model, args.baseline_per_genre, args.yamnet)
    write_json(args.output, model)
    write_json(report_path, report)
    test = report["test"]
    print(f"Test macro-F1 {test['macro_f1']:.3f}, accuracy {test['accuracy']:.3f}, coverage {test['coverage']:.3f}", flush=True)
    print(f"Model: {args.output}\nReport: {report_path}", flush=True)


if __name__ == "__main__":
    main()
