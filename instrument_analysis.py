"""Per-section timbre estimates from the bundled YAMNet classifier and DSP."""

from __future__ import annotations

from collections.abc import Iterable
from math import gcd, isfinite
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np
from scipy.signal import resample_poly

from style_analysis import identify_style, undetermined_style


MODEL_RATE = 16_000
WINDOW = 15_600
HOP = 8_000
MIN_SECTION = MODEL_RATE // 2
MAX_WINDOWS = 256
UNDETECTED = "—"

# The first applicable entry in each family is used. A specific class must
# retain enough of the strongest score in its family to replace the parent.
FAMILIES = {
    "voice": (
        ("Choir", "Choir", 0.37),
        ("Rapping", "Rap Vocals", 0.48),
        ("Synthetic singing", "Synthetic Vocals", 0.66),
        ("Singing", "Singing Vocals", 0.0),
    ),
    "guitar": (
        ("Electric guitar", "Electric Guitar", 0.56),
        ("Acoustic guitar", "Acoustic Guitar", 0.60),
        ("Steel guitar, slide guitar", "Slide Guitar", 0.72),
        ("Guitar", "Guitar", 0.60),
        ("Strum", "Strummed Strings", 0.58),
        ("Plucked string instrument", "Plucked Strings", 0.0),
    ),
    "bass": (("Bass guitar", "Bass Guitar", 0.0),),
    "keys": (
        ("Electric piano", "Electric Piano", 0.72),
        ("Harpsichord", "Harpsichord", 0.82),
        ("Piano", "Piano", 0.64),
        ("Organ", "Organ", 0.70),
        ("Keyboard (musical)", "Keyboard", 0.0),
    ),
    "electronic": (
        ("Synthesizer", "Synthesizer", 0.0),
        ("Sampler", "Sampler", 0.72),
    ),
    "strings": (
        ("String section", "String Section", 0.60),
        ("Cello", "Cello", 0.69),
        ("Violin, fiddle", "Violin", 0.69),
        ("Double bass", "Double Bass", 0.75),
        ("Bowed string instrument", "Bowed Strings", 0.0),
    ),
    "brass": (
        ("French horn", "French Horn", 0.64),
        ("Trumpet", "Trumpet", 0.64),
        ("Trombone", "Trombone", 0.64),
        ("Brass instrument", "Brass", 0.0),
    ),
    "woodwinds": (
        ("Flute", "Flute", 0.65),
        ("Saxophone", "Saxophone", 0.65),
        ("Clarinet", "Clarinet", 0.65),
        ("Wind instrument, woodwind instrument", "Woodwinds", 0.0),
    ),
    "drums": (
        ("Drum machine", "Drum Machine", 0.62),
        ("Drum kit", "Drum Kit", 0.56),
        ("Timpani", "Timpani", 0.62),
        ("Hand percussion", "Hand Percussion", 0.63),
        ("Percussion", "Percussion", 0.0),
    ),
    "harp": (("Harp", "Harp", 0.0),),
    "mallets": (("Mallet percussion", "Mallet Percussion", 0.0),),
    "orchestra": (("Orchestra", "Orchestral Textures", 0.0),),
}

ALIASES = {
    "Singing": ("Humming", "Yodeling", "Child singing"),
    "Organ": ("Electronic organ", "Hammond organ"),
    "Percussion": ("Drum", "Bass drum", "Snare drum", "Cymbal", "Hi-hat", "Drum roll"),
    "Hand percussion": ("Tambourine", "Maraca", "Rattle (instrument)", "Tabla"),
    "Mallet percussion": ("Marimba, xylophone", "Glockenspiel", "Vibraphone", "Steelpan"),
}

ELECTRONIC_STYLES = ("Electronic music", "Electronic dance music", "Electronica", "Dubstep", "Techno")


def _load_model(path: Path):
    if not path.is_file():
        return None
    try:
        from ai_edge_litert.interpreter import Interpreter

        with ZipFile(path) as model_archive:
            labels = model_archive.read("yamnet_label_list.txt").decode("utf-8").splitlines()
        interpreter = Interpreter(model_path=str(path), num_threads=2)
        interpreter.allocate_tensors()
        model_input = interpreter.get_input_details()[0]
        model_output = interpreter.get_output_details()[0]
        if tuple(model_input["shape"]) != (WINDOW,) or int(model_output["shape"][-1]) != len(labels):
            return None
        return interpreter, model_input["index"], model_output["index"], labels
    except (ImportError, OSError, ValueError, RuntimeError, BadZipFile, KeyError, IndexError):
        return None


def _model_input(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray | None]:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    elif audio.ndim != 2:
        raise ValueError("Expected one-dimensional or two-dimensional audio samples")
    elif audio.shape[0] > 8 and audio.shape[1] <= 8:
        audio = audio.T
    if audio.shape[0] == 0 or audio.shape[1] == 0:
        return np.empty(0, dtype=np.float32), None
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    side = (audio[0] - audio[1]) * 0.5 if audio.shape[0] == 2 else None
    mono = audio.mean(axis=0, dtype=np.float32)
    if side is not None and np.max(np.abs(mono)) < 0.08 * np.max(np.abs(side)):
        mono, side = side, None
    if sample_rate != MODEL_RATE:
        divisor = gcd(sample_rate, MODEL_RATE)
        mono = resample_poly(mono, MODEL_RATE // divisor, sample_rate // divisor)
        if side is not None:
            side = resample_poly(side, MODEL_RATE // divisor, sample_rate // divisor)
    return mono, side


def _predict(model, segment: np.ndarray, rms_floor: float) -> np.ndarray | None:
    if model is None or len(segment) < MIN_SECTION:
        return None
    interpreter, input_index, output_index, _ = model
    last_start = max(0, len(segment) - WINDOW)
    starts = np.arange(0, last_start + 1, HOP)
    if last_start - starts[-1] >= HOP // 2:
        starts = np.append(starts, last_start)
    if len(starts) > MAX_WINDOWS:
        starts = np.linspace(0, last_start, MAX_WINDOWS, dtype=int)
    scores = []
    for start in starts:
        frame = segment[start:start + WINDOW]
        if np.sqrt(np.mean(frame ** 2)) < rms_floor:
            continue
        if len(frame) < WINDOW:
            frame = np.pad(frame, (0, WINDOW - len(frame)))
        interpreter.set_tensor(input_index, np.clip(frame, -1.0, 1.0))
        interpreter.invoke()
        scores.append(interpreter.get_tensor(output_index).reshape(-1))
    return np.stack(scores) if scores else None


def _support(scores: np.ndarray | None, labels: list[str]) -> dict[str, float]:
    if scores is None:
        return {}
    mean = scores.mean(axis=0)
    percentile = np.percentile(scores, 80, axis=0)
    coverage = (scores >= 0.015).mean(axis=0)
    # Require repeat or sustained evidence, while keeping clear instrument
    # entries when they occupy a small portion of a long section.
    qualifies = ((coverage >= 0.26) & (percentile >= 0.013)) | (mean >= 0.024)
    if len(scores) == 1:
        qualifies &= mean >= 0.041
    strengths = 0.60 * percentile + 0.40 * mean
    return {label: float(value) for label, value, valid in zip(labels, strengths, qualifies, strict=True) if valid}


def _pick_instruments(mono: dict[str, float], side: dict[str, float]) -> dict[str, tuple[str, float, float]]:
    def strength(scores: dict[str, float], name: str) -> float:
        names = (name, *ALIASES.get(name, ()))
        return max(scores.get(label, 0.0) for label in names)

    instruments = {}
    for family, entries in FAMILIES.items():
        broad_name = entries[-1][0]
        side_family_only = family in {"strings", "drums", "brass", "woodwinds"}
        candidates = []
        for name, display, specificity in entries:
            mono_score = strength(mono, name)
            side_score = strength(side, name)
            if family == "voice":
                side_score = min(side_score, max(0.023, 2.0 * mono_score))
            if side_family_only:
                if name == broad_name:
                    # The stereo side can support the family without
                    # identifying an exact instrument in the original mix.
                    side_score = max(strength(side, entry[0]) for entry in entries)
                elif mono_score < 0.012:
                    continue
            rank = max(mono_score, 0.75 * side_score)
            candidates.append((display, specificity, rank, mono_score))
        maximum = max(candidate[2] for candidate in candidates)
        if maximum < 0.012:
            continue
        display, _, rank, mono_score = next(
            candidate for candidate in candidates
            if candidate[2] >= max(0.012, candidate[1] * maximum)
        )
        instruments[family] = (display, rank, mono_score)
    if "electronic" in instruments and "keys" in instruments and instruments["keys"][0] == "Keyboard":
        del instruments["keys"]
    return instruments


def _describe_audio(segment: np.ndarray) -> tuple[bool, bool]:
    """Detect sustained low-band energy and recurrent broadband transients."""
    frame_size = 1024
    hop = 256
    if len(segment) < frame_size:
        return False, False
    frames = np.lib.stride_tricks.sliding_window_view(segment, frame_size)[::hop]
    # A bounded sample keeps spectral analysis practical for long recordings.
    step = max(1, (len(frames) + 4095) // 4096)
    frames = frames[::step]
    spectrum = np.abs(np.fft.rfft(frames * np.hanning(frame_size), axis=1)) ** 2
    freq = np.fft.rfftfreq(frame_size, 1 / MODEL_RATE)
    bands = np.stack([spectrum[:, (freq >= lo) & (freq < hi)].sum(axis=1) for lo, hi in (
        (25, 155), (155, 400), (400, 1800), (1800, 5000), (5000, 7500),
    )], axis=1)
    power = bands.sum(axis=1)
    active = power >= max(float(power.max()) * 0.003, 1e-8)
    if np.count_nonzero(active) < 8:
        return False, False
    low_share = bands[active, 0] / power[active]
    bass = float(np.median(low_share)) >= 0.18 and float(np.percentile(low_share, 75)) >= 0.32

    log_bands = np.log1p(bands / np.maximum(np.percentile(bands, 70, axis=0), 1e-6))
    growth = np.maximum(np.diff(log_bands, axis=0), 0.0)
    loud_edges = active[1:] & (np.count_nonzero(growth[:, 1:] > 0.50, axis=1) >= 2)
    interval = step * hop / MODEL_RATE
    onsets = np.flatnonzero(loud_edges) * interval
    if len(onsets):
        onsets = onsets[np.r_[True, np.diff(onsets) > 0.09]]
    independent_hits = len(onsets)
    percussive = 3 <= independent_hits and independent_hits / (len(segment) / MODEL_RATE) >= 0.45
    return bass, percussive


def _result(main: str = UNDETECTED, backing: Iterable[str] = (), confidence: str = "Undetected", note: str = "") -> dict[str, str]:
    return {
        "main_instruments": main,
        "accompaniment_instruments": ", ".join(backing) or UNDETECTED,
        "instrument_confidence": confidence,
        "instrument_note": note,
    }


def analyze_instruments(
    audio: np.ndarray,
    sample_rate: int,
    sections: Iterable[tuple[float, float]],
    model_path: str | Path | None = None,
) -> list[dict[str, str]]:
    """Return timbre/role estimates for each half-open section in seconds.

    Audio is a normalized floating-point waveform. Two-channel audio may use
    either channel-first or sample-first layout. The role fields describe the
    most prominent identified melody-capable timbre and supporting clues.
    """
    section_list = list(sections)
    if not section_list:
        return []
    return analyze_instruments_and_style(audio, sample_rate, section_list, model_path)[0]


def analyze_instruments_and_style(
    audio: np.ndarray,
    sample_rate: int,
    sections: Iterable[tuple[float, float]],
    model_path: str | Path | None = None,
) -> tuple[list[dict[str, str]], dict]:
    """Reuse section classifier windows to estimate the track's style."""
    sample_rate = int(sample_rate)
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive")
    section_list = list(sections)
    mono, side = _model_input(audio, sample_rate)
    peak = float(np.max(np.abs(mono), initial=0.0))
    rms_floor = max(8e-5, peak * 0.0015)
    if peak < rms_floor:
        return (
            [_result(note="The audio is too quiet to identify instruments.") for _ in section_list],
            undetermined_style("The audio is too quiet to estimate a style."),
        )
    if model_path is None:
        model_path = Path(__file__).resolve().parent / "models" / "yamnet.tflite"
    model = _load_model(Path(model_path))
    labels = model[3] if model is not None else []
    duration = len(mono) / MODEL_RATE
    style_windows = []
    output = []
    for start, end in section_list:
        if not isfinite(start) or not isfinite(end):
            output.append(_result(note="Invalid time range."))
            continue
        a = round(np.clip(start, 0, duration) * MODEL_RATE)
        b = round(np.clip(end, 0, duration) * MODEL_RATE)
        segment = mono[a:b]
        if len(segment) < MIN_SECTION or np.sqrt(np.mean(segment ** 2)) < rms_floor:
            output.append(_result(note="The section is too short or too quiet to identify instruments."))
            continue

        mono_scores, side_scores = {}, {}
        if model is not None:
            try:
                predictions = _predict(model, segment, rms_floor)
                if predictions is not None:
                    style_windows.append(predictions)
                    mono_scores = _support(predictions, labels)
                if side is not None:
                    side_segment = side[a:b]
                    if np.sqrt(np.mean(side_segment ** 2)) >= max(rms_floor, 0.12 * np.sqrt(np.mean(segment ** 2))):
                        side_scores = _support(_predict(model, side_segment * 1.35, rms_floor), labels)
            except (OSError, RuntimeError, ValueError):
                model = None

        instruments = _pick_instruments(mono_scores, side_scores)
        bass, percussive = _describe_audio(segment)
        style_score = max(mono_scores.get(label, 0.0) for label in ELECTRONIC_STYLES)
        electronic = style_score >= 0.030 and (bass or percussive)
        ranked = sorted(instruments, key=lambda family: instruments[family][1], reverse=True)
        melody_candidates = [family for family in ranked if family not in {"drums", "bass", "mallets", "orchestra"}]
        main_family = next((family for family in melody_candidates if instruments[family][2] >= 0.012), None)
        main = UNDETECTED
        confidence = "Undetected"
        if main_family is not None:
            main, _, mono_support = instruments[main_family]
            confidence = "Medium" if mono_support >= 0.045 else "Low"
        elif electronic:
            main = "Electronic Textures"
            confidence = "Low"

        backing = [instruments[family][0] for family in ranked if family != main_family][:3]
        if electronic and "electronic" not in instruments and main_family is not None:
            backing.append("Electronic Textures")
        keys_only = main_family == "keys" and not backing
        if bass and "bass" not in instruments and not keys_only:
            backing.append("Bass Layer")
        if percussive and "drums" not in instruments and not keys_only:
            backing.append("Percussive Transients")
        if backing and confidence == "Undetected":
            confidence = "Low"
        if model is None:
            note = "The instrument model is unavailable. Estimates use the mix spectrum and transients."
        elif main == UNDETECTED and not backing:
            note = "The mix provides too little evidence to identify instruments."
        else:
            note = "Lead and accompaniment roles are estimated from timbre and prominence in the mix."
        output.append(_result(main, backing[:4], confidence, note))

    if not section_list and model is not None:
        try:
            predictions = _predict(model, mono, rms_floor)
            if predictions is not None:
                style_windows.append(predictions)
        except (OSError, RuntimeError, ValueError):
            model = None
    if style_windows:
        return output, identify_style(np.concatenate(style_windows), labels)
    note = "Too little audible audio was available to estimate a style."
    if model is None:
        note = "The offline style model is unavailable."
    return output, undetermined_style(note)
