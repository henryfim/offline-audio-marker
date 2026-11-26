"""Estimate an offline tempo and a section-alignment grid from audio onsets."""

from __future__ import annotations

import math

import numpy as np
from scipy import fft, ndimage, signal


# The low bands retain kick and bass attacks. The remaining bands add snare,
# hi-hat, and pitched onsets without allowing a single bright cymbal to win.
_BAND_EDGES = (25, 90, 190, 480, 1450, 4400, 9000, 12000)
_BAND_WEIGHTS = np.array([2, 2, 3, 3, 3, 1, 0.5], dtype=float)
_MIN_BPM = 42.0
_MAX_BPM = 220.0


def _spectral_flux(y: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    hop = max(1, round(sr / 150))
    window_length = min(4096, 2 ** round(math.log2(sr * 0.09)))
    window = signal.windows.hann(window_length, sym=False).astype(np.float32)
    padded = np.pad(y, window_length // 2)
    frames = np.lib.stride_tricks.sliding_window_view(padded, window_length)[::hop]
    frequencies = fft.rfftfreq(window_length, 1 / sr)
    bands = [(frequencies >= lo) & (frequencies < hi) for lo, hi in zip(_BAND_EDGES, _BAND_EDGES[1:])]
    flux = np.zeros((len(bands), len(frames)), dtype=np.float32)
    previous = None
    for start in range(0, len(frames), 2048):
        end = min(start + 2048, len(frames))
        magnitude = np.abs(fft.rfft(frames[start:end] * window))
        if previous is None:
            previous = magnitude[:1]
        positive = np.maximum(np.diff(magnitude, axis=0, prepend=previous), 0.0)
        previous = magnitude[-1:]
        for index, band in enumerate(bands):
            if np.any(band):
                flux[index, start:end] = positive[:, band].mean(axis=1)
    return flux, sr / hop


def _interpolate(grid: np.ndarray, values: np.ndarray, samples: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(samples, grid, row, left=0.0, right=0.0) for row in values])


def _normalized_flux(flux: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scales = np.quantile(flux, 0.95, axis=1)
    active = scales > max(float(np.max(scales)) * 0.0001, 1e-7)
    weights = _BAND_WEIGHTS * active
    weights /= max(float(weights.sum()), 1e-9)
    normalized = np.minimum(flux / np.maximum(scales[:, None], 1e-7), 3.0)
    return normalized.astype(np.float32), weights


def _rank_tempos(
    envelopes: np.ndarray, weights: np.ndarray, frame_rate: float
) -> tuple[list[tuple[float, float]], float]:
    # Remove volume swells without suppressing the principal beat periods.
    centered = envelopes - ndimage.gaussian_filter1d(envelopes, frame_rate * 0.6, axis=1)
    frame_count = centered.shape[1]
    transform_size = fft.next_fast_len(max(frame_count, round(frame_rate * 960)))
    frequency_bpm = 60 * fft.rfftfreq(transform_size, 1 / frame_rate)
    amplitude = np.abs(fft.rfft(centered, n=transform_size, axis=1))
    amplitude /= np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-7)
    amplitude /= np.sqrt(frame_count)
    autocorrelation = fft.irfft(np.abs(fft.rfft(centered, n=fft.next_fast_len(2 * frame_count), axis=1)) ** 2, axis=1)
    autocorrelation = autocorrelation[:, : min(frame_count, round(frame_rate * 6))]
    autocorrelation /= np.maximum(autocorrelation[:, :1].copy(), 1e-7)

    tempos = np.arange(_MIN_BPM, _MAX_BPM + 0.02, 0.04)
    direct = _interpolate(frequency_bpm, amplitude, tempos)
    half = weights @ _interpolate(frequency_bpm, amplitude, tempos * 0.5)
    quarter = weights @ _interpolate(frequency_bpm, amplitude, tempos * 0.25)
    doubled = weights @ _interpolate(frequency_bpm, amplitude, tempos * 2)
    quadrupled = weights @ _interpolate(frequency_bpm, amplitude, tempos * 4)
    fundamental = weights @ direct
    bass = 0.55 * direct[0] + 0.45 * direct[1]
    lag = 60 * frame_rate / tempos
    lags = np.arange(autocorrelation.shape[1])
    beat_repeat = weights @ _interpolate(lags, autocorrelation, lag)
    meter_repeat = weights @ _interpolate(lags, autocorrelation, lag * 4)
    # A pulse found mainly in subdivisions has less metrical evidence than a
    # pulse accompanied by equally regular low-frequency accents on the beats.
    strength = 0.8 * fundamental + 0.6 * doubled + 0.18 * quadrupled + 0.2 * half + 0.3 * quarter
    strength += 0.65 * bass + 0.09 * beat_repeat + 0.06 * meter_repeat
    peaks, _ = signal.find_peaks(strength)
    peaks = np.r_[0, peaks, len(tempos) - 1]
    peaks = peaks[np.argsort(strength[peaks])[::-1]]
    ranked = []
    for peak in peaks:
        tempo = float(tempos[peak])
        if any(abs(tempo - previous) < max(2.0, previous * 0.024) for previous, _ in ranked):
            continue
        if 0 < peak < len(tempos) - 1:
            left, center, right = strength[peak - 1:peak + 2]
            curve = left - 2 * center + right
            if curve < 0:
                tempo += float(np.clip((left - right) / (2 * curve), -0.5, 0.5)) * 0.04
        ranked.append((tempo, float(strength[peak])))
        if len(ranked) >= 6:
            break
    peak = peaks[0]
    periodicity = float(fundamental[peak] + 0.45 * doubled[peak] + 0.25 * max(0.0, beat_repeat[peak]))
    return ranked, periodicity


def _fold(values: np.ndarray, positions: np.ndarray, size: int) -> np.ndarray:
    indices = np.floor(np.remainder(positions, 1.0) * size).astype(int)
    distribution = np.bincount(indices, weights=values, minlength=size)
    distribution /= np.maximum(np.bincount(indices, minlength=size), 1)
    return ndimage.gaussian_filter1d(distribution, 1.25, mode="wrap")


def _phase_distance(a: float, b: float, period: float) -> float:
    return float(abs((a - b + period / 2) % period - period / 2))


def _meter_grid(
    envelopes: np.ndarray, weights: np.ndarray, frame_rate: float, bpm: float, tempo_confidence: float,
) -> dict:
    grid = {
        "time_signature": None,
        "meter_confidence": 0.0,
        "beat_phase": None,
        "beat_grid_confidence": 0.0,
        "downbeat_phase": None,
        "bar_period": None,
        "grid_confidence": 0.0,
    }
    beat_period = 60 / bpm
    time = np.arange(envelopes.shape[1]) / frame_rate
    blended = weights @ envelopes
    folded = _fold(blended, time / beat_period, 96)
    # Positive spectral flux is centered slightly before its corresponding
    # transient. Use the same offset for the global and local phase estimates.
    flux_delay = 0.012
    pulse = (float(np.argmax(folded)) + 0.5) / 96 * beat_period
    maximum = float(np.max(folded))
    alignment = (maximum - float(np.median(folded))) / max(maximum, 1e-7)
    beat_numbers = np.arange(max(0, math.ceil(-pulse / beat_period)), math.floor((time[-1] - pulse) / beat_period) + 1)
    beat_times = pulse + beat_numbers * beat_period
    if len(beat_times) < 10 or alignment < 0.25:
        return grid
    grid["beat_phase"] = (pulse + flux_delay) % beat_period
    low = np.average(envelopes[:3], axis=0, weights=[0.40, 0.35, 0.25])
    accent_signal = 0.60 * low + 0.40 * blended
    smoothed = ndimage.gaussian_filter1d(accent_signal, frame_rate * 0.025)
    accents = np.interp(beat_times, time, smoothed)
    accents = accents / np.maximum(ndimage.gaussian_filter1d(accents, 8.0), 0.035)

    choices = []
    for size, signature in ((3, "3/4"), (4, "4/4")):
        phases = np.arange(size)
        mean = np.array([np.mean(accents[beat_numbers % size == phase]) for phase in phases])
        phase = int(np.argmax(mean))
        # Test against repeated two-beat accents. Those support duple meter
        # and yield a weaker indication of the exact four-beat bar start.
        spread = float(np.std(mean))
        margin = float(mean[phase] - np.sort(mean)[-2])
        choices.append((spread + 0.2 * margin, phase, signature, size, margin))
    # Two beats with ternary subdivision suggest compound duple meter.
    # Require both subdivision peaks to exceed the binary offbeat so a single
    # syncopated accent is insufficient to infer a compound meter.
    pulse_index = int(np.argmax(folded))
    baseline = float(np.median(folded))
    thirds = min(folded[(pulse_index + offset) % 96] for offset in (32, 64))
    binary = float(folded[(pulse_index + 48) % 96])
    ternary = float(max(0.0, thirds - max(binary, baseline)) / max(maximum, 1e-7))
    if ternary >= 0.18:
        half_bar = np.array([np.mean(accents[beat_numbers % 2 == phase]) for phase in range(2)])
        phase = int(np.argmax(half_bar))
        margin = float(abs(half_bar[0] - half_bar[1]))
        if margin >= 0.12:
            choices.append((0.6 * margin + ternary, phase, "6/8", 2, margin))
    choices.sort(reverse=True)
    score, phase, signature, size, margin = choices[0]
    other = choices[1][0]
    meter_confidence = np.clip((score - other) * 3 + score * 0.5, 0.0, 1.0)
    grid["meter_confidence"] = round(float(meter_confidence), 3)
    if score >= 0.075 and meter_confidence >= 0.14:
        bar_period = size * beat_period
        grid["time_signature"] = signature
        grid["bar_period"] = bar_period
        grid["downbeat_phase"] = (pulse + phase * beat_period + flux_delay) % bar_period

    local_periods = []
    local_phases = []
    span = max(10 * size * beat_period, 12.0)
    for start in np.arange(0.0, time[-1], span):
        window = (time >= start) & (time < start + span)
        local_beat_mask = (beat_times >= start) & (beat_times < start + span)
        if window.sum() < 0.65 * frame_rate * span or local_beat_mask.sum() < 3 * size:
            continue
        local = _fold(blended[window], time[window] / beat_period, 96)
        if float(local.max() - np.median(local)) < 0.025:
            continue
        local_pulse = (float(np.argmax(local)) + 0.5) / 96 * beat_period
        local_periods.append(_phase_distance(local_pulse, pulse, beat_period) <= 0.13 * beat_period)
        local_numbers = beat_numbers[local_beat_mask]
        local_accents = accents[local_beat_mask]
        phase_means = [float(np.mean(local_accents[local_numbers % size == index])) for index in range(size)]
        local_phases.append(phase_means[phase] >= max(phase_means) * 0.90)
    pulse_stability = float(np.mean(local_periods)) if local_periods else 0.0
    grid["beat_grid_confidence"] = round(tempo_confidence * pulse_stability * math.sqrt(alignment), 3)
    if grid["time_signature"] is not None and local_phases:
        bar_stability = float(min(pulse_stability, np.mean(local_phases)))
        confidence = tempo_confidence * bar_stability * math.sqrt(float(meter_confidence) * alignment)
        grid["grid_confidence"] = round(confidence * min(1.0, margin * 5), 3)
    return grid


def analyze_rhythm(y: np.ndarray, sr: int, bpm_override: float | None = None) -> dict:
    """Return tempo, half-time alternatives, meter, and an optional bar grid.

    Timing values use seconds from the first sample. Tempo confidence describes
    the audio estimate, while an override sets the tempo used for grid fitting.
    Only use downbeat_phase + n * bar_period to snap sections if grid_confidence
    supports it. All return values are JSON serializable.
    """
    y = np.asarray(y, dtype=np.float32)
    if y.ndim != 1 or not 1000 <= sr <= 384000:
        raise ValueError("Expected mono audio and a valid sample rate.")
    override = None if bpm_override is None else float(bpm_override)
    if override is not None and (not math.isfinite(override) or override <= 0):
        raise ValueError("BPM override must be a finite positive number.")
    result = {
        "bpm": override,
        "half_bpm": None if override is None else round(override / 2, 3),
        "estimated_bpm": None,
        "tempo_confidence": 0.0,
        "tempo_candidates": [],
        "time_signature": None,
        "meter_confidence": 0.0,
        "beat_period": None if override is None else 60 / override,
        "beat_phase": None,
        "beat_grid_confidence": 0.0,
        "bar_period": None,
        "downbeat_phase": None,
        "grid_confidence": 0.0,
    }
    if len(y) < 4 * sr or float(np.max(np.abs(y))) < 1e-5:
        return result
    # A steady tone can generate periodic leakage between FFT bins even when
    # the waveform has no transients. Check the short-term amplitude first.
    block_size = max(1, round(sr * 0.50))
    blocks = y[:len(y) - len(y) % block_size].reshape(-1, block_size)
    block_rms = np.sqrt(np.mean(blocks ** 2, axis=1))
    if float(np.std(block_rms)) < float(np.mean(block_rms)) * 0.015:
        return result
    envelopes, frame_rate = _spectral_flux(y, sr)
    edge = max(1, round(frame_rate * 0.08))
    envelopes[:, :edge] = 0
    envelopes[:, -edge:] = 0
    envelopes, weights = _normalized_flux(envelopes)
    if not np.any(weights):
        return result
    tempos, periodicity = _rank_tempos(envelopes, weights, frame_rate)
    best, strength = tempos[0]
    runner_up = tempos[1][1] if len(tempos) > 1 else 0.0
    relative_margin = (strength - runner_up) / max(strength, 0.01)
    periodic_confidence = float(np.clip((periodicity - 0.04) / 0.13, 0.0, 1.0))
    confidence = periodic_confidence * (0.68 + 0.32 * min(1.0, relative_margin / 0.25))
    result["tempo_confidence"] = round(confidence, 3)
    if confidence < 0.20:
        return result
    result["estimated_bpm"] = round(best, 3)
    result["tempo_candidates"] = [
        {"bpm": round(tempo, 3), "confidence": round(confidence * max(0.0, value) / strength, 3)}
        for tempo, value in tempos[:4] if value > strength * 0.55
    ]
    bpm = override if override is not None else best
    result.update(bpm=round(bpm, 3), half_bpm=round(bpm / 2, 3), beat_period=60 / bpm)
    result.update(_meter_grid(envelopes, weights, frame_rate, bpm, confidence))
    return result
