"""Refine structural timestamps on a song's locally estimated phrase grid."""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import stft


def _phrase_cues(waveform: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Spectral and loudness changes before and after each candidate downbeat."""
    if waveform.ndim != 1 or len(waveform) < sr:
        return np.array([]), np.array([])
    # A 50 ms hop resolves sub-beat changes without retaining a full spectrogram.
    fft_size = min(2048, max(256, sr // 10))
    hop = min(fft_size // 2, max(1, round(sr * 0.045)))
    frequencies, times, bins = stft(
        waveform, fs=sr, nperseg=fft_size, noverlap=fft_size - hop, boundary=None, padded=False,
    )
    power = np.abs(bins) ** 2
    bands = []
    for low, high in zip((25, 100, 240, 600, 1500, 3600), (100, 240, 600, 1500, 3600, sr / 2)):
        band = (frequencies >= low) & (frequencies < high)
        if np.any(band):
            bands.append(10 * np.log10(np.maximum(np.mean(power[band], axis=0), 1e-9)))
    if not bands:
        return np.array([]), np.array([])
    trace = np.vstack(bands)
    novelty = np.zeros(len(times))
    for seconds, weight in ((0.4, 0.36), (1.25, 0.35), (2.4, 0.29)):
        radius = max(1, round(seconds * sr / hop))
        smooth = gaussian_filter1d(trace, radius / 2, axis=1)
        i = np.arange(len(times))
        left = smooth[:, np.maximum(0, i - radius)]
        right = smooth[:, np.minimum(len(times) - 1, i + radius)]
        novelty += weight * np.sqrt(np.mean((right - left) ** 2, axis=0))
    low, high = np.percentile(novelty, [45, 95])
    novelty = np.clip((novelty - low) / max(high - low, 0.7), 0, 1.5)
    return times, novelty


def refine_phrase_grid(
    boundaries: list[float],
    strengths: list[float],
    onset: float | None,
    duration: float,
    rhythm: dict,
    *,
    evidence_audio: np.ndarray | None = None,
    sr: int | None = None,
) -> tuple[list[float], list[float]]:
    """Use repeated phrase phases to restore phrase starts within broad sections.

    The first and last elements are the full analysis interval. A reliable
    off-grid accent may be kept as a pickup alongside the regular downbeat.
    """
    period = rhythm.get("bar_period")
    phase = rhythm.get("downbeat_phase")
    pulse_confidence = rhythm.get("beat_grid_confidence", rhythm.get("grid_confidence", 0.0))
    if (
        not period or phase is None or onset is None or duration < 28 or len(boundaries) < 5
        or rhythm.get("tempo_confidence", 0.0) < 0.44 or pulse_confidence < 0.30
    ):
        return boundaries, strengths

    # Favor musically conventional lengths near a 14-second phrase. Half
    # phrases are allowed when the arrangement shifts phase on their start.
    phrase_bars = int(2 ** np.clip(round(math.log2(14 / period)), 1, 3))
    half_bars = max(1, phrase_bars // 2)
    raw_points = np.asarray(boundaries[1:-1])
    raw_strength = np.asarray(strengths[1:-1])
    measured = (raw_points - phase) / period
    hypotheses = np.arange(half_bars)
    errors = np.abs((measured[:, None] - hypotheses + half_bars / 2) % half_bars - half_bars / 2)
    votes = (raw_strength[:, None] * np.exp(-0.5 * (errors / 0.42) ** 2)).sum(axis=0)
    origin = int(np.argmax(votes))
    if votes[origin] < 0.33 * max(float(raw_strength.sum()), 1e-8):
        return boundaries, strengths

    cue_times, cues = _phrase_cues(evidence_audio, sr) if evidence_audio is not None and sr else (np.array([]), np.array([]))
    def acoustic_cue(time: float) -> float:
        if not cues.size:
            return 0.0
        left, right = np.searchsorted(cue_times, [time - .22, time + .32])
        return float(cues[left:right].max(initial=0))

    onset_bar = (onset - phase) / period
    anchors = {}
    pickups = []
    residual_cues = []
    anchor_residual = {}
    for position, point, strength in zip(measured, raw_points, raw_strength):
        nearby = origin + half_bars * round((position - origin) / half_bars)
        delta = nearby - position
        tolerance = min(1.12, 0.53 * half_bars)
        if abs(delta) <= tolerance:
            anchors[nearby] = max(anchors.get(nearby, 0.0), float(strength))
            anchor_residual[nearby] = min(anchor_residual.get(nearby, math.inf), abs(delta))
            if 0.15 < delta <= 0.77 and strength >= 0.87:
                residual_cues.append((nearby, float(point), float(strength)))
        else:
            # Quiet tail entrances can reset the phrase phase.
            final_entrance = point > duration - 2.1 * phrase_bars * period
            nearest_bar = round(position)
            if strength >= 0.78 and (abs(nearest_bar - position) <= 0.28 or final_entrance):
                anchors[nearest_bar] = max(anchors.get(nearest_bar, 0.0), float(strength))

    if len(anchors) < 3:
        return boundaries, strengths
    ordered = sorted(anchors)
    first_candidate = origin + half_bars * math.ceil((onset_bar - 0.04 - origin) / half_bars)
    if 0 <= first_candidate - onset_bar <= phrase_bars and ordered[0] - first_candidate >= half_bars:
        first_offset = first_candidate * period + phase - onset
        if 0.35 < first_offset < 2.0 and ordered[0] - first_candidate >= phrase_bars:
            anchors[first_candidate] = 0.55
        elif abs(first_candidate - onset_bar) <= 0.25:
            anchors[first_candidate] = 0.6

    # Remove a half-phrase mark between two full-phrase endpoints when the
    # audio offers little corroboration. This retains arrangement shifts that
    # start a new phrase phase without marking every detected energy peak.
    ordered = sorted(anchors)
    for left, middle, right in zip(ordered, ordered[1:], ordered[2:]):
        left_gap, right_gap = middle - left, right - middle
        if left_gap != half_bars and right_gap != half_bars:
            if (right - left) % phrase_bars == 0 and (middle - left) % phrase_bars != 0 and anchors[middle] < 0.45:
                anchors.pop(middle, None)
            continue
        cue = acoustic_cue(middle * period + phase)
        half_mark = left_gap == right_gap == half_bars
        if half_mark and anchors[middle] < 0.58:
            anchors.pop(middle, None)
        elif (right - left) % phrase_bars == 0 and anchors[middle] < 0.80:
            alternative = left + phrase_bars
            if alternative < right and (
                cue + .10 < acoustic_cue(alternative * period + phase)
                or anchor_residual.get(middle, 0) > 0.75
            ):
                anchors.pop(middle, None)
        elif half_mark and anchors[left] > 0.64 and anchors[right] > 0.64 and cue < 0.23:
            anchors.pop(middle, None)

    ordered = sorted(anchors)
    for left, right in zip(ordered, ordered[1:]):
        remaining = right - left
        if right % half_bars != origin and remaining <= phrase_bars + half_bars:
            continue
        cursor = left
        while remaining > phrase_bars:
            step = phrase_bars
            # Where the ending is off the prevailing phrase phase, use the
            # stronger local change to decide which half comes first.
            if remaining == phrase_bars + half_bars:
                earlier = (cursor + half_bars) * period + phase
                later = (cursor + phrase_bars) * period + phase
                if acoustic_cue(earlier) > acoustic_cue(later) + 0.23:
                    step = half_bars
            cursor += step
            remaining = right - cursor
            anchors[cursor] = 0.54
    ordered = sorted(anchors)
    for left, right in zip(ordered, ordered[1:]):
        if right - left == phrase_bars and right % half_bars == origin:
            midpoint = left + half_bars
            time = midpoint * period + phase
            cue = acoustic_cue(time)
            weak_endpoint = min(anchors[left], anchors[right]) <= .56
            early_section_change = time < duration * .20 and cue >= .84
            if half_bars >= 4 and (cue >= .22 and weak_endpoint or early_section_change):
                anchors[midpoint] = max(anchors.get(midpoint, 0.0), 0.56)

    for bar, point, strength in residual_cues:
        if bar not in anchors or anchors[bar] < 0.72:
            continue
        offset = bar * period + phase - point
        grid_cue = acoustic_cue(bar * period + phase)
        salient_pickup = acoustic_cue(point) >= .65 and grid_cue < 1.2
        independent_pickup = 0.95 < offset < 1.7 and period < 2.2
        if 0.25 < offset < min(1.7, period * 0.70) and (salient_pickup or independent_pickup):
            pickups.append((point, strength))

    grid = [(index * period + phase, strength) for index, strength in anchors.items()]
    candidates = [(point, strength) for point, strength in grid + pickups if onset + .25 < point < duration - .3]
    refined = [0.0]
    confidence = [strengths[0]]
    for point, strength in sorted(candidates):
        if point - refined[-1] < .25:
            continue
        refined.append(point)
        confidence.append(strength)
    return refined + [duration], confidence + [strengths[-1]]


def estimate_phrase_end(
    waveform: np.ndarray, sr: int, duration: float, final_section_start: float, rhythm: dict,
) -> float:
    """Locate a late musical cutoff, leaving the file duration available separately."""
    if waveform.ndim != 1 or sr <= 0 or duration - final_section_start < 3:
        return duration
    hop = max(1, sr // 20)
    width = max(1, round(sr * .16))
    first = max(round(final_section_start * sr), len(waveform) - sr * 45)
    windows = np.lib.stride_tricks.sliding_window_view(waveform[first:], width)[::hop]
    rms = np.sqrt(np.mean(windows * windows, axis=1))
    times = (first + (np.arange(len(rms)) * hop + width / 2)) / sr
    period = rhythm.get("bar_period")
    phase = rhythm.get("downbeat_phase")
    if not period or phase is None or not rms.size:
        return duration
    indices = np.arange(math.ceil((final_section_start + min(2.0, period) - phase) / period),
                        math.floor((duration - 1.3 - phase) / period) + 1)
    for index in reversed(indices):
        candidate = index * period + phase
        before = rms[(times >= candidate - min(3.5, 1.7 * period)) & (times < candidate - .1)]
        after = rms[(times >= candidate + .4) & (times < min(duration, candidate + min(3.6, 2 * period)))]
        remaining = rms[times >= candidate + .6]
        if min(len(before), len(after)) < 12:
            continue
        recent = float(np.median(before))
        decline = float(np.median(after)) / max(recent, 1e-8)
        if recent >= .025 and decline < .30 and np.quantile(remaining, .9) <= max(.016, .4 * recent):
            return float(candidate)
    return duration
