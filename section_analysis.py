"""Offline music structure detection with waveform-based timestamp refinement."""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


@dataclass
class SectionAnalysis:
    boundaries: list[float]
    first_audio_onset: float | None
    boundary_strengths: list[float]
    repeated_sections: list[dict[str, float | int]]


def _first_audio_onset(y: np.ndarray, sr: int, reference: float) -> float | None:
    block_size = max(1, round(sr * 0.002))
    padding = -len(y) % block_size
    blocks = np.pad(y, (0, padding)).reshape(-1, block_size)
    rms = np.sqrt(np.einsum('ij,ij->i', blocks, blocks) / block_size)
    # Scale the silence threshold to the recording level and require a small
    # cluster of active blocks to reject codec ringing and isolated clicks.
    threshold = max(1e-6, reference * 0.018)
    active = rms >= threshold
    minimum_active = min(4, len(rms))
    sustained = np.convolve(active, np.ones(min(12, len(rms))), mode='full')
    candidates = np.flatnonzero(active & (sustained[min(12, len(rms)) - 1:] >= minimum_active))
    if not candidates.size:
        return None
    first_sample = int(candidates[0]) * block_size
    return first_sample / sr


def _spectral_features(y: np.ndarray, sr: int, derive_chroma: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    hop = max(1, round(sr * 0.01))
    fft_size = int(2 ** np.ceil(np.log2(max(32, sr * 0.033))))
    mel_filter = librosa.filters.mel(sr=sr, n_fft=fft_size, n_mels=40, fmin=min(28.0, sr / 8))
    chroma_filter = librosa.filters.chroma(sr=sr, n_fft=fft_size) if derive_chroma else None
    padded = np.pad(y, (fft_size // 2, fft_size // 2))
    frames = np.lib.stride_tricks.sliding_window_view(padded, fft_size)[::hop]
    window = np.hanning(fft_size).astype(np.float32)
    mel = np.empty((len(frames), len(mel_filter)), dtype=np.float32)
    chroma = np.empty((len(frames), 12), dtype=np.float32) if derive_chroma else None
    for start in range(0, len(frames), 2048):
        stop = start + 2048
        spectrum = abs(np.fft.rfft(frames[start:stop] * window, axis=1)).astype(np.float32)
        power = spectrum * spectrum
        mel[start:stop] = power @ mel_filter.T
        if chroma is not None:
            chroma[start:stop] = power @ chroma_filter.T
    return np.arange(len(mel)) * (hop / sr), mel, chroma


def _scale_features(features: np.ndarray, spread_floor: float) -> np.ndarray:
    low, median, high = np.percentile(features, [15, 50, 85], axis=0)
    return np.clip((features - median) / np.maximum(high - low, spread_floor), -2.0, 2.0)


def _normalized_novelty(values: np.ndarray) -> np.ndarray:
    median, high = np.percentile(values, [50, 95])
    return np.clip((values - median) / max(high - median, 0.10), 0.0, 2.0)


def _mean_change(features: np.ndarray, radius: int) -> np.ndarray:
    length = len(features)
    prefix = np.vstack((np.zeros((1, features.shape[1])), np.cumsum(features, axis=0, dtype=float)))
    center = np.arange(length)
    left = np.maximum(0, center - radius)
    right = np.minimum(length, center + radius)
    before = (prefix[center] - prefix[left]) / np.maximum(1, center - left)[:, None]
    after = (prefix[right] - prefix[center]) / np.maximum(1, right - center)[:, None]
    coverage = np.minimum(center - left, right - center) / radius
    return np.linalg.norm(after - before, axis=1) * np.sqrt(coverage / features.shape[1])


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sum(left * right) / max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-8))


def _sample_interval(features: np.ndarray, first: int, last: int, count: int) -> np.ndarray:
    points = np.linspace(first, last, count + 1, dtype=int)
    return np.array([features[start:max(start + 1, stop)].mean(axis=0) for start, stop in zip(points, points[1:])])


def _repeat_candidates(features: np.ndarray, peaks: np.ndarray, step: float) -> np.ndarray:
    """Reward transitions that lead into a repeat with a different preceding texture."""
    scores = np.zeros(len(peaks), dtype=float)
    context = round(5.0 / step)
    indices = np.flatnonzero((peaks >= context) & (peaks + context <= len(features)))
    if not indices.size:
        return scores
    positions = peaks[indices]
    before = np.array([_sample_interval(features, point - context, point, 5).ravel() for point in positions], dtype=np.float32)
    after = np.array([_sample_interval(features, point, point + context, 5).ravel() for point in positions], dtype=np.float32)
    before /= np.maximum(np.linalg.norm(before, axis=1, keepdims=True), 1e-8)
    after /= np.maximum(np.linalg.norm(after, axis=1, keepdims=True), 1e-8)
    # Evaluate in blocks to keep memory bounded for long recordings.
    for start in range(0, len(indices), 256):
        stop = start + 256
        similarity = after[start:stop] @ after.T
        previous = before[start:stop] @ before.T
        change = np.maximum(0.0, similarity - np.maximum(0.25, previous))
        values = np.clip((similarity - 0.52) * change * 2.0, 0.0, 0.22)
        values[abs(positions[start:stop, None] - positions[None, :]) * step < 10.0] = 0.0
        scores[indices[start:stop]] = values.max(axis=1)
    return scores


def _phrase_candidates(
    features: np.ndarray, novelty: np.ndarray, downbeats: np.ndarray, step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compare adjacent bars and estimate recurring four- and eight-bar phases."""
    if downbeats.size < 13:
        return np.array([], dtype=int), np.array([]), np.array([], dtype=bool)
    period = float(np.median(np.diff(downbeats)))
    if period < 0.5 or np.max(abs(np.diff(downbeats) - period)) > 0.30 * period:
        return np.array([], dtype=int), np.array([]), np.array([], dtype=bool)
    bar_frames = np.rint(downbeats / step).astype(int)
    bars = np.array([features[start:end].mean(axis=0) for start, end in zip(bar_frames[:-1], bar_frames[1:])])
    two_bars = np.array([_mean_change(bars, width) for width in (1, 2, 4)])
    bar_novelty = sum(weight * _normalized_novelty(change) for weight, change in zip((0.18, 0.38, 0.44), two_bars))
    nearby = np.array([novelty[max(0, frame - 3):frame + 4].max() for frame in bar_frames[:-1]])
    evidence = 0.56 * bar_novelty + 0.44 * nearby

    # Avoid learning a phase from one dramatic event by clipping its vote.
    # A phase must recur through the recording to influence boundary selection.
    indices = np.arange(len(bars))
    votes = np.minimum(evidence, 1.10)
    phase_values = np.array([np.mean(votes[indices % 4 == phase]) for phase in range(4)])
    phase = int(np.argmax(phase_values))
    separation = phase_values[phase] - np.median(phase_values)
    if separation < 0.045:
        return bar_frames[2:-1], evidence[2:], np.zeros(len(bars) - 2, dtype=bool)
    four = indices % 4 == phase
    first, second = np.mean(votes[indices % 8 == phase]), np.mean(votes[indices % 8 == phase + 4])
    phase_eight = phase + 4 * (second > first)
    eight = indices % 8 == phase_eight
    evidence += 0.11 * four + 0.12 * eight
    return bar_frames[2:-1], evidence[2:], four[2:]


def _outro_boundaries(
    bands: np.ndarray, candidates: np.ndarray, step: float, strengths: np.ndarray,
) -> list[int]:
    """Find the final loss of density and any distinct entrance after a gap."""
    window = round(4.0 / step)
    candidates_late = (candidates >= len(bands) * 0.65) & (candidates <= len(bands) * 0.94)
    possible = []
    for point, strength in zip(candidates[candidates_late], strengths[candidates_late]):
        before, after = bands[point - window:point], bands[point:point + window]
        if not len(before) or not len(after) or strength < 0.55:
            continue
        baseline = np.median(before, axis=0)
        loss = baseline - np.median(after, axis=0)
        stays_quiet = np.mean((baseline[2:] - bands[point:, 2:]).mean(axis=1) >= 5.0)
        if loss[0] >= 3.0 and loss[2:].mean() >= 7.0 and stays_quiet >= 0.68:
            possible.append((point, min(loss[2:].mean(), 12.0) + 5.0 * strength + 3.0 * stays_quiet))
    if not possible:
        return []
    first = possible[0][0]
    # Prefer the most evident cut near the earliest sustainable change.
    outro = int(max((item for item in possible if (item[0] - first) * step <= 1.2), key=lambda item: item[1])[0])
    gap_window = round(1.8 / step)
    reference = float(np.median(bands[outro - window:outro, 0]))
    for point in candidates[(candidates > outro + window) & (candidates < len(bands) - window)]:
        before = bands[point - gap_window:point, 0]
        after = bands[point:point + gap_window, 0]
        remains_quiet = np.quantile(before, 0.85) < reference - 24.0
        has_new_entrance = np.median(after) > np.median(before) + 13.0
        if remains_quiet and has_new_entrance:
            return [outro, int(point)]
    return [outro]


def _refine_boundary(
    estimate: float, times: np.ndarray, short_novelty: np.ndarray,
    transient: np.ndarray, downbeats: np.ndarray,
) -> float:
    radius = 0.70
    if downbeats.size and np.min(abs(downbeats - estimate)) < 0.09:
        radius = 0.46
    first, last = np.searchsorted(times, [estimate - radius, estimate + radius])
    local_times = times[first:last]
    if not local_times.size:
        return estimate
    # A sudden spectral change sharpens the cut. A supplied downbeat can
    # resolve near ties, while drum hits far from the change remain weak cues.
    local_score = 0.56 * short_novelty[first:last] + 0.44 * transient[first:last]
    local_score -= 0.60 * ((local_times - estimate) / radius) ** 2
    if downbeats.size:
        index = np.searchsorted(downbeats, local_times)
        nearest_after = downbeats[np.minimum(index, len(downbeats) - 1)]
        nearest_before = downbeats[np.maximum(index - 1, 0)]
        error = np.minimum(abs(nearest_after - local_times), abs(nearest_before - local_times))
        local_score += 0.22 * np.exp(-0.5 * (error / 0.095) ** 2)
    best = int(np.argmax(local_score))
    if best == 0 or best == len(local_times) - 1:
        return float(local_times[best])
    left, peak, right = local_score[best - 1:best + 2]
    curvature = left - 2.0 * peak + right
    offset = (left - right) / (2.0 * curvature) if curvature < -1e-8 else 0.0
    return float(local_times[best] + offset * (local_times[1] - local_times[0]))


def _repeated_sections(boundaries: list[float], features: np.ndarray, step: float) -> list[dict[str, float | int]]:
    fingerprints = {}
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        duration = end - start
        if duration < 5.0:
            continue
        frame_start = int(round(start / step))
        frame_end = min(int(round(end / step)), len(features))
        samples = _sample_interval(features, frame_start, frame_end, 24)
        fingerprints[index] = duration, samples

    matches: list[dict[str, float | int]] = []
    for index, (duration, samples) in fingerprints.items():
        for other, (other_duration, other_samples) in fingerprints.items():
            if other <= index or min(duration, other_duration) / max(duration, other_duration) < 0.65:
                continue
            texture = _cosine_similarity(samples.mean(axis=0), other_samples.mean(axis=0))
            similarity = 0.65 * _cosine_similarity(samples, other_samples) + 0.35 * texture
            if similarity >= 0.67:
                matches.append({'first_index': index, 'second_index': other, 'similarity': round(similarity, 3)})
    return sorted(matches, key=lambda match: -match['similarity'])


def discover_sections(
    y: np.ndarray, sr: int, *, chroma: np.ndarray | None = None,
    chroma_times: np.ndarray | None = None, downbeat_times: np.ndarray | None = None,
    use_phrase_grid: bool = True,
) -> SectionAnalysis:
    """Find arrangement boundaries, a leading audio onset and repeated sections.

    Input is a mono waveform. Times are measured from its first sample.
    The first and last boundaries are 0 and duration. Chroma, when supplied,
    has shape (12, frames) and frame-center times in seconds. Optional
    downbeats are seconds from a trusted metrical beat grid. Set
    use_phrase_grid=False to require audio evidence for candidate discovery
    while still using downbeats during timestamp refinement.
    """
    y = np.asarray(y, dtype=np.float32)
    if y.ndim != 1 or sr <= 0:
        raise ValueError('Section analysis expects a mono waveform and a positive sample rate.')
    if y.size == 0:
        return SectionAnalysis([0.0], None, [1.0], [])
    duration = len(y) / sr
    block_size = max(1, round(sr * 0.03))
    waveform_blocks = np.pad(y, (0, -len(y) % block_size)).reshape(-1, block_size)
    block_power = np.einsum('ij,ij->i', waveform_blocks, waveform_blocks) / block_size
    reference = float(np.sqrt(np.quantile(block_power, 0.90)))
    onset = _first_audio_onset(y, sr, reference)
    if duration < 7.0 or onset is None:
        return SectionAnalysis([0.0, duration], onset, [1.0, 1.0], [])

    if chroma is not None and chroma_times is not None:
        chroma = np.asarray(chroma)
        chroma_times = np.asarray(chroma_times)
        if chroma.ndim != 2 or chroma.shape != (12, chroma_times.size) or chroma_times.ndim != 1:
            raise ValueError('Chroma must contain 12 pitch rows and one time per column.')
    supplied_chroma = chroma is not None and chroma_times is not None and chroma_times.size > 0
    times, mel, derived_chroma = _spectral_features(y, sr, not supplied_chroma)
    step = 0.10
    centers = np.arange(0.0, duration, step)
    mel_floor = max(float(np.max(mel)) * 1e-7, 1e-14)
    smoothed = gaussian_filter1d(mel, 0.055 / (times[1] - times[0]), axis=0)
    loudness = 10.0 * np.log10(np.maximum(smoothed.mean(axis=1), mel_floor))
    log_mel = 10.0 * np.log10(np.maximum(smoothed, mel_floor))
    frame_spectrum = log_mel - log_mel.mean(axis=1, keepdims=True)
    flux = np.maximum(np.diff(log_mel, axis=0, prepend=log_mel[:1]), 0).mean(axis=1)
    fine_features = np.column_stack((loudness, frame_spectrum[:, ::2]))
    fine_features = _scale_features(fine_features, 2.5)
    center_frames = np.minimum(np.round(centers / (times[1] - times[0])).astype(int), len(times) - 1)
    coarse_loudness = _scale_features(gaussian_filter1d(loudness, 12)[center_frames, None], 3.0)
    coarse_spectrum = _scale_features(gaussian_filter1d(frame_spectrum, 12, axis=0)[center_frames], 3.0)
    coarse_flux = _scale_features(gaussian_filter1d(flux, 15)[center_frames, None], 0.35)
    if supplied_chroma:
        harmonic = np.column_stack([np.interp(centers, chroma_times, pitch) for pitch in chroma])
    else:
        harmonic = derived_chroma[center_frames]
    harmonic = harmonic / np.maximum(harmonic.sum(axis=1, keepdims=True), 1e-9)
    harmonic = _scale_features(gaussian_filter1d(harmonic, 2, axis=0), 0.065)

    band_log = np.column_stack([
        10.0 * np.log10(np.maximum(mel[:, start:end].mean(axis=1), mel_floor))
        for start, end in ((0, 5), (5, 13), (13, 26), (26, 40))
    ])
    band_levels = gaussian_filter1d(band_log, 12, axis=0)[center_frames]
    coarse_bands = _scale_features(band_levels, 3.0)
    groups = (coarse_loudness, coarse_spectrum, coarse_bands, coarse_flux, harmonic)
    weights = (0.15, 0.34, 0.20, 0.10, 0.21)
    fingerprints = np.column_stack([features * np.sqrt(weight / features.shape[1]) for features, weight in zip(groups, weights)])
    novelty = np.zeros(len(centers))
    for context, scale_weight in ((0.60, 0.22), (1.35, 0.32), (2.60, 0.28), (4.60, 0.18)):
        for group, group_weight in zip(groups, weights):
            change = _mean_change(group, round(context / step))
            novelty += scale_weight * group_weight * _normalized_novelty(change)
    novelty = gaussian_filter1d(novelty, 0.70)
    peaks, properties = find_peaks(novelty, distance=round(1.1 / step), prominence=0.10)
    downbeats = np.asarray(downbeat_times if downbeat_times is not None else [], dtype=float).ravel()
    downbeats = np.unique(downbeats[np.isfinite(downbeats) & (downbeats >= 0.0) & (downbeats < centers[-1])])
    phrase_grid = downbeats if use_phrase_grid else downbeats[:0]
    phrase_points, phrase_scores, on_phrase = _phrase_candidates(fingerprints, novelty, phrase_grid, step)
    if phrase_points.size:
        # Require a change that persists across several seconds. Chroma alone
        # can flag an ordinary chord change and subdivide an otherwise steady verse.
        context = round(3.8 / step)
        level_change = _mean_change(coarse_loudness, context)[phrase_points]
        texture_change = _mean_change(coarse_spectrum, context)[phrase_points]
        bands_change = _mean_change(coarse_bands, context)[phrase_points]
        bar_transition = (texture_change >= 0.29) & ((level_change >= 0.33) | (bands_change >= 0.26))
        band_transition = (bands_change >= 0.31) & (level_change >= 0.04)
        keep = bar_transition | band_transition
        phrase_points = phrase_points[keep]
        phrase_scores = phrase_scores[keep]
        on_phrase = on_phrase[keep]
    candidates = np.union1d(peaks, phrase_points)
    repeat_bonus = _repeat_candidates(fingerprints, candidates, step)
    strength = novelty[candidates] + repeat_bonus
    strength[np.searchsorted(candidates, peaks)] += 0.24 * properties['prominences']
    # Compare unusual changes to their local context so a sparse intro or a
    # burst of edits near the end cannot set the threshold for the whole song.
    local_baseline = gaussian_filter1d(novelty, 12.0 / step)
    strength += np.clip(0.65 * (novelty[candidates] - local_baseline[candidates] - 0.22), 0.0, 0.24)
    if phrase_points.size:
        phrase_indices = np.searchsorted(candidates, phrase_points)
        strength[phrase_indices] = np.maximum(strength[phrase_indices], phrase_scores + 0.60 * repeat_bonus[phrase_indices])
        if np.any(on_phrase):
            phase_times = phrase_points[on_phrase] * step
            candidate_times = candidates * step
            index = np.searchsorted(phase_times, candidate_times)
            nearest_after = phase_times[np.minimum(index, len(phase_times) - 1)]
            nearest_before = phase_times[np.maximum(index - 1, 0)]
            proximity = np.minimum(abs(nearest_after - candidate_times), abs(nearest_before - candidate_times))
            strength += 0.23 * np.exp(-0.5 * (proximity / 0.48) ** 2)
    cutoff = max(0.46, float(np.quantile(strength, 0.39))) if strength.size else 0.46
    minimum_gap = min(12.0, max(7.0, duration * 0.045))
    if np.any(on_phrase) and downbeats.size >= 2:
        minimum_gap = min(minimum_gap, float(np.median(np.diff(downbeats))) * 3.7)
    ending_gap = min(9.0, duration * 0.14)
    outro_boundaries = _outro_boundaries(band_levels, candidates, step, strength)
    selected: list[tuple[float, float]] = []
    for outro in outro_boundaries:
        outro_strength = strength[np.searchsorted(candidates, outro)]
        selected.append((float(centers[outro]), round(float(np.clip(outro_strength / 1.45, 0.0, 1.0)), 3)))
    for index in np.argsort(strength)[::-1]:
        point = candidates[index]
        time = float(centers[point])
        if strength[index] < cutoff:
            break
        if outro_boundaries and point >= outro_boundaries[0]:
            continue
        if time - onset >= min(5.0, minimum_gap) and duration - time >= ending_gap and all(abs(time - other) >= minimum_gap for other, _ in selected):
            selected.append((time, round(float(np.clip(strength[index] / 1.45, 0.0, 1.0)), 3)))

    fine_step = times[1] - times[0]
    short_novelty = _normalized_novelty(_mean_change(fine_features, max(2, round(0.115 / fine_step))))
    transient = _normalized_novelty(gaussian_filter1d(flux, 0.85))
    boundaries = [0.0]
    boundary_strengths = [1.0]
    for estimate, amount in sorted(selected):
        boundaries.append(_refine_boundary(estimate, times, short_novelty, transient, downbeats))
        boundary_strengths.append(amount)
    boundaries.append(duration)
    boundary_strengths.append(1.0)
    repeated_sections = _repeated_sections(boundaries, fingerprints, step)
    return SectionAnalysis(boundaries, onset, boundary_strengths, repeated_sections)
