"""Deterministic acoustic features shared by local genre training and inference."""

from __future__ import annotations

from functools import lru_cache
from math import gcd

import numpy as np
from scipy import fft, signal

FEATURE_VERSION = 1
SAMPLE_RATE = 16000
FFT_SIZE = 1024
HOP = 320
MEL_BANDS = 40


@lru_cache(maxsize=1)
def _filter_bank() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequencies = fft.rfftfreq(FFT_SIZE, 1 / SAMPLE_RATE)
    mel = lambda hz: 2595 * np.log10(1 + hz / 700)
    edges = 700 * (10 ** (np.linspace(mel(30), mel(7800), MEL_BANDS + 2) / 2595) - 1)
    rising = (frequencies[:, None] - edges[:-2]) / (edges[1:-1] - edges[:-2])
    falling = (edges[2:] - frequencies[:, None]) / (edges[2:] - edges[1:-1])
    filters = np.maximum(0, np.minimum(rising, falling)).astype(np.float32)
    filters /= np.maximum(filters.sum(axis=0), 1e-8)
    pitches = np.rint(69 + 12 * np.log2(np.maximum(frequencies, 1) / 440)).astype(int) % 12
    chroma = np.eye(12, dtype=np.float32)[pitches]
    chroma[(frequencies < 70) | (frequencies > 2200)] = 0
    return frequencies.astype(np.float32), filters, chroma


def _summary(frames: np.ndarray) -> np.ndarray:
    return np.concatenate((frames.mean(axis=0), frames.std(axis=0),
                           np.quantile(frames, (0.10, 0.50, 0.90), axis=0).ravel()))


def _audio_excerpt(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        mixed = audio.mean(axis=0)
        if np.max(np.abs(mixed), initial=0) < 0.08 * np.max(np.abs(audio), initial=0):
            mixed = audio[np.argmax(np.max(np.abs(audio), axis=1))]
        audio = mixed
    if audio.ndim != 1 or sample_rate <= 0 or audio.size < sample_rate * 2:
        return np.array([], dtype=np.float32)
    audio = np.nan_to_num(audio, copy=True)
    length = 6 * sample_rate
    if len(audio) > 5 * length:
        starts = np.linspace(0, len(audio) - length, 5, dtype=int)
        audio = np.concatenate([audio[start:start + length] for start in starts])
    if sample_rate != SAMPLE_RATE:
        divisor = gcd(int(sample_rate), SAMPLE_RATE)
        audio = signal.resample_poly(audio, SAMPLE_RATE // divisor, int(sample_rate) // divisor)
    return audio


def extract_style_features(audio: np.ndarray, sample_rate: int) -> np.ndarray | None:
    """Summarize up to 30 seconds across a track in a fixed feature vector."""
    audio = _audio_excerpt(audio, sample_rate)
    if audio.size < FFT_SIZE or np.max(np.abs(audio)) < 1e-5:
        return None
    audio /= max(float(np.quantile(np.abs(audio), 0.99)), 1e-5)
    frames = np.lib.stride_tricks.sliding_window_view(audio, FFT_SIZE)[::HOP]
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    active = rms > max(float(np.quantile(rms, 0.90)) * 0.025, 0.0002)
    if active.sum() < 40:
        return None

    frequencies, filters, chroma_filter = _filter_bank()
    magnitude = np.abs(fft.rfft(frames * np.hanning(FFT_SIZE).astype(np.float32), axis=1))
    power = magnitude ** 2
    spectrum = power / np.maximum(power.sum(axis=1, keepdims=True), 1e-9)
    mel = np.log(np.maximum(power @ filters, 1e-7))
    mfcc = fft.dct(mel, type=2, norm="ortho", axis=1)[:, :16]
    chroma = spectrum @ chroma_filter
    chroma /= np.maximum(chroma.sum(axis=1, keepdims=True), 1e-9)
    centroid = spectrum @ frequencies
    spread = np.sqrt(np.maximum(spectrum @ frequencies ** 2 - centroid ** 2, 0))
    rolloff = frequencies[np.argmax(np.cumsum(spectrum, axis=1) >= 0.85, axis=1)]
    flatness = np.exp(np.log(np.maximum(spectrum, 1e-10)).mean(axis=1)) / np.maximum(spectrum.mean(axis=1), 1e-10)
    crossing = (np.signbit(frames[:, 1:]) != np.signbit(frames[:, :-1])).mean(axis=1)
    band_edges = (0, 90, 220, 500, 1200, 2600, 5200, SAMPLE_RATE)
    bands = np.column_stack([
        spectrum[:, (frequencies >= left) & (frequencies < right)].sum(axis=1)
        for left, right in zip(band_edges, band_edges[1:])
    ])
    local = np.column_stack((centroid / 8000, spread / 8000, rolloff / 8000,
                             flatness, crossing, np.log(np.maximum(rms, 1e-5)), bands))
    deltas = np.diff(mfcc, axis=0)
    stable = active[1:] & active[:-1]
    if not np.any(stable):
        return None

    flux = np.maximum(np.diff(mel, axis=0, prepend=mel[:1]), 0)
    envelopes = np.column_stack((flux[:, :10].mean(axis=1),
                                 flux[:, 10:24].mean(axis=1),
                                 flux[:, 24:].mean(axis=1)))
    envelopes -= envelopes.mean(axis=0, keepdims=True)
    transform = fft.rfft(envelopes, n=fft.next_fast_len(2 * len(envelopes)), axis=0)
    correlation = fft.irfft(np.abs(transform) ** 2, axis=0)
    correlation /= np.maximum(correlation[:1], 1e-8)
    lag = 60 * SAMPLE_RATE / (HOP * np.array([55, 65, 75, 85, 95, 105, 115, 130, 150, 175, 200]))
    pulse = np.column_stack([
        np.interp(lag, np.arange(len(correlation)), column) for column in correlation.T
    ])
    max_pitch = np.argmax(chroma[active].mean(axis=0))
    features = np.concatenate((
        _summary(mfcc[active]), _summary(deltas[stable, 1:12]), _summary(local[active]),
        _summary(np.roll(chroma[active], -max_pitch, axis=1)),
        pulse.ravel(), np.array([active.mean()]),
    )).astype(np.float32)
    return features if np.all(np.isfinite(features)) else None
