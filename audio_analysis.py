"""Offline audio marker analysis engine.

The engine intentionally uses interpretable DSP features so it can run without
an internet connection or a cloud API.  Section names are feature-based
estimates and are exported together with confidence values.
"""

from __future__ import annotations

import csv
import json
import math
import re
import struct
import subprocess
from pathlib import Path
from typing import Callable

import librosa
import numpy as np
import soundfile
from imageio_ffmpeg import get_ffmpeg_exe
from scipy.ndimage import median_filter

from instrument_analysis import analyze_instruments_and_style
from local_style import combine_local_style
from phrase_analysis import estimate_phrase_end, refine_phrase_grid
from reference_annotations import apply_reference, find_reference
from rhythm_analysis import analyze_rhythm
from section_analysis import discover_sections


ProgressCallback = Callable[[int, str], None]

PITCH_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
KEY_NAMES = tuple(
    [f"{name} major" for name in PITCH_NAMES]
    + [f"{name} minor" for name in PITCH_NAMES]
)

SECTION_COLUMNS = ("Timestamp (s)", "Section", "BPM", "Key / Mode", "Time Signature", "Lead Instruments", "Accompaniment")

CHORD_TEMPLATES = {
    "": (0, 4, 7),
    "m": (0, 3, 7),
    "7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "m7": (0, 3, 7, 10),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "dim": (0, 3, 6),
    "5": (0, 7),
}


def _report(progress: ProgressCallback | None, value: int, message: str) -> None:
    if progress:
        progress(value, message)


def timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes:02d}:{remainder:04.1f}"


def _read_flp_bpm(audio_path: Path) -> int | None:
    """Read a project BPM when the FLP exposes an explicit decimal value.

    FLP versions vary in their top-level binary layout. Known project tempo
    fields are read only from a project with the same stem as the audio.
    """
    project = audio_path.with_suffix(".flp")
    if not project.is_file():
        return None
    try:
        raw = project.read_bytes()
    except OSError:
        return None
    text = raw.decode("latin1", errors="ignore")
    values = []
    for match in re.finditer(r'TEMPO="([0-9a-fA-F]{8})"', text, flags=re.IGNORECASE):
        try:
            values.append(struct.unpack(">f", bytes.fromhex(match.group(1)))[0])
        except (ValueError, struct.error):
            continue
    values = [value for value in values if 30.0 <= value <= 300.0]
    # FL Studio stores the project tempo as a U32 event with id 156. The
    # event stream starts after the 22-byte FLP header. Its value is BPM*1000.
    # Reading this single event is stable across FLP versions and avoids
    # mistaking plugin parameters for tempo metadata.
    offset = 22
    try:
        while offset < len(raw):
            event_id = raw[offset]
            offset += 1
            if event_id < 64:
                size = 1
            elif event_id < 128:
                size = 2
            elif event_id < 192:
                size = 4
            else:
                # Variable-length strings and data are encoded with a 7-bit
                # continuation length prefix.
                length = 0
                shift = 0
                while offset < len(raw):
                    byte = raw[offset]
                    offset += 1
                    length |= (byte & 0x7F) << shift
                    if byte < 0x80:
                        break
                    shift += 7
                    if shift > 35:
                        return None
                size = length
            value = raw[offset:offset + size]
            offset += size
            if event_id == 156 and len(value) == 4:
                tempo = int.from_bytes(value, "little") / 1000.0
                if 30.0 <= tempo <= 300.0:
                    return int(round(tempo))
    except (IndexError, ValueError):
        pass
    if values:
        return int(round(float(np.median(values))))
    return None


def _normalize_chroma(chroma: np.ndarray) -> np.ndarray:
    chroma = np.maximum(chroma, 0.0)
    norm = np.sum(chroma, axis=0, keepdims=True)
    return chroma / np.maximum(norm, 1e-9)


def _key_scores(chroma_vector: np.ndarray) -> list[tuple[str, float, int]]:
    """Correlate with mean-centered profiles to discount a flat chroma floor."""
    vector = np.asarray(chroma_vector, dtype=float)
    vector = vector - vector.mean()
    vector /= max(float(np.linalg.norm(vector)), 1e-9)
    candidates = []
    for offset, profile in ((0, MAJOR_PROFILE), (12, MINOR_PROFILE)):
        centered = profile - profile.mean()
        centered /= np.linalg.norm(centered)
        for pitch in range(12):
            score = float(vector @ np.roll(centered, pitch))
            candidates.append((KEY_NAMES[offset + pitch], score, offset + pitch))
    return sorted(candidates, key=lambda item: item[1], reverse=True)


def _key_at(chroma: np.ndarray, times: np.ndarray, start: float, end: float) -> tuple[str, float]:
    mask = (times >= start) & (times < end)
    if not np.any(mask):
        return "Undetermined", 0.0
    profile = chroma[:, mask].mean(axis=1)
    best, second = _key_scores(profile)[:2]
    if best[1] < 0.42 or (np.max(profile) - np.min(profile)) < 0.028:
        return "Undetermined", 0.0
    separation = min(1.0, (best[1] - second[1]) / 0.20)
    return best[0], float(np.clip(best[1] * (0.72 + 0.28 * separation), 0, 1))


def _detect_key_segments(
    chroma: np.ndarray,
    melodic_chroma: np.ndarray,
    times: np.ndarray,
    boundaries: list[float],
    global_key: str,
    global_score: float,
) -> list[dict]:
    """Require sustained evidence before changing the active tonal center."""
    local = []
    current_key = global_key
    current_score = global_score
    for start, end in zip(boundaries, boundaries[1:]):
        evidence_end = min(times[-1], start + 32)
        centers = np.arange(start + 8, evidence_end - 5, 4)
        windows = []
        for center in centers:
            mask = (times >= center - 8) & (times < center + 8)
            all_scores = {key: score for key, score, _ in _key_scores(chroma[:, mask].mean(axis=1))}
            melody_scores = {key: score for key, score, _ in _key_scores(melodic_chroma[:, mask].mean(axis=1))}
            combined = {key: 0.32 * score + 0.68 * melody_scores[key] for key, score in all_scores.items()}
            candidate = max(combined, key=combined.get)
            if combined[candidate] >= 0.55 and combined[candidate] - combined.get(current_key, 0) >= 0.075:
                windows.append((candidate, combined[candidate], all_scores[candidate]))
        if end - start >= 14 and windows and (local or current_key == "Undetermined"):
            candidate = max(KEY_NAMES, key=lambda key: sum(key == item[0] for item in windows))
            votes = [item for item in windows if item[0] == candidate]
            mode = candidate.split()[1]
            same_mode = current_key.endswith(mode)
            average = float(np.mean([item[1] for item in votes]))
            full_range = float(np.mean([item[2] for item in votes]))
            tonic, _, previous_mode = current_key.partition(" ")
            related_major = previous_mode == mode == "major" and (
                (PITCH_NAMES.index(candidate.split()[0]) - PITCH_NAMES.index(tonic)) % 12 in (5, 7)
            )
            sustained = len(votes) >= max(2, math.ceil(len(centers) * 0.66))
            trusted = full_range >= 0.55 and average >= 0.64
            strong_same_mode = same_mode and average >= 0.72 and len(votes) >= 4
            confirmed_modulation = not related_major or end - start >= 60
            if sustained and (trusted or strong_same_mode) and confirmed_modulation:
                current_key, current_score = candidate, average
        local.append({"start": round(start, 3), "end": round(end, 3),
                      "key": current_key, "confidence": round(current_score, 3)})

    segments = []
    for item in local:
        if segments and segments[-1]["key"] == item["key"]:
            previous = segments[-1]
            a = previous["end"] - previous["start"]
            b = item["end"] - item["start"]
            previous["confidence"] = round((a * previous["confidence"] + b * item["confidence"]) / max(a + b, 1e-9), 3)
            previous["end"] = item["end"]
        else:
            segments.append(item.copy())
    return segments


def _chord_match(chroma_vector: np.ndarray) -> tuple[str, float, int]:
    vector = np.maximum(np.asarray(chroma_vector, dtype=float), 0.0)
    vector = vector / max(float(np.linalg.norm(vector)), 1e-9)
    candidates: list[tuple[str, float, int]] = []
    for root in range(12):
        for suffix, intervals in CHORD_TEMPLATES.items():
            template = np.zeros(12, dtype=float)
            template[[(root + interval) % 12 for interval in intervals]] = 1.0
            template /= np.linalg.norm(template)
            candidates.append((f"{PITCH_NAMES[root]}{suffix}", float(np.dot(vector, template)), root))
    return max(candidates, key=lambda item: item[1])


def _chord_progression(
    chroma: np.ndarray, times: np.ndarray, duration: float, bar_period: float | None, phase: float, key: str,
) -> dict:
    step = float(np.clip(bar_period or 2.0, 0.8, 5.0))
    starts = np.r_[0.0, np.arange(phase % step or step, duration, step)]
    collapsed = []
    for start, end in zip(starts, np.r_[starts[1:], duration]):
        mask = (times >= start + min(0.08, (end - start) / 4)) & (times < end)
        if not np.any(mask):
            continue
        chord, score, _ = _chord_match(chroma[:, mask].mean(axis=1))
        if score < 0.50:
            continue
        if collapsed and collapsed[-1]["chord"] == chord:
            collapsed[-1]["end"] = round(float(end), 3)
        else:
            collapsed.append({
                "start": round(float(start), 3), "end": round(float(end), 3),
                "chord": chord, "confidence": round(score, 3),
            })
    names = [item["chord"] for item in collapsed]
    summary = "–".join(names[:16]) + ("–…" if len(names) > 16 else "") if names else "No stable chord progression detected"
    functional = ""
    if key in KEY_NAMES and names:
        index = KEY_NAMES.index(key)
        semitones = ("I", "♭II", "II", "♭III", "III", "IV", "♯IV", "V", "♭VI", "VI", "♭VII", "VII")
        degrees = []
        for chord in names[:16]:
            root = re.match(r"[A-G][#b]?", chord).group()
            degree = semitones[(PITCH_NAMES.index(root) - index % 12) % 12]
            suffix = chord[len(root):]
            degrees.append(degree.lower() if suffix in ("m", "m7", "dim") else degree)
        functional = f"{key} · " + "–".join(degrees)
    return {"summary": summary, "functional_summary": functional, "segments": collapsed}


def _scene_mood(style: str, key: str, bpm: float, energy: float, centroid: float) -> dict:
    minor = key.endswith("minor")
    if bpm >= 125 and energy >= 0.16:
        scene = ["Nightclub", "Sports", "City at night"]
        mood = ["Excited", "Tense", "Intense"] if minor else ["Excited", "Bright", "Energetic"]
    elif minor:
        scene = ["Night", "Rain", "Introspective"]
        mood = ["Melancholic", "Mysterious", "Restrained"]
    else:
        scene = ["Everyday life", "Bright spaces", "Travel"]
        mood = ["Warm", "Bright", "Romantic"]
    if centroid > 3000:
        scene.append("Science fiction or neon lighting")
    return {"scene_tags": scene, "scene_description": ", ".join(scene), "mood_tags": mood, "mood_description": ", ".join(mood)}


def _infer_sections(analysis, rhythm: dict, instruments: list[dict], times: np.ndarray, rms: np.ndarray, vocal_cues=None) -> list[dict]:
    boundaries = analysis.boundaries
    ranges = list(zip(boundaries, boundaries[1:]))
    energies = []
    for start, end in ranges:
        values = rms[(times >= start) & (times < end)]
        energies.append(float(np.median(values)) if values.size else 0.0)
    low, high = np.quantile(energies, [0.20, 0.75])
    span = max(high - low, 0.01)
    peak = max(energies)
    repeated = {}
    for pair in analysis.repeated_sections:
        first, second = pair["first_index"], pair["second_index"]
        similarity = pair["similarity"]
        repeated[first] = max(repeated.get(first, 0), similarity)
        repeated[second] = max(repeated.get(second, 0), similarity)

    sections = []
    counts = {}
    for index, (start, end) in enumerate(ranges):
        energy = energies[index]
        previous = energies[index - 1] if index else energy
        next_energy = energies[index + 1] if index + 1 < len(energies) else energy
        timbre = instruments[index]
        front = timbre["main_instruments"].lower()
        singing = any(word in front for word in ("vocal", "rap", "singing"))
        vocal_share = None
        if vocal_cues is not None:
            within = (vocal_cues.times >= start) & (vocal_cues.times < end)
            if np.any(within):
                vocal_share = float(np.mean(vocal_cues.strength[within] >= 0.50))
                if vocal_share >= 0.38 and front in ("—", "electronic textures"):
                    singing = True
                    timbre = {**timbre, "main_instruments": "Vocal",
                              "instrument_note": "Separated vocal activity supports the lead assignment."}
        choral = "choir" in front
        electronic = any(word in front for word in ("electronic", "synth"))
        recurrence = repeated.get(index, 0)
        active = energy >= low + span * 0.65 or (peak > 0 and energy / peak >= 0.78)
        if index == 0:
            label = "intro" if len(ranges) > 1 else "section"
            note = "First audible musical or vocal material."
        elif index == len(ranges) - 1 and (energy < previous * 0.88 or not singing):
            label, note = "outro", "Main material fades or gives way to the ending."
        elif electronic and active and (energy >= 1.25 * previous or recurrence >= 0.5):
            label, note = "drop", "Electronic textures intensify or a high-energy passage returns."
        elif singing and active and energy > 1.27 * max(previous, 0.005) and end - start >= 12:
            label, note = "chorus", "Sustained vocal material enters with a thicker arrangement."
        elif recurrence >= 0.45 and active:
            label = "chorus" if singing else "hook"
            note = "Similar textures and motifs recur elsewhere in the track."
        elif choral and active:
            label, note = "theme", "Choral textures are prominent in the mix."
        elif next_energy > 1.35 * max(energy, 0.005) and energy > previous * 1.08:
            label, note = "build-up", "Energy rises ahead of a stronger section."
        elif singing:
            label, note = "verse", "Vocals appear to lead the mix."
        elif energy < low + span * 0.20 or energy < previous * 0.68:
            label, note = "interlude", "Instrumentation thins or energy drops."
        elif active:
            label, note = "theme", "A prominent motif or thicker arrangement enters."
        else:
            label, note = "section", "Texture, rhythm or harmony changes."
        counts[label] = counts.get(label, 0) + 1
        if label in {"verse", "chorus", "drop", "hook", "theme"}:
            label = f"{label} {counts[label]}"
        marker = analysis.first_audio_onset if index == 0 else start
        if marker is None or marker >= end:
            marker = start
        sections.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "marker_seconds": round(marker, 3),
            "label": label,
            "bpm": _display_bpm(rhythm["bpm"]),
            "time_signature": rhythm["time_signature"],
            "energy": round(energy, 4),
            "confidence": round(float(analysis.boundary_strengths[index]), 3),
            "recurrence": round(float(recurrence), 3),
            "vocal_activity": round(vocal_share, 3) if vocal_share is not None else None,
            "note": note,
            **timbre,
        })
    return sections


def _display_bpm(bpm: float | None) -> float | None:
    if bpm is None:
        return None
    rounded = round(bpm)
    return rounded if abs(bpm - rounded) < 0.03 else round(bpm, 2)


def _load_audio(path: Path, sample_rate: int) -> np.ndarray:
    try:
        audio, _ = librosa.load(path, sr=sample_rate, mono=False)
        return audio
    except soundfile.LibsndfileError:
        try:
            ffmpeg = get_ffmpeg_exe()
        except RuntimeError:
            raise ValueError(
                "此文件需要 FFmpeg 解码。请安装 FFmpeg 并添加到 PATH，或设置 IMAGEIO_FFMPEG_EXE。"
            ) from None
        command = [
            ffmpeg, "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0",
            "-ac", "2", "-ar", str(sample_rate), "-f", "f32le", "pipe:1",
        ]
        result = subprocess.run(command, capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            raise ValueError(detail or f"Could not decode {path.name}.")
        return np.frombuffer(result.stdout, dtype="<f4").reshape(-1, 2).T.copy()


def _ending_marker(rms: np.ndarray, times: np.ndarray, duration: float, rhythm: dict) -> float:
    audible = np.flatnonzero(rms > max(float(np.quantile(rms, 0.90)) * 0.018, 1e-6))
    if not audible.size:
        return duration
    last_audio = min(duration, times[audible[-1]])
    if duration - last_audio < 0.70:
        return duration
    period = rhythm["bar_period"]
    phase = rhythm["downbeat_phase"]
    if period and phase is not None:
        downbeat = phase + round((last_audio - phase) / period) * period
        if abs(downbeat - last_audio) <= 0.30:
            last_audio = downbeat
    return float(last_audio)


def analyze_audio(
    path: str | Path, progress: ProgressCallback | None = None, *,
    use_vocals: bool = True, use_reference: bool = True,
) -> dict:
    path = Path(path)
    _report(progress, 5, "Reading audio…")
    sr = 22050
    audio = _load_audio(path, sr)
    if audio.size == 0:
        raise ValueError("The audio file is empty.")
    y = librosa.to_mono(audio)
    if audio.ndim == 2 and np.max(np.abs(y)) < 0.08 * np.max(np.abs(audio)):
        y = max(audio, key=lambda channel: np.linalg.norm(channel))
    duration = len(y) / sr
    hop = 512
    _report(progress, 18, "Estimating BPM, time signature and bar grid…")
    project_bpm = _read_flp_bpm(path)
    rhythm = analyze_rhythm(y, sr, bpm_override=project_bpm)
    downbeats = None
    if rhythm["grid_confidence"] >= 0.54 and rhythm["bar_period"] and rhythm["downbeat_phase"] is not None:
        downbeats = np.arange(rhythm["downbeat_phase"] % rhythm["bar_period"], duration, rhythm["bar_period"])

    _report(progress, 38, "Extracting harmonic and timbral features…")
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    times = librosa.times_like(rms, sr=sr, hop_length=hop)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    if len(y) >= sr:
        harmonic = librosa.effects.harmonic(y)
        fmin = librosa.note_to_hz("C1")
        tuning = librosa.estimate_tuning(y=harmonic, sr=sr) if np.any(y) else 0.0
        cqt = np.abs(librosa.cqt(y=harmonic, sr=sr, hop_length=hop, fmin=fmin,
                                n_bins=7 * 36, bins_per_octave=36, tuning=tuning))
        raw_chroma = librosa.feature.chroma_cqt(C=cqt, sr=sr, hop_length=hop, fmin=fmin)
        melodic_octaves = cqt.reshape(7, 36, -1)[3:, ::3]
        melodic_raw = np.linalg.norm(melodic_octaves, axis=0)
        melodic_chroma = _normalize_chroma(melodic_raw)
    else:
        raw_chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop)
        melodic_chroma = _normalize_chroma(raw_chroma)
    chroma = _normalize_chroma(median_filter(raw_chroma, size=(1, 5), mode="nearest"))
    active = rms[:chroma.shape[1]] > max(float(np.max(rms)) * 0.014, 1e-5)
    chroma[:, np.flatnonzero(~active)] = 0
    melodic_chroma[:, np.flatnonzero(~active)] = 0
    chroma_times = librosa.times_like(chroma, sr=sr, hop_length=hop)
    global_key, global_key_score = _key_at(chroma, chroma_times, 0, duration)

    vocals = None
    vocal_analysis = "disabled"
    if use_vocals and duration >= 5 and np.any(y):
        from vocal_separation import describe_vocals, separate_vocals

        _report(progress, 48, "Separating vocals locally…")
        separated = separate_vocals(audio, sr, progress=lambda fraction: _report(
            progress, 48 + int(fraction * 8), "Separating vocals locally…",
        ))
        if separated is not None:
            vocals = describe_vocals(separated.vocals, separated.accompaniment, sr)
            vocal_analysis = "separated"
            del separated
        else:
            vocal_analysis = "unavailable"
    elif use_vocals:
        vocal_analysis = "not_applicable"

    _report(progress, 57, "Matching repeated sections and locating onsets…")
    structure = discover_sections(
        y, sr, chroma=chroma, chroma_times=chroma_times, downbeat_times=downbeats,
    )
    original_ranges = list(zip(structure.boundaries, structure.boundaries[1:]))
    refined, strengths = refine_phrase_grid(
        structure.boundaries, structure.boundary_strengths,
        structure.first_audio_onset, duration, rhythm, evidence_audio=y, sr=sr,
    )
    if refined is not structure.boundaries:
        mapping = []
        for left, right in zip(refined, refined[1:]):
            mapping.append(max(range(len(original_ranges)), key=lambda index: max(
                0.0, min(right, original_ranges[index][1]) - max(left, original_ranges[index][0]),
            )))
        recurrence = {(pair["first_index"], pair["second_index"]): pair["similarity"]
                      for pair in structure.repeated_sections}
        structure.repeated_sections = [
            {"first_index": left, "second_index": right, "similarity": similarity}
            for left, first in enumerate(mapping)
            for right, second in enumerate(mapping[left + 1:], start=left + 1)
            if (similarity := recurrence.get((first, second))) is not None
        ]
        structure.boundaries, structure.boundary_strengths = refined, strengths
    key_changes = _detect_key_segments(chroma, melodic_chroma, chroma_times, structure.boundaries, global_key, global_key_score)
    _report(progress, 73, "Estimating section instruments with the offline model…")
    ranges = list(zip(structure.boundaries, structure.boundaries[1:]))
    timbres, style_details = analyze_instruments_and_style(audio, sr, ranges)
    style_details = combine_local_style(audio, sr, style_details)
    sections = _infer_sections(structure, rhythm, timbres, times, rms, vocals)
    for section in sections:
        section["key"] = next((
            item["key"] for item in key_changes if item["start"] <= section["start"] < item["end"]
        ), global_key)
    if structure.first_audio_onset is None:
        sections = []
    elif duration >= 8:
        phrase_end = estimate_phrase_end(y, sr, duration, sections[-1]["marker_seconds"], rhythm)
        final_audio = min(phrase_end, _ending_marker(rms, times, duration, rhythm))
        ending = max(sections[-1]["marker_seconds"] + 0.001, final_audio)
        if ending <= duration:
            sections[-1]["end"] = round(ending, 3)
            sections.append({
                "start": round(ending, 3), "end": round(duration, 3), "marker_seconds": round(ending, 3),
                "label": "end", "bpm": None, "key": None, "time_signature": None,
                "main_instruments": None, "accompaniment_instruments": None, "note": "End of audible material.",
            })

    chord_data = _chord_progression(
        chroma, chroma_times, duration, rhythm["bar_period"], rhythm["downbeat_phase"] or 0.0, global_key,
    )
    energy = float(np.mean(rms))
    bpm = _display_bpm(rhythm["bpm"])
    alternate = None
    if bpm is not None:
        alternate = round(bpm * 2 if bpm < 100 else bpm / 2, 2)
    names = list(dict.fromkeys(
        name.strip()
        for item in timbres
        for part in ("main_instruments", "accompaniment_instruments")
        for name in item[part].split(",")
        if name.strip() and name.strip() != "—"
    ))
    instrument_description = ", ".join(names) or "Undetected"
    atmosphere = _scene_mood(style_details["style"], global_key, bpm or 0, energy, float(np.mean(centroid)))
    _report(progress, 88, "Formatting point markers…")
    result = {
        "file": str(path),
        "name": path.name,
        "duration": round(duration, 3),
        "duration_text": timestamp(duration),
        **rhythm,
        "bpm": bpm,
        "alternate_bpm": alternate,
        "project_bpm": project_bpm,
        "annotation_source": "automatic",
        "vocal_analysis": vocal_analysis,
        "global_key": global_key,
        "global_key_confidence": round(global_key_score, 3),
        "key_changes": key_changes,
        **style_details,
        "chord_progression": chord_data["summary"],
        "functional_chord_progression": chord_data["functional_summary"],
        "chord_segments": chord_data["segments"],
        "instruments": names,
        "instrument_description": instrument_description,
        **atmosphere,
        "sections": sections,
        "analysis_notes": [
            "Timestamps show three decimal places in seconds. The intro marker follows the first audible onset. Later markers estimate arrangement changes.",
            "Each timestamp is a point marker. JSON stores the section interval in start and end, and the point in marker_seconds.",
            "A same-name FL Studio project can supply the tempo. Instruments, section names, time signatures and local keys are estimates from the mix.",
        ],
    }
    if use_reference and (reference := find_reference(path)):
        apply_reference(result, *reference)
    _report(progress, 100, "Analysis complete")
    return result


def section_row(result: dict, item: dict) -> list[str]:
    bpm = item.get("bpm", result.get("bpm"))
    return [
        f"{item.get('marker_seconds', item['start']):.3f}",
        item["label"],
        f"{bpm:g}" if bpm is not None else "—",
        item.get("key", result.get("global_key", "Undetermined")) or "—",
        item.get("time_signature", result.get("time_signature")) or "—",
        item.get("main_instruments") or "—",
        item.get("accompaniment_instruments") or "—",
    ]


def export_result(result: dict, output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(result["name"]).stem
    json_path = output_dir / f"{stem}_analysis.json"
    md_path = output_dir / f"{stem}_markers.md"
    csv_path = output_dir / f"{stem}_sections.csv"
    labels_path = output_dir / f"{stem}_audacity_labels.txt"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = result["sections"]
    key_confidence = result["global_key_confidence"]
    key_detail = f" (confidence {key_confidence})" if key_confidence is not None else " (reference)"
    style_confidence = result.get("style_confidence")
    style_detail = f" ({style_confidence.lower()} confidence)" if style_confidence in {"High", "Medium", "Low"} else ""
    lines = [
        f"# {result['name']} Audio Markers",
        "",
        f"- Duration · {result['duration_text']}",
        f"- BPM · {result['bpm'] if result['bpm'] is not None else 'Undetermined'}",
        f"- Time signature · {result.get('time_signature') or 'Undetermined'}",
        f"- Key · {result['global_key']}{key_detail}",
        f"- Annotation source · {result.get('annotation_source', 'automatic').title()}",
        f"- Vocal analysis · {result.get('vocal_analysis', 'disabled').replace('_', ' ').title()}",
        f"- Style · {result['style']}{style_detail}",
        f"- Style source · {'Locally trained classifier' if result.get('style_source') == 'local_classifier' else 'YAMNet and audio rules'}",
        f"- Chords · {result.get('chord_progression', 'Undetermined')}",
        f"- Roman numerals · {result.get('functional_chord_progression', 'Undetermined')}",
        f"- Instruments · {result.get('instrument_description', 'Undetermined')}",
        f"- Scene · {result.get('scene_description', 'Undetermined')}",
        f"- Mood · {result.get('mood_description', 'Undetermined')}",
        "",
        "## Key Changes",
        "",
        "| Timestamp (s) | Key | Confidence |",
        "|---|---|---|",
    ]
    for item in result["key_changes"]:
        confidence = item["confidence"] if item["confidence"] is not None else "—"
        lines.append(f"| {item['start']:.3f} | {item['key']} | {confidence} |")
    lines += ["", "## Sections", "", "Each row gives the section start in seconds.", "",
              "| " + " | ".join(SECTION_COLUMNS) + " |", "|---|---|---:|---|---|---|---|"]
    for item in rows:
        cells = (str(cell).replace("|", "\\|").replace("\n", " ") for cell in section_row(result, item))
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Chord Progression", "", f"{result.get('chord_progression', 'Undetermined')}.",
              f"Roman numerals · {result.get('functional_chord_progression') or 'Undetermined'}.",
              "", "## Analysis Notes", "", *(f"- {note}" for note in result.get("analysis_notes", []))]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(SECTION_COLUMNS)
        writer.writerows(section_row(result, item) for item in rows)
    point_labels = []
    for item in rows:
        marker = section_row(result, item)[0]
        point_labels.append(f"{marker}\t{marker}\t{item['label']}\n")
    labels_path.write_text("".join(point_labels), encoding="utf-8")
    return {"json": json_path, "markdown": md_path, "csv": csv_path, "labels": labels_path}
