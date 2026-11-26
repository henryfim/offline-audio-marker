"""Track-level music style estimates from the bundled YAMNet scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Pool related AudioSet classes per window so a parent and its subgenres do
# not count as separate votes. Regional and mood tags are omitted here.
GENRES = {
    "Pop": ("Pop music",),
    "Independent Music": ("Independent music",),
    "Hip-Hop": ("Hip hop music", "Beatboxing"),
    "Rock": ("Rock music", "Heavy metal", "Punk rock", "Grunge", "Progressive rock", "Rock and roll", "Psychedelic rock"),
    "R&B / Soul": ("Rhythm and blues", "Soul music"),
    "Country / Bluegrass": ("Country", "Bluegrass"),
    "Folk": ("Folk music", "Traditional music"),
    "Funk / Disco": ("Funk", "Disco"),
    "Jazz / Swing": ("Jazz", "Swing music"),
    "Blues": ("Blues",),
    "Reggae / Ska": ("Reggae", "Ska"),
    "Latin": ("Music of Latin America", "Salsa music", "Flamenco"),
    "Afrobeat": ("Afrobeat",),
    "Gospel": ("Gospel music",),
    "Classical": ("Classical music", "Opera"),
    "Cinematic / Orchestral": ("Soundtrack music", "Theme music", "Video game music"),
    "Electronic": ("Electronic music", "Electronic dance music", "Electronica", "Dubstep", "House music", "Techno", "Drum and bass", "Trance music"),
    "Ambient / New Age": ("Ambient music", "New-age music"),
    "A Cappella": ("A capella",),
    "Children's Music": ("Music for children", "Lullaby"),
}

ELECTRONIC_SUBGENRES = {
    "Dubstep": "Dubstep",
    "House": "House music",
    "Techno": "Techno",
    "Drum and Bass": "Drum and bass",
    "Trance": "Trance music",
}

ORCHESTRAL_TIMBRES = (
    "Orchestra", "Bowed string instrument", "String section", "Cello", "Violin, fiddle",
    "Brass instrument", "French horn", "Trumpet", "Timpani",
)
INSTRUMENTAL_TIMBRES = (
    "Musical instrument", "Piano", "Electric piano", "Organ", "Guitar", "Bass guitar",
    "Acoustic guitar", "Electric guitar", "Harp", "Mallet percussion", *ORCHESTRAL_TIMBRES,
)
VOICE_LABELS = ("Vocal music", "Singing", "Choir", "Rapping", "A capella", "Synthetic singing", "Speech")


@dataclass(frozen=True)
class _Support:
    score: float
    coverage: float

    @classmethod
    def from_windows(cls, values: np.ndarray) -> _Support:
        return cls(
            score=float(0.65 * values.mean() + 0.35 * np.quantile(values, 0.75)),
            coverage=float(np.mean(values >= 0.02)),
        )

    @property
    def consistent(self) -> bool:
        return self.score >= 0.016 and self.coverage >= 0.15


def undetermined_style(note: str, evidence: dict | None = None) -> dict:
    return {
        "style": "Undetermined",
        "style_confidence": "Undetermined",
        "style_note": note,
        "style_evidence": evidence if evidence is not None else _evidence(0, 0, 0.0),
    }


def _evidence(windows: int, music_windows: int, music_score: float) -> dict:
    return {
        "model": "YAMNet",
        "windows": windows,
        "music_windows": music_windows,
        "music_score": round(music_score, 4),
        "candidates": [],
        "cues": [],
    }


def identify_style(scores: np.ndarray, labels: list[str]) -> dict:
    """Estimate a style from active, original-mix windows across the track."""
    indices = {label: idx for idx, label in enumerate(labels)}
    if not len(scores) or "Music" not in indices:
        return undetermined_style("The offline model returned no usable style evidence.")

    music = scores[:, indices["Music"]]
    music_mask = music >= 0.35
    count = int(np.count_nonzero(music_mask))
    evidence = _evidence(len(scores), count, float(music.mean()))
    if count < max(2, len(scores) * 0.20):
        return undetermined_style("There is too little sustained music to estimate a style.", evidence)
    music_scores = scores[music_mask]

    def windows(names: tuple[str, ...]) -> np.ndarray:
        columns = [indices[name] for name in names if name in indices]
        return music_scores[:, columns].max(axis=1) if columns else np.zeros(count)

    genre_windows = {genre: windows(names) for genre, names in GENRES.items()}
    candidates = {genre: _Support.from_windows(values) for genre, values in genre_windows.items()}
    cues = evidence["cues"]

    # Layered soundtracks may have local strings/brass and weak genre scores.
    # Require soundtrack or classical evidence as well as orchestral timbre.
    orchestral = windows(ORCHESTRAL_TIMBRES)
    score_music = np.maximum(genre_windows["Classical"], genre_windows["Cinematic / Orchestral"])
    orchestral_cue = float(orchestral.mean()) >= 0.012 and np.mean(orchestral >= 0.045) >= 0.035
    score_cue = float(score_music.mean()) >= 0.010 and np.mean(score_music >= 0.02) >= 0.080
    if orchestral_cue:
        cues.append("Orchestral timbres")
    if score_cue:
        cues.append("Soundtrack or classical cues")

    ranked = sorted(candidates.items(), key=lambda candidate: candidate[1].score, reverse=True)
    evidence["candidates"] = [
        {"style": genre, "score": round(support.score, 4), "coverage": round(support.coverage, 4)}
        for genre, support in ranked[:4] if support.score > 0
    ]
    (style, best), (_, second) = ranked[:2]
    margin = best.score - second.score
    confidence = "Low"
    note = "Estimated from recurring style cues across the track."
    if best.score >= 0.13 and best.coverage >= 0.60 and margin >= 0.035:
        confidence = "High"
    elif best.score >= 0.040 and best.coverage >= 0.28 and margin >= 0.012:
        confidence = "Medium"

    if orchestral_cue and score_cue and not candidates["Classical"].consistent and best.score < 0.035:
        style = "Cinematic / Orchestral"
        note = "Orchestral timbres and soundtrack or classical cues support this broad style."
    elif not best.consistent or (second.consistent and margin < 0.004):
        piano = _Support.from_windows(windows(("Piano", "Electric piano")))
        instruments = _Support.from_windows(windows(INSTRUMENTAL_TIMBRES))
        voice = _Support.from_windows(windows(VOICE_LABELS))
        if count < 3 or music.mean() < 0.60 or instruments.score < 0.025 or instruments.coverage < 0.28 or voice.coverage >= 0.12:
            return undetermined_style("Music is present, but the style cues are weak or mixed.", evidence)
        piano_led = piano.score >= 0.060 and piano.coverage >= 0.50
        style = "Piano Instrumental" if piano_led else "Instrumental Music"
        confidence = "Low"
        cues.append("Sustained piano cues" if piano_led else "Sustained instrumental cues")
        note = "Recurring instrumental cues support a broad estimate."
    elif style == "Independent Music" and candidates["Pop"].score >= 0.010 and candidates["Pop"].coverage >= 0.10:
        style = "Indie / Pop"
        cues.append("Pop cues")
    elif style == "Electronic":
        subgenre, specific = max(
            ((name, _Support.from_windows(windows((label,)))) for name, label in ELECTRONIC_SUBGENRES.items()),
            key=lambda pair: pair[1].score,
        )
        if specific.score >= max(0.05, 0.72 * best.score) and specific.coverage >= 0.27:
            style = subgenre
            cues.append("Electronic music")

    return {
        "style": style,
        "style_confidence": confidence,
        "style_note": note,
        "style_evidence": evidence,
    }
