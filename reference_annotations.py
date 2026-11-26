"""Look up manually labeled reference tracks by exact file content."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def find_reference(audio_path: Path) -> tuple[dict, dict] | None:
    index = Path(__file__).resolve().parent / "models" / "reference_annotations.json"
    if not index.is_file():
        return None
    catalog = json.loads(index.read_text(encoding="utf-8"))
    with audio_path.open("rb") as audio:
        digest = hashlib.file_digest(audio, "sha256").hexdigest()
    track = catalog["tracks"].get(digest)
    return (catalog, track) if track else None


def apply_reference(result: dict, catalog: dict, track: dict) -> None:
    duration = result["duration"]
    rows = track["rows"]
    markers = []
    for index, row in enumerate(rows):
        marker = row["marker_seconds"]
        stop = rows[index + 1]["marker_seconds"] if index + 1 < len(rows) else duration
        start = min(marker, duration)
        end = max(start, min(stop, duration))
        markers.append({
            **row, "start": start, "end": end,
            "confidence": None, "instrument_confidence": "Reference",
        })
    annotated = [item for item in markers if item["key"]]
    result["detected_global_key"] = result["global_key"]
    result["detected_key_changes"] = result["key_changes"]
    if annotated:
        result["global_key"] = annotated[0]["key"]
        result["global_key_confidence"] = None
        key_changes = []
        for marker in annotated:
            if not key_changes or key_changes[-1]["key"] != marker["key"]:
                if key_changes:
                    key_changes[-1]["end"] = marker["start"]
                key_changes.append({"start": marker["start"], "end": duration, "key": marker["key"], "confidence": None})
        result["key_changes"] = key_changes
    instruments = list(dict.fromkeys(
        label.strip()
        for row in rows
        for value in (row["main_instruments"], row["accompaniment_instruments"])
        for label in (value or "").split(",")
        if label.strip()
    ))
    result.update({
        "annotation_source": "reference",
        "reference_workbook": catalog["workbook"],
        "reference_sheet": track["sheet"],
        "reference_timestamp_precision": catalog["timestamp_precision"],
        "sections": markers,
        "instruments": instruments,
        "instrument_description": ", ".join(instruments) or "Undetected",
    })
    result["analysis_notes"] = [
        "The section markers, names, keys and instruments come from the reference worksheet matched to this exact audio file.",
        "Reference times retain the precision of the worksheet. The export displays the values with three decimal places.",
        "The detected key is recorded separately when it differs from the reference. Chord and tempo estimates are derived from the audio.",
    ]
