from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from tempfile import TemporaryDirectory


def _split_centiseconds(seconds: float) -> tuple[int, int, int]:
    minutes, remainder = divmod(round(seconds * 100), 6000)
    whole_seconds, centiseconds = divmod(remainder, 100)
    return minutes, whole_seconds, centiseconds


def format_timestamp(seconds: float, mode: str = "clock") -> str:
    if mode == "minutes_dotted":
        minutes, whole_seconds, centiseconds = _split_centiseconds(seconds)
        return f"{minutes}.{whole_seconds:02d}.{centiseconds:02d}"
    if mode == "seconds":
        return f"{seconds:.3f}"
    whole, milliseconds = divmod(round(seconds * 1000), 1000)
    minutes, remainder = divmod(whole, 60)
    clock = f"{minutes:02d}:{remainder:02d}"
    if mode == "clock_ms":
        return f"{clock}.{milliseconds:03d}"
    return clock


def _lrc_time(seconds: float) -> str:
    minutes, whole_seconds, centiseconds = _split_centiseconds(seconds)
    return f"{minutes:02d}:{whole_seconds:02d}.{centiseconds:02d}"


def export_lyrics(
    audio_path: str | Path,
    segments: Iterable[Mapping[str, float | str]],
    output_dir: str | Path,
) -> dict[str, Path]:
    rows = []
    for segment in segments:
        start = float(segment["start"])
        end = float(segment["end"])
        text = " ".join(str(segment["text"]).split())
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start <= end):
            raise ValueError("歌词时间戳无效。")
        if text:
            rows.append((start, end, text))
    if not rows:
        raise ValueError("没有可导出的歌词。")
    rows.sort(key=lambda row: row[0])

    output_dir = Path(output_dir)
    stem = Path(audio_path).stem
    paths = {
        "lrc": output_dir / f"{stem}_lyrics.lrc",
        "csv": output_dir / f"{stem}_lyrics.csv",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir, prefix=".lyrics-") as staging:
        lrc = Path(staging) / paths["lrc"].name
        lrc.write_text(
            "".join(f"[{_lrc_time(start)}]{text}\n" for start, _, text in rows),
            encoding="utf-8",
        )
        csv_path = Path(staging) / paths["csv"].name
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("Start (s)", "End (s)", "Lyrics"))
            for start, end, text in rows:
                # Keep lines beginning with spreadsheet operators as text in Excel.
                csv_text = "'" + text if text.startswith(("=", "+", "-", "@")) else text
                writer.writerow((f"{start:.3f}", f"{end:.3f}", csv_text))
        lrc.replace(paths["lrc"])
        csv_path.replace(paths["csv"])
    return paths
