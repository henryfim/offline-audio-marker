"""Run the Windows package against a generated, known-key audio clip."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import wave
from array import array
from pathlib import Path
from tempfile import TemporaryDirectory


SAMPLE_RATE = 22_050
DURATION = 7
CHORD_HZ = (261.63, 329.63, 392.0)


def write_c_major_fixture(destination: Path) -> None:
    samples = array("h")
    for frame in range(DURATION * SAMPLE_RATE):
        seconds = frame / SAMPLE_RATE
        gain = 0.24 * math.exp(-(seconds % 0.5) * 8)
        chord = sum(math.sin(math.tau * frequency * seconds) for frequency in CHORD_HZ)
        samples.append(round(32767 * gain * chord))
    with wave.open(str(destination), "wb") as handle:
        handle.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
        handle.writeframes(samples.tobytes())


def verify_reports(artifact: Path, *, expect_vocals: bool = False) -> None:
    with TemporaryDirectory(prefix="offline-audio-marker-smoke-") as workspace:
        root = Path(workspace)
        fixture = root / "c-major.wav"
        reports = root / "reports"
        write_c_major_fixture(fixture)
        env = os.environ.copy()
        env["LOCALAPPDATA"] = str(root / "local-app-data")
        args = [str(artifact), "--analyze", str(reports), "--no-reference", str(fixture)]
        subprocess.run(
            args,
            check=True,
            cwd=artifact.parent,
            creationflags=subprocess.CREATE_NO_WINDOW,
            env=env,
            timeout=120,
        )
        analysis = json.loads((reports / "c-major_analysis.json").read_text(encoding="utf-8"))
        if analysis["global_key"] != "C major" or not analysis["sections"]:
            raise ValueError("The fixture should have a C major key and at least one section.")
        expected_vocals = "separated" if expect_vocals else "unavailable"
        if analysis["vocal_analysis"] != expected_vocals:
            raise ValueError(f"Expected vocal analysis {expected_vocals!r}, got {analysis['vocal_analysis']!r}.")
        if any("instrument model is unavailable" in item["instrument_note"] for item in analysis["sections"]):
            raise ValueError("The offline timbre model could not be loaded.")
        with (reports / "c-major_sections.csv").open(encoding="utf-8-sig", newline="") as handle:
            if len(list(csv.reader(handle))) != len(analysis["sections"]) + 1:
                raise ValueError("CSV rows and JSON section counts do not match.")
        labels = (reports / "c-major_audacity_labels.txt").read_text(encoding="utf-8").splitlines()
        timestamps = [row.split("\t", 2)[:2] for row in labels]
        if len(timestamps) != len(analysis["sections"]) or any(start != end for start, end in timestamps):
            raise ValueError("Audacity labels should contain one point per section.")
        if not (reports / "c-major_markers.md").read_text(encoding="utf-8").strip():
            raise ValueError("The Markdown report is empty.")
        print(f"Verified four reports, C major, {len(labels)} section marker(s), and vocals {expected_vocals}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, help="Path to the packaged executable.")
    parser.add_argument("--expect-vocals", action="store_true", help="Require the ONNX vocal model to run.")
    args = parser.parse_args()
    verify_reports(args.artifact.resolve(strict=True), expect_vocals=args.expect_vocals)


if __name__ == "__main__":
    main()
