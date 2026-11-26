"""Bounded storage for reusable analysis metadata."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path

from runtime_paths import vocal_model_path


CACHE_REVISION = 5
MAX_FILES = 128
MAX_BYTES = 32 * 1024 * 1024
MAX_RESULT_BYTES = 4 * 1024 * 1024
MAX_KERNEL_BYTES = 128 * 1024 * 1024
ORPHAN_AGE_SECONDS = 24 * 60 * 60
ENTRY_NAME = re.compile(r"[0-9a-f]{64}\.json")
TEMP_NAME = re.compile(r"\.[0-9a-f]{64}\.[0-9a-z_]+\.tmp")
ANALYSIS_MODULES = (
    "audio_analysis.py",
    "phrase_analysis.py",
    "section_analysis.py",
    "rhythm_analysis.py",
    "instrument_analysis.py",
    "style_analysis.py",
    "style_features.py",
    "local_style.py",
    "vocal_separation.py",
    "reference_annotations.py",
)


def _file_digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _asset_state(app_dir: Path) -> list[tuple[str, int, int]]:
    """Track the bundled model assets and editable analysis modules."""
    paths = [app_dir / name for name in ANALYSIS_MODULES]
    models = app_dir / "models"
    if models.is_dir():
        paths.extend(file for file in models.rglob("*") if file.is_file())
    state = []
    for path in sorted(paths):
        if path.is_file():
            info = path.stat()
            state.append((path.relative_to(app_dir).as_posix(), info.st_size, info.st_mtime_ns))
    return state


class AnalysisCache:
    def __init__(self, app_dir: Path, root: Path | None = None, executable: Path | None = None) -> None:
        self.app_dir = app_dir
        self.executable = executable
        if root is None:
            base = Path(os.environ.get("LOCALAPPDATA") or tempfile.gettempdir())
            root = base / "OfflineAudioMarker" / "analysis-cache"
        self.root = root

    def key_for(self, audio: Path, *, use_vocals: bool, use_reference: bool) -> str | None:
        try:
            project = audio.with_suffix(".flp")
            executable = self.executable.stat() if self.executable else None
            external_assets = []
            if use_vocals and self.executable:
                model = vocal_model_path()
                if model.is_file() and not model.is_relative_to(self.app_dir):
                    info = model.stat()
                    external_assets = [(str(model), info.st_size, info.st_mtime_ns)]
            key_data = [
                CACHE_REVISION,
                (executable.st_size, executable.st_mtime_ns) if executable else None,
                _file_digest(audio),
                _file_digest(project) if project.is_file() and project != audio else None,
                use_vocals,
                use_reference,
                _asset_state(self.app_dir),
                external_assets,
            ]
            return hashlib.sha256(json.dumps(key_data, ensure_ascii=False).encode("utf-8")).hexdigest()
        except OSError:
            return None

    def _safe_root(self, *, create: bool = False) -> Path:
        # Reject symlinks and junctions for the app-owned cache directories.
        expected = self.root.parent.parent.resolve() / self.root.parent.name / self.root.name
        if self.root.resolve() != expected:
            raise OSError("Cache directory redirects to another location.")
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def _entry(self, key: str) -> Path:
        name = f"{key}.json"
        if ENTRY_NAME.fullmatch(name) is None:
            raise ValueError("Invalid cache key.")
        return self._safe_root() / name

    def configure_kernel_cache(self) -> None:
        root = self._kernel_root()
        root.mkdir(parents=True, exist_ok=True)
        os.environ["NUMBA_CACHE_DIR"] = str(root)

    def _kernel_root(self) -> Path:
        root = self._safe_root().resolve() / "kernels"
        if root.resolve() != root:
            raise OSError("Kernel cache directory redirects to another location.")
        return root

    def load(self, key: str | None, audio: Path) -> dict | None:
        if key is None:
            return None
        try:
            path = self._entry(key)
            if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_RESULT_BYTES:
                return None
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("key") != key:
                return None
            result = payload.get("result")
            required = {"sections", "key_changes", "duration_text", "bpm", "global_key", "global_key_confidence", "style"}
            if (
                not isinstance(result, dict)
                or not required <= result.keys()
                or not isinstance(result["sections"], list)
                or not isinstance(result["key_changes"], list)
            ):
                return None
            os.utime(path, None)
            return {**result, "file": str(audio), "name": audio.name}
        except (OSError, ValueError):
            return None

    def store(self, key: str | None, result: dict) -> None:
        if key is None:
            return
        scratch = None
        try:
            value = json.dumps({"key": key, "result": result}, ensure_ascii=False, allow_nan=False).encode("utf-8")
            if len(value) > MAX_RESULT_BYTES:
                return
            destination = self._entry(key)
            with tempfile.NamedTemporaryFile(
                dir=self._safe_root(create=True), prefix=f".{key}.", suffix=".tmp", delete=False,
            ) as handle:
                scratch = Path(handle.name)
                handle.write(value)
            os.replace(scratch, destination)
            self._prune()
        except (OSError, ValueError):
            pass
        finally:
            if scratch is not None:
                try:
                    scratch.unlink(missing_ok=True)
                except OSError:
                    pass

    def _owned_files(self) -> list[tuple[Path, os.stat_result, bool]]:
        root = self._safe_root()
        if not root.is_dir():
            return []
        entries = []
        now = time.time()
        for path in root.iterdir():
            temporary = TEMP_NAME.fullmatch(path.name) is not None
            if not temporary and ENTRY_NAME.fullmatch(path.name) is None:
                continue
            try:
                if path.is_symlink() or not path.is_file():
                    continue
                info = path.stat()
            except FileNotFoundError:
                continue
            if not temporary or now - info.st_mtime >= ORPHAN_AGE_SECONDS:
                entries.append((path, info, temporary))
        return entries

    def _prune(self) -> None:
        entries = sorted(self._owned_files(), key=lambda item: item[1].st_mtime, reverse=True)
        count = used = 0
        for path, info, temporary in entries:
            if temporary or count >= MAX_FILES or used + info.st_size > MAX_BYTES:
                path.unlink(missing_ok=True)
            else:
                count += 1
                used += info.st_size

    def _kernel_files(self) -> list[tuple[Path, os.stat_result]]:
        root = self._kernel_root()
        if not root.is_dir():
            return []
        return self._jit_files(root, root.iterdir())

    def _legacy_kernel_files(self) -> list[tuple[Path, os.stat_result]]:
        if self.executable is None:
            return []
        librosa_dir = self.app_dir / "librosa"
        if librosa_dir.resolve().parent != self.app_dir.resolve() or librosa_dir.is_symlink():
            return []
        return self._jit_files(librosa_dir, librosa_dir.rglob("__pycache__"))

    @staticmethod
    def _jit_files(root: Path, directories) -> list[tuple[Path, os.stat_result]]:
        entries = []
        root = root.resolve()
        for directory in directories:
            if directory.is_symlink() or not directory.is_dir() or directory.resolve() != directory.absolute():
                continue
            if not directory.resolve().is_relative_to(root):
                continue
            for path in directory.iterdir():
                if path.suffix not in (".nbc", ".nbi") or path.is_symlink() or not path.is_file():
                    continue
                try:
                    entries.append((path, path.stat()))
                except FileNotFoundError:
                    continue
        return entries

    def prune_kernels(self) -> None:
        try:
            entries = sorted(self._kernel_files(), key=lambda item: item[1].st_mtime, reverse=True)
            used = 0
            for path, info in entries:
                if used + info.st_size <= MAX_KERNEL_BYTES:
                    used += info.st_size
                else:
                    path.unlink(missing_ok=True)
        except OSError:
            pass

    def clear(self) -> tuple[int, int]:
        removed = released_bytes = 0
        owned = [(path, info) for path, info, _ in self._owned_files()]
        for path, info in owned + self._kernel_files() + self._legacy_kernel_files():
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            removed += 1
            released_bytes += info.st_size
        return removed, released_bytes
