"""Register the existing Tk window for native file drops."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable

try:
    from tkinterdnd2 import COPY, DND_FILES, REFUSE_DROP, TkinterDnD
except ImportError:
    TkinterDnD = None


if TkinterDnD is None:
    AudioDropWindow = tk.Tk
else:
    class AudioDropWindow(tk.Tk, TkinterDnD.DnDWrapper):
        pass


def install_audio_drop(window: AudioDropWindow, import_paths: Callable[[tuple[str, ...]], int]) -> bool:
    if TkinterDnD is None:
        return False

    def on_drop(event: tk.Event) -> str:
        try:
            paths = window.tk.splitlist(event.data)
        except tk.TclError:
            return REFUSE_DROP
        return COPY if paths and import_paths(paths) else REFUSE_DROP

    try:
        TkinterDnD._require(window)
        window.drop_target_register(DND_FILES)
        window.dnd_bind("<<Drop>>", on_drop)
    except (RuntimeError, tk.TclError):
        return False
    return True
