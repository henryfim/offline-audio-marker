from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from collections.abc import Iterable
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter import font as tkfont

from analysis_cache import AnalysisCache
from audio_drop import AudioDropWindow, install_audio_drop
from deepseek_credentials import delete_key, load_key, save_key
from lyrics_analysis import format_timestamp
from runtime_paths import vocal_model_path


FROZEN_EXE = Path(sys.executable) if getattr(sys, "frozen", False) else None
APP_DIR = (FROZEN_EXE or Path(__file__)).resolve().parent
RESOURCE_DIR = Path(__file__).resolve().parent
APP_CACHE = AnalysisCache(RESOURCE_DIR, executable=FROZEN_EXE)
DEFAULT_OUTPUT = APP_DIR / "Analysis Results"
if FROZEN_EXE and sys.argv[1:] == ["--installed"]:
    DEFAULT_OUTPUT = Path.home() / "Documents" / "OfflineAudioMarker" / "Analysis Results"
SUPPORTED = ("*.mp3", "*.wav", "*.flac", "*.m4a", "*.aac", "*.ogg", "*.aiff", "*.wma")
SUPPORTED_SUFFIXES = {pattern[1:] for pattern in SUPPORTED}
SECTION_HEADINGS = ("时间戳", "段落", "BPM", "调性调式", "节拍", "主奏乐器", "伴奏乐器")
KEY_HEADINGS = ("时间戳", "调性", "置信度")
LYRICS_HEADINGS = ("起点", "终点", "歌词")
TIME_FORMATS = {
    "分.秒.百分秒": "minutes_dotted",
    "分:秒": "clock",
    "分:秒.毫秒": "clock_ms",
    "秒.毫秒": "seconds",
}
STYLE_CONFIDENCE = {"High": "高", "Medium": "中", "Low": "低", "Undetermined": "未确定"}
ANALYSIS_STAGES = {
    5: "读取音频",
    18: "估算速度、拍号和小节网格",
    38: "提取和声与音色特征",
    57: "匹配段落并定位起点",
    73: "识别各段乐器",
    88: "整理标注",
    100: "分析完成",
}


class AudioMarkerApp(AudioDropWindow):
    def __init__(self) -> None:
        super().__init__()
        self.title("离线音频标注器")
        self.geometry("1180x760")
        self.minsize(940, 620)
        self.files: list[str] = []
        self.results: dict[str, dict] = {}
        self.lyrics: dict[str, list[dict]] = {}
        self.output_dir = DEFAULT_OUTPUT
        self.cache = APP_CACHE
        self.messages: queue.Queue = queue.Queue()
        self.current_result: dict | None = None
        self._analysis_running = False
        self._selected_result_text: tk.Text | None = None
        self.result_menu = tk.Menu(self, tearoff=False)
        self.result_menu.add_command(label="复制所选文本", accelerator="Ctrl+C")
        self.result_menu.add_command(label="全选", accelerator="Ctrl+A")
        self.use_vocals = tk.BooleanVar(value=vocal_model_path().is_file())
        self.use_reference = tk.BooleanVar(value=(RESOURCE_DIR / "models" / "reference_annotations.json").is_file())
        self.time_format = tk.StringVar(value=next(iter(TIME_FORMATS)))
        self._build_style()
        self._build_ui()
        if install_audio_drop(self, self._import_audio_paths):
            self.file_heading.configure(text="音频文件（可拖入窗口）")
        self.after(100, self._poll_messages)

    def _build_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("vista")
        except tk.TclError:
            pass
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 16, "bold"))
        self._summary_background = style.lookup("TLabel", "background") or self.cget("background")
        self._row_font = tkfont.Font(self, family="Microsoft YaHei UI", size=10)
        self._heading_font = tkfont.Font(self, family="Microsoft YaHei UI", size=10, weight="bold")

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=14)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="离线音频标注器", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            outer,
            text="离线分析段落、速度、调性和配器，导出英文标注。",
        ).pack(anchor="w", pady=(4, 12))

        toolbar = ttk.Frame(outer)
        toolbar.pack(fill="x", pady=(0, 10))
        ttk.Button(toolbar, text="添加音频", command=self._add_files).pack(side="left")
        ttk.Button(toolbar, text="移除所选", command=self._remove_selected).pack(side="left", padx=6)
        ttk.Button(toolbar, text="清空列表", command=self._clear_files).pack(side="left")
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=12)
        ttk.Button(toolbar, text="分析全部", command=self._analyze_all).pack(side="left")
        ttk.Button(toolbar, text="分析所选", command=self._analyze_current).pack(side="left", padx=6)
        self.lyrics_button = ttk.Button(toolbar, text="识别歌词（DeepSeek）", command=self._start_lyrics)
        self.lyrics_button.pack(side="left")
        ttk.Button(toolbar, text="选择结果目录", command=self._choose_output).pack(side="right")
        ttk.Button(toolbar, text="打开结果目录", command=self._open_output).pack(side="right", padx=6)

        options = ttk.Frame(outer)
        options.pack(fill="x", pady=(0, 10))
        ttk.Checkbutton(options, text="离线人声分离", variable=self.use_vocals).pack(side="left")
        ttk.Checkbutton(options, text="使用匹配的参考标注", variable=self.use_reference).pack(side="left", padx=12)
        ttk.Label(options, text="时间显示").pack(side="left", padx=(4, 6))
        time_format = ttk.Combobox(
            options, textvariable=self.time_format, values=tuple(TIME_FORMATS), state="readonly", width=12,
        )
        time_format.pack(side="left")
        time_format.bind("<<ComboboxSelected>>", self._refresh_timestamp_tables)
        self.clear_cache_button = ttk.Button(options, text="清理缓存", command=self._clear_cache)
        self.clear_cache_button.pack(side="right")
        ttk.Button(options, text="API Key 设置", command=self._open_deepseek_settings).pack(side="right", padx=6)

        content = ttk.PanedWindow(outer, orient="horizontal")
        content.pack(fill="both", expand=True)
        left = ttk.Frame(content, padding=(0, 0, 12, 0))
        right = ttk.Frame(content)
        content.add(left, weight=1)
        content.add(right, weight=4)

        self.file_heading = ttk.Label(left, text="音频文件")
        self.file_heading.pack(anchor="w")
        list_frame = ttk.Frame(left)
        list_frame.pack(fill="both", expand=True, pady=(5, 0))
        self.file_list = tk.Listbox(
            list_frame, selectmode="extended", exportselection=False, activestyle="dotbox", font=self._row_font,
        )
        self.file_list.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.file_list.yview)
        scroll.pack(side="right", fill="y")
        self.file_list.configure(yscrollcommand=scroll.set)
        self.file_list.bind("<<ListboxSelect>>", self._on_file_select)
        self.output_label = ttk.Label(left, text=self._output_text(), wraplength=280, foreground="#555")
        self.output_label.pack(anchor="w", pady=(10, 0))

        summary = ttk.LabelFrame(right, text="分析概览", padding=10)
        summary.pack(fill="x")
        self.summary_values: dict[str, tk.Text] = {}
        for column, (key, title) in enumerate((("name", "文件"), ("bpm", "BPM"), ("key", "调性"), ("style", "风格"))):
            padding = (0, 12) if key != "style" else 0
            ttk.Label(summary, text=title, foreground="#666").grid(row=0, column=column, sticky="w", padx=padding)
            value = self._make_result_text(
                summary, wrap="word", font=("Microsoft YaHei UI", 11), width=1, height=1,
                background=self._summary_background, relief="flat", borderwidth=0, highlightthickness=0,
                padx=0, pady=0,
            )
            value.grid(row=1, column=column, sticky="ew", padx=padding, pady=(2, 0))
            value.bind("<Configure>", lambda event: self._fit_summary_value(event.widget))
            summary.columnconfigure(column, weight=2 if key in {"name", "style"} else 1, uniform="summary")
            self.summary_values[key] = value

        self.progress = ttk.Progressbar(right, mode="determinate", maximum=100)
        self.progress.pack(fill="x", pady=(10, 3))
        result_controls = ttk.Frame(right)
        result_controls.pack(fill="x")
        self.copy_button = ttk.Button(result_controls, text="复制所选文本", command=self._copy_result_selection)
        self.copy_button.pack(side="right", padx=(6, 0))
        self.status = ttk.Label(result_controls, text="添加音频后开始分析。", foreground="#555")
        self.status.pack(side="left", fill="x", expand=True)

        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill="both", expand=True, pady=(10, 0))
        self.key_tab = ttk.Frame(self.notebook, padding=8)
        self.section_tab = ttk.Frame(self.notebook, padding=8)
        self.raw_tab = ttk.Frame(self.notebook, padding=8)
        self.info_tab = ttk.Frame(self.notebook, padding=10)
        self.lyrics_tab = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(self.section_tab, text="段落结构")
        self.notebook.add(self.key_tab, text="调性变化")
        self.notebook.add(self.raw_tab, text="原始 JSON")
        self.notebook.add(self.info_tab, text="和弦与乐器")
        self.notebook.add(self.lyrics_tab, text="歌词")
        self.section_text = self._make_table(self.section_tab)
        self.key_text = self._make_table(self.key_tab)
        self.raw_text = self._make_scrolled_text(self.raw_tab, wrap="none", font=("Consolas", 9))
        self.info_text = self._make_scrolled_text(self.info_tab, wrap="word", font=self._row_font)
        lyrics_controls = ttk.Frame(self.lyrics_tab)
        lyrics_controls.pack(fill="x", pady=(0, 6))
        self.lyrics_export_button = ttk.Button(lyrics_controls, text="导出 LRC / CSV", command=self._export_current_lyrics)
        self.lyrics_export_button.pack(side="right")
        self.lyrics_label = ttk.Label(lyrics_controls, foreground="#555")
        self.lyrics_label.pack(side="left", fill="x", expand=True, padx=(0, 8))
        lyrics_table = ttk.Frame(self.lyrics_tab)
        lyrics_table.pack(fill="both", expand=True)
        self.lyrics_text = self._make_table(lyrics_table)
        self._clear_result_view()
        self._show_lyrics(None)

    def _make_result_text(self, parent: ttk.Frame, **options: object) -> tk.Text:
        text = tk.Text(parent, state="disabled", exportselection=False, **options)
        text.configure(inactiveselectbackground=text.cget("selectbackground"))
        text.bind("<Button-1>", self._activate_result_text)
        text.bind("<FocusIn>", self._activate_result_text)
        text.bind("<<Selection>>", self._on_result_selection)
        text.bind("<<Copy>>", self._copy_result_event)
        text.bind("<<SelectAll>>", self._select_all_result_event)
        text.bind("<Button-3>", self._result_context_menu)
        return text

    def _make_scrolled_text(self, parent: ttk.Frame, *, wrap: str, **options: object) -> tk.Text:
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        text = self._make_result_text(parent, wrap=wrap, width=1, height=1, **options)
        text.grid(row=0, column=0, sticky="nsew")
        vertical = ttk.Scrollbar(parent, orient="vertical", command=text.yview)
        vertical.grid(row=0, column=1, sticky="ns")
        text.configure(yscrollcommand=vertical.set)
        if wrap == "none":
            horizontal = ttk.Scrollbar(parent, orient="horizontal", command=text.xview)
            horizontal.grid(row=1, column=0, sticky="ew")
            text.configure(xscrollcommand=horizontal.set)
        return text

    def _make_table(self, parent: ttk.Frame) -> tk.Text:
        text = self._make_scrolled_text(parent, wrap="none", font=self._row_font, padx=6, spacing1=4, spacing3=4)
        text.tag_configure("heading", font=self._heading_font, background="#e7edf4")
        text.tag_configure("alternate_row", background="#f3f6f9")
        text.tag_raise("sel")
        return text

    def _set_table(self, text: tk.Text, headings: tuple[str, ...], rows: list[list[str]]) -> None:
        tab_stops = []
        for column, heading in enumerate(headings[:-1]):
            width = max([self._heading_font.measure(heading), *(self._row_font.measure(row[column]) for row in rows)])
            tab_stops.append(width + 28 + (tab_stops[-1] if tab_stops else 0))
        text.configure(state="normal", tabs=tab_stops, tabstyle="tabular")
        text.delete("1.0", "end")
        text.insert("end-1c", "\t".join(headings) + "\n", "heading")
        for index, row in enumerate(rows):
            if index:
                text.insert("end-1c", "\n")
            text.insert("end-1c", "\t".join(row), "alternate_row" if index % 2 else ())
        text.configure(state="disabled")
        text.xview_moveto(0)
        text.yview_moveto(0)

    @staticmethod
    def _set_result_text(text: tk.Text, value: str) -> None:
        text.configure(state="normal")
        text.delete("1.0", "end")
        text.insert("1.0", value)
        text.configure(state="disabled")
        text.xview_moveto(0)
        text.yview_moveto(0)

    @staticmethod
    def _fit_summary_value(text: tk.Text) -> None:
        count = text.count("1.0", "end", "displaylines")
        height = min(4, max(1, count[0] if count else 1))
        if int(text.cget("height")) != height:
            text.configure(height=height)

    def _on_result_selection(self, event: tk.Event) -> None:
        if event.widget.tag_ranges("sel"):
            self._activate_result_text(event)

    def _activate_result_text(self, event: tk.Event) -> None:
        previous = self._selected_result_text
        self._selected_result_text = event.widget
        if previous is not None and previous is not event.widget:
            previous.tag_remove("sel", "1.0", "end")

    def _copy_result_selection(self, text: tk.Text | None = None) -> None:
        if text is None:
            text = self._selected_result_text
        if text is None or not text.winfo_viewable() or not text.tag_ranges("sel"):
            self.bell()
            return
        selected = text.get("sel.first", "sel.last")
        self.clipboard_clear()
        self.clipboard_append(selected)

    def _copy_result_event(self, event: tk.Event) -> str:
        self._copy_result_selection(event.widget)
        return "break"

    def _select_all_result_event(self, event: tk.Event) -> str:
        text = event.widget
        text.tag_remove("sel", "1.0", "end")
        text.tag_add("sel", "1.0", "end-1c")
        return "break"

    def _result_context_menu(self, event: tk.Event) -> str:
        text = event.widget
        text.focus_set()
        self.result_menu.entryconfigure(
            0, command=lambda: self._copy_result_selection(text),
            state="normal" if text.tag_ranges("sel") else "disabled",
        )
        self.result_menu.entryconfigure(1, command=lambda: text.event_generate("<<SelectAll>>"))
        try:
            self.result_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.result_menu.grab_release()
        return "break"

    def _output_text(self) -> str:
        return f"结果目录\n{self.output_dir}"

    def _add_files(self) -> None:
        paths = filedialog.askopenfilenames(title="选择音频文件", filetypes=[("音频文件", " ".join(SUPPORTED)), ("所有文件", "*.*")])
        self._import_audio_paths(paths)

    def _import_audio_paths(self, paths: Iterable[str]) -> int:
        known = {os.path.normcase(path) for path in self.files}
        added = 0
        duplicates = 0
        invalid = 0
        for path in paths:
            try:
                audio = Path(path).resolve(strict=True)
                supported = audio.suffix.lower() in SUPPORTED_SUFFIXES and audio.is_file()
            except (OSError, RuntimeError, ValueError):
                supported = False
            if not supported:
                invalid += 1
                continue
            path_key = os.path.normcase(str(audio))
            if path_key in known:
                duplicates += 1
                continue
            known.add(path_key)
            self.files.append(str(audio))
            self.file_list.insert("end", audio.name)
            added += 1
        if added or duplicates or invalid:
            status = [f"已添加 {added} 个音频文件"]
            if duplicates:
                status.append(f"跳过 {duplicates} 个重复文件")
            if invalid:
                status.append(f"跳过 {invalid} 个不可用或不支持的文件")
            self.status.configure(text="，".join(status) + "。")
        return added

    def _remove_selected(self) -> None:
        selected = list(self.file_list.curselection())
        for index in reversed(selected):
            self.files.pop(index)
            self.file_list.delete(index)
        self.current_result = None
        self._on_file_select(None)

    def _clear_files(self) -> None:
        self.files.clear()
        self.results.clear()
        self.lyrics.clear()
        self.file_list.delete(0, "end")
        self._clear_result_view()
        self._show_lyrics(None)

    def _choose_output(self) -> None:
        chosen = filedialog.askdirectory(title="选择结果目录", initialdir=str(self.output_dir))
        if chosen:
            self.output_dir = Path(chosen)
            self.output_label.configure(text=self._output_text())

    def _open_output(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if sys.platform.startswith("win"):
            os.startfile(self.output_dir)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(self.output_dir)])
        else:
            subprocess.Popen(["xdg-open", str(self.output_dir)])

    def _analyze_all(self) -> None:
        self._start_analysis(self.files)

    def _analyze_current(self) -> None:
        selected = self.file_list.curselection()
        self._start_analysis([self.files[index] for index in selected])

    def _selected_lyrics_path(self) -> str | None:
        selected = self.file_list.curselection()
        return self.files[selected[0]] if len(selected) == 1 else None

    def _open_deepseek_settings(self) -> None:
        try:
            stored_key = load_key()
        except OSError as error:
            messagebox.showerror("无法读取 DeepSeek 设置", str(error), parent=self)
            return

        dialog = tk.Toplevel(self)
        dialog.title("DeepSeek API Key 设置")
        dialog.transient(self)
        dialog.resizable(False, False)
        panel = ttk.Frame(dialog, padding=16)
        panel.pack(fill="both", expand=True)
        ttk.Label(
            panel, text="密钥保存在 Windows 凭据管理器。识别歌词时，音频会发送到 DeepSeek，接口可能计费。",
            wraplength=370, justify="left",
        ).pack(anchor="w", pady=(0, 12))
        has_environment_key = bool(os.environ.get("DEEPSEEK_API_KEY", "").strip())
        key_status = "已保存密钥。填写新密钥可替换。" if stored_key else "请填写 DeepSeek API Key。"
        if not stored_key and has_environment_key:
            key_status = "正在使用系统环境变量中的密钥。"
        ttk.Label(panel, text=key_status).pack(anchor="w", pady=(0, 6))
        entry = ttk.Entry(panel, show="*", width=48)
        entry.pack(fill="x")

        def save() -> None:
            value = entry.get().strip()
            if not value:
                messagebox.showinfo("API Key 设置", "请输入 DeepSeek API Key。", parent=dialog)
                return
            try:
                save_key(value)
            except OSError as error:
                messagebox.showerror("无法保存密钥", str(error).replace(value, "***"), parent=dialog)
                return
            self.status.configure(text="已保存 DeepSeek API Key。")
            dialog.destroy()

        def clear() -> None:
            try:
                delete_key()
            except OSError as error:
                messagebox.showerror("无法清除密钥", str(error), parent=dialog)
                return
            status = "已清除保存的密钥。"
            if has_environment_key:
                status += "识别时会使用系统环境变量中的密钥。"
            self.status.configure(text=status)
            dialog.destroy()

        buttons = ttk.Frame(panel)
        buttons.pack(fill="x", pady=(12, 0))
        ttk.Button(buttons, text="保存", command=save).pack(side="right")
        ttk.Button(buttons, text="关闭", command=dialog.destroy).pack(side="right", padx=6)
        clear_button = ttk.Button(buttons, text="清除已保存的密钥", command=clear)
        clear_button.pack(side="left")
        if not stored_key:
            clear_button.state(["disabled"])
        dialog.bind("<Return>", lambda _event: save())
        dialog.bind("<Escape>", lambda _event: dialog.destroy())
        dialog.grab_set()
        entry.focus_set()
        self.wait_window(dialog)

    def _start_lyrics(self) -> None:
        if self._analysis_running:
            return
        path = self._selected_lyrics_path()
        if path is None:
            messagebox.showinfo("识别歌词", "请从列表中选择一首音频。", parent=self)
            return

        try:
            api_key = load_key() or os.environ.get("DEEPSEEK_API_KEY", "").strip()
        except OSError as error:
            messagebox.showerror("无法读取 DeepSeek 设置", str(error), parent=self)
            return
        if not api_key:
            self._open_deepseek_settings()
            return

        audio = Path(path)
        self.progress.configure(value=0)
        self.status.configure(text=f"正在准备识别 {audio.name} 的歌词…")
        self.notebook.select(self.lyrics_tab)
        self._set_analysis_running(True)
        thread = threading.Thread(
            target=self._lyrics_worker,
            args=(path, self.output_dir, api_key, self.use_vocals.get()),
            daemon=True,
        )
        thread.start()

    def _lyrics_worker(self, path: str, output_dir: Path, api_key: str, use_vocals: bool) -> None:
        def progress(value: int, stage: str) -> None:
            self.messages.put(("progress", value, stage))

        try:
            from lyrics_deepseek import recognize_lyrics

            segments = recognize_lyrics(path, api_key=api_key, use_vocals=use_vocals, progress=progress)
        except Exception as error:
            status_code = getattr(error, "status_code", None)
            self.messages.put(("lyrics_error", str(error).replace(api_key, "***"), status_code))
            return

        try:
            from lyrics_analysis import export_lyrics

            if segments:
                export_lyrics(path, segments, output_dir)
        except Exception as error:
            self.messages.put(("lyrics_result", path, segments, f"导出失败，{error}"))
            return
        status = f"已识别 {len(segments)} 句歌词并导出 LRC / CSV。" if segments else "没有识别到歌词。"
        self.messages.put(("lyrics_result", path, segments, status))

    def _show_lyrics(self, path: str | None) -> None:
        segments = self.lyrics.get(path) if path is not None else None
        if segments is None:
            detail = "选择一首音频后，点击“识别歌词（DeepSeek）”。"
        else:
            detail = f"{Path(path).name}（{len(segments)} 句）"
        self.lyrics_label.configure(text=detail)
        mode = TIME_FORMATS[self.time_format.get()]
        rows = [
            [format_timestamp(segment["start"], mode), format_timestamp(segment["end"], mode), segment["text"]]
            for segment in segments or ()
        ]
        self._set_table(self.lyrics_text, LYRICS_HEADINGS, rows)
        self.lyrics_export_button.state(["!disabled"] if segments else ["disabled"])

    def _export_current_lyrics(self) -> None:
        path = self._selected_lyrics_path()
        if path is None or not self.lyrics.get(path) or self._analysis_running:
            return
        try:
            from lyrics_analysis import export_lyrics

            export_lyrics(path, self.lyrics[path], self.output_dir)
        except Exception as error:
            messagebox.showerror("无法导出歌词", str(error), parent=self)
            return
        self.status.configure(text=f"LRC / CSV 已保存到 {self.output_dir}")

    def _set_analysis_running(self, running: bool) -> None:
        self._analysis_running = running
        state = ["disabled"] if running else ["!disabled"]
        self.clear_cache_button.state(state)
        self.lyrics_button.state(state)

    def _clear_cache(self) -> None:
        if self._analysis_running:
            return
        try:
            removed, size = self.cache.clear()
        except OSError as error:
            messagebox.showerror("无法清理缓存", str(error))
            return
        if removed:
            self.status.configure(text=f"已清理 {removed} 个缓存文件，释放 {size / 1024:.1f} KB。")
        else:
            self.status.configure(text="没有可清理的分析缓存。")

    def _start_analysis(self, paths: list[str]) -> None:
        if self._analysis_running:
            return
        if not paths:
            messagebox.showinfo("未选择音频", "请添加或选择要分析的音频文件。")
            return
        self.progress.configure(value=0)
        self.status.configure(text=f"准备分析 {len(paths)} 个文件…")
        self._set_analysis_running(True)
        thread = threading.Thread(
            target=self._worker,
            args=(list(paths), self.output_dir, self.use_vocals.get(), self.use_reference.get()),
            daemon=True,
        )
        thread.start()

    def _worker(self, paths: list[str], output_dir: Path, use_vocals: bool, use_reference: bool) -> None:
        try:
            from audio_analysis import analyze_audio, export_result
        except (ImportError, OSError) as error:
            self.messages.put(("done", len(paths), len(paths), str(error)))
            return

        failures = 0
        last_error = ""
        for index, path in enumerate(paths):
            def progress(value: int, message: str, index: int = index) -> None:
                overall = (index * 100 + value) / len(paths)
                if "separat" in message.lower():
                    stage = "离线分离人声"
                elif "reference" in message.lower():
                    stage = "匹配参考标注"
                else:
                    stage = ANALYSIS_STAGES.get(value, "正在分析")
                self.messages.put(("progress", overall, f"{Path(path).name}：{stage}"))

            try:
                audio = Path(path)
                options = {"use_vocals": use_vocals, "use_reference": use_reference}
                cache_key = self.cache.key_for(audio, **options)
                result = self.cache.load(cache_key, audio)
                if result is None:
                    result = analyze_audio(audio, progress, **options)
                    can_cache = cache_key and result.get("vocal_analysis") != "unavailable"
                    if can_cache and cache_key == self.cache.key_for(audio, **options):
                        self.cache.store(cache_key, result)
                else:
                    overall = (index + 1) * 100 / len(paths)
                    self.messages.put(("progress", overall, f"{audio.name}：已读取分析缓存"))
                exported = export_result(result, output_dir)
                self.messages.put(("result", path, result, exported))
            except Exception as exc:  # show the concrete file error in the UI
                failures += 1
                last_error = f"{Path(path).name}，{exc}"
                self.messages.put(("error", last_error))
        self.cache.prune_kernels()
        self.messages.put(("done", len(paths), failures, last_error))

    def _poll_messages(self) -> None:
        try:
            while True:
                message = self.messages.get_nowait()
                kind = message[0]
                if kind == "progress":
                    self.progress.configure(value=message[1])
                    self.status.configure(text=message[2])
                elif kind == "result":
                    _, path, result, exported = message
                    self.results[path] = result
                    self._show_result(result)
                    self.status.configure(text=f"{result['name']} 已保存到 {exported['markdown'].parent}")
                elif kind == "error":
                    self.status.configure(text=f"分析失败。{message[1]}")
                elif kind == "done":
                    self._set_analysis_running(False)
                    self.progress.configure(value=100)
                    _, total, failures, last_error = message
                    status = f"已完成 {total - failures}/{total} 个文件。"
                    if failures:
                        status += f" 最近一次失败，{last_error}"
                    self.status.configure(text=status)
                elif kind == "lyrics_error":
                    _, error, status_code = message
                    if status_code in {401, 403}:
                        error = f"{error}\n请在“API Key 设置”中检查密钥。"
                    self._set_analysis_running(False)
                    self.progress.configure(value=0)
                    self.status.configure(text="识别歌词失败。")
                    messagebox.showerror("无法识别歌词", error, parent=self)
                elif kind == "lyrics_result":
                    _, path, segments, status = message
                    self._set_analysis_running(False)
                    self.progress.configure(value=100)
                    if path in self.files:
                        self.lyrics[path] = segments
                    if self._selected_lyrics_path() == path:
                        self._show_lyrics(path)
                    self.status.configure(text=status)
        except queue.Empty:
            pass
        self.after(100, self._poll_messages)

    def _on_file_select(self, _event: object) -> None:
        selected = self.file_list.curselection()
        if selected and self.files[selected[0]] in self.results:
            self._show_result(self.results[self.files[selected[0]]])
        self._show_lyrics(self._selected_lyrics_path())

    def _clear_result_view(self) -> None:
        self.current_result = None
        self._selected_result_text = None
        for text in self.summary_values.values():
            self._set_result_text(text, "—")
            self._fit_summary_value(text)
        self._set_table(self.section_text, SECTION_HEADINGS, [])
        self._set_table(self.key_text, KEY_HEADINGS, [])
        self._set_result_text(self.raw_text, "")
        self._set_result_text(self.info_text, "选择已分析的音频查看和弦与乐器。")

    def _show_result(self, result: dict) -> None:
        self.current_result = result
        self._selected_result_text = None
        bpm_detail = "未确定"
        if result["bpm"] is not None:
            bpm_detail = f"{result['bpm']:g} BPM"
            if result.get("alternate_bpm") is not None:
                bpm_detail += f"\n参考速度 {result['alternate_bpm']:g} BPM"
        if result.get("project_bpm"):
            bpm_detail += f"\n工程速度 {result['project_bpm']}\n音频估算 {result.get('estimated_bpm', '—')}"
        key_confidence = result["global_key_confidence"]
        key_detail = "参考标注" if key_confidence is None else f"置信度 {key_confidence}"
        style_detail = result["style"]
        style_confidence = STYLE_CONFIDENCE.get(result.get("style_confidence"))
        if style_confidence:
            style_detail += f"\n置信度 {style_confidence}"
        style_info = style_detail
        if result.get("style_source") == "local_classifier":
            style_info += "\n来源 本地训练的风格分类器"
        if result.get("style_note"):
            style_info += f"\n{result['style_note']}"
        for key, value in (
            ("name", result["name"]), ("bpm", bpm_detail),
            ("key", f"{result['global_key']}\n{key_detail}"), ("style", style_detail),
        ):
            text = self.summary_values[key]
            self._set_result_text(text, value)
            self._fit_summary_value(text)
        self._show_analysis_tables(result)
        self._set_result_text(self.raw_text, json.dumps(result, ensure_ascii=False, indent=2))
        source = "参考标注" if result.get("annotation_source") == "reference" else "自动分析"
        vocal_status = {
            "disabled": "已关闭",
            "separated": "已在本机分离",
            "unavailable": "模型不可用",
            "not_applicable": "未运行",
        }.get(result.get("vocal_analysis"), "未运行")
        info = [
            f"标注来源\n{source}",
            "",
            f"人声分析\n{vocal_status}",
            "",
            f"风格\n{style_info}",
            "",
            f"和弦走向\n{result.get('chord_progression', '—')}",
            "",
            f"乐器音色\n{result.get('instrument_description', '—')}",
            "",
            f"场景\n{result.get('scene_description', '—')}",
            "",
            f"情绪\n{result.get('mood_description', '—')}",
        ]
        self._set_result_text(self.info_text, "\n".join(info))

    def _show_analysis_tables(self, result: dict) -> None:
        from audio_analysis import section_row

        mode = TIME_FORMATS[self.time_format.get()]
        key_rows = []
        for item in result["key_changes"]:
            confidence = item["confidence"]
            key_rows.append([
                format_timestamp(item["start"], mode), item["key"], "—" if confidence is None else str(confidence),
            ])
        self._set_table(self.key_text, KEY_HEADINGS, key_rows)
        section_rows = []
        for item in result["sections"]:
            row = section_row(result, item)
            row[0] = format_timestamp(item.get("marker_seconds", item["start"]), mode)
            section_rows.append(row)
        self._set_table(self.section_text, SECTION_HEADINGS, section_rows)

    def _refresh_timestamp_tables(self, _event: object) -> None:
        if self.current_result is not None:
            self._show_analysis_tables(self.current_result)
        self._show_lyrics(self._selected_lyrics_path())


def main() -> None:
    try:
        APP_CACHE.configure_kernel_cache()
    except OSError:
        # Inaccessible cache storage does not prevent offline analysis.
        pass
    if len(sys.argv) > 1 and sys.argv[1] == "--analyze":
        from audio_analysis import analyze_audio, export_result

        arguments = sys.argv[3:]
        allowed = {"--no-reference", "--no-vocals"}
        options = {argument for argument in arguments if argument in allowed}
        sources = [argument for argument in arguments if argument not in allowed]
        if not sources:
            raise SystemExit("Usage: OfflineAudioMarker --analyze OUTPUT_DIR [--no-reference] [--no-vocals] AUDIO [AUDIO ...]")
        try:
            for source in sources:
                result = analyze_audio(source, use_reference="--no-reference" not in options, use_vocals="--no-vocals" not in options)
                export_result(result, sys.argv[2])
        finally:
            APP_CACHE.prune_kernels()
        return
    app = AudioMarkerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
