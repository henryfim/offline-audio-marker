"""Transcribe local audio phrases through DeepSeek's official ASR endpoint."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.ndimage import binary_closing, gaussian_filter1d

ASR_URL = "https://api.deepseek.com/v1/tools/asr"
SAMPLE_RATE = 16000
MAX_SEGMENT_SECONDS = 12.0
FRAME_SECONDS = 0.025


class DeepSeekASRError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        return None


def _transcribe(wav: bytes, api_key: str) -> str:
    boundary = uuid.uuid4().hex
    opening = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="segment.wav"\r\n'
               'Content-Type: audio/wav\r\n\r\n').encode("ascii")
    body = opening + wav + f"\r\n--{boundary}--\r\n".encode("ascii")
    request = urllib.request.Request(
        ASR_URL, data=body, method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json",
                 "Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.build_opener(_NoRedirect()).open(request, timeout=90) as response:
            data = json.loads(response.read(1024 * 1024))
    except urllib.error.HTTPError as error:
        messages = {
            400: "DeepSeek 拒绝了音频片段。",
            401: "DeepSeek API Key 无效或已过期。",
            402: "DeepSeek 账户余额不足。",
            403: "此 DeepSeek API Key 无权使用语音识别。",
            429: "DeepSeek 请求过多，请稍后重试。",
            503: "DeepSeek 服务繁忙，请稍后重试。",
        }
        raise DeepSeekASRError(messages.get(error.code, f"DeepSeek 请求失败（HTTP {error.code}）。"), error.code) from None
    except (urllib.error.URLError, TimeoutError):
        raise DeepSeekASRError("无法连接 DeepSeek，或请求已超时。") from None
    except (UnicodeError, ValueError):
        raise DeepSeekASRError("DeepSeek 返回了无法解析的响应。") from None
    if not isinstance(data, dict) or not isinstance(data.get("text"), str):
        raise DeepSeekASRError("DeepSeek 响应缺少转写文本。")
    return " ".join(data["text"].split())


def _wav(audio: np.ndarray) -> bytes:
    samples = np.rint(np.clip(audio, -1, 1) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
        output.writeframes(samples.tobytes())
    return buffer.getvalue()


def _phrase_boundaries(mono: np.ndarray, *, vocal_activity=None) -> list[tuple[int, int]]:
    hop = round(SAMPLE_RATE * FRAME_SECONDS)
    frames = np.pad(mono, (0, (-len(mono)) % hop)).reshape(-1, hop)
    envelope = np.sqrt(gaussian_filter1d(np.mean(frames ** 2, axis=1), 1.2))
    level = float(np.quantile(envelope, 0.90))
    if float(envelope.max()) < 0.0001:
        return []
    threshold = max(0.00015, level * (0.12 if vocal_activity is not None else 0.04))
    active = envelope >= threshold
    if vocal_activity is not None and vocal_activity.times.size:
        times = (np.arange(len(frames)) + 0.5) * FRAME_SECONDS
        relative = np.interp(times, vocal_activity.times, vocal_activity.strength)
        active &= relative >= 0.30
    if len(active) > 12:
        # Join very short pauses so adjacent words stay in the same request.
        active = binary_closing(np.pad(active, (6, 6)), structure=np.ones(13, dtype=bool))[6:-6] | active
    edges = np.flatnonzero(np.diff(np.r_[False, active, False].astype(np.int8)))
    spans: list[tuple[int, int]] = []
    min_frames = round(0.20 / FRAME_SECONDS)
    max_frames = round(MAX_SEGMENT_SECONDS / FRAME_SECONDS)
    lower_cut = round(4.0 / FRAME_SECONDS)
    preferred_cut = round(7.5 / FRAME_SECONDS)
    upper_cut = round(9.5 / FRAME_SECONDS)
    for start, end in zip(edges[::2], edges[1::2]):
        if end - start < min_frames:
            continue
        while end - start > max_frames:
            candidates = np.arange(start + lower_cut, start + upper_cut)
            cost = envelope[candidates] / max(level, 0.0001)
            cost += 0.12 * np.abs(candidates - start - preferred_cut) / lower_cut
            cut = int(candidates[np.argmin(cost)])
            spans.append((int(start) * hop, cut * hop))
            start = cut
        spans.append((int(start) * hop, min(int(end) * hop, len(mono))))
    return spans


def recognize_lyrics(
    path: str | Path, *, api_key: str, use_vocals: bool = True,
    progress: Callable[[int, str], None] | None = None,
) -> list[dict]:
    """Return DeepSeek text aligned to each locally chosen phrase interval."""
    from audio_analysis import _load_audio
    from vocal_separation import describe_vocals, separate_vocals

    def report(value: int, message: str) -> None:
        if progress:
            progress(value, message)

    api_key = api_key.strip()
    if not api_key or not api_key.isascii() or not api_key.isprintable() or any(char.isspace() for char in api_key):
        raise DeepSeekASRError("请输入有效的 DeepSeek API Key。", 401)
    report(5, "读取音频")
    audio = _load_audio(Path(path), SAMPLE_RATE)
    if audio.size == 0:
        return []
    vocal_activity = None
    if use_vocals and audio.shape[-1] >= 4 * SAMPLE_RATE and np.max(np.abs(audio)) >= 0.0001:
        report(10, "本地分离人声")
        separated = separate_vocals(
            audio, SAMPLE_RATE,
            progress=lambda fraction: report(10 + round(38 * fraction), "本地分离人声"),
        )
        if separated is None:
            raise DeepSeekASRError("人声分离模型不可用。可取消勾选「离线人声分离」，再用原音频识别。")
        vocal_activity = describe_vocals(separated.vocals, separated.accompaniment, SAMPLE_RATE)
        audio = separated.vocals
        del separated

    mono = np.asarray(audio.mean(axis=0) if audio.ndim > 1 else audio, dtype=np.float32)
    if audio.ndim > 1 and np.max(np.abs(mono)) < 0.08 * np.max(np.abs(audio)):
        mono = audio[np.argmax(np.max(np.abs(audio), axis=1))]
    spans = _phrase_boundaries(mono, vocal_activity=vocal_activity)
    segments = []
    context = round(SAMPLE_RATE * 0.075)
    for index, (start, end) in enumerate(spans, 1):
        report(50 + round(49 * (index - 1) / len(spans)), f"DeepSeek 转写片段 {index}/{len(spans)}")
        clip = mono[max(0, start - context):min(len(mono), end + context)]
        text = _transcribe(_wav(clip), api_key)
        if text:
            segments.append({"start": start / SAMPLE_RATE, "end": end / SAMPLE_RATE, "text": text})
    report(100, "歌词识别完成")
    return segments
