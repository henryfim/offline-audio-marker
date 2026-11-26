"""Store the DeepSeek ASR API key in the current user's Windows credentials."""

from __future__ import annotations

import ctypes as ct
import sys
from ctypes import wintypes as wt
from functools import lru_cache

TARGET = "OfflineAudioMarker/DeepSeekASR"
GENERIC_CREDENTIAL = 1
LOCAL_MACHINE = 2
ERROR_NOT_FOUND = 1168


class Credential(ct.Structure):
    _fields_ = [
        ("Flags", wt.DWORD), ("Type", wt.DWORD), ("TargetName", wt.LPWSTR),
        ("Comment", wt.LPWSTR), ("LastWritten", wt.FILETIME),
        ("CredentialBlobSize", wt.DWORD), ("CredentialBlob", ct.POINTER(ct.c_ubyte)),
        ("Persist", wt.DWORD), ("AttributeCount", wt.DWORD), ("Attributes", ct.c_void_p),
        ("TargetAlias", wt.LPWSTR), ("UserName", wt.LPWSTR),
    ]


@lru_cache(maxsize=1)
def _windows_api():
    if sys.platform != "win32":
        raise OSError("此系统没有 Windows 凭据管理器。")
    api = ct.WinDLL("advapi32", use_last_error=True)
    api.CredReadW.argtypes = [wt.LPCWSTR, wt.DWORD, wt.DWORD, ct.POINTER(ct.POINTER(Credential))]
    api.CredReadW.restype = wt.BOOL
    api.CredWriteW.argtypes = [ct.POINTER(Credential), wt.DWORD]
    api.CredWriteW.restype = wt.BOOL
    api.CredDeleteW.argtypes = [wt.LPCWSTR, wt.DWORD, wt.DWORD]
    api.CredDeleteW.restype = wt.BOOL
    api.CredFree.argtypes = [ct.c_void_p]
    api.CredFree.restype = None
    return api


def load_key() -> str | None:
    api = _windows_api()
    credential = ct.POINTER(Credential)()
    if not api.CredReadW(TARGET, GENERIC_CREDENTIAL, 0, ct.byref(credential)):
        error = ct.get_last_error()
        if error == ERROR_NOT_FOUND:
            return None
        raise ct.WinError(error)
    try:
        return ct.string_at(credential.contents.CredentialBlob, credential.contents.CredentialBlobSize).decode("utf-8")
    except UnicodeError:
        raise OSError("Windows 凭据中的 DeepSeek API Key 无法读取。") from None
    finally:
        api.CredFree(credential)


def save_key(value: str) -> None:
    value = value.strip()
    if not value or not value.isascii() or not value.isprintable() or any(char.isspace() for char in value) or len(value) > 512:
        raise OSError("请输入有效的 DeepSeek API Key。")
    blob = (ct.c_ubyte * len(value)).from_buffer_copy(value.encode("utf-8"))
    credential = Credential(Type=GENERIC_CREDENTIAL, TargetName=TARGET,
                            CredentialBlobSize=len(blob), CredentialBlob=blob, Persist=LOCAL_MACHINE)
    if not _windows_api().CredWriteW(ct.byref(credential), 0):
        raise ct.WinError(ct.get_last_error())


def delete_key() -> None:
    if not _windows_api().CredDeleteW(TARGET, GENERIC_CREDENTIAL, 0):
        error = ct.get_last_error()
        if error != ERROR_NOT_FOUND:
            raise ct.WinError(error)
