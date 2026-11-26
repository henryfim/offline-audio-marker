# -*- mode: python ; coding: utf-8 -*-
import os
from importlib.metadata import distributions
from pathlib import Path, PurePosixPath

from PyInstaller.utils.hooks import collect_dynamic_libs

source = Path(SPECPATH)
include_local_assets = os.environ.get("OFFLINE_AUDIO_MARKER_LOCAL_ASSETS") == "1"
public_models = ("yamnet.tflite", "local_style.json")
local_models = ("UVR_MDXNET_1_9703.onnx", "reference_annotations.json")
models = public_models + local_models if include_local_assets else public_models
datas = [(str(source / "models" / name), "models") for name in models]

# Include vendor notices and license files supplied by the installed wheels.
for distribution in distributions():
    package = distribution.metadata["Name"]
    for entry in distribution.files or ():
        if ".." in entry.parts:
            continue
        if "licenses" in entry.parent.parts or entry.name.lower().startswith(
            ("license", "licence", "copying", "notice", "thirdpartynotices")
        ):
            location = distribution.locate_file(entry)
            if location.is_file():
                datas.append((str(location), f"licenses/{package}/{entry.parent.as_posix()}"))

# The waveform interpreter uses CPU inference. WebGPU and OpenVINO
# plug-ins only serve explicit accelerator requests.
lite_rt_dlls = {"liblitert.dll", "pywrap_litert_common.dll"}
binaries = [
    (path, destination)
    for path, destination in collect_dynamic_libs("ai_edge_litert")
    if Path(path).name.lower() in lite_rt_dlls
]

a = Analysis(
    [str(source / "app.py")],
    pathex=[str(source)],
    binaries=binaries,
    datas=datas,
    hiddenimports=["ai_edge_litert.interpreter"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["matplotlib", "pandas", "IPython", "jupyter", "lxml", "PIL"],
    noarchive=False,
    optimize=0,
)

# The upstream imageio_ffmpeg hook adds its bundled FFmpeg executable.
# Public builds retain the Python finder and resolve a user-provided executable.
if not include_local_assets:
    def is_ffmpeg_binary(entry):
        destination = PurePosixPath(entry[0].replace("\\", "/"))
        return (
            destination.parts[:2] == ("imageio_ffmpeg", "binaries")
            and destination.suffix.lower() == ".exe"
        )

    a.datas = [entry for entry in a.datas if not is_ffmpeg_binary(entry)]
    a.binaries = [entry for entry in a.binaries if not is_ffmpeg_binary(entry)]

pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="OfflineAudioMarker",
    console=False,
    upx=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="OfflineAudioMarker",
    upx=False,
)
