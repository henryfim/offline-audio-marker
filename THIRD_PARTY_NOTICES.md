# Third-party notices

OfflineAudioMarker uses the components below. The Windows installer includes license texts from the Python runtime and installed wheels. Each component retains its upstream license.

## Model files and training data

| Component | Use in OfflineAudioMarker | Attribution |
| --- | --- | --- |
| YAMNet TFLite | Audio event scores for instrument and genre estimates. | Google, Apache License 2.0. The bundled file is Google's YAMNet classification model with metadata. [Model file](https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/audio_classification/android/lite-model_yamnet_classification_tflite_1.tflite) · [YAMNet source](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet) · [License](https://github.com/tensorflow/models/blob/master/LICENSE) |
| FMA Small | Training and evaluation of the local genre classifier. The distributed model contains numeric coefficients and feature normalization statistics. | Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst and Xavier Bresson, *FMA: A Dataset for Music Analysis*, ISMIR 2017. [Paper](https://arxiv.org/abs/1612.01840) · [Dataset and license details](https://github.com/mdeff/fma#license) |
| UVR MDX-Net | Optional vocal separation using a model obtained separately. | Copyright © 2022 Ultimate Vocal Remover. MIT License. Thanks to the Ultimate Vocal Remover developers for the model. [Project and model-use terms](https://github.com/Anjok07/ultimatevocalremovergui#license) · [Model download](https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR_MDXNET_1_9703.onnx) |

The FMA metadata is licensed under CC BY 4.0. The training manifest selects 320 entries, assigns train, validation and test splits, and includes identifying metadata and track-specific licenses. The archive and sampling method are recorded in `training/fma_provenance.json`. Use the track-specific license for any reuse of a recording. [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## Windows runtime

The package includes CPython 3.13.2, Tcl/Tk 8.6.15, and Python libraries for audio I/O, signal processing, inference, and the interface. The bundled wheel licenses are under `_internal/licenses/`. Additional texts for YAMNet, UVR, CPython, Tcl/Tk and the libsndfile codecs are in `third-party-licenses/`.

| Component | Upstream source and terms |
| --- | --- |
| CPython | [Python 3.13.2 source](https://github.com/python/cpython/tree/v3.13.2) · [Terms](https://docs.python.org/3.13/license.html) |
| Tcl and Tk | [Tcl 8.6.15](https://github.com/tcltk/tcl/tree/core-8-6-15) · [Tk 8.6.15](https://github.com/tcltk/tk/tree/core-8-6-15) |
| NumPy, SciPy, librosa, Numba, llvmlite | [NumPy](https://numpy.org/) · [SciPy](https://scipy.org/) · [librosa](https://librosa.org/) · [Numba](https://numba.pydata.org/) · [llvmlite](https://llvmlite.readthedocs.io/) |
| SoundFile 0.14.0 and libsndfile 1.2.2 | [Python binding](https://github.com/bastibe/python-soundfile/tree/0.14.0) · [libsndfile source and build information](#audio-runtime-source) |
| Python-SoXR 1.1.0 | [Upstream project](https://github.com/dofuuz/python-soxr/tree/v1.1.0) · [Source archive](https://files.pythonhosted.org/packages/ed/11/27cebce4a108f77afea7c80545115536b45e3f11ebfb914f638fdd9ba847/soxr-1.1.0.tar.gz) |
| LiteRT and ONNX Runtime | [LiteRT](https://github.com/google-ai-edge/LiteRT) · [ONNX Runtime](https://github.com/microsoft/onnxruntime) |
| TkinterDnD2 and tkdnd | [Python binding](https://github.com/Eliav2/tkinterdnd2) · [tkdnd](https://github.com/petasis/tkdnd) |
| imageio-ffmpeg | The Python interface locates an external FFmpeg executable for additional formats. [Python interface](https://github.com/imageio/imageio-ffmpeg) · [FFmpeg downloads](https://ffmpeg.org/download.html) |

## Audio runtime source

The release asset `OfflineAudioMarker-LGPL-Source.zip` contains audio library source archives and the corresponding build recipe.

The Windows build uses `_internal/_soundfile_data/libsndfile_x64.dll` from the SoundFile 0.14.0 Windows wheel. To use a compatible build of libsndfile, close the application and replace the DLL with the new one using the same file name. Python-SoXR can likewise be rebuilt from its source archive with the matching Python ABI, then used in place of `_internal/soxr/soxr_ext.pyd`. The LGPL license terms apply to these components. [SoundFile source](https://github.com/bastibe/python-soundfile/tree/0.14.0) · [libsndfile source archive](https://github.com/libsndfile/libsndfile/releases/download/1.2.2/libsndfile-1.2.2.tar.xz) · [Windows DLL build workflow and triplets](https://github.com/bastibe/libsndfile-binaries/tree/a3e6f9769d0c7e91d2d036cf0fdbe5b4bbf18b87) · [Python-SoXR source archive](https://files.pythonhosted.org/packages/ed/11/27cebce4a108f77afea7c80545115536b45e3f11ebfb914f638fdd9ba847/soxr-1.1.0.tar.gz)

The libsndfile DLL includes statically linked codecs. Its upstream Windows recipe uses vcpkg plus a custom triplet. The December 2024 vcpkg port definitions contain the upstream revisions, hashes and patches for the matching codec versions. [Build workflow](https://github.com/bastibe/libsndfile-binaries/blob/a3e6f9769d0c7e91d2d036cf0fdbe5b4bbf18b87/.github/workflows/build-libs.yml) · [x64 triplet](https://github.com/bastibe/libsndfile-binaries/blob/a3e6f9769d0c7e91d2d036cf0fdbe5b4bbf18b87/triplets/x64-windows-custom.cmake) · [vcpkg port snapshot](https://github.com/microsoft/vcpkg/tree/2024.12.16/ports)

| Codec library | Source and license |
| --- | --- |
| FLAC 1.4.3 | [Code](https://github.com/xiph/flac/tree/1.4.3) · [Library license](https://github.com/xiph/flac/blob/1.4.3/COPYING.Xiph) |
| Ogg 1.3.5 | [Code and license](https://github.com/xiph/ogg/tree/v1.3.5) |
| Vorbis 1.3.7 | [Code and license](https://github.com/xiph/vorbis/tree/v1.3.7) |
| Opus 1.5.2 | [Code and license](https://github.com/xiph/opus/tree/v1.5.2) |
| mpg123 1.32.9 | [Source archive](https://sourceforge.net/projects/mpg123/files/mpg123/1.32.9/) |
| LAME 3.100 | [Source archive](https://sourceforge.net/projects/lame/files/lame/3.100/) |

