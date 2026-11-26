# Offline Audio Marker

Offline Audio Marker 是一个 Windows 音频标注工具。把音频拖进窗口，即可生成带时间点的段落表，估算速度和调性，并给出主奏与伴奏的乐器线索。需要做歌词时间轴时，可以先在本机分离人声，再手动提交片段给 DeepSeek 转写。

原型来自一个具体的听辨任务：逐首核对段落从哪一拍开始，再把乐器和调性填进表格。同一份标注既要方便在界面里查看，也要能导入音频编辑工具，所以程序把分析结果存为秒数，显示时才转成 `1.13.52` 这样的格式。

## 安装和使用

1. 在 [Releases](../../releases) 下载 Windows x64 安装包及 SHA-256 文件。可在 PowerShell 中运行 `Get-FileHash .\OfflineAudioMarker-Setup-x64.msi -Algorithm SHA256`，与发布页提供的值核对。
2. 安装后从开始菜单打开软件。把音频拖入窗口或点「添加音频」，再点「分析所选」。表格允许选中任意一段文本复制。
3. 在「时间显示」中切换格式。默认把 73.52 秒显示成 `1.13.52`。结果目录由界面指定，默认写入用户的文档目录。
4. WAV、FLAC、MP3 等可由 SoundFile 读取。M4A、AAC、WMA 等解码失败时会尝试外部 FFmpeg。安装 FFmpeg 并加入 PATH，也可以设置 `IMAGEIO_FFMPEG_EXE` 指向可执行文件。
5. 需要本地人声分离时，把对应的 UVR 模型放入安装目录下的 `models` 文件夹，重启后勾选「离线人声分离」。模型文件名及下载来源见 [第三方声明](THIRD_PARTY_NOTICES.md)。

### 歌词时间轴

在「API Key 设置」保存 DeepSeek Key，选中一首歌，再点「识别歌词（DeepSeek）」。应用先用本地音量包络找出活动片段。启用离线人声分离时，人声活动曲线也参与切分。只有手动识别歌词时，片段才会发给 DeepSeek。

每条歌词的起止来自本地切分，可能覆盖多行歌词。导出的 LRC 使用 `[01:13.52]` 格式，CSV 保留起点和终点的秒数。API Key 保存在 Windows 凭据管理器中。

## 音频分析怎么做

解码后，程序先估算节拍、小节网格和调性，再通过重复结构和响度变化定位段落。网格置信度足够时，会用小节起拍辅助定位。YAMNet 输出的音频类别和分频段特征给乐器识别提供线索，自动标注适合先定位可疑段落，再进行人工核对。

主要分析任务在线程中运行，界面用队列接收进度和结果。分离的音轨在内存中处理，分析缓存按文件内容、设置和模型状态区分，界面的「清理缓存」可以清除分析记录和编译缓存。段落报告可导出为 Markdown、CSV 或 JSON，也可以生成 Audacity 点标签。

| 模块 | 负责的工作 |
| --- | --- |
| `app.py`、`analysis_cache.py` | 拖入文件、后台任务、进度回传、缓存 |
| `audio_analysis.py`、`rhythm_analysis.py`、`section_analysis.py` | 解码、声学特征、小节网格和段落定位 |
| `instrument_analysis.py`、`vocal_separation.py` | YAMNet 音频标签、可选的 ONNX 人声分离 |
| `local_style.py`、`train_style.py` | 风格分类推理和训练 |
| `lyrics_deepseek.py`、`lyrics_analysis.py` | 音频切片、ASR、时间显示和歌词导出 |

## 本地训练和验证

风格分类的补充模型使用 FMA Small 的 320 首子集，提取 294 个声学特征，训练八类多项逻辑回归。训练、验证、测试分别包含 192、64、64 首。64 首测试集上的无拒判准确率为 31.25%，Macro-F1 为 0.312。应用只在风格规则未给出结果且分类置信度达到 0.40 时采用它。这个测试分数仅对应风格分类。

`training/` 中保存了采样清单、数据来源和训练报告。按各录音许可获取训练数据后，可以从同一份清单重新训练。读取报告时，可用 `annotation_source` 查看段落来源。本地参考资料只对文件哈希完全匹配的录音生效，公开安装包使用自动分析路径。

本次验证用合成 WAV 和 M4A 检查了两条解码路径及四种报告导出。也测试了分钟进位、LRC 时间格式和没有模型时的人声分离选项。用 `py -3.13 verify_release.py .\dist-public\OfflineAudioMarker\OfflineAudioMarker.exe` 可复现 WAV 测试。

## 从源码运行

需要 Windows x64 和 Python 3.13。首次运行用 `run.bat` 建环境，或在 PowerShell 执行下列命令。

```powershell
.\install_dependencies.ps1
.\.venv\Scripts\python.exe .\app.py
```

```powershell
.\.venv\Scripts\python.exe .\app.py --analyze .\output --no-vocals --no-reference .\sample.wav
.\build_exe.ps1
```

打包脚本默认使用独立的构建环境，生成的程序在 `dist-public/OfflineAudioMarker`。公开版解压后的程序文件为 333.94 MiB，原安装的程序文件为 510.63 MiB，体积减少 34.6%。执行 `installer\build_installer.ps1` 可用 WiX 6 构建约 103 MiB 的 MSI 和 SHA-256 校验文件。

## 项目材料

项目介绍包含音频链路的设计取舍和测试证据，可在 [项目介绍 Word 文档](docs/OfflineAudioMarker-Project.docx) 中查看。FMA 数据、模型和运行库的来源记录在 [第三方声明](THIRD_PARTY_NOTICES.md)。

