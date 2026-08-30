# Voice-Denoise Splitter

Windows desktop tool for extracting voice from audio/video files and cleaning it
with local neural models.

```text
video/audio -> ffmpeg extracted WAV -> Demucs voice split -> neural speech denoise
```

The interface follows the cyber-metal rack style of M-A Splitter, but this is a
separate project focused on voices from video, interviews, podcasts, streams and
mixed audio.

## Features

- Accepts common video and audio formats: MP4, MKV, MOV, AVI, WEBM, WMV, MP3,
  WAV, FLAC, M4A, AAC, OGG, OPUS and more.
- Downloads/uses ffmpeg through `imageio-ffmpeg`.
- Splits voice with Demucs `--two-stems=vocals`.
- Cleans voice with a local neural speech denoiser.
- Saves:
  - `audio/extracted.wav`
  - `audio/voice_raw.wav`
  - `audio/voice_clean.wav`
  - `audio/background_no_voice.wav` when enabled
- Scans the system during installation and installs CUDA Torch wheels when an
  NVIDIA GPU is detected.

## Install From EXE

Build the installer:

```powershell
.\build_installer.ps1
```

The generated installer is:

```text
dist\Install Voice-Denoise Splitter.exe
```

When launched, the installer copies the app to:

```text
%LOCALAPPDATA%\Voice-Denoise Splitter
```

Then it creates a virtual environment, installs dependencies, downloads ffmpeg
support, creates a desktop shortcut and selects CUDA/CPU Torch wheels based on
the local system scan.

## Run From Source

```powershell
.\run_from_source.ps1
```

Or manually:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\pythonw.exe gui.py
```

Use CPU wheels instead of the CUDA index on machines without NVIDIA GPUs.

## CLI

```powershell
python pipeline.py input.mp4 --out output --device auto
```

## Licenses And Models

- Demucs is used for source separation.
- Facebook Research Denoiser is used for neural speech denoising.
- The bundled Oxanium font is distributed under SIL Open Font License 1.1.

See the upstream projects for their model/code licenses before redistributing
pretrained weights or commercial bundles.
