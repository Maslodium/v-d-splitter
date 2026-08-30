# V-D Splitter

VOICE-DENOISE Splitter is a local desktop tool for extracting speech from
audio/video files, separating the voice stem, denoising it with neural models
and polishing the result for editing.

```text
video/audio -> ffmpeg WAV -> Demucs voice split -> neural denoise -> voice polish
```

The interface uses the same cyber-metal rack language as M-A Splitter: dark
working panels, brushed-metal rails, custom title bar, Oxanium display font and
cyan/magenta control accents.

## Features

- Accepts common video and audio formats: MP4, MKV, MOV, AVI, WEBM, WMV, MP3,
  WAV, FLAC, M4A, AAC, OGG, OPUS and more.
- Downloads/uses ffmpeg through `imageio-ffmpeg`.
- Splits voice with Demucs `--two-stems=vocals`.
- Cleans voice with a local neural speech denoiser.
- Optional Reference Match: provide a lav/recorder sample from the same shoot,
  and V-D Splitter matches broad tone, RMS level and dynamics after denoise.
- Saves `extracted.wav`, `voice_raw.wav`, `voice_clean.wav`,
  `background_no_voice.wav` when enabled, and `reference.wav` when used.
- Windows installer scans for NVIDIA GPU and installs CUDA Torch wheels when
  available.
- macOS build supports PyTorch `mps` device when Apple Metal acceleration is
  available, with CPU fallback.

## Install

Windows:

```powershell
.\build_installer.ps1
```

Output:

```text
dist\Install V-D Splitter.exe
```

The Windows bootstrap installs into:

```text
%LOCALAPPDATA%\V-D Splitter
```

macOS builds must be produced on macOS:

```bash
bash build_macos.sh
```

Output:

```text
dist/V-D-Splitter-macOS.zip
```

The macOS bootstrap installs into:

```text
~/Applications/V-D Splitter
```

Both bootstrap installers create a venv, install PyTorch, install app
dependencies, provide ffmpeg support and create a launch shortcut/script.

## Run From Source

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\pythonw.exe gui.py
```

CPU machines can install Torch without the CUDA index. macOS can use standard
Torch wheels and the `auto`/`mps` device path.

## CLI

```powershell
python pipeline.py input.mp4 --out output --device auto
python pipeline.py camera.wav --reference-audio lav_take.wav --out output
```

## Audio Roadmap

- Neural denoise strength presets for speech, interviews, room noise and heavy
  camera hiss.
- Compressor, limiter, de-esser and loudness target controls for broadcast-like
  exports.
- Reference Match can evolve into a learned same-shoot restoration model: give
  it camera audio plus clean lav samples from other takes, and it learns the
  spectral/dynamic gap between the camera and lav chain.
- Batch mode for folders, with per-file logs and failed-file recovery.

## Licenses And Models

- Demucs is used for source separation.
- Facebook Research Denoiser is used for neural speech denoising.
- Oxanium is bundled under the SIL Open Font License 1.1.

Check upstream model/code licenses before redistributing pretrained weights or
commercial bundles.

## Русское Описание

**V-D Splitter** расшифровывается как **VOICE-DENOISE Splitter**. Это локальная
программа для Windows и macOS, которая достаёт звук из видео или аудиофайла,
отделяет голос, чистит его нейросетью и готовит дорожку к монтажу.

Интерфейс приведён к стилю M-A Splitter: тёмные рабочие области без лишней
текстуры, металлические полосы, кастомная верхняя панель окна, Oxanium для
заголовков и киберпанк-акценты в палитре.

## Возможности

- Поддержка популярных видео и аудио форматов: MP4, MKV, MOV, AVI, WEBM, WMV,
  MP3, WAV, FLAC, M4A, AAC, OGG, OPUS и другие.
- `ffmpeg` подтягивается через `imageio-ffmpeg`.
- Голос отделяется через Demucs.
- Шум убирается локальным нейросетевым denoiser.
- Режим Reference Match: можно дать программе образец с петлички или рекордера
  из той же съёмки, и она подгонит очищенный камерный звук по уровню, тембру и
  мягкой динамике.
- На Windows установщик сканирует NVIDIA GPU и выбирает CUDA/CPU Torch.
- На macOS поддержан режим `mps` для Apple Silicon, если PyTorch видит Metal.

## Что Добавить Дальше

- Ручки denoise strength, компрессии, лимитера, de-esser и target loudness.
- Пресеты: интервью, съёмка на камеру, подкаст, шумная улица, помещение.
- Более умный Reference Match: обучаемый локальный профиль камеры и петлички по
  нескольким дублям одной съёмки.
- Пакетная обработка папки с видео.
