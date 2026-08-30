# V-D Splitter

Desktop voice extraction and denoise tool for video/audio files.

V-D means VOICE-DENOISE. The tool extracts audio through ffmpeg, separates the
voice stem with Demucs, cleans it with a local neural denoiser and can match the
result to a same-shoot lavalier or recorder reference.

The interface follows the same dark cyber-metal rack style as M-A Splitter:
custom title bar, brushed-metal section rails, dark work panels, Oxanium display
font and cyan/magenta accents.

Maintained by Maslodium.

## Features

- Video/audio import through ffmpeg via `imageio-ffmpeg`.
- Voice / no-voice separation through Demucs `--two-stems=vocals`.
- Local neural speech denoise through Facebook Research Denoiser.
- Reference Match for same-shoot lavalier or recorder samples.
- Output files: `extracted.wav`, `voice_raw.wav`, `voice_clean.wav`,
  optional `background_no_voice.wav` and `reference.wav`.
- Windows CUDA/CPU dependency selection during bootstrap install.
- macOS `mps` / CPU device path for Apple Silicon machines.

## Requirements

- Windows 10/11 or macOS.
- Python 3.10+ available through the system launcher.
- Internet connection on first install for Python packages, Torch wheels,
  Demucs/Denoiser weights and ffmpeg support.
- NVIDIA GPU is optional on Windows. Apple Silicon acceleration is optional on
  macOS.

## Installers

Current release:

```text
https://github.com/Maslodium/v-d-splitter/releases/tag/v0.1.0
```

Windows asset:

```text
Install-V-D-Splitter.exe
```

macOS asset:

```text
V-D-Splitter-macOS.zip
```

## Run From Source

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\pythonw.exe gui.py
```

CPU machines can install Torch without the CUDA index. On macOS, use standard
Torch wheels and `--device auto` or `--device mps`.

## CLI

```powershell
python pipeline.py input.mp4 --out output --device auto
python pipeline.py camera.wav --reference-audio lav_take.wav --out output
```

## Notes

Reference Match currently performs conservative level, broad spectral and soft
dynamics matching. It is not yet a trained camera-to-lav restoration model.

Good next audio controls are denoise strength, compressor, limiter, de-esser,
loudness target and batch processing.

Check upstream model/code licenses before redistributing pretrained weights or
commercial bundles.

---

# V-D Splitter

Настольная утилита для извлечения и очистки голоса из видео и аудио.

V-D означает VOICE-DENOISE. Программа достаёт аудио через ffmpeg, отделяет
голос через Demucs, чистит его локальным нейросетевым denoiser и может
подгонять результат под образец с петлички или рекордера из той же съёмки.

Интерфейс сделан в том же тёмном cyber-metal rack стиле, что и M-A Splitter:
собственная верхняя панель окна, металлические полосы разделов, тёмные рабочие
панели, шрифт Oxanium и cyan/magenta акценты.

Поддерживает Maslodium.

## Возможности

- Импорт видео и аудио через ffmpeg / `imageio-ffmpeg`.
- Разделение voice / no-voice через Demucs `--two-stems=vocals`.
- Локальная нейросетевая очистка речи через Facebook Research Denoiser.
- Reference Match для образцов с петлички или рекордера из той же съёмки.
- Выходные файлы: `extracted.wav`, `voice_raw.wav`, `voice_clean.wav`,
  опционально `background_no_voice.wav` и `reference.wav`.
- На Windows установщик выбирает CUDA/CPU зависимости.
- На macOS есть путь `mps` / CPU для Apple Silicon.

## Требования

- Windows 10/11 или macOS.
- Python 3.10+ в системном запускателе.
- Интернет при первой установке для Python-пакетов, Torch, весов моделей и
  ffmpeg.
- NVIDIA GPU на Windows не обязателен. Apple Silicon acceleration на macOS тоже
  опционален.

## Установщики

Текущий релиз:

```text
https://github.com/Maslodium/v-d-splitter/releases/tag/v0.1.0
```

Windows:

```text
Install-V-D-Splitter.exe
```

macOS:

```text
V-D-Splitter-macOS.zip
```

## Запуск из исходников

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\pythonw.exe gui.py
```

## Примечания

Reference Match сейчас делает аккуратное выравнивание громкости, широкого тембра
и мягкой динамики. Это ещё не обученная модель восстановления `camera -> lav`.

Ближайшие полезные регуляторы: сила denoise, compressor, limiter, de-esser,
loudness target и пакетная обработка.
