# V-D Splitter

Desktop voice extraction and denoise tool for video/audio files.

V-D means VOICE-DENOISE. The tool extracts audio through ffmpeg, separates the
voice stem with Demucs, cleans it with a local neural denoiser and can match the
result to a same-shoot lavalier or recorder reference.

Maintained by Maslodium.

## Features

- Imports video and audio through ffmpeg via `imageio-ffmpeg`.
- Supports common formats: MP4, MKV, MOV, AVI, WEBM, WMV, MP3, WAV, FLAC, M4A,
  AAC, OGG, OPUS and more.
- Separates voice / no-voice stems through Demucs `--two-stems=vocals`.
- Cleans speech with Facebook Research Denoiser.
- Can use a reference recording from the same shoot to match level, broad tone
  and soft dynamics.
- Saves `extracted.wav`, `voice_raw.wav`, `voice_clean.wav`, optional
  `background_no_voice.wav` and optional `reference.wav`.
- Selects CUDA or CPU Torch wheels during Windows bootstrap install.
- Supports `auto`, `cuda`, `mps` and `cpu` device modes.

## Requirements

- Windows 10/11 or macOS.
- Python 3.10+ available through the system launcher.
- Internet connection on first install for Python packages, Torch wheels,
  Demucs/Denoiser weights and ffmpeg support.
- NVIDIA GPU is optional on Windows.
- Apple Silicon acceleration is optional on macOS.

## Installers

Current release:

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

## Possible Improvements

- Denoise strength presets for light cleanup, room noise and heavy camera hiss.
- Compressor, limiter, de-esser and loudness target controls.
- Better Reference Match based on several lavalier samples from the same shoot.
- Learned camera-to-lav restoration profile.
- Batch processing for folders.

## Notes

Reference Match currently performs conservative level, broad spectral and soft
dynamics matching. It is not yet a trained camera-to-lav restoration model.

Check upstream model/code licenses before redistributing pretrained weights or
commercial bundles.

---

# V-D Splitter

Настольная утилита для извлечения и очистки голоса из видео и аудио.

V-D означает VOICE-DENOISE. Программа достаёт аудио через ffmpeg, отделяет
голос через Demucs, чистит его локальным нейросетевым denoiser и может
подгонять результат под образец с петлички или рекордера из той же съёмки.

Поддерживает Maslodium.

## Возможности

- Импорт видео и аудио через ffmpeg / `imageio-ffmpeg`.
- Поддержка популярных форматов: MP4, MKV, MOV, AVI, WEBM, WMV, MP3, WAV, FLAC,
  M4A, AAC, OGG, OPUS и другие.
- Разделение voice / no-voice через Demucs `--two-stems=vocals`.
- Очистка речи через Facebook Research Denoiser.
- Подгонка уровня, широкого тембра и мягкой динамики по референсной записи из
  той же съёмки.
- Сохранение `extracted.wav`, `voice_raw.wav`, `voice_clean.wav`,
  опционально `background_no_voice.wav` и `reference.wav`.
- Выбор CUDA или CPU Torch во время Windows bootstrap install.
- Режимы устройства: `auto`, `cuda`, `mps`, `cpu`.

## Требования

- Windows 10/11 или macOS.
- Python 3.10+ в системном запускателе.
- Интернет при первой установке для Python-пакетов, Torch, весов моделей и
  ffmpeg.
- NVIDIA GPU на Windows не обязателен.
- Apple Silicon acceleration на macOS не обязателен.

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

## CLI

```powershell
python pipeline.py input.mp4 --out output --device auto
python pipeline.py camera.wav --reference-audio lav_take.wav --out output
```

## Возможные Доработки

- Пресеты силы denoise для лёгкой чистки, комнатного шума и сильного шума
  камеры.
- Compressor, limiter, de-esser и loudness target.
- Более точный Reference Match по нескольким образцам петлички из той же
  съёмки.
- Обучаемый профиль восстановления `camera -> lav`.
- Пакетная обработка папок.

## Примечания

Reference Match сейчас делает аккуратное выравнивание громкости, широкого тембра
и мягкой динамики. Это ещё не обученная модель восстановления `camera -> lav`.

Перед распространением весов моделей или коммерческой сборкой нужно отдельно
проверить лицензии upstream-проектов.
