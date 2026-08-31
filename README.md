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
- Optional Resemble Enhance backend for experimental speech enhancement.
- Can use a reference recording from the same shoot to match level, broad tone
  and soft dynamics.
- Can apply reusable shoot/reference profiles and trained `model.pt` reference
  models, including a profile learned from other takes when the target lav track
  was lost.
- Voice polish stage with compressor, de-esser, peak limiter and approximate
  loudness matching.
- Folder input for batch processing supported audio/video files.
- Community dataset tooling for Hugging Face style paired audio datasets.
- Baseline training script with checkpoint resume for camera-to-reference
  experiments.
- Explicit Hugging Face model download/publish commands.
- Saves `extracted.wav`, `voice_raw.wav`, `voice_clean.wav`, optional
  `background_no_voice.wav` and optional `reference.wav`.
- Selects CUDA or CPU Torch wheels during Windows bootstrap install.
- Supports `auto`, `cuda`, `mps` and `cpu` device modes.

## Requirements

- Windows 10/11, macOS or Linux.
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

Linux:

```text
V-D-Splitter-linux.tar.gz
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

Optional experimental backend:

```powershell
.\.venv\Scripts\python.exe install_optional.py --backend resemble-enhance
```

## CLI

```powershell
python pipeline.py input.mp4 --out output --device auto
python pipeline.py input.mp4 --denoise-backend resemble-enhance --out output
python pipeline.py camera.wav --reference-audio lav_take.wav --out output
python pipeline.py camera.wav --reference-profile profiles/camera_lav.json --out output
python pipeline.py camera.wav --reference-model models/community/model.pt --out output
python pipeline.py takes_folder --polish-preset camera-hiss --out output
```

## Community Training

```powershell
python community_training.py prepare-dataset --camera-dir camera --reference-dir reference --out vd_dataset
python community_training.py build-profile --dataset-dir vd_dataset --out profiles/camera_lav.json
python community_training.py build-shoot-profile --dataset-dir vd_dataset --out profiles/shoot.json
python train_reference_model.py --dataset-dir vd_dataset --out models/camera_lav --epochs 8
python model_manager.py publish-model --folder models/camera_lav --repo-id Maslodium/v-d-splitter-models
python model_manager.py download-model --repo-id Maslodium/v-d-splitter-models --out models/community/model.pt
```

See `docs/community-training.md`.

## Privacy

The program does not upload recordings, datasets, checkpoints or profiles by
itself. Publishing to Hugging Face is an explicit command. Only publish material
you have rights to share. If raw audio is private, share a trained model,
checkpoint or aggregate reference profile instead of the source recordings.

## Possible Improvements

- Stronger open restoration model trained on community paired camera/reference
  material.
- Combined mode where paired training learns voice/noise transfer and shoot
  profiles adapt the model to a concrete camera, room and microphone setup.
- Better no-reference speech enhancement before Demucs/after Demucs.
- Optional DeepFilterNet/ONNX backend after packaging tests.
- Real LUFS/true-peak metering.
- Optional cloud training workflow through Hugging Face Jobs or other donated
  compute.
- Export presets for podcast, YouTube, broadcast and dialogue edit.

## Notes

Reference Match currently performs conservative level, broad spectral and soft
dynamics matching. The baseline `model.pt` path is experimental and intended as
a foundation for community training, not an Auphonic-level model yet.

Check upstream model/code licenses before redistributing pretrained weights or
commercial bundles.

---

# V-D Splitter

Настольная утилита для извлечения и очистки голоса из видео и аудио.

V-D означает VOICE-DENOISE. Программа достает аудио через ffmpeg, отделяет
голосовой stem через Demucs, чистит его локальным нейросетевым denoiser и может
подгонять результат под запись с петлички или рекордера из той же съемки.

Поддерживает Maslodium.

## Возможности

- Импорт видео и аудио через ffmpeg / `imageio-ffmpeg`.
- Поддержка популярных форматов: MP4, MKV, MOV, AVI, WEBM, WMV, MP3, WAV, FLAC,
  M4A, AAC, OGG, OPUS и других.
- Разделение voice / no-voice через Demucs `--two-stems=vocals`.
- Очистка речи через Facebook Research Denoiser.
- Опциональный Resemble Enhance backend для experimental speech enhancement.
- Подгонка уровня, широкого тембра и мягкой динамики по референсной записи из
  той же съемки.
- Применение многоразовых shoot/reference profiles и обученных `model.pt`,
  включая профиль по другим дублям, когда петличка целевого дубля потеряна.
- Финальная обработка голоса: compressor, de-esser, peak limiter и примерное
  loudness matching.
- Обработка папки с поддержанными аудио/видео файлами.
- Инструменты подготовки community dataset в формате, удобном для Hugging Face.
- Baseline-скрипт обучения с resume из checkpoint для экспериментов
  `camera -> reference`.
- Явные команды скачивания и публикации моделей через Hugging Face.
- Сохранение `extracted.wav`, `voice_raw.wav`, `voice_clean.wav`,
  опционально `background_no_voice.wav` и `reference.wav`.
- Выбор CUDA или CPU Torch во время Windows bootstrap install.
- Режимы устройства: `auto`, `cuda`, `mps`, `cpu`.

## Требования

- Windows 10/11, macOS или Linux.
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

Linux:

```text
V-D-Splitter-linux.tar.gz
```

## Запуск из исходников

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\pythonw.exe gui.py
```

На CPU можно ставить Torch без CUDA index. На macOS используйте обычные Torch
wheels и `--device auto` или `--device mps`.

Опциональный experimental backend:

```powershell
.\.venv\Scripts\python.exe install_optional.py --backend resemble-enhance
```

## CLI

```powershell
python pipeline.py input.mp4 --out output --device auto
python pipeline.py input.mp4 --denoise-backend resemble-enhance --out output
python pipeline.py camera.wav --reference-audio lav_take.wav --out output
python pipeline.py camera.wav --reference-profile profiles/camera_lav.json --out output
python pipeline.py camera.wav --reference-model models/community/model.pt --out output
python pipeline.py takes_folder --polish-preset camera-hiss --out output
```

## Community Training

```powershell
python community_training.py prepare-dataset --camera-dir camera --reference-dir reference --out vd_dataset
python community_training.py build-profile --dataset-dir vd_dataset --out profiles/camera_lav.json
python community_training.py build-shoot-profile --dataset-dir vd_dataset --out profiles/shoot.json
python train_reference_model.py --dataset-dir vd_dataset --out models/camera_lav --epochs 8
python model_manager.py publish-model --folder models/camera_lav --repo-id Maslodium/v-d-splitter-models
python model_manager.py download-model --repo-id Maslodium/v-d-splitter-models --out models/community/model.pt
```

Подробнее: `docs/community-training.md`.

## Приватность

Программа не загружает записи, датасеты, checkpoints или profiles сама.
Публикация на Hugging Face выполняется только явной командой. Публикуйте только
материалы, на которые у вас есть права. Если исходное аудио приватное, лучше
делиться обученной моделью, checkpoint или aggregate reference profile, а не
исходными записями.

## Возможные доработки

- Более сильная открытая restoration model, обученная на community paired
  camera/reference material.
- Combined mode: парное обучение учит transfer голоса/шума, а shoot profiles
  адаптируют модель под конкретную камеру, комнату и микрофон.
- Лучшее no-reference speech enhancement до Demucs и после Demucs.
- Опциональный DeepFilterNet/ONNX backend после packaging tests.
- Настоящий LUFS/true-peak metering.
- Опциональный cloud training workflow через Hugging Face Jobs или другие
  donated compute.
- Export presets для подкаста, YouTube, broadcast и диалогового монтажа.

## Примечания

Reference Match сейчас делает аккуратное выравнивание громкости, широкого тембра
и мягкой динамики. Путь с baseline `model.pt` экспериментальный: это фундамент
для community training, а не модель уровня Auphonic на текущем этапе.

Перед распространением весов моделей или коммерческой сборкой нужно отдельно
проверить лицензии upstream-проектов.
