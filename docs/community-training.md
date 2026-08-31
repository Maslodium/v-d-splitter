# Community Training

V-D Splitter can grow through paired same-shoot audio:

```text
camera audio -> lavalier/recorder reference
```

The practical short-term target is not a huge public model first. The first
target is shared dataset format, local reference profiles, resumable
checkpoints, and small published models. A stronger restoration model can be
trained later from the same folder format.

Long-running training should be versioned rather than edited in-place. Keep the
dataset in a Hugging Face dataset repository, train from checkpoints, publish
new model revisions/tags, and keep older releases available for comparison.

## Prepare A Dataset

Put files with matching names into two folders:

```text
camera/take01.mp4
reference/take01.wav
camera/take02.mp4
reference/take02.wav
```

Then run:

```powershell
python community_training.py prepare-dataset --camera-dir camera --reference-dir reference --out vd_dataset --license other
```

The output folder contains:

```text
vd_dataset/
  README.md
  metadata.csv
  data/train/camera/*.wav
  data/train/reference/*.wav
```

## Build A Local Profile

```powershell
python community_training.py build-profile --dataset-dir vd_dataset --out profiles/my_camera_lav.json
```

This profile stores average RMS and spectral differences between camera and
reference recordings. It is small and can be used locally. Share it only when
you are comfortable treating it as a public derived artifact.

## Train The Baseline Model

```powershell
python train_reference_model.py --dataset-dir vd_dataset --out models/camera_lav --epochs 8
```

Resume later:

```powershell
python train_reference_model.py --dataset-dir vd_dataset --out models/camera_lav --resume models/camera_lav/checkpoint.pt --epochs 16
```

The baseline writes:

```text
models/camera_lav/checkpoint.pt
models/camera_lav/model.pt
models/camera_lav/README.md
```

This is deliberately small. A 3 GB universal model may become useful later, but
community work should start with datasets, checkpoints, smaller profiles and
repeatable training runs.

## Publish And Download Models

Login once:

```powershell
hf auth login
```

Publish a model folder:

```powershell
python model_manager.py publish-model --folder models/camera_lav --repo-id Maslodium/v-d-splitter-models
```

Download a public model:

```powershell
python model_manager.py download-model --repo-id Maslodium/v-d-splitter-models --out models/community/model.pt
```

Use it:

```powershell
python pipeline.py camera.wav --reference-model models/community/model.pt --out output
```

## Upload Datasets

Upload a dataset only when the audio can be published:

```powershell
python model_manager.py publish-dataset --folder vd_dataset --repo-id Maslodium/v-d-splitter-community
```

Hugging Face supports model and dataset repositories, model cards, dataset
cards, CLI uploads, pull requests/discussions, and Jobs for cloud-side runs.
That makes it a good place to host public versions of the work, but it should
not be treated as a silent telemetry backend.

## Privacy Model

Default behavior:

- No automatic upload.
- No background dataset sync.
- No hidden contribution of user recordings.

Safe contribution path:

- Train locally or on explicitly chosen cloud compute.
- Publish only the model/checkpoint/profile when it is acceptable for the
  result to be public.
- Publish raw paired audio only when every voice and project owner has agreed.

Even trained weights can sometimes leak information about training data. Treat
models and profiles as derived public artifacts, not as guaranteed anonymized
secrets.

---

# Community Training

V-D Splitter можно развивать через парные записи с одной съемки:

```text
звук камеры -> петличка или рекордер
```

Ближайшая практическая цель — не сразу огромная публичная модель. Сначала нужен
общий формат датасета, локальные reference profiles, resumable checkpoints и
небольшие опубликованные модели. Более сильную модель восстановления можно
обучать позже из того же формата папок.

Длительное обучение лучше вести версиями, а не перезаписью одного живого файла.
Датасет растет в Hugging Face dataset repository, обучение продолжается из
checkpoints, а модель публикуется новыми revisions/tags. Старые релизы остаются
доступны для сравнения.

## Подготовка Датасета

Положите файлы с одинаковыми именами в две папки:

```text
camera/take01.mp4
reference/take01.wav
camera/take02.mp4
reference/take02.wav
```

Запуск:

```powershell
python community_training.py prepare-dataset --camera-dir camera --reference-dir reference --out vd_dataset --license other
```

На выходе:

```text
vd_dataset/
  README.md
  metadata.csv
  data/train/camera/*.wav
  data/train/reference/*.wav
```

## Локальный Профиль

```powershell
python community_training.py build-profile --dataset-dir vd_dataset --out profiles/my_camera_lav.json
```

Профиль хранит среднюю разницу по громкости и спектру между камерой и
референсом. Его можно использовать локально. Делиться им стоит только если вы
готовы считать его публичным производным артефактом.

## Обучение Baseline-Модели

```powershell
python train_reference_model.py --dataset-dir vd_dataset --out models/camera_lav --epochs 8
```

Продолжить позже:

```powershell
python train_reference_model.py --dataset-dir vd_dataset --out models/camera_lav --resume models/camera_lav/checkpoint.pt --epochs 16
```

На выходе:

```text
models/camera_lav/checkpoint.pt
models/camera_lav/model.pt
models/camera_lav/README.md
```

Это специально небольшая baseline-модель. Большая универсальная модель на
несколько гигабайт может понадобиться позже, но комьюнити лучше начинать с
датасетов, checkpoints, небольших профилей и воспроизводимых запусков обучения.

## Публикация И Скачивание Моделей

Один раз авторизоваться:

```powershell
hf auth login
```

Опубликовать папку модели:

```powershell
python model_manager.py publish-model --folder models/camera_lav --repo-id Maslodium/v-d-splitter-models
```

Скачать публичную модель:

```powershell
python model_manager.py download-model --repo-id Maslodium/v-d-splitter-models --out models/community/model.pt
```

Использовать ее:

```powershell
python pipeline.py camera.wav --reference-model models/community/model.pt --out output
```

## Загрузка Датасетов

Загружайте датасет только когда аудио можно публиковать:

```powershell
python model_manager.py publish-dataset --folder vd_dataset --repo-id Maslodium/v-d-splitter-community
```

Hugging Face подходит для model repos, dataset repos, model cards, dataset
cards, CLI uploads, pull requests/discussions и Jobs для cloud-side runs. Это
хорошее место для публичных версий работы, но не скрытый telemetry backend.

## Модель Приватности

Поведение по умолчанию:

- Нет автоматической загрузки.
- Нет фоновой синхронизации датасета.
- Нет скрытого вклада пользовательских записей.

Безопасный путь вклада:

- Обучить локально или на явно выбранных cloud compute.
- Публиковать только model/checkpoint/profile, если допустимо считать результат
  публичным.
- Публиковать raw paired audio только если все голоса и владельцы проекта дали
  согласие.

Даже веса модели иногда могут раскрывать информацию о тренировочных данных.
Относитесь к моделям и профилям как к производным публичным артефактам, а не
как к гарантированно анонимизированным секретам.
