# Community Training

V-D Splitter can grow through paired same-shoot audio:

```text
camera audio -> lavalier/recorder reference
```

The practical target combines two paths. Paired takes teach the model what
camera noise and useful voice should become. Shoot profiles adapt that learned
transfer to a concrete room, camera and microphone setup when the clean track
for the target take is lost.

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

## Build A Shoot Profile

```powershell
python community_training.py build-shoot-profile --dataset-dir vd_dataset --out profiles/shoot.json
```

A shoot profile is for the lost-audio case: the target take has only camera
audio, while other takes from the same shoot still have camera/reference pairs.
It stores camera tone, approximate noise floor, reference tone and the transfer
ratio between them. It does not need the same words as the target take.

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

## Seed An Empty Model Repository

If the Hugging Face account has no V-D model repository yet, create a valid
untrained seed model first:

```powershell
python model_manager.py create-seed-model --out models/seed
```

Then publish it:

```powershell
hf auth login
python model_manager.py publish-model --folder models/seed --repo-id Maslodium/v-d-splitter-models --create-repo
```

The seed is not a quality model. It only gives the project a correct `model.pt`,
model card and manifest so downloads, releases and inference can be tested
before real training replaces the weights.

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

Use a model alone:

```powershell
python pipeline.py camera.wav --reference-model models/community/model.pt --out output
```

Use a shoot profile and model together:

```powershell
python pipeline.py lost_take_camera.wav --reference-profile profiles/shoot.json --reference-model models/community/model.pt --out output
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

Практическая цель совмещает два пути. Парные дубли учат модель тому, во что
должны превращаться шум камеры и полезная нагрузка — голос. Shoot profiles
адаптируют этот transfer под конкретную комнату, камеру и микрофон, когда
чистая дорожка целевого дубля потеряна.

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

## Профиль Съемки

```powershell
python community_training.py build-shoot-profile --dataset-dir vd_dataset --out profiles/shoot.json
```

Shoot profile нужен для аварийного случая: у целевого дубля есть только звук
камеры, а у других дублей той же съемки сохранились пары камера/референс. Он
хранит тембр камеры, примерный noise floor, тембр референса и transfer ratio
между ними. Ему не нужны те же слова, что в целевом дубле.

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

## Старт Пустого Репозитория Модели

Если в Hugging Face аккаунте еще нет model repository для V-D, сначала создайте
валидную необученную seed-модель:

```powershell
python model_manager.py create-seed-model --out models/seed
```

Затем опубликуйте ее:

```powershell
hf auth login
python model_manager.py publish-model --folder models/seed --repo-id Maslodium/v-d-splitter-models --create-repo
```

Seed не является качественной моделью. Он только дает проекту правильный
`model.pt`, model card и manifest, чтобы проверить скачивание, релизы и
inference до замены весов настоящим обучением.

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

Использовать только модель:

```powershell
python pipeline.py camera.wav --reference-model models/community/model.pt --out output
```

Использовать shoot profile и модель вместе:

```powershell
python pipeline.py lost_take_camera.wav --reference-profile profiles/shoot.json --reference-model models/community/model.pt --out output
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
