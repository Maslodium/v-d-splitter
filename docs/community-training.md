# Community Training

V-D Splitter can grow through paired same-shoot audio:

```text
camera audio -> lavalier/recorder reference
```

The practical short-term target is not a huge public model first. The first
target is shared datasets and local reference profiles. A stronger restoration
model can be trained later from the same folder format.

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
reference recordings. It is small, local and safe to share when the underlying
audio cannot be published.

## Upload To Hugging Face

Login once:

```powershell
hf auth login
```

Upload a dataset:

```powershell
python community_training.py upload --folder vd_dataset --repo-id Maslodium/vd-same-shoot-pairs --repo-type dataset
```

Hugging Face supports model and dataset repositories, dataset cards, model cards
and CLI uploads. Audio datasets can be uploaded as raw audio files with metadata
or packaged for larger scale.

## Consent And Rights

Only publish material you have rights to share. Do not upload private voices,
client footage, unreleased film material or identifiable recordings without
explicit permission.

---

# Community Training

V-D Splitter можно развивать через парные записи с одной съёмки:

```text
звук камеры -> петличка или рекордер
```

Ближайшая цель - не сразу большая публичная модель, а общий формат датасетов и
локальные reference profiles. Потом по тем же папкам можно обучать более
сильную модель восстановления.

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
референсом. Его можно использовать локально или делиться им, если исходные
записи публиковать нельзя.

## Загрузка На Hugging Face

Один раз авторизоваться:

```powershell
hf auth login
```

Загрузить датасет:

```powershell
python community_training.py upload --folder vd_dataset --repo-id Maslodium/vd-same-shoot-pairs --repo-type dataset
```

Hugging Face подходит для model repos, dataset repos, model cards, dataset cards
и загрузки через CLI. Аудиодатасеты можно хранить как raw audio + metadata.

## Права И Согласие

Публикуйте только материалы, на которые есть права. Не загружайте частные
голоса, клиентские съёмки, невыпущенные материалы и узнаваемые записи без
явного разрешения.
