"""
Community dataset and profile tools for V-D Splitter.

The first open-training target is paired same-shoot material:

    camera/noisy take -> lavalier or recorder reference take

This script prepares a Hugging Face friendly dataset folder and can build a
small local reference profile that V-D Splitter can apply without training a
large model.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from pipeline import SUPPORTED_AUDIO, SUPPORTED_VIDEO, extract_audio


SUPPORTED = SUPPORTED_AUDIO | SUPPORTED_VIDEO


@dataclass(frozen=True)
class Pair:
    camera: Path
    reference: Path
    stem: str


def safe_name(value: str) -> str:
    keep = []
    for ch in value.strip():
        if ch.isalnum() or ch in {"-", "_"}:
            keep.append(ch)
        elif ch in {" ", ".", "+"}:
            keep.append("_")
    cleaned = "".join(keep).strip("_")
    return cleaned or "take"


def collect_files(path: Path) -> dict[str, Path]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    files: dict[str, Path] = {}
    for item in sorted(path.rglob("*")):
        if item.is_file() and item.suffix.lower() in SUPPORTED:
            files.setdefault(item.stem.lower(), item)
    return files


def pair_by_stem(camera_dir: Path, reference_dir: Path) -> list[Pair]:
    cameras = collect_files(camera_dir)
    refs = collect_files(reference_dir)
    pairs = []
    for key in sorted(set(cameras) & set(refs)):
        pairs.append(Pair(cameras[key], refs[key], safe_name(key)))
    return pairs


def wav_info(path: Path) -> tuple[int, float]:
    info = sf.info(str(path))
    return int(info.samplerate), float(info.frames / info.samplerate) if info.samplerate else 0.0


def prepare_dataset(camera_dir: Path, reference_dir: Path, out_dir: Path,
                    dataset_name: str, license_name: str, maintainer: str,
                    notes: str = "") -> int:
    pairs = pair_by_stem(camera_dir, reference_dir)
    if not pairs:
        raise RuntimeError("No matching camera/reference pairs found. Match filenames by stem, for example take01.mp4 and take01.wav.")

    cam_out = out_dir / "data" / "train" / "camera"
    ref_out = out_dir / "data" / "train" / "reference"
    cam_out.mkdir(parents=True, exist_ok=True)
    ref_out.mkdir(parents=True, exist_ok=True)

    rows = []
    for pair in pairs:
        camera_wav = extract_audio(pair.camera, cam_out / f"{pair.stem}.wav")
        ref_wav = extract_audio(pair.reference, ref_out / f"{pair.stem}.wav")
        sr, duration = wav_info(camera_wav)
        rows.append({
            "camera": f"data/train/camera/{pair.stem}.wav",
            "reference": f"data/train/reference/{pair.stem}.wav",
            "source_id": pair.stem,
            "sample_rate": sr,
            "duration": f"{duration:.3f}",
            "split": "train",
            "license": license_name,
            "notes": notes,
        })

    with (out_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    write_dataset_card(out_dir / "README.md", dataset_name, license_name, maintainer, len(rows))
    print(f"[ok] dataset folder: {out_dir}")
    print(f"[ok] pairs: {len(rows)}")
    return len(rows)


def write_dataset_card(path: Path, dataset_name: str, license_name: str,
                       maintainer: str, pairs: int) -> None:
    path.write_text(
        f"""---
license: {license_name}
task_categories:
- audio-to-audio
pretty_name: {dataset_name}
---

# {dataset_name}

Paired camera and lavalier/recorder audio for speech restoration experiments.

Maintained by {maintainer}.

## Data

- `camera`: noisy or distant camera audio.
- `reference`: cleaner same-shoot lavalier or recorder audio.
- pairs: {pairs}

Only upload material you have rights to share. Do not publish private voices,
client footage or identifiable recordings without permission.

---

# {dataset_name}

Парный датасет камерного звука и петлички/рекордера для экспериментов по
восстановлению речи.

Поддерживает {maintainer}.

Публикуйте только материалы, на которые у вас есть права. Не выкладывайте
частные голоса, клиентские съёмки и узнаваемые записи без разрешения.
""",
        encoding="utf-8",
    )


def build_profile(dataset_dir: Path, out_json: Path) -> Path:
    metadata = dataset_dir / "metadata.csv"
    if not metadata.is_file():
        raise FileNotFoundError(metadata)
    gains: list[float] = []
    ratios: list[np.ndarray] = []
    sample_rate = 44100
    fft_size = 8192

    with metadata.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            camera = dataset_dir / row["camera"]
            ref = dataset_dir / row["reference"]
            y, sr = sf.read(str(camera), always_2d=True)
            r, ref_sr = sf.read(str(ref), always_2d=True)
            if sr != ref_sr:
                continue
            sample_rate = sr
            y_mono = np.mean(y, axis=1)
            r_mono = np.mean(r, axis=1)
            n = min(len(y_mono), len(r_mono), sr * 45)
            if n <= 4096:
                continue
            gains.append(float((np.sqrt(np.mean(r_mono[:n] ** 2)) + 1e-6) / (np.sqrt(np.mean(y_mono[:n] ** 2)) + 1e-6)))
            y_spec = np.abs(np.fft.rfft(y_mono[:n] * np.hanning(n), n=fft_size)) + 1e-7
            r_spec = np.abs(np.fft.rfft(r_mono[:n] * np.hanning(n), n=fft_size)) + 1e-7
            ratio = np.clip(r_spec / y_spec, 0.35, 2.8)
            ratios.append(ratio)

    if not ratios:
        raise RuntimeError("No usable profile pairs found.")
    avg_ratio = np.mean(np.stack(ratios), axis=0)
    kernel = np.ones(33, dtype=np.float64) / 33
    avg_ratio = np.convolve(avg_ratio, kernel, mode="same")
    profile = {
        "format": "v-d-reference-profile-v1",
        "sample_rate": sample_rate,
        "fft_size": fft_size,
        "rms_gain": float(np.clip(np.median(gains), 0.2, 5.0)),
        "spectral_ratio": [float(v) for v in avg_ratio],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] profile: {out_json}")
    return out_json


def build_shoot_profile(dataset_dir: Path, out_json: Path) -> Path:
    metadata = dataset_dir / "metadata.csv"
    if not metadata.is_file():
        raise FileNotFoundError(metadata)

    sample_rate = 44100
    fft_size = 8192
    camera_specs: list[np.ndarray] = []
    reference_specs: list[np.ndarray] = []
    transfer_ratios: list[np.ndarray] = []
    camera_rms: list[float] = []
    reference_rms: list[float] = []
    camera_floor: list[float] = []

    with metadata.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            camera = dataset_dir / row["camera"]
            ref = dataset_dir / row["reference"]
            y, sr = sf.read(str(camera), always_2d=True)
            r, ref_sr = sf.read(str(ref), always_2d=True)
            if sr != ref_sr:
                continue
            sample_rate = sr
            y_mono = np.mean(y, axis=1)
            r_mono = np.mean(r, axis=1)
            n = min(len(y_mono), len(r_mono), sr * 45)
            if n <= 4096:
                continue

            y_part = y_mono[:n]
            r_part = r_mono[:n]
            camera_rms.append(_rms_np(y_part))
            reference_rms.append(_rms_np(r_part))
            camera_floor.append(float(np.percentile(np.abs(y_part), 12)))

            y_spec = np.abs(np.fft.rfft(y_part * np.hanning(n), n=fft_size)) + 1e-7
            r_spec = np.abs(np.fft.rfft(r_part * np.hanning(n), n=fft_size)) + 1e-7
            camera_specs.append(y_spec / max(np.median(y_spec), 1e-7))
            reference_specs.append(r_spec / max(np.median(r_spec), 1e-7))
            transfer_ratios.append(np.clip(r_spec / y_spec, 0.35, 2.8))

    if not transfer_ratios:
        raise RuntimeError("No usable shoot profile pairs found.")

    kernel = np.ones(33, dtype=np.float64) / 33
    transfer = np.convolve(np.mean(np.stack(transfer_ratios), axis=0), kernel, mode="same")
    camera_tone = np.convolve(np.mean(np.stack(camera_specs), axis=0), kernel, mode="same")
    reference_tone = np.convolve(np.mean(np.stack(reference_specs), axis=0), kernel, mode="same")
    profile = {
        "format": "v-d-shoot-profile-v1",
        "sample_rate": sample_rate,
        "fft_size": fft_size,
        "pairs": len(transfer_ratios),
        "camera": {
            "rms": float(np.median(camera_rms)),
            "noise_floor": float(np.median(camera_floor)),
            "spectral_shape": [float(v) for v in camera_tone],
        },
        "reference": {
            "rms": float(np.median(reference_rms)),
            "spectral_shape": [float(v) for v in reference_tone],
        },
        "transfer": {
            "rms_gain": float(np.clip(np.median(reference_rms) / max(np.median(camera_rms), 1e-7), 0.2, 5.0)),
            "spectral_ratio": [float(v) for v in transfer],
        },
        "use_case": "Learn shoot/microphone character from other takes, then apply it to a lost-reference camera take.",
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] shoot profile: {out_json}")
    return out_json


def _rms_np(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64))) if x.size else 0.0


def upload_to_hf(folder: Path, repo_id: str, repo_type: str = "dataset") -> None:
    if shutil.which("hf") is None:
        raise RuntimeError("Hugging Face CLI not found. Install with: python -m pip install huggingface_hub")
    run = ["hf", "upload", repo_id, str(folder), "--repo-type", repo_type]
    print("$ " + " ".join(run))
    subprocess.check_call(run)


def main() -> int:
    parser = argparse.ArgumentParser(description="prepare V-D Splitter community datasets and reference profiles")
    sub = parser.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare-dataset")
    prep.add_argument("--camera-dir", type=Path, required=True)
    prep.add_argument("--reference-dir", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--dataset-name", default="V-D Same-Shoot Camera/Lav Pairs")
    prep.add_argument("--license", default="other")
    prep.add_argument("--maintainer", default="Maslodium")
    prep.add_argument("--notes", default="")

    prof = sub.add_parser("build-profile")
    prof.add_argument("--dataset-dir", type=Path, required=True)
    prof.add_argument("--out", type=Path, required=True)

    shoot = sub.add_parser("build-shoot-profile")
    shoot.add_argument("--dataset-dir", type=Path, required=True)
    shoot.add_argument("--out", type=Path, required=True)

    up = sub.add_parser("upload")
    up.add_argument("--folder", type=Path, required=True)
    up.add_argument("--repo-id", required=True)
    up.add_argument("--repo-type", choices=["dataset", "model"], default="dataset")

    args = parser.parse_args()
    try:
        if args.cmd == "prepare-dataset":
            prepare_dataset(args.camera_dir, args.reference_dir, args.out, args.dataset_name, args.license, args.maintainer, args.notes)
        elif args.cmd == "build-profile":
            build_profile(args.dataset_dir, args.out)
        elif args.cmd == "build-shoot-profile":
            build_shoot_profile(args.dataset_dir, args.out)
        elif args.cmd == "upload":
            upload_to_hf(args.folder, args.repo_id, args.repo_type)
    except Exception as exc:
        print(f"[error] {exc!r}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
