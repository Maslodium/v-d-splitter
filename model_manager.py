"""
Download and publish V-D Splitter community models.

All network operations are explicit commands. The app never uploads user audio
or checkpoints by itself.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_MODEL_REPO = "Maslodium/v-d-splitter-models"
DEFAULT_DATASET_REPO = "Maslodium/v-d-splitter-community"


def find_hf_cli() -> str:
    found = shutil.which("hf")
    if found:
        return found
    suffix = ".exe" if sys.platform.startswith("win") else ""
    local = Path(sys.executable).with_name(f"hf{suffix}")
    if local.exists():
        return str(local)
    raise RuntimeError("Hugging Face CLI not found. Install with: python -m pip install huggingface_hub")


def require_hf() -> str:
    try:
        return find_hf_cli()
    except RuntimeError:
        raise RuntimeError("Hugging Face CLI not found. Install with: python -m pip install huggingface_hub")


def hf_download(repo_id: str, filename: str, out: Path, repo_type: str,
                revision: str | None = None) -> Path:
    from huggingface_hub import hf_hub_download

    out.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        revision=revision,
    )
    shutil.copy2(downloaded, out)
    print(f"[ok] downloaded {repo_id}/{filename} -> {out}")
    return out


def create_hf_repo(repo_id: str, repo_type: str) -> None:
    from huggingface_hub import HfApi

    HfApi().create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)


def hf_upload(folder: Path, repo_id: str, repo_type: str,
              create_repo_first: bool = False,
              message: str = "Upload V-D Splitter artifact") -> None:
    hf_cli = require_hf()
    if not folder.exists():
        raise FileNotFoundError(folder)
    if create_repo_first:
        create_hf_repo(repo_id, repo_type)
    cmd = [
        hf_cli,
        "upload",
        repo_id,
        str(folder),
        "--repo-type",
        repo_type,
        "--commit-message",
        message,
    ]
    print("$ " + " ".join(cmd))
    subprocess.check_call(cmd)


def write_seed_model_card(path: Path, hidden: int, sample_rate: int,
                          n_fft: int, hop_length: int) -> None:
    path.write_text(
        f"""---
license: other
tags:
- audio
- speech-enhancement
- denoise
- seed-model
---

# V-D Splitter Seed Model

Bootstrap model repository for V-D Splitter community training.

Maintained by Maslodium.

This artifact is an untrained `v-d-spectral-mapper-v1` model. It exists so the
Hugging Face repository, download path and local inference wiring have a valid
starting checkpoint before community training produces useful weights.

## Contents

- `model.pt`: untrained spectral mapper weights.
- `seed_manifest.json`: model format and configuration.

## Configuration

- sample rate: {sample_rate}
- FFT: {n_fft}
- hop length: {hop_length}
- hidden size: {hidden}

Do not judge restoration quality from this seed. It should be replaced by a
trained model produced from paired camera/reference datasets.

---

# V-D Splitter Seed Model

Стартовый репозиторий модели для community training в V-D Splitter.

Поддерживает Maslodium.

Этот артефакт содержит необученную модель формата `v-d-spectral-mapper-v1`.
Она нужна, чтобы у Hugging Face репозитория, скачивания и локального inference
уже была валидная точка входа до появления полезных весов от обучения.

## Содержимое

- `model.pt`: необученные веса spectral mapper.
- `seed_manifest.json`: формат модели и конфигурация.

## Конфигурация

- sample rate: {sample_rate}
- FFT: {n_fft}
- hop length: {hop_length}
- hidden size: {hidden}

По этому seed нельзя оценивать качество восстановления. Его нужно заменить
обученной моделью, полученной на парных датасетах камера/reference.
""",
        encoding="utf-8",
    )


def create_seed_model(out_dir: Path, hidden: int = 512, sample_rate: int = 16000,
                      n_fft: int = 1024, hop_length: int = 256) -> Path:
    import torch
    from train_reference_model import SpectralMapper

    out_dir.mkdir(parents=True, exist_ok=True)
    bins = n_fft // 2 + 1
    model = SpectralMapper(bins, hidden)
    cfg = {
        "sample_rate": sample_rate,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "hidden": hidden,
        "batch_size": 128,
        "epochs": 0,
        "lr": 1e-3,
        "max_frames_per_pair": 1800,
    }
    model_path = out_dir / "model.pt"
    torch.save({
        "format": "v-d-spectral-mapper-v1",
        "config": cfg,
        "model_state_dict": model.state_dict(),
        "trained": False,
    }, model_path)
    manifest = {
        "format": "v-d-seed-model-v1",
        "model_format": "v-d-spectral-mapper-v1",
        "project": "V-D Splitter",
        "description": "VOICE-DENOISE bootstrap model for community training.",
        "author": "Maslodium",
        "trained": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": ["model.pt", "README.md", "seed_manifest.json"],
        "config": cfg,
        "purpose": "Bootstrap the Hugging Face model repository and validate downloader/inference wiring.",
    }
    (out_dir / "seed_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_seed_model_card(out_dir / "README.md", hidden, sample_rate, n_fft, hop_length)
    print(f"[ok] seed model: {model_path}")
    return model_path


def main() -> int:
    parser = argparse.ArgumentParser(description="download and publish V-D Splitter community models")
    sub = parser.add_subparsers(dest="cmd", required=True)

    seed = sub.add_parser("create-seed-model")
    seed.add_argument("--out", type=Path, default=Path("models") / "seed")
    seed.add_argument("--hidden", type=int, default=512)
    seed.add_argument("--sample-rate", type=int, default=16000)
    seed.add_argument("--n-fft", type=int, default=1024)
    seed.add_argument("--hop-length", type=int, default=256)

    dl_model = sub.add_parser("download-model")
    dl_model.add_argument("--repo-id", default=DEFAULT_MODEL_REPO)
    dl_model.add_argument("--filename", default="model.pt")
    dl_model.add_argument("--out", type=Path, default=Path("models") / "community" / "model.pt")
    dl_model.add_argument("--revision", default=None)

    dl_profile = sub.add_parser("download-profile")
    dl_profile.add_argument("--repo-id", default=DEFAULT_MODEL_REPO)
    dl_profile.add_argument("--filename", default="profiles/community.json")
    dl_profile.add_argument("--out", type=Path, default=Path("profiles") / "community.json")
    dl_profile.add_argument("--revision", default=None)

    pub_model = sub.add_parser("publish-model")
    pub_model.add_argument("--folder", type=Path, required=True)
    pub_model.add_argument("--repo-id", default=DEFAULT_MODEL_REPO)
    pub_model.add_argument("--message", default="Upload V-D Splitter model")
    pub_model.add_argument("--create-repo", action="store_true")

    pub_dataset = sub.add_parser("publish-dataset")
    pub_dataset.add_argument("--folder", type=Path, required=True)
    pub_dataset.add_argument("--repo-id", default=DEFAULT_DATASET_REPO)
    pub_dataset.add_argument("--message", default="Upload V-D Splitter dataset")
    pub_dataset.add_argument("--create-repo", action="store_true")

    args = parser.parse_args()
    try:
        if args.cmd == "create-seed-model":
            create_seed_model(args.out, args.hidden, args.sample_rate, args.n_fft, args.hop_length)
        elif args.cmd == "download-model":
            hf_download(args.repo_id, args.filename, args.out, "model", args.revision)
        elif args.cmd == "download-profile":
            hf_download(args.repo_id, args.filename, args.out, "model", args.revision)
        elif args.cmd == "publish-model":
            hf_upload(args.folder, args.repo_id, "model", args.create_repo, args.message)
        elif args.cmd == "publish-dataset":
            hf_upload(args.folder, args.repo_id, "dataset", args.create_repo, args.message)
    except Exception as exc:
        print(f"[error] {exc!r}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
