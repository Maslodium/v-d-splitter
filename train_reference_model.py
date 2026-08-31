"""
Train a small camera-to-reference spectral mapper for V-D Splitter.

This is a lightweight research baseline, not a replacement for a full speech
enhancement model. It learns a frequency-bin gain curve from paired
camera/reference WAV files prepared by community_training.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class TrainConfig:
    sample_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 256
    hidden: int = 512
    batch_size: int = 128
    epochs: int = 8
    lr: float = 1e-3
    max_frames_per_pair: int = 1800


class SpectralMapper(nn.Module):
    def __init__(self, bins: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(bins, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, bins),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PairFrames(Dataset):
    def __init__(self, dataset_dir: Path, cfg: TrainConfig) -> None:
        self.frames: list[tuple[np.ndarray, np.ndarray]] = []
        metadata = dataset_dir / "metadata.csv"
        if not metadata.is_file():
            raise FileNotFoundError(metadata)
        rng = np.random.default_rng(1337)
        with metadata.open("r", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                camera = dataset_dir / row["camera"]
                reference = dataset_dir / row["reference"]
                cam = load_mono(camera, cfg.sample_rate)
                ref = load_mono(reference, cfg.sample_rate)
                n = min(len(cam), len(ref))
                if n < cfg.n_fft * 2:
                    continue
                cam_mag = stft_mag(cam[:n], cfg)
                ref_mag = stft_mag(ref[:n], cfg)
                target = np.clip(np.log1p(ref_mag) - np.log1p(cam_mag), -2.0, 2.0)
                inputs = np.log1p(cam_mag)
                count = min(inputs.shape[0], cfg.max_frames_per_pair)
                idx = rng.choice(inputs.shape[0], size=count, replace=False)
                self.frames.extend((inputs[i].astype(np.float32), target[i].astype(np.float32)) for i in idx)
        if not self.frames:
            raise RuntimeError("No usable training frames found.")

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.frames[index]
        return torch.from_numpy(x), torch.from_numpy(y)


def load_mono(path: Path, sample_rate: int) -> np.ndarray:
    y, sr = sf.read(str(path), always_2d=True)
    y = np.mean(y, axis=1).astype(np.float32)
    if sr == sample_rate:
        return y
    import librosa

    return librosa.resample(y, orig_sr=sr, target_sr=sample_rate).astype(np.float32)


def stft_mag(y: np.ndarray, cfg: TrainConfig) -> np.ndarray:
    window = torch.hann_window(cfg.n_fft)
    wav = torch.from_numpy(y)
    spec = torch.stft(
        wav,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.n_fft,
        window=window,
        return_complex=True,
    )
    return spec.abs().transpose(0, 1).numpy()


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
                    cfg: TrainConfig, epoch: int, loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "v-d-spectral-mapper-v1",
        "epoch": epoch,
        "loss": loss,
        "config": cfg.__dict__,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)


def load_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer) -> int:
    data = torch.load(path, map_location="cpu")
    if data.get("format") != "v-d-spectral-mapper-v1":
        raise ValueError(f"unsupported checkpoint: {data.get('format')}")
    model.load_state_dict(data["model_state_dict"])
    optimizer.load_state_dict(data["optimizer_state_dict"])
    return int(data.get("epoch", 0)) + 1


def train(dataset_dir: Path, out_dir: Path, cfg: TrainConfig, resume: Path | None = None) -> Path:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = PairFrames(dataset_dir, cfg)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    bins = cfg.n_fft // 2 + 1
    model = SpectralMapper(bins, cfg.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    start_epoch = 0
    if resume:
        start_epoch = load_checkpoint(resume, model, optimizer)

    loss_fn = nn.SmoothL1Loss()
    ckpt = out_dir / "checkpoint.pt"
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        losses = []
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(losses))
        print(f"epoch={epoch + 1}/{cfg.epochs} loss={mean_loss:.6f}")
        save_checkpoint(ckpt, model, optimizer, cfg, epoch, mean_loss)

    export = out_dir / "model.pt"
    torch.save({
        "format": "v-d-spectral-mapper-v1",
        "config": cfg.__dict__,
        "model_state_dict": model.cpu().state_dict(),
    }, export)
    write_model_card(out_dir / "README.md", cfg, len(dataset))
    print(f"[ok] model: {export}")
    return export


def write_model_card(path: Path, cfg: TrainConfig, frames: int) -> None:
    path.write_text(
        f"""---
license: other
tags:
- audio
- speech-enhancement
- denoise
---

# V-D Spectral Mapper

Small camera-to-reference spectral mapper baseline for V-D Splitter.

Maintained by Maslodium.

## Training

- frames: {frames}
- sample rate: {cfg.sample_rate}
- FFT: {cfg.n_fft}
- hop length: {cfg.hop_length}

This is a research baseline. It is meant for experiments with paired
camera/lavalier datasets, not as a finished universal restoration model.

---

# V-D Spectral Mapper

Небольшая baseline-модель для подгонки камерного звука к референсу в V-D
Splitter.

Поддерживает Maslodium.

Это исследовательская заготовка для парных датасетов камера/петличка, а не
готовая универсальная модель восстановления речи.
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="train V-D Splitter spectral mapper baseline")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    try:
        train(args.dataset_dir, args.out, cfg, resume=args.resume)
    except Exception as exc:
        print(f"[error] {exc!r}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
