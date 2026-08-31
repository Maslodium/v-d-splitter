"""
Video Voice Cleaner pipeline.

video/audio -> extracted WAV -> voice stem -> neural denoise -> normalized clean WAV
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


APP_NAME = "V-D Splitter"
SUPPORTED_VIDEO = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".wmv", ".mpg", ".mpeg",
    ".ts", ".mts", ".m2ts", ".flv", ".3gp", ".ogv",
}
SUPPORTED_AUDIO = {
    ".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wma", ".aiff", ".aif",
    ".opus", ".alac", ".amr",
}


@dataclass(frozen=True)
class PipelineResult:
    input_path: Path
    result_dir: Path
    extracted_audio: Path
    voice_stem: Path
    cleaned_voice: Path
    instrumental: Path | None
    reference_audio: Path | None = None


POLISH_PRESETS = {
    "speech": {"compressor": True, "deesser": True, "target_lufs": -16.0, "peak": 0.95},
    "web": {"compressor": True, "deesser": True, "target_lufs": -16.0, "peak": 0.93},
    "broadcast": {"compressor": True, "deesser": True, "target_lufs": -23.0, "peak": 0.90},
    "camera-hiss": {"compressor": True, "deesser": True, "target_lufs": -18.0, "peak": 0.92},
    "raw": {"compressor": False, "deesser": False, "target_lufs": None, "peak": 0.95},
}


def find_ffmpeg() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def run_command(cmd: list[str], cwd: Path | None = None) -> None:
    print("$ " + " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert proc.stdout is not None
    lines: list[str] = []
    for line in proc.stdout:
        lines.append(line)
        sys.stdout.write(line)
    code = proc.wait()
    if code:
        tail = "".join(lines[-25:])
        raise RuntimeError(f"command failed with code {code}\n{tail}")


def extract_audio(input_path: Path, out_wav: Path, sample_rate: int = 44100) -> Path:
    input_path = Path(input_path)
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found. Install imageio-ffmpeg or add ffmpeg to PATH.")

    if input_path.suffix.lower() in SUPPORTED_AUDIO and input_path.suffix.lower() == ".wav":
        shutil.copy2(input_path, out_wav)
        return out_wav

    run_command([
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "2",
        "-ar",
        str(sample_rate),
        "-sample_fmt",
        "s16",
        str(out_wav),
    ])
    return out_wav


def pick_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    try:
        import torch

        if requested in {"auto", "cuda"} and torch.cuda.is_available():
            return "cuda"
        if requested in {"auto", "mps"} and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if requested == "mps":
            print("[warn] MPS requested but unavailable; using CPU.")
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    if requested == "cuda":
        print("[warn] CUDA requested but unavailable; using CPU.")
    return "cpu"


def separate_voice(audio_wav: Path, work_dir: Path, device: str, model: str = "htdemucs_ft",
                   segment: int = 7) -> tuple[Path, Path | None]:
    stems_root = work_dir / "stems"
    run_command([
        sys.executable,
        "-u",
        "-m",
        "demucs",
        "-n",
        model,
        "-d",
        device,
        "--two-stems",
        "vocals",
        "--segment",
        str(segment),
        "-o",
        str(stems_root),
        str(audio_wav),
    ])
    stem_dir = stems_root / model / audio_wav.stem
    vocals = stem_dir / "vocals.wav"
    instrumental = stem_dir / "no_vocals.wav"
    if not vocals.is_file():
        raise RuntimeError(f"Demucs did not produce vocals at {vocals}")
    return vocals, instrumental if instrumental.is_file() else None


def denoise_voice(voice_wav: Path, work_dir: Path, device: str = "cpu",
                  model: str = "dns64", dry: float = 0.0) -> Path:
    noisy_dir = work_dir / "denoise_input"
    out_dir = work_dir / "denoised"
    noisy_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    staged = noisy_dir / voice_wav.name
    shutil.copy2(voice_wav, staged)

    model_flag = {
        "dns48": "--dns48",
        "dns64": "--dns64",
        "master64": "--master64",
        "valentini_nc": "--valentini_nc",
    }.get(model, "--dns64")

    run_command([
        sys.executable,
        "-u",
        "-m",
        "denoiser.enhance",
        model_flag,
        "--device",
        device,
        "--dry",
        str(dry),
        "--num_workers",
        "0",
        "--out_dir",
        str(out_dir),
        "--noisy_dir",
        str(noisy_dir),
    ])

    candidates = sorted(out_dir.rglob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError(f"Denoiser did not produce a wav in {out_dir}")
    return candidates[0]


def normalize_voice(wav: Path, out_wav: Path, target_peak: float = 0.95) -> Path:
    import numpy as np
    import soundfile as sf

    y, sr = sf.read(str(wav), always_2d=False)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    if peak > 0:
        y = y * min(1.0, target_peak / peak)
    sf.write(str(out_wav), y, sr)
    return out_wav


def _db_to_amp(db: float) -> float:
    return 10 ** (db / 20)


def _amp_to_db(value: float) -> float:
    import math

    return 20 * math.log10(max(value, 1e-12))


def _rms(x) -> float:
    import numpy as np

    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64))) if x.size else 0.0


def apply_compressor(y, threshold_db: float = -18.0, ratio: float = 2.6):
    import numpy as np

    threshold = _db_to_amp(threshold_db)
    sign = np.sign(y)
    mag = np.abs(y)
    over = mag > threshold
    compressed = mag.copy()
    compressed[over] = threshold * np.power(mag[over] / threshold, 1.0 / ratio)
    return sign * compressed


def apply_deesser(y, sr: int, threshold_db: float = -24.0, amount: float = 0.45):
    import numpy as np
    from scipy.signal import butter, sosfilt

    if sr < 16000 or not y.size:
        return y
    hi = min(11000, int(sr * 0.45))
    if hi <= 5000:
        return y
    sos = butter(3, [5000, hi], btype="bandpass", fs=sr, output="sos")
    band = sosfilt(sos, y, axis=0)
    threshold = _db_to_amp(threshold_db)
    mask = np.abs(band) > threshold
    return np.where(mask, y - band * amount, y)


def match_loudness(y, target_lufs: float | None):
    if target_lufs is None:
        return y
    import numpy as np

    mono = np.mean(y, axis=1) if y.ndim == 2 else y
    current = _amp_to_db(_rms(mono))
    gain_db = max(-18.0, min(18.0, target_lufs - current))
    return y * _db_to_amp(gain_db)


def limit_peak(y, peak: float = 0.95):
    import numpy as np

    ceiling = max(0.1, min(0.99, peak))
    current = float(np.max(np.abs(y))) if y.size else 0.0
    if current > ceiling:
        y = y * (ceiling / current)
    return np.clip(y, -ceiling, ceiling)


def polish_voice(wav: Path, out_wav: Path, preset: str = "speech",
                 compressor: bool | None = None, deesser: bool | None = None,
                 target_lufs: float | None = None, peak: float | None = None) -> Path:
    import soundfile as sf

    settings = dict(POLISH_PRESETS.get(preset, POLISH_PRESETS["speech"]))
    if compressor is not None:
        settings["compressor"] = compressor
    if deesser is not None:
        settings["deesser"] = deesser
    if target_lufs is not None:
        settings["target_lufs"] = target_lufs
    if peak is not None:
        settings["peak"] = peak

    y, sr = sf.read(str(wav), always_2d=True)
    if settings["deesser"]:
        y = apply_deesser(y, sr)
    if settings["compressor"]:
        y = apply_compressor(y)
    y = match_loudness(y, settings["target_lufs"])
    y = limit_peak(y, settings["peak"])

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), y.squeeze() if y.shape[1] == 1 else y, sr)
    return out_wav


def match_reference_voice(wav: Path, reference_wav: Path, out_wav: Path,
                          target_peak: float = 0.95) -> Path:
    """
    Match a camera voice track toward a same-shoot reference recording.

    This is intentionally conservative: RMS match, broad spectral tilt match,
    then a soft knee compressor/limiter. It is a good place to later swap in a
    learned voice-restoration model while keeping the app workflow stable.
    """
    import numpy as np
    import soundfile as sf

    y, sr = sf.read(str(wav), always_2d=True)
    ref, ref_sr = sf.read(str(reference_wav), always_2d=True)
    if ref_sr != sr:
        import librosa

        ref = np.stack([librosa.resample(ref[:, ch], orig_sr=ref_sr, target_sr=sr)
                        for ch in range(ref.shape[1])], axis=1)

    target_mono = np.mean(y, axis=1)
    ref_mono = np.mean(ref, axis=1)
    target_rms = max(_rms(target_mono), 1e-6)
    ref_rms = max(_rms(ref_mono), 1e-6)
    gain = min(5.0, max(0.2, ref_rms / target_rms))
    y = y * gain

    n = min(len(target_mono), sr * 45)
    ref_n = min(len(ref_mono), sr * 45)
    if n > 4096 and ref_n > 4096:
        fft_size = 8192
        target_spec = np.abs(np.fft.rfft(target_mono[:n] * np.hanning(n), n=fft_size)) + 1e-7
        ref_spec = np.abs(np.fft.rfft(ref_mono[:ref_n] * np.hanning(ref_n), n=fft_size)) + 1e-7
        ratio = np.clip(ref_spec / target_spec, 0.35, 2.8)
        kernel = np.ones(33, dtype=np.float64) / 33
        ratio = np.convolve(ratio, kernel, mode="same")
        freqs = np.fft.rfftfreq(fft_size, 1.0 / sr)
        ratio[(freqs < 70) | (freqs > 14500)] = 1.0

        for ch in range(y.shape[1]):
            spec = np.fft.rfft(y[:, ch], n=max(fft_size, len(y)))
            interp = np.interp(
                np.fft.rfftfreq(max(fft_size, len(y)), 1.0 / sr),
                freqs,
                ratio,
                left=1.0,
                right=1.0,
            )
            y[:, ch] = np.fft.irfft(spec * interp, n=max(fft_size, len(y)))[:len(y)]

    threshold = 0.55
    abs_y = np.abs(y)
    over = abs_y > threshold
    y[over] = np.sign(y[over]) * (threshold + (abs_y[over] - threshold) * 0.45)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0:
        y = y * min(1.0, target_peak / peak)

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), y.squeeze() if y.shape[1] == 1 else y, sr)
    return out_wav


def apply_reference_profile(wav: Path, profile_json: Path, out_wav: Path,
                            target_peak: float = 0.95) -> Path:
    import json
    import numpy as np
    import soundfile as sf

    profile = json.loads(Path(profile_json).read_text(encoding="utf-8"))
    if profile.get("format") != "v-d-reference-profile-v1":
        raise ValueError(f"unsupported reference profile: {profile.get('format')}")

    y, sr = sf.read(str(wav), always_2d=True)
    y = y * float(profile.get("rms_gain", 1.0))

    fft_size = int(profile.get("fft_size", 8192))
    ratio = np.asarray(profile.get("spectral_ratio", []), dtype=np.float64)
    if ratio.size == (fft_size // 2 + 1):
        source_freqs = np.fft.rfftfreq(fft_size, 1.0 / float(profile.get("sample_rate", sr)))
        target_n = max(fft_size, len(y))
        target_freqs = np.fft.rfftfreq(target_n, 1.0 / sr)
        interp = np.interp(target_freqs, source_freqs, ratio, left=1.0, right=1.0)
        interp[(target_freqs < 70) | (target_freqs > 14500)] = 1.0
        for ch in range(y.shape[1]):
            spec = np.fft.rfft(y[:, ch], n=target_n)
            y[:, ch] = np.fft.irfft(spec * interp, n=target_n)[:len(y)]

    y = limit_peak(y, target_peak)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), y.squeeze() if y.shape[1] == 1 else y, sr)
    return out_wav


def process_video(input_path: Path, out_root: Path, device: str = "auto",
                  model: str = "htdemucs_ft", segment: int = 7,
                  keep_instrumental: bool = True,
                  denoise_model: str = "dns64",
                  denoise_dry: float = 0.0,
                  reference_audio: Path | None = None,
                  reference_profile: Path | None = None,
                  polish_preset: str = "speech",
                  compressor: bool = True,
                  deesser: bool = True,
                  target_lufs: float | None = -16.0,
                  peak_ceiling: float = 0.95) -> PipelineResult:
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    suffix = input_path.suffix.lower()
    if suffix not in SUPPORTED_VIDEO and suffix not in SUPPORTED_AUDIO:
        raise ValueError(f"unsupported input type: {suffix}")

    out_root = Path(out_root)
    result_dir = out_root / input_path.stem
    work_dir = result_dir / "_work"
    audio_dir = result_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"[input] {input_path}")
    extracted = extract_audio(input_path, audio_dir / "extracted.wav")
    print(f"[audio] extracted -> {extracted}")

    actual_device = pick_device(device)
    print(f"[voice] separating with Demucs ({model}, {actual_device})")
    vocals, instrumental = separate_voice(extracted, work_dir, actual_device, model=model, segment=segment)
    voice_stem = audio_dir / "voice_raw.wav"
    shutil.copy2(vocals, voice_stem)
    inst_out = None
    if keep_instrumental and instrumental:
        inst_out = audio_dir / "background_no_voice.wav"
        shutil.copy2(instrumental, inst_out)

    print(f"[denoise] neural speech enhancement ({denoise_model})")
    denoised = denoise_voice(voice_stem, work_dir, device=actual_device, model=denoise_model, dry=denoise_dry)
    ref_out = None
    cleaned = audio_dir / "voice_clean.wav"
    polish_src = work_dir / "voice_before_polish.wav"
    if reference_audio:
        reference_audio = Path(reference_audio)
        if not reference_audio.is_file():
            raise FileNotFoundError(reference_audio)
        print(f"[match] reference tone/dynamics -> {reference_audio}")
        ref_out = extract_audio(reference_audio, audio_dir / "reference.wav")
        match_reference_voice(denoised, ref_out, polish_src)
    elif reference_profile:
        reference_profile = Path(reference_profile)
        if not reference_profile.is_file():
            raise FileNotFoundError(reference_profile)
        print(f"[profile] applying reference profile -> {reference_profile}")
        apply_reference_profile(denoised, reference_profile, polish_src)
    else:
        normalize_voice(denoised, polish_src)
    print(f"[polish] preset={polish_preset} compressor={compressor} deesser={deesser} target_lufs={target_lufs}")
    polish_voice(
        polish_src,
        cleaned,
        preset=polish_preset,
        compressor=compressor,
        deesser=deesser,
        target_lufs=target_lufs,
        peak=peak_ceiling,
    )
    print(f"[done] cleaned voice -> {cleaned}")
    return PipelineResult(input_path, result_dir, extracted, voice_stem, cleaned, inst_out, ref_out)


def iter_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    supported = SUPPORTED_VIDEO | SUPPORTED_AUDIO
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in supported)


def main() -> int:
    parser = argparse.ArgumentParser(description="extract voices from video and denoise them")
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, default=Path("output"))
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--model", default="htdemucs_ft")
    parser.add_argument("--segment", type=int, default=7)
    parser.add_argument("--denoise-model", choices=["dns48", "dns64", "master64", "valentini_nc"], default="dns64")
    parser.add_argument("--denoise-dry", type=float, default=0.0)
    parser.add_argument("--reference-audio", type=Path, default=None,
                        help="same-shoot lav/recorder sample used for post-denoise tone and dynamics matching")
    parser.add_argument("--reference-profile", type=Path, default=None,
                        help="profile JSON created by community_training.py build-profile")
    parser.add_argument("--polish-preset", choices=sorted(POLISH_PRESETS), default="speech")
    parser.add_argument("--no-compressor", action="store_true")
    parser.add_argument("--no-deesser", action="store_true")
    parser.add_argument("--target-lufs", type=float, default=None)
    parser.add_argument("--no-loudness", action="store_true")
    parser.add_argument("--peak-ceiling", type=float, default=0.95)
    parser.add_argument("--no-instrumental", action="store_true")
    args = parser.parse_args()

    try:
        inputs = iter_inputs(args.input)
        if not inputs:
            raise RuntimeError(f"no supported audio/video files in {args.input}")
        for input_path in inputs:
            process_video(
                input_path,
                args.out,
                device=args.device,
                model=args.model,
                segment=args.segment,
                keep_instrumental=not args.no_instrumental,
                denoise_model=args.denoise_model,
                denoise_dry=args.denoise_dry,
                reference_audio=args.reference_audio,
                reference_profile=args.reference_profile,
                polish_preset=args.polish_preset,
                compressor=not args.no_compressor,
                deesser=not args.no_deesser,
                target_lufs=None if args.no_loudness else args.target_lufs,
                peak_ceiling=args.peak_ceiling,
            )
    except Exception as exc:
        print(f"[error] {exc!r}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
