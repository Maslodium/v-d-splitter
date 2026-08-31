from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
import soundfile as sf

import pipeline


class PipelineSmokeTests(unittest.TestCase):
    def test_supported_video_and_audio_sets_are_broad(self) -> None:
        self.assertIn(".mp4", pipeline.SUPPORTED_VIDEO)
        self.assertIn(".mkv", pipeline.SUPPORTED_VIDEO)
        self.assertIn(".opus", pipeline.SUPPORTED_AUDIO)
        self.assertIn(".wav", pipeline.SUPPORTED_AUDIO)

    def test_normalize_voice_writes_peak_limited_wav(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "voice.wav"
            dst = Path(td) / "voice_clean.wav"
            sr = 16000
            y = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
            sf.write(str(src), y, sr)

            pipeline.normalize_voice(src, dst, target_peak=0.5)

            cleaned, out_sr = sf.read(str(dst), always_2d=False)
            self.assertEqual(out_sr, sr)
            self.assertLessEqual(float(np.max(np.abs(cleaned))), 0.51)

    def test_find_ffmpeg_uses_bundled_imageio_when_available(self) -> None:
        self.assertIsNotNone(pipeline.find_ffmpeg())

    def test_match_reference_voice_writes_limited_wav(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "camera.wav"
            ref = Path(td) / "lav.wav"
            dst = Path(td) / "matched.wav"
            sr = 16000
            t = np.linspace(0, 1, sr, endpoint=False, dtype=np.float32)
            camera = 0.08 * np.sin(2 * np.pi * 260 * t)
            lav = 0.22 * np.sin(2 * np.pi * 260 * t) + 0.03 * np.sin(2 * np.pi * 2500 * t)
            sf.write(str(src), camera, sr)
            sf.write(str(ref), lav, sr)

            pipeline.match_reference_voice(src, ref, dst, target_peak=0.5)

            matched, out_sr = sf.read(str(dst), always_2d=False)
            self.assertEqual(out_sr, sr)
            self.assertLessEqual(float(np.max(np.abs(matched))), 0.51)

    def test_polish_voice_writes_peak_limited_wav(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "voice.wav"
            dst = Path(td) / "polished.wav"
            sr = 16000
            t = np.linspace(0, 1, sr, endpoint=False, dtype=np.float32)
            voice = 0.8 * np.sin(2 * np.pi * 260 * t)
            voice += 0.25 * np.sin(2 * np.pi * 7200 * t)
            sf.write(str(src), voice, sr)

            pipeline.polish_voice(src, dst, preset="speech", target_lufs=-18.0, peak=0.5)

            polished, out_sr = sf.read(str(dst), always_2d=False)
            self.assertEqual(out_sr, sr)
            self.assertLessEqual(float(np.max(np.abs(polished))), 0.51)

    def test_iter_inputs_supports_batch_folders(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "take.mp4").write_bytes(b"fake")
            (root / "voice.wav").write_bytes(b"fake")
            (root / "notes.txt").write_text("ignore", encoding="utf-8")

            found = [p.name for p in pipeline.iter_inputs(root)]

            self.assertEqual(found, ["take.mp4", "voice.wav"])

    def test_apply_reference_profile_writes_wav(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "voice.wav"
            profile = Path(td) / "profile.json"
            dst = Path(td) / "matched.wav"
            sr = 16000
            y = np.zeros(sr, dtype=np.float32)
            y[100:200] = 0.1
            sf.write(str(src), y, sr)
            profile.write_text(json.dumps({
                "format": "v-d-reference-profile-v1",
                "sample_rate": sr,
                "fft_size": 8192,
                "rms_gain": 2.0,
                "spectral_ratio": [1.0] * 4097,
            }), encoding="utf-8")

            pipeline.apply_reference_profile(src, profile, dst, target_peak=0.5)

            matched, out_sr = sf.read(str(dst), always_2d=False)
            self.assertEqual(out_sr, sr)
            self.assertLessEqual(float(np.max(np.abs(matched))), 0.51)

    def test_apply_shoot_profile_writes_wav(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "voice.wav"
            profile = Path(td) / "shoot.json"
            dst = Path(td) / "matched.wav"
            sr = 16000
            y = np.zeros(sr, dtype=np.float32)
            y[100:200] = 0.1
            sf.write(str(src), y, sr)
            profile.write_text(json.dumps({
                "format": "v-d-shoot-profile-v1",
                "sample_rate": sr,
                "fft_size": 8192,
                "pairs": 2,
                "camera": {
                    "rms": 0.1,
                    "noise_floor": 0.01,
                    "spectral_shape": [1.0] * 4097,
                },
                "reference": {
                    "rms": 0.2,
                    "spectral_shape": [1.0] * 4097,
                },
                "transfer": {
                    "rms_gain": 2.0,
                    "spectral_ratio": [1.0] * 4097,
                },
            }), encoding="utf-8")

            pipeline.apply_reference_profile(src, profile, dst, target_peak=0.5)

            matched, out_sr = sf.read(str(dst), always_2d=False)
            self.assertEqual(out_sr, sr)
            self.assertLessEqual(float(np.max(np.abs(matched))), 0.51)

    def test_build_shoot_profile_writes_transfer_profile(self) -> None:
        import csv
        import community_training

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cam_dir = root / "data" / "train" / "camera"
            ref_dir = root / "data" / "train" / "reference"
            cam_dir.mkdir(parents=True)
            ref_dir.mkdir(parents=True)
            sr = 16000
            t = np.linspace(0, 1, sr, endpoint=False, dtype=np.float32)
            camera = 0.08 * np.sin(2 * np.pi * 260 * t)
            reference = 0.18 * np.sin(2 * np.pi * 260 * t)
            sf.write(str(cam_dir / "take01.wav"), camera, sr)
            sf.write(str(ref_dir / "take01.wav"), reference, sr)
            with (root / "metadata.csv").open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=["camera", "reference"])
                writer.writeheader()
                writer.writerow({
                    "camera": "data/train/camera/take01.wav",
                    "reference": "data/train/reference/take01.wav",
                })

            out = root / "shoot.json"
            community_training.build_shoot_profile(root, out)

            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(data["format"], "v-d-shoot-profile-v1")
            self.assertEqual(data["pairs"], 1)
            self.assertIn("camera", data)
            self.assertIn("reference", data)
            self.assertIn("transfer", data)

    def test_apply_reference_model_writes_wav(self) -> None:
        import torch
        from train_reference_model import SpectralMapper

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "voice.wav"
            model_path = root / "model.pt"
            dst = root / "matched.wav"
            sr = 16000
            n_fft = 256
            hidden = 16
            t = np.linspace(0, 0.5, sr // 2, endpoint=False, dtype=np.float32)
            y = 0.1 * np.sin(2 * np.pi * 260 * t)
            sf.write(str(src), y, sr)

            model = SpectralMapper(n_fft // 2 + 1, hidden)
            torch.save({
                "format": "v-d-spectral-mapper-v1",
                "config": {
                    "sample_rate": sr,
                    "n_fft": n_fft,
                    "hop_length": 64,
                    "hidden": hidden,
                },
                "model_state_dict": model.state_dict(),
            }, model_path)

            pipeline.apply_reference_model(src, model_path, dst, target_peak=0.5)

            matched, out_sr = sf.read(str(dst), always_2d=False)
            self.assertEqual(out_sr, sr)
            self.assertGreater(len(matched), 0)
            self.assertLessEqual(float(np.max(np.abs(matched))), 0.51)


if __name__ == "__main__":
    unittest.main()
