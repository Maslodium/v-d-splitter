from __future__ import annotations

import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
