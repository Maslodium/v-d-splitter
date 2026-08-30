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


if __name__ == "__main__":
    unittest.main()
