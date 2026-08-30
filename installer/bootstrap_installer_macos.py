"""
Voice-Denoise Splitter macOS bootstrap installer.

Packaged as a small .app on macOS. It copies the app payload into
~/Applications/Voice-Denoise Splitter, creates a venv, installs dependencies,
and creates a Desktop launcher.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path


APP_NAME = "Voice-Denoise Splitter"
INSTALL_DIR = Path.home() / "Applications" / APP_NAME


def payload_root() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / "payload"
    return Path(__file__).resolve().parents[1]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("$ " + " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def find_host_python() -> list[str]:
    if not getattr(sys, "frozen", False):
        return [sys.executable]
    candidates = [
        ["python3.12"],
        ["python3.11"],
        ["python3"],
        ["python"],
    ]
    for cmd in candidates:
        try:
            proc = subprocess.run(
                [*cmd, "-c", "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if proc.returncode == 0:
                return cmd
        except Exception:
            pass
    raise RuntimeError("Python 3.10+ was not found. Install Python 3 from python.org or Homebrew and run again.")


def copy_payload(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    ignored = {".git", ".venv", "build", "dist", "payload", "__pycache__"}
    for item in src.iterdir():
        if item.name in ignored or item.name.endswith(".spec"):
            continue
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def create_launcher(app_dir: Path, pyw: Path) -> None:
    desktop = Path.home() / "Desktop"
    desktop.mkdir(parents=True, exist_ok=True)
    launcher = desktop / f"{APP_NAME}.command"
    script = app_dir / "gui.py"
    launcher.write_text(
        "#!/bin/zsh\n"
        f"cd {sh_quote(str(app_dir))}\n"
        f"exec {sh_quote(str(pyw))} {sh_quote(str(script))}\n",
        encoding="utf-8",
    )
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def sh_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def install_dependencies(app_dir: Path) -> None:
    venv = app_dir / ".venv"
    py = venv / "bin" / "python"
    pythonw = venv / "bin" / "pythonw"
    pyw = pythonw if pythonw.exists() else py
    if not py.exists():
        run([*find_host_python(), "-m", "venv", str(venv)])

    run([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    print("[scan] macOS detected; installing PyTorch with CPU/MPS support.")
    run([str(py), "-m", "pip", "install", "torch==2.6.0", "torchaudio==2.6.0"])
    run([str(py), "-m", "pip", "install", "-r", str(app_dir / "requirements.txt")])
    create_launcher(app_dir, pyw)


def main() -> int:
    print(f"{APP_NAME} macOS installer")
    src = payload_root()
    print(f"[copy] {src} -> {INSTALL_DIR}")
    copy_payload(src, INSTALL_DIR)
    install_dependencies(INSTALL_DIR)
    print(f"\nInstalled to: {INSTALL_DIR}")
    print(f"Desktop launcher: {APP_NAME}.command")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
