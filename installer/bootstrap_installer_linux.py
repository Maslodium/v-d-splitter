"""
V-D Splitter Linux bootstrap installer.

Copies the project into ~/.local/share/V-D Splitter, creates a venv, installs
dependencies and writes a Desktop launcher when possible.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path


APP_NAME = "V-D Splitter"
INSTALL_DIR = Path.home() / ".local" / "share" / APP_NAME


def payload_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("$ " + " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def find_host_python() -> list[str]:
    for cmd in (["python3.12"], ["python3.11"], ["python3"], ["python"]):
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
    raise RuntimeError("Python 3.10+ was not found.")


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


def create_launcher(app_dir: Path, py: Path) -> None:
    launcher_dir = Path.home() / ".local" / "share" / "applications"
    launcher_dir.mkdir(parents=True, exist_ok=True)
    desktop_file = launcher_dir / "v-d-splitter.desktop"
    desktop_file.write_text(
        "[Desktop Entry]\n"
        "Type=Application\n"
        f"Name={APP_NAME}\n"
        "Comment=VOICE-DENOISE speech extraction and cleanup tool\n"
        f"Exec={py} {app_dir / 'gui.py'}\n"
        f"Path={app_dir}\n"
        "Terminal=false\n"
        "Categories=AudioVideo;Audio;\n",
        encoding="utf-8",
    )
    desktop_file.chmod(desktop_file.stat().st_mode | stat.S_IXUSR)


def install_dependencies(app_dir: Path) -> None:
    venv = app_dir / ".venv"
    py = venv / "bin" / "python"
    if not py.exists():
        run([*find_host_python(), "-m", "venv", str(venv)])
    run([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    run([str(py), "-m", "pip", "install", "torch==2.6.0", "torchaudio==2.6.0"])
    run([str(py), "-m", "pip", "install", "-r", str(app_dir / "requirements.txt")])
    create_launcher(app_dir, py)


def main() -> int:
    print(f"{APP_NAME} Linux installer")
    src = payload_root()
    print(f"[copy] {src} -> {INSTALL_DIR}")
    copy_payload(src, INSTALL_DIR)
    install_dependencies(INSTALL_DIR)
    print(f"\nInstalled to: {INSTALL_DIR}")
    print("Launcher: ~/.local/share/applications/v-d-splitter.desktop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
