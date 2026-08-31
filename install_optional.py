"""
Install optional V-D Splitter backends inside the current virtual environment.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
OPTIONAL = HERE / "requirements-optional.txt"


def run(args: list[str]) -> None:
    print("$ " + " ".join(args), flush=True)
    subprocess.check_call(args)


def main() -> int:
    parser = argparse.ArgumentParser(description="install optional V-D Splitter backends")
    parser.add_argument("--backend", choices=["resemble-enhance", "all"], default="resemble-enhance")
    args = parser.parse_args()

    py = sys.executable
    if args.backend == "all":
        run([py, "-m", "pip", "install", "-r", str(OPTIONAL)])
    else:
        run([py, "-m", "pip", "install", "resemble-enhance"])
    print("\nOptional backend dependencies are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
