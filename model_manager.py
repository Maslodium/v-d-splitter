"""
Download and publish V-D Splitter community models.

All network operations are explicit commands. The app never uploads user audio
or checkpoints by itself.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


DEFAULT_MODEL_REPO = "Maslodium/v-d-splitter-models"
DEFAULT_DATASET_REPO = "Maslodium/v-d-splitter-community"


def require_hf() -> None:
    if shutil.which("hf") is None:
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


def hf_upload(folder: Path, repo_id: str, repo_type: str,
              message: str = "Upload V-D Splitter artifact") -> None:
    require_hf()
    if not folder.exists():
        raise FileNotFoundError(folder)
    cmd = [
        "hf",
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


def main() -> int:
    parser = argparse.ArgumentParser(description="download and publish V-D Splitter community models")
    sub = parser.add_subparsers(dest="cmd", required=True)

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

    pub_dataset = sub.add_parser("publish-dataset")
    pub_dataset.add_argument("--folder", type=Path, required=True)
    pub_dataset.add_argument("--repo-id", default=DEFAULT_DATASET_REPO)
    pub_dataset.add_argument("--message", default="Upload V-D Splitter dataset")

    args = parser.parse_args()
    try:
        if args.cmd == "download-model":
            hf_download(args.repo_id, args.filename, args.out, "model", args.revision)
        elif args.cmd == "download-profile":
            hf_download(args.repo_id, args.filename, args.out, "model", args.revision)
        elif args.cmd == "publish-model":
            hf_upload(args.folder, args.repo_id, "model", args.message)
        elif args.cmd == "publish-dataset":
            hf_upload(args.folder, args.repo_id, "dataset", args.message)
    except Exception as exc:
        print(f"[error] {exc!r}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
