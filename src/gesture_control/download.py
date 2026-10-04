"""Download the MediaPipe hand-landmark model."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from .config import HAND_MODEL_PATH, HAND_MODEL_URL


def download_model(path: Path = HAND_MODEL_PATH, force: bool = False) -> Path:
    if path.exists() and not force:
        print(f"Model already present: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {HAND_MODEL_URL}")
    urllib.request.urlretrieve(HAND_MODEL_URL, path)
    print(f"Saved to {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    download_model(force=parser.parse_args().force)


if __name__ == "__main__":
    main()
