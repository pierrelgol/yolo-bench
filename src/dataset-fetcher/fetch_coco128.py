#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics.utils.downloads import safe_download

COCO128_URL = "https://ultralytics.com/assets/coco128.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch the coco128 dataset into the repository dataset directory."
    )
    parser.add_argument(
        "--dataset-dir",
        default="dataset",
        help="Top-level directory where coco128 should be stored.",
    )
    return parser.parse_args()


def dataset_ready(dataset_dir: Path) -> bool:
    coco_root = dataset_dir / "coco128"
    return coco_root.exists() and any(coco_root.rglob("*.jpg"))


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    dataset_dir = (repo_root / args.dataset_dir).resolve()
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if dataset_ready(dataset_dir):
        print(f"coco128 already present in {dataset_dir / 'coco128'}")
        return 0

    print(f"Downloading coco128 into {dataset_dir}", flush=True)
    safe_download(COCO128_URL, dir=dataset_dir, unzip=True, delete=True, exist_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
