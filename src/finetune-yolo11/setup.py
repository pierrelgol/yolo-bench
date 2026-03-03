#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmark_common import config
from benchmark_common.metrics import write_json
from benchmark_common.paths import make_run_name, model_root, repo_root, runs_root, write_latest_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the YOLO11 benchmark run context.")
    parser.add_argument("--run-name", default="", help="Optional explicit run name.")
    return parser.parse_args()


def ensure_nvidia() -> None:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError("NVIDIA GPU is required. No GPU was detected with nvidia-smi.")


def validate_paths(root: Path) -> dict[str, Path]:
    dataset_dir = root / config.DATASET_ROOT
    dataset_yaml = root / config.DATASET_YAML
    class_file = root / "dataset" / "class.txt"
    required = {
        "dataset_dir": dataset_dir,
        "dataset_yaml": dataset_yaml,
        "class_file": class_file,
        "train_images": dataset_dir / "images" / "train2017",
        "val_images": dataset_dir / "images" / "val2017",
        "train_labels": dataset_dir / "labels" / "train2017",
        "val_labels": dataset_dir / "labels" / "val2017",
    }
    for name, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required {name}: {path}")
    return required


def load_class_map(class_file: Path) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for line_number, line in enumerate(class_file.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            raw_id, class_name = stripped.split(maxsplit=1)
            class_id = int(raw_id)
        except ValueError as exc:
            raise ValueError(f"Malformed class mapping at line {line_number}: {line!r}") from exc
        mapping[class_id] = class_name
    if not mapping:
        raise ValueError(f"Class map is empty: {class_file}")
    return mapping


def create_relative_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(source.resolve())


def remap_label_line(line: str, original_to_train: dict[int, int]) -> str:
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"Expected 5 columns in YOLO label, got {len(parts)}: {line!r}")
    original_id = int(parts[0])
    try:
        mapped_id = original_to_train[original_id]
    except KeyError as exc:
        raise ValueError(f"Unknown class id {original_id} in dataset label") from exc
    parts[0] = str(mapped_id)
    return " ".join(parts)


def build_remapped_dataset(required: dict[str, Path], run_root: Path, class_map: dict[int, str]) -> tuple[Path, Path, Path]:
    dataset_root = run_root / "setup" / "dataset_yolo11"
    train_images_out = dataset_root / "images" / "train2017"
    val_images_out = dataset_root / "images" / "val2017"
    train_labels_out = dataset_root / "labels" / "train2017"
    val_labels_out = dataset_root / "labels" / "val2017"

    original_ids = sorted(class_map)
    original_to_train = {original_id: index for index, original_id in enumerate(original_ids)}
    train_to_name = {index: class_map[original_id] for original_id, index in original_to_train.items()}

    for source_dir, destination_dir in [
        (required["train_images"], train_images_out),
        (required["val_images"], val_images_out),
    ]:
        destination_dir.mkdir(parents=True, exist_ok=True)
        for image_path in sorted(source_dir.glob("*")):
            if image_path.is_file():
                create_relative_symlink(image_path, destination_dir / image_path.name)

    for source_dir, destination_dir in [
        (required["train_labels"], train_labels_out),
        (required["val_labels"], val_labels_out),
    ]:
        destination_dir.mkdir(parents=True, exist_ok=True)
        for label_path in sorted(source_dir.glob("*.txt")):
            text = label_path.read_text(encoding="utf-8").strip()
            if not text:
                (destination_dir / label_path.name).write_text("", encoding="utf-8")
                continue
            remapped_lines = [remap_label_line(line, original_to_train) for line in text.splitlines() if line.strip()]
            (destination_dir / label_path.name).write_text("\n".join(remapped_lines) + "\n", encoding="utf-8")

    dataset_yaml = dataset_root / "dataset.yaml"
    names_list = [train_to_name[index] for index in sorted(train_to_name)]
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {dataset_root.resolve()}",
                "train: images/train2017",
                "val: images/val2017",
                f"nc: {len(names_list)}",
                f"names: {json.dumps(names_list)}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    class_map_json = dataset_root / "class_map.json"
    write_json(
        class_map_json,
        {
            "original_to_train": {str(k): v for k, v in original_to_train.items()},
            "train_to_name": {str(k): v for k, v in train_to_name.items()},
        },
    )
    return dataset_root, dataset_yaml, class_map_json


def main() -> int:
    args = parse_args()
    root = repo_root()
    ensure_nvidia()
    required = validate_paths(root)
    class_map = load_class_map(required["class_file"])

    model_dir = model_root(config.YOLO11_MODEL_NAME)
    run_name = args.run_name or make_run_name(config.YOLO11_MODEL_NAME)
    run_root = runs_root(config.YOLO11_MODEL_NAME) / run_name
    for relative in ["setup", "train", "eval", "infer", "bench", "exports", "checkpoints"]:
        (run_root / relative).mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    dataset_root, dataset_yaml, class_map_json = build_remapped_dataset(required, run_root, class_map)
    context = {
        "model_name": config.YOLO11_MODEL_NAME,
        "run_name": run_name,
        "repo_root": str(root),
        "benchmark": {
            "dataset_root": str((root / config.DATASET_ROOT).resolve()),
            "dataset_yaml": str((root / config.DATASET_YAML).resolve()),
            "epochs": config.EPOCHS,
            "eval_every": config.EVAL_EVERY,
            "img_size": config.IMG_SIZE,
            "batch_size": config.BATCH_SIZE,
            "workers": config.NUM_WORKERS,
            "seed": config.SEED,
            "device": config.CUDA_DEVICE,
            "subset_size": config.BENCHMARK_SUBSET_SIZE,
            "wandb_project": config.WANDB_PROJECT,
        },
        "paths": {
            "run_root": str(run_root),
            "setup_dir": str(run_root / "setup"),
            "train_dir": str(run_root / "train"),
            "eval_dir": str(run_root / "eval"),
            "infer_dir": str(run_root / "infer"),
            "bench_dir": str(run_root / "bench"),
            "exports_dir": str(run_root / "exports"),
            "checkpoints_dir": str(run_root / "checkpoints"),
            "prepared_dataset_dir": str(dataset_root),
            "prepared_dataset_yaml": str(dataset_yaml),
            "prepared_class_map": str(class_map_json),
        },
        "model": {
            "variant": config.YOLO11_VARIANT,
            "weights": config.YOLO11_WEIGHTS,
        },
        "training": {
            "completed_epochs": 0,
            "train_total_sec": 0.0,
            "eval_total_sec": 0.0,
            "history": [],
        },
        "inference": {},
        "benchmark_results": {},
    }
    write_json(run_root / "setup" / "run_context.json", context)
    write_latest_run(config.YOLO11_MODEL_NAME, context)
    print(f"Prepared YOLO11 run: {run_name}")
    print(f"Run root: {run_root}")
    print(f"Prepared dataset: {dataset_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
