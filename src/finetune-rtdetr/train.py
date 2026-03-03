#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmark_common import config
from benchmark_common.metrics import write_json
from benchmark_common.paths import load_latest_run, write_latest_run


def _load_local_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RT-DETR on the shared benchmark dataset.")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Total epochs to reach.")
    return parser.parse_args()


def ensure_cuda_runtime() -> None:
    result = subprocess.run(["nvidia-smi"], check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("NVIDIA GPU is required. nvidia-smi failed.")
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for RT-DETR training. Install a CUDA-enabled torch build first.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This benchmark does not support CPU fallback.")


def _first_value(row: dict[str, str], keys: list[str]) -> float:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    raise KeyError(f"None of the expected metric keys were found: {keys}")


def parse_results_rows(results_file: Path) -> list[dict[str, float]]:
    with results_file.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No training results found in {results_file}")
    parsed_rows: list[dict[str, float]] = []
    previous_time = 0.0
    for row in rows:
        train_loss = 0.0
        val_loss = 0.0
        for key, value in row.items():
            if value in (None, ""):
                continue
            if key.startswith("train/") and key.endswith("loss"):
                train_loss += float(value)
            if key.startswith("val/") and key.endswith("loss"):
                val_loss += float(value)
        cumulative_time = float(row.get("time", 0.0) or 0.0)
        parsed_rows.append(
            {
                "epoch": int(float(row["epoch"])),
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/precision": _first_value(row, ["metrics/precision(B)", "metrics/precision"]),
                "val/recall": _first_value(row, ["metrics/recall(B)", "metrics/recall"]),
                "val/map50": _first_value(row, ["metrics/mAP50(B)", "metrics/mAP50"]),
                "val/map50_95": _first_value(row, ["metrics/mAP50-95(B)", "metrics/mAP50-95"]),
                "time/cumulative_sec": cumulative_time,
                "time/chunk_sec": cumulative_time - previous_time,
            }
        )
        previous_time = cumulative_time
    return parsed_rows


def train_once(context: dict, epochs: int) -> float:
    from ultralytics import RTDETR, settings

    train_dir = Path(context["paths"]["train_dir"])
    train_dir.mkdir(parents=True, exist_ok=True)
    settings.update({"wandb": False})
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "disabled"

    start = time.perf_counter()
    model = RTDETR(context["model"]["weights"])
    model.train(
        data=context["paths"]["prepared_dataset_yaml"],
        epochs=epochs,
        imgsz=context["benchmark"]["img_size"],
        batch=context["benchmark"]["batch_size"],
        workers=context["benchmark"]["workers"],
        device=context["benchmark"]["device"],
        project=str(train_dir),
        name="native",
        exist_ok=True,
        seed=context["benchmark"]["seed"],
        deterministic=True,
        plots=False,
        val=True,
        save=True,
        save_period=-1,
        verbose=True,
    )
    return time.perf_counter() - start


def reset_training_outputs(context: dict) -> None:
    for key in ["train_dir", "eval_dir", "infer_dir", "bench_dir", "checkpoints_dir"]:
        path = Path(context["paths"][key])
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def snapshot_checkpoints(context: dict) -> tuple[Path, Path]:
    native_weights_dir = Path(context["paths"]["train_dir"]) / "native" / "weights"
    last_source = native_weights_dir / "last.pt"
    best_source = native_weights_dir / "best.pt"
    if not last_source.exists():
        raise FileNotFoundError(f"Missing Ultralytics last checkpoint: {last_source}")
    checkpoints_dir = Path(context["paths"]["checkpoints_dir"])
    last_checkpoint = checkpoints_dir / f"epoch_{context['benchmark']['epochs']:03d}_last.pt"
    best_checkpoint = checkpoints_dir / f"epoch_{context['benchmark']['epochs']:03d}_best.pt"
    shutil.copy2(last_source, last_checkpoint)
    if best_source.exists():
        shutil.copy2(best_source, best_checkpoint)
    else:
        shutil.copy2(last_source, best_checkpoint)
    return last_checkpoint, best_checkpoint


def main() -> int:
    args = parse_args()
    ensure_cuda_runtime()
    context = load_latest_run(config.RTDETR_MODEL_NAME)
    reset_training_outputs(context)
    context["training"] = {
        "completed_epochs": 0,
        "train_total_sec": 0.0,
        "eval_total_sec": 0.0,
        "history": [],
    }
    context["inference"] = {}
    context["benchmark_results"] = {}

    wandb_module = _load_local_module("finetune_rtdetr_wandb_local", Path(__file__).with_name("wandb.py"))

    run = wandb_module.init_run(
        project=context["benchmark"]["wandb_project"],
        run_name=context["run_name"],
        job_type="train",
        config_payload=context["benchmark"],
        tags=["model:rtdetr", "dataset:augment", "runtime:onnx", f"variant:{context['model']['variant']}"],
        run_id=None,
    )
    context["wandb"] = {"run_id": run.id, "project": context["benchmark"]["wandb_project"]}
    write_latest_run(config.RTDETR_MODEL_NAME, context)

    results_file = Path(context["paths"]["train_dir"]) / "native" / "results.csv"
    train_duration = train_once(context, args.epochs)
    rows = parse_results_rows(results_file)
    last_checkpoint, best_checkpoint = snapshot_checkpoints(context)

    context["training"]["completed_epochs"] = args.epochs
    context["training"]["train_total_sec"] = train_duration
    context["training"]["eval_total_sec"] = 0.0
    context["training"]["last_checkpoint"] = str(last_checkpoint)
    context["training"]["best_checkpoint"] = str(best_checkpoint)
    context["training"]["results_file"] = str(results_file)

    for target_epoch in range(config.EVAL_EVERY, args.epochs + 1, config.EVAL_EVERY):
        row = rows[target_epoch - 1]
        start_index = target_epoch - config.EVAL_EVERY
        chunk_rows = rows[start_index:target_epoch]
        train_chunk_sec = sum(item["time/chunk_sec"] for item in chunk_rows)
        eval_metrics = {
            "epoch": target_epoch,
            "val/loss": row["val/loss"],
            "val/precision": row["val/precision"],
            "val/recall": row["val/recall"],
            "val/map50": row["val/map50"],
            "val/map50_95": row["val/map50_95"],
            "time/eval_epoch_sec": 0.0,
        }
        context["training"]["history"].append(
            {
                "epoch": target_epoch,
                "train_duration_sec": train_chunk_sec,
                "eval_duration_sec": 0.0,
                "train_metrics": row,
                "eval_metrics": eval_metrics,
            }
        )
        log_payload = {
            "epoch": target_epoch,
            "train/loss": row["train/loss"],
            "val/loss": row["val/loss"],
            "val/precision": row["val/precision"],
            "val/recall": row["val/recall"],
            "val/map50": row["val/map50"],
            "val/map50_95": row["val/map50_95"],
            "time/train_epoch_sec": train_chunk_sec / config.EVAL_EVERY,
            "time/eval_epoch_sec": 0.0,
        }
        wandb_module.log_metrics(run, log_payload, step=target_epoch)

    context["training"]["last_eval"] = {
        "epoch": args.epochs,
        "val/loss": rows[-1]["val/loss"],
        "val/precision": rows[-1]["val/precision"],
        "val/recall": rows[-1]["val/recall"],
        "val/map50": rows[-1]["val/map50"],
        "val/map50_95": rows[-1]["val/map50_95"],
        "time/eval_epoch_sec": 0.0,
    }
    write_latest_run(config.RTDETR_MODEL_NAME, context)

    summary_path = Path(context["paths"]["train_dir"]) / "summary.json"
    write_json(summary_path, context["training"])
    wandb_module.update_summary(
        run,
        {
            "checkpoint/last": context["training"].get("last_checkpoint", ""),
            "checkpoint/best": context["training"].get("best_checkpoint", ""),
            "bench/train_total_sec": context["training"]["train_total_sec"],
            "bench/eval_total_sec": context["training"]["eval_total_sec"],
        },
    )
    wandb_module.finish_run(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
