#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import os
import re
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
    parser = argparse.ArgumentParser(description="Evaluate the latest YOLOv7 checkpoint.")
    parser.add_argument("--checkpoint", default="", help="Optional checkpoint path to evaluate.")
    parser.add_argument("--epoch", type=int, default=0, help="Epoch to associate with this evaluation.")
    parser.add_argument("--name", default="manual", help="Output name inside the eval artifact directory.")
    return parser.parse_args()


def ensure_nvidia() -> None:
    result = subprocess.run(["nvidia-smi"], check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("NVIDIA GPU is required. nvidia-smi failed.")


def _lookup_val_loss(context: dict, epoch: int) -> float:
    history = context.get("training", {}).get("history", [])
    if epoch:
        for item in history:
            if item.get("epoch") == epoch:
                return float(item.get("train_metrics", {}).get("val/loss", 0.0))
    if history:
        return float(history[-1].get("train_metrics", {}).get("val/loss", 0.0))
    return 0.0


def _parse_eval_stdout(stdout: str) -> dict[str, float]:
    metrics_line = None
    speed_line = None
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("all"):
            metrics_line = stripped
        if stripped.startswith("Speed:"):
            speed_line = stripped

    if metrics_line is None:
        raise RuntimeError(f"Failed to parse YOLOv7 evaluation metrics from output:\n{stdout}")

    fields = metrics_line.split()
    if len(fields) < 7:
        raise RuntimeError(f"Unexpected YOLOv7 metrics line: {metrics_line}")
    metrics = {
        "val/precision": float(fields[3]),
        "val/recall": float(fields[4]),
        "val/map50": float(fields[5]),
        "val/map50_95": float(fields[6]),
    }
    if speed_line:
        match = re.search(r"Speed:\s*([0-9.]+)/([0-9.]+)/([0-9.]+)\s*ms", speed_line)
        if match:
            metrics["speed/inference_ms"] = float(match.group(1))
            metrics["speed/nms_ms"] = float(match.group(2))
            metrics["speed/total_ms"] = float(match.group(3))
    return metrics


def run_native_eval(context: dict, checkpoint: Path, epoch: int, name: str) -> dict:
    eval_dir = Path(context["paths"]["eval_dir"])
    output_name = f"{name}-epoch{epoch:03d}" if epoch else name
    yolov7_root = Path(context["yolov7_root"])
    command = [
        sys.executable,
        "test.py",
        "--weights",
        str(checkpoint),
        "--data",
        context["paths"]["prepared_dataset_yaml"],
        "--batch-size",
        str(context["benchmark"]["batch_size"]),
        "--img-size",
        str(context["benchmark"]["img_size"]),
        "--device",
        context["benchmark"]["device"],
        "--project",
        str(eval_dir),
        "--name",
        output_name,
        "--exist-ok",
        "--task",
        "val",
        "--no-trace",
        "--v5-metric",
    ]
    env = dict(os.environ)
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    start = time.perf_counter()
    result = subprocess.run(command, cwd=yolov7_root, env=env, check=False, capture_output=True, text=True)
    duration = time.perf_counter() - start
    if result.returncode != 0:
        raise RuntimeError(f"YOLOv7 evaluation failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    metrics = _parse_eval_stdout(result.stdout)
    val_loss = _lookup_val_loss(context, epoch)
    base_val_loss = context.get("training", {}).get("loss_baseline", {}).get("val/loss") or val_loss or 1.0
    metrics["val/loss"] = val_loss
    metrics["val/loss_norm"] = val_loss / base_val_loss if base_val_loss else 0.0
    metrics["epoch"] = epoch
    metrics["time/eval_epoch_sec"] = duration
    payload = {
        "checkpoint": str(checkpoint),
        "metrics": metrics,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    write_json(eval_dir / f"{output_name}.json", payload)
    return payload


def main() -> int:
    args = parse_args()
    ensure_nvidia()
    context = load_latest_run(config.YOLO7_MODEL_NAME)
    checkpoint = Path(args.checkpoint) if args.checkpoint else Path(context["training"].get("last_checkpoint", ""))
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    payload = run_native_eval(context, checkpoint, args.epoch, args.name)
    context["training"]["last_eval"] = payload["metrics"]

    wandb_module = _load_local_module("finetune_yolo7_wandb_local", Path(__file__).with_name("wandb.py"))
    run_id = context.get("wandb", {}).get("run_id")
    if run_id:
        run = wandb_module.init_run(
            project=context["benchmark"]["wandb_project"],
            run_name=context["run_name"],
            job_type="eval",
            config_payload=context["benchmark"],
            tags=["model:yolo7", "dataset:augment", "stage:eval"],
            run_id=run_id,
        )
        step = payload["metrics"]["epoch"] or None
        if step is not None and step <= int(context["training"].get("completed_epochs", 0)):
            step = None
        wandb_module.log_metrics(run, payload["metrics"], step=step)
        wandb_module.finish_run(run)

    write_latest_run(config.YOLO7_MODEL_NAME, context)
    print(payload["metrics"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
