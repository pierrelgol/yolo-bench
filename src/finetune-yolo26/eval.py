#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
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
    parser = argparse.ArgumentParser(description="Evaluate the latest YOLO26 checkpoint.")
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


def run_native_eval(context: dict, checkpoint: Path, epoch: int, name: str) -> dict:
    from ultralytics import YOLO, settings

    settings.update({"wandb": False})
    eval_dir = Path(context["paths"]["eval_dir"])
    output_name = f"{name}-epoch{epoch:03d}" if epoch else name
    eval_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    model = YOLO(str(checkpoint))
    results = model.val(
        data=context["paths"]["prepared_dataset_yaml"],
        imgsz=context["benchmark"]["img_size"],
        batch=context["benchmark"]["batch_size"],
        workers=context["benchmark"]["workers"],
        device=context["benchmark"]["device"],
        project=str(eval_dir),
        name=output_name,
        exist_ok=True,
        split="val",
        plots=False,
        verbose=False,
    )
    duration = time.perf_counter() - start
    results_dict = results.results_dict
    val_loss = _lookup_val_loss(context, epoch)
    base_val_loss = context.get("training", {}).get("history", [{}])[0].get("train_metrics", {}).get("val/loss", 0.0) or val_loss or 1.0
    metrics = {
        "epoch": epoch,
        "val/loss": val_loss,
        "val/loss_norm": val_loss / base_val_loss,
        "val/precision": float(results_dict["metrics/precision(B)"]),
        "val/recall": float(results_dict["metrics/recall(B)"]),
        "val/map50": float(results_dict["metrics/mAP50(B)"]),
        "val/map50_95": float(results_dict["metrics/mAP50-95(B)"]),
        "time/eval_epoch_sec": duration,
    }
    payload = {
        "checkpoint": str(checkpoint),
        "metrics": metrics,
        "raw_metrics": {key: float(value) for key, value in results_dict.items()},
        "speed": getattr(results, "speed", {}),
    }
    write_json(eval_dir / f"{output_name}.json", payload)
    return payload


def main() -> int:
    args = parse_args()
    ensure_nvidia()
    context = load_latest_run(config.YOLO26_MODEL_NAME)
    checkpoint = Path(args.checkpoint) if args.checkpoint else Path(context["training"].get("last_checkpoint", ""))
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    payload = run_native_eval(context, checkpoint, args.epoch, args.name)
    context["training"]["last_eval"] = payload["metrics"]

    wandb_module = _load_local_module("finetune_yolo26_wandb_local", Path(__file__).with_name("wandb.py"))
    run_id = context.get("wandb", {}).get("run_id")
    if run_id:
        run = wandb_module.init_run(
            project=context["benchmark"]["wandb_project"],
            run_name=context["run_name"],
            job_type="eval",
            config_payload=context["benchmark"],
            tags=["model:yolo26", "dataset:augment", "stage:eval"],
            run_id=run_id,
        )
        step = payload["metrics"]["epoch"] or None
        if step is not None and step <= int(context["training"].get("completed_epochs", 0)):
            step = None
        wandb_module.log_metrics(run, payload["metrics"], step=step)
        wandb_module.finish_run(run)

    write_latest_run(config.YOLO26_MODEL_NAME, context)
    print(payload["metrics"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
