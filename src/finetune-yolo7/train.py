#!/usr/bin/env python3

from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Train YOLOv7 on the shared benchmark dataset.")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Total epochs to reach.")
    return parser.parse_args()


def ensure_cuda_runtime() -> None:
    result = subprocess.run(["nvidia-smi"], check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("NVIDIA GPU is required. nvidia-smi failed.")
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for YOLOv7 training. Install a CUDA-enabled torch build first.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This benchmark does not support CPU fallback.")


def parse_results_line(results_file: Path) -> dict[str, float]:
    lines = [line.strip() for line in results_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"No training results found in {results_file}")
    fields = lines[-1].split()
    if len(fields) < 15:
        raise RuntimeError(f"Unexpected results.txt line: {lines[-1]}")

    epoch_token = fields[0]
    epoch = int(epoch_token.split("/")[0]) + 1
    return {
        "epoch": epoch,
        "train/box_loss": float(fields[2]),
        "train/obj_loss": float(fields[3]),
        "train/cls_loss": float(fields[4]),
        "train/loss": float(fields[5]),
        "val/precision": float(fields[8]),
        "val/recall": float(fields[9]),
        "val/map50": float(fields[10]),
        "val/map50_95": float(fields[11]),
        "val/box_loss": float(fields[12]),
        "val/obj_loss": float(fields[13]),
        "val/cls_loss": float(fields[14]),
        "val/loss": float(fields[12]) + float(fields[13]) + float(fields[14]),
    }


def train_chunk(context: dict, weights: Path, target_epoch: int) -> float:
    train_dir = Path(context["paths"]["train_dir"])
    yolov7_root = Path(context["yolov7_root"])
    command = [
        sys.executable,
        "train.py",
        "--weights",
        str(weights),
        "--data",
        context["paths"]["prepared_dataset_yaml"],
        "--hyp",
        context["model"]["hyp"],
        "--epochs",
        str(target_epoch),
        "--batch-size",
        str(context["benchmark"]["batch_size"]),
        "--img-size",
        str(context["benchmark"]["img_size"]),
        str(context["benchmark"]["img_size"]),
        "--device",
        context["benchmark"]["device"],
        "--workers",
        str(context["benchmark"]["workers"]),
        "--project",
        str(train_dir),
        "--name",
        "native",
        "--exist-ok",
        "--notest",
        "--v5-metric",
    ]
    env = dict(os.environ)
    env["WANDB_DISABLED"] = "true"
    env["WANDB_MODE"] = "disabled"
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    start = time.perf_counter()
    result = subprocess.run(command, cwd=yolov7_root, env=env, check=False, capture_output=True, text=True)
    duration = time.perf_counter() - start
    if result.returncode != 0:
        raise RuntimeError(f"YOLOv7 training failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    return duration


def main() -> int:
    args = parse_args()
    ensure_cuda_runtime()
    context = load_latest_run(config.YOLO7_MODEL_NAME)

    wandb_module = _load_local_module("finetune_yolo7_wandb_local", Path(__file__).with_name("wandb.py"))
    eval_module = _load_local_module("finetune_yolo7_eval_local", Path(__file__).with_name("eval.py"))

    run = wandb_module.init_run(
        project=context["benchmark"]["wandb_project"],
        run_name=context["run_name"],
        job_type="train",
        config_payload=context["benchmark"],
        tags=["model:yolo7", "dataset:augment", "runtime:onnx", f"variant:{context['model']['variant']}"],
        run_id=context.get("wandb", {}).get("run_id"),
    )
    context["wandb"] = {"run_id": run.id, "project": context["benchmark"]["wandb_project"]}
    write_latest_run(config.YOLO7_MODEL_NAME, context)

    checkpoints_dir = Path(context["paths"]["checkpoints_dir"])
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    native_weights_dir = Path(context["paths"]["train_dir"]) / "native" / "weights"
    results_file = Path(context["paths"]["train_dir"]) / "native" / "results.txt"
    base_train_loss = context["training"].get("loss_baseline", {}).get("train/loss")
    base_val_loss = context["training"].get("loss_baseline", {}).get("val/loss")

    completed = int(context["training"].get("completed_epochs", 0))
    for target_epoch in range(completed + config.EVAL_EVERY, args.epochs + 1, config.EVAL_EVERY):
        source_weights = (
            Path(context["model"]["weights"])
            if target_epoch == config.EVAL_EVERY and completed == 0
            else Path(context["training"]["last_checkpoint"])
        )
        train_duration = train_chunk(context, source_weights, target_epoch)
        chunk_metrics = parse_results_line(results_file)
        eval_payload = eval_module.run_native_eval(
            context=context,
            checkpoint=native_weights_dir / "last.pt",
            epoch=target_epoch,
            name="periodic",
        )
        if base_train_loss is None:
            base_train_loss = chunk_metrics["train/loss"] or 1.0
        if base_val_loss is None:
            base_val_loss = chunk_metrics["val/loss"] or 1.0

        last_checkpoint = checkpoints_dir / f"epoch_{target_epoch:03d}_last.pt"
        best_checkpoint = checkpoints_dir / f"epoch_{target_epoch:03d}_best.pt"
        if (native_weights_dir / "last.pt").exists():
            shutil.copy2(native_weights_dir / "last.pt", last_checkpoint)
        if (native_weights_dir / "best.pt").exists():
            shutil.copy2(native_weights_dir / "best.pt", best_checkpoint)

        context["training"]["completed_epochs"] = target_epoch
        context["training"]["train_total_sec"] += train_duration
        context["training"]["eval_total_sec"] += eval_payload["metrics"]["time/eval_epoch_sec"]
        context["training"]["last_checkpoint"] = str(last_checkpoint)
        context["training"]["best_checkpoint"] = str(best_checkpoint if best_checkpoint.exists() else last_checkpoint)
        context["training"]["results_file"] = str(results_file)
        context["training"]["loss_baseline"] = {"train/loss": base_train_loss, "val/loss": base_val_loss}
        context["training"]["history"].append(
            {
                "epoch": target_epoch,
                "train_duration_sec": train_duration,
                "eval_duration_sec": eval_payload["metrics"]["time/eval_epoch_sec"],
                "train_metrics": {
                    **chunk_metrics,
                    "train/loss_norm": chunk_metrics["train/loss"] / base_train_loss,
                    "val/loss_norm": chunk_metrics["val/loss"] / base_val_loss,
                },
                "eval_metrics": {
                    **eval_payload["metrics"],
                    "val/loss_norm": chunk_metrics["val/loss"] / base_val_loss,
                },
            }
        )

        log_payload = {
            "epoch": target_epoch,
            "train/loss": chunk_metrics["train/loss"],
            "train/loss_norm": chunk_metrics["train/loss"] / base_train_loss,
            "val/loss": chunk_metrics["val/loss"],
            "val/loss_norm": chunk_metrics["val/loss"] / base_val_loss,
            "val/precision": eval_payload["metrics"]["val/precision"],
            "val/recall": eval_payload["metrics"]["val/recall"],
            "val/map50": eval_payload["metrics"]["val/map50"],
            "val/map50_95": eval_payload["metrics"]["val/map50_95"],
            "time/train_epoch_sec": train_duration / config.EVAL_EVERY,
            "time/eval_epoch_sec": eval_payload["metrics"]["time/eval_epoch_sec"],
        }
        wandb_module.log_metrics(run, log_payload, step=target_epoch)
        write_latest_run(config.YOLO7_MODEL_NAME, context)

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
