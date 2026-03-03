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
from benchmark_common.runtime import benchmark_ultralytics_runtime, ensure_tensorrt


def _load_local_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the exported YOLO26 TensorRT runtime.")
    parser.add_argument("--warmup", type=int, default=config.BENCHMARK_WARMUP, help="Warmup iterations.")
    return parser.parse_args()


def ensure_runtime() -> None:
    result = subprocess.run(["nvidia-smi"], check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("NVIDIA GPU is required. nvidia-smi failed.")
    try:
        import ultralytics  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError("Ultralytics is required for TensorRT benchmarking.") from exc
    ensure_tensorrt()


def main() -> int:
    args = parse_args()
    ensure_runtime()
    context = load_latest_run(config.YOLO26_MODEL_NAME)
    engine_path = Path(context.get("inference", {}).get("engine_path", ""))
    onnx_path = Path(context.get("inference", {}).get("onnx_path", ""))
    if not engine_path.exists():
        raise FileNotFoundError("Missing TensorRT engine. Run yolo26-infer first.")

    bench_runtime = benchmark_ultralytics_runtime(context, engine_path, args.warmup)
    summary = {
        "bench/infer_latency_ms": bench_runtime["latency_ms"],
        "bench/infer_throughput_img_s": bench_runtime["throughput_img_s"],
        "bench/trt_infer_latency_ms": bench_runtime["latency_ms"],
        "bench/trt_infer_throughput_img_s": bench_runtime["throughput_img_s"],
        "bench/onnx_file_size_mb": onnx_path.stat().st_size / (1024 * 1024) if onnx_path.exists() else 0.0,
        "bench/trt_engine_size_mb": engine_path.stat().st_size / (1024 * 1024),
        "bench/train_total_sec": context["training"].get("train_total_sec", 0.0),
        "bench/eval_total_sec": context["training"].get("eval_total_sec", 0.0),
        "subset": bench_runtime["subset"],
        "onnx_path": str(onnx_path),
        "engine_path": str(engine_path),
    }
    bench_dir = Path(context["paths"]["bench_dir"])
    write_json(bench_dir / "summary.json", summary)
    context["benchmark_results"] = summary

    wandb_module = _load_local_module("finetune_yolo26_wandb_local", Path(__file__).with_name("wandb.py"))
    run_id = context.get("wandb", {}).get("run_id")
    if run_id:
        run = wandb_module.init_run(
            project=context["benchmark"]["wandb_project"],
            run_name=context["run_name"],
            job_type="bench",
            config_payload=context["benchmark"],
            tags=["model:yolo26", "dataset:augment", "stage:bench", "runtime:onnx"],
            run_id=run_id,
        )
        wandb_module.log_metrics(run, summary)
        wandb_module.update_summary(run, summary)
        wandb_module.finish_run(run)

    write_latest_run(config.YOLO26_MODEL_NAME, context)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
