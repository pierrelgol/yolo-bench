#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmark_common import config
from benchmark_common.metrics import write_json
from benchmark_common.paths import load_latest_run, write_latest_run
from benchmark_common.runtime import ensure_tensorrt, export_engine_from_onnx, run_ultralytics_inference


def _load_local_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO11 to ONNX and TensorRT, then run Ultralytics inference.")
    parser.add_argument("--checkpoint", default="best", choices=["best", "last"], help="Checkpoint kind to export.")
    return parser.parse_args()


def ensure_runtime() -> None:
    try:
        import onnx  # noqa: F401
        import ultralytics  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError("ONNX and Ultralytics are required for inference/export.") from exc
    ensure_tensorrt()
    result = subprocess.run(["nvidia-smi"], check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("NVIDIA GPU is required. nvidia-smi failed.")


def export_onnx(context: dict, checkpoint: Path) -> Path:
    from ultralytics import YOLO, settings

    settings.update({"wandb": False})
    model = YOLO(str(checkpoint))
    exported = Path(
        model.export(
            format="onnx",
            imgsz=context["benchmark"]["img_size"],
            device=context["benchmark"]["device"],
            simplify=True,
        )
    )
    if not exported.exists():
        raise FileNotFoundError(f"Exported ONNX model not found: {exported}")
    exports_dir = Path(context["paths"]["exports_dir"])
    exports_dir.mkdir(parents=True, exist_ok=True)
    final_path = exports_dir / f"{context['model']['variant']}.onnx"
    if exported.resolve() != final_path.resolve():
        final_path.write_bytes(exported.read_bytes())
    return final_path


def main() -> int:
    args = parse_args()
    ensure_runtime()
    context = load_latest_run(config.YOLO11_MODEL_NAME)

    checkpoint = Path(context["training"][f"{args.checkpoint}_checkpoint"])
    exported_onnx = export_onnx(context, checkpoint)
    engine_path = export_engine_from_onnx(
        onnx_path=exported_onnx,
        engine_path=Path(context["paths"]["exports_dir"]) / f"{context['model']['variant']}.engine",
        img_size=context["benchmark"]["img_size"],
        half=True,
    )
    inference_payload = run_ultralytics_inference(context, engine_path)
    inference_payload.update(
        {
            "checkpoint": str(checkpoint),
            "onnx_path": str(exported_onnx),
            "patched_onnx": str(exported_onnx),
            "engine_path": str(engine_path),
        }
    )
    write_json(Path(context["paths"]["infer_dir"]) / "summary.json", inference_payload)
    context["inference"] = inference_payload

    wandb_module = _load_local_module("finetune_yolo11_wandb_local", Path(__file__).with_name("wandb.py"))
    run_id = context.get("wandb", {}).get("run_id")
    if run_id:
        run = wandb_module.init_run(
            project=context["benchmark"]["wandb_project"],
            run_name=context["run_name"],
            job_type="infer",
            config_payload=context["benchmark"],
            tags=["model:yolo11", "dataset:augment", "stage:infer", "runtime:onnx"],
            run_id=run_id,
        )
        wandb_module.update_summary(
            run,
            {
                "artifact/onnx": str(exported_onnx),
                "artifact/engine": str(engine_path),
                "artifact/checkpoint": str(checkpoint),
            },
        )
        wandb_module.finish_run(run)

    write_latest_run(config.YOLO11_MODEL_NAME, context)
    print(f"TensorRT engine written to {engine_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
