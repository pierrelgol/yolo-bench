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


def _load_local_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export RT-DETR to ONNX and run Ultralytics inference.")
    parser.add_argument("--checkpoint", default="best", choices=["best", "last"], help="Checkpoint kind to export.")
    return parser.parse_args()


def ensure_runtime() -> None:
    try:
        import onnx  # noqa: F401
        import ultralytics  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError("ONNX and Ultralytics are required for inference/export.") from exc
    result = subprocess.run(["nvidia-smi"], check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("NVIDIA GPU is required. nvidia-smi failed.")


def export_onnx(context: dict, checkpoint: Path) -> Path:
    from ultralytics import RTDETR, settings

    settings.update({"wandb": False})
    model = RTDETR(str(checkpoint))
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


def fixed_subset(context: dict) -> list[Path]:
    val_dir = Path(context["benchmark"]["dataset_root"]) / "images" / "val2017"
    return sorted(val_dir.glob("*.jpg"))[: context["benchmark"]["subset_size"]]


def run_ultralytics_inference(context: dict, onnx_path: Path) -> dict:
    from ultralytics import YOLO, settings

    settings.update({"wandb": False})
    subset = fixed_subset(context)
    model = YOLO(str(onnx_path), task="detect")
    prediction_summary = []
    for image_path in subset:
        results = model.predict(
            source=str(image_path),
            device=int(context["benchmark"]["device"]),
            imgsz=context["benchmark"]["img_size"],
            verbose=False,
        )
        count = 0
        if results and results[0].boxes is not None:
            count = len(results[0].boxes)
        prediction_summary.append({"image": image_path.name, "detections": count})
    return {"subset": [path.name for path in subset], "predictions": prediction_summary}


def main() -> int:
    args = parse_args()
    ensure_runtime()
    context = load_latest_run(config.RTDETR_MODEL_NAME)

    checkpoint = Path(context["training"][f"{args.checkpoint}_checkpoint"])
    exported_onnx = export_onnx(context, checkpoint)
    inference_payload = run_ultralytics_inference(context, exported_onnx)
    inference_payload.update(
        {
            "checkpoint": str(checkpoint),
            "patched_onnx": str(exported_onnx),
        }
    )
    write_json(Path(context["paths"]["infer_dir"]) / "summary.json", inference_payload)
    context["inference"] = inference_payload

    wandb_module = _load_local_module("finetune_rtdetr_wandb_local", Path(__file__).with_name("wandb.py"))
    run_id = context.get("wandb", {}).get("run_id")
    if run_id:
        run = wandb_module.init_run(
            project=context["benchmark"]["wandb_project"],
            run_name=context["run_name"],
            job_type="infer",
            config_payload=context["benchmark"],
            tags=["model:rtdetr", "dataset:augment", "stage:infer", "runtime:onnx"],
            run_id=run_id,
        )
        wandb_module.update_summary(
            run,
            {
                "artifact/onnx": str(exported_onnx),
                "artifact/checkpoint": str(checkpoint),
            },
        )
        wandb_module.finish_run(run)

    write_latest_run(config.RTDETR_MODEL_NAME, context)
    print(f"ONNX written to {exported_onnx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
