from __future__ import annotations

import time
from pathlib import Path


def fixed_subset(context: dict) -> list[Path]:
    val_dir = Path(context["benchmark"]["dataset_root"]) / "images" / "val2017"
    return sorted(val_dir.glob("*.jpg"))[: context["benchmark"]["subset_size"]]


def ensure_tensorrt() -> None:
    try:
        import tensorrt  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TensorRT Python bindings are required for engine export and benchmarking. "
            "Install the NVIDIA TensorRT Python package in this environment first."
        ) from exc


def export_engine_from_onnx(
    *,
    onnx_path: Path,
    engine_path: Path,
    img_size: int,
    half: bool = True,
    workspace: int | None = None,
) -> Path:
    ensure_tensorrt()
    from ultralytics.utils.export import onnx2engine

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    onnx2engine(
        str(onnx_path),
        engine_file=str(engine_path),
        workspace=workspace,
        half=half,
        shape=(1, 3, img_size, img_size),
    )
    if not engine_path.exists():
        raise FileNotFoundError(f"TensorRT engine export did not produce {engine_path}")
    return engine_path


def run_ultralytics_inference(context: dict, model_path: Path) -> dict:
    from ultralytics import YOLO, settings

    settings.update({"wandb": False})
    subset = fixed_subset(context)
    model = YOLO(str(model_path), task="detect")
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


def benchmark_ultralytics_runtime(context: dict, model_path: Path, warmup: int) -> dict:
    from ultralytics import YOLO, settings

    settings.update({"wandb": False})
    subset = fixed_subset(context)
    model = YOLO(str(model_path), task="detect")
    warmup_source = str(subset[0])
    for _ in range(warmup):
        model.predict(
            source=warmup_source,
            device=int(context["benchmark"]["device"]),
            imgsz=context["benchmark"]["img_size"],
            verbose=False,
        )

    total = 0.0
    for image_path in subset:
        start = time.perf_counter()
        model.predict(
            source=str(image_path),
            device=int(context["benchmark"]["device"]),
            imgsz=context["benchmark"]["img_size"],
            verbose=False,
        )
        total += time.perf_counter() - start

    latency_ms = (total / len(subset)) * 1000.0
    throughput = len(subset) / total if total > 0 else 0.0
    return {
        "subset": [path.name for path in subset],
        "latency_ms": latency_ms,
        "throughput_img_s": throughput,
    }
