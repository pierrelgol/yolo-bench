#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import os
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
    parser = argparse.ArgumentParser(description="Export YOLOv7 to ONNX and run Ultralytics inference.")
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
    yolov7_root = Path(context["yolov7_root"])
    command = [
        sys.executable,
        "export.py",
        "--weights",
        str(checkpoint),
        "--grid",
        "--end2end",
        "--simplify",
        "--topk-all",
        "100",
        "--iou-thres",
        "0.65",
        "--conf-thres",
        "0.35",
        "--img-size",
        str(context["benchmark"]["img_size"]),
        str(context["benchmark"]["img_size"]),
        "--max-wh",
        str(context["benchmark"]["img_size"]),
        "--device",
        context["benchmark"]["device"],
    ]
    env = dict(os.environ)
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    result = subprocess.run(command, cwd=yolov7_root, env=env, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"YOLOv7 export failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    onnx_path = checkpoint.with_suffix(".onnx")
    if not onnx_path.exists():
        raise FileNotFoundError(f"Exported ONNX model not found: {onnx_path}")
    return onnx_path


def patch_onnx_model(source_path: Path, output_path: Path) -> None:
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(str(source_path))
    graph = model.graph
    input_shape = graph.input[0].type.tensor_type.shape
    input_shape.dim[0].dim_value = 1
    original_output_name = graph.output[0].name
    sliced_output_name = f"{original_output_name}_sliced"

    graph.initializer.extend(
        [
            numpy_helper.from_array(np.array([1], dtype=np.int64), name="slice_start"),
            numpy_helper.from_array(np.array([7], dtype=np.int64), name="slice_end"),
            numpy_helper.from_array(np.array([1], dtype=np.int64), name="slice_axes"),
            numpy_helper.from_array(np.array([1], dtype=np.int64), name="slice_steps"),
        ]
    )
    graph.node.append(
        helper.make_node(
            "Slice",
            inputs=[original_output_name, "slice_start", "slice_end", "slice_axes", "slice_steps"],
            outputs=[sliced_output_name],
            name="SliceNode",
        )
    )
    graph.initializer.extend(
        [
            numpy_helper.from_array(np.array([0], dtype=np.int64), name="seg1_start"),
            numpy_helper.from_array(np.array([4], dtype=np.int64), name="seg1_end"),
            numpy_helper.from_array(np.array([4], dtype=np.int64), name="seg2_start"),
            numpy_helper.from_array(np.array([5], dtype=np.int64), name="seg2_end"),
            numpy_helper.from_array(np.array([5], dtype=np.int64), name="seg3_start"),
            numpy_helper.from_array(np.array([6], dtype=np.int64), name="seg3_end"),
        ]
    )

    segment_1_name = f"{sliced_output_name}_segment1"
    segment_2_name = f"{sliced_output_name}_segment2"
    segment_3_name = f"{sliced_output_name}_segment3"
    graph.node.extend(
        [
            helper.make_node(
                "Slice",
                inputs=[sliced_output_name, "seg1_start", "seg1_end", "slice_axes", "slice_steps"],
                outputs=[segment_1_name],
                name="SliceSegment1",
            ),
            helper.make_node(
                "Slice",
                inputs=[sliced_output_name, "seg2_start", "seg2_end", "slice_axes", "slice_steps"],
                outputs=[segment_2_name],
                name="SliceSegment2",
            ),
            helper.make_node(
                "Slice",
                inputs=[sliced_output_name, "seg3_start", "seg3_end", "slice_axes", "slice_steps"],
                outputs=[segment_3_name],
                name="SliceSegment3",
            ),
        ]
    )

    concat_output_name = f"{sliced_output_name}_concat"
    graph.node.append(
        helper.make_node(
            "Concat",
            inputs=[segment_1_name, segment_3_name, segment_2_name],
            outputs=[concat_output_name],
            axis=1,
            name="ConcatSwapped",
        )
    )
    graph.initializer.append(numpy_helper.from_array(np.array([1, -1, 6], dtype=np.int64), name="reshape_shape"))
    final_output_name = f"{concat_output_name}_batched"
    graph.node.append(
        helper.make_node(
            "Reshape",
            inputs=[concat_output_name, "reshape_shape"],
            outputs=[final_output_name],
            name="AddBatchDimension",
        )
    )
    shape_node_name = f"{final_output_name}_shape"
    graph.node.append(helper.make_node("Shape", inputs=[final_output_name], outputs=[shape_node_name], name="GetShapeDim"))
    graph.initializer.append(numpy_helper.from_array(np.array([1], dtype=np.int64), name="dim_1_index"))
    second_dim_name = f"{final_output_name}_dim1"
    graph.node.append(
        helper.make_node(
            "Gather",
            inputs=[shape_node_name, "dim_1_index"],
            outputs=[second_dim_name],
            name="GatherSecondDim",
        )
    )
    graph.initializer.append(numpy_helper.from_array(np.array([100], dtype=np.int64), name="target_size"))
    pad_size_name = f"{second_dim_name}_padsize"
    graph.node.append(
        helper.make_node("Sub", inputs=["target_size", second_dim_name], outputs=[pad_size_name], name="CalculatePadSize")
    )
    graph.initializer.append(numpy_helper.from_array(np.array([0, 0, 0], dtype=np.int64), name="pad_starts"))
    graph.initializer.append(numpy_helper.from_array(np.array([0], dtype=np.int64), name="zero_scalar"))
    graph.node.append(
        helper.make_node("Concat", inputs=["zero_scalar", pad_size_name, "zero_scalar"], outputs=["pad_ends"], axis=0, name="ConcatPadEnds")
    )
    graph.node.append(
        helper.make_node("Concat", inputs=["pad_starts", "pad_ends"], outputs=["pad_values"], axis=0, name="ConcatPadStartsEnds")
    )
    graph.initializer.append(numpy_helper.from_array(np.array([0.0], dtype=np.float32), name="pad_constant_value"))
    pad_output_name = f"{final_output_name}_padded"
    graph.node.append(
        helper.make_node(
            "Pad",
            inputs=[final_output_name, "pad_values", "pad_constant_value"],
            outputs=[pad_output_name],
            mode="constant",
            name="PadToFixedSize",
        )
    )

    new_output_type = onnx.helper.make_tensor_type_proto(
        elem_type=graph.output[0].type.tensor_type.elem_type,
        shape=[1, 100, 6],
    )
    graph.output.pop()
    graph.output.extend([onnx.helper.make_value_info(name=pad_output_name, type_proto=new_output_type)])
    onnx.save(model, str(output_path))


def fixed_subset(context: dict) -> list[Path]:
    val_dir = Path(context["benchmark"]["dataset_root"]) / "images" / "val2017"
    return sorted(val_dir.glob("*.jpg"))[: context["benchmark"]["subset_size"]]


def run_ultralytics_inference(context: dict, onnx_path: Path) -> dict:
    from ultralytics import YOLO

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
    context = load_latest_run(config.YOLO7_MODEL_NAME)

    checkpoint = Path(context["training"][f"{args.checkpoint}_checkpoint"])
    exported_onnx = export_onnx(context, checkpoint)
    exports_dir = Path(context["paths"]["exports_dir"])
    exports_dir.mkdir(parents=True, exist_ok=True)
    patched_onnx = exports_dir / f"{context['model']['variant']}-ultralytics.onnx"
    patch_onnx_model(exported_onnx, patched_onnx)
    inference_payload = run_ultralytics_inference(context, patched_onnx)
    inference_payload.update(
        {
            "checkpoint": str(checkpoint),
            "exported_onnx": str(exported_onnx),
            "patched_onnx": str(patched_onnx),
        }
    )
    write_json(Path(context["paths"]["infer_dir"]) / "summary.json", inference_payload)
    context["inference"] = inference_payload

    wandb_module = _load_local_module("finetune_yolo7_wandb_local", Path(__file__).with_name("wandb.py"))
    run_id = context.get("wandb", {}).get("run_id")
    if run_id:
        run = wandb_module.init_run(
            project=context["benchmark"]["wandb_project"],
            run_name=context["run_name"],
            job_type="infer",
            config_payload=context["benchmark"],
            tags=["model:yolo7", "dataset:augment", "stage:infer", "runtime:onnx"],
            run_id=run_id,
        )
        wandb_module.update_summary(
            run,
            {
                "artifact/onnx": str(patched_onnx),
                "artifact/checkpoint": str(checkpoint),
            },
        )
        wandb_module.finish_run(run)

    write_latest_run(config.YOLO7_MODEL_NAME, context)
    print(f"Patched ONNX written to {patched_onnx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
