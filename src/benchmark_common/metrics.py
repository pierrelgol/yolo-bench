from __future__ import annotations

import json
from pathlib import Path

CANONICAL_METRICS = [
    "epoch",
    "train/loss",
    "train/loss_norm",
    "val/loss",
    "val/loss_norm",
    "val/precision",
    "val/recall",
    "val/map50",
    "val/map50_95",
    "time/train_epoch_sec",
    "time/eval_epoch_sec",
    "bench/infer_latency_ms",
    "bench/infer_throughput_img_s",
    "bench/onnx_file_size_mb",
    "bench/trt_infer_latency_ms",
    "bench/trt_infer_throughput_img_s",
    "bench/trt_engine_size_mb",
    "bench/train_total_sec",
    "bench/eval_total_sec",
]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))
