#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmark_common import config
from benchmark_common.metrics import write_json
from benchmark_common.paths import artifacts_root, latest_run_file


MODELS = [
    config.YOLO7_MODEL_NAME,
    config.YOLO26_MODEL_NAME,
    config.YOLO11_MODEL_NAME,
    config.RTDETR_MODEL_NAME,
]


def extract_row(model_name: str) -> dict | None:
    path = latest_run_file(model_name)
    if not path.exists():
        return None
    import json

    context = json.loads(path.read_text(encoding="utf-8"))
    training = context.get("training", {})
    last_eval = training.get("last_eval", {})
    benchmark = context.get("benchmark_results", {})
    return {
        "model": model_name,
        "run_name": context.get("run_name", ""),
        "val/precision": last_eval.get("val/precision"),
        "val/recall": last_eval.get("val/recall"),
        "val/map50": last_eval.get("val/map50"),
        "val/map50_95": last_eval.get("val/map50_95"),
        "bench/trt_infer_latency_ms": benchmark.get("bench/trt_infer_latency_ms"),
        "bench/trt_infer_throughput_img_s": benchmark.get("bench/trt_infer_throughput_img_s"),
        "bench/trt_engine_size_mb": benchmark.get("bench/trt_engine_size_mb"),
        "bench/train_total_sec": benchmark.get("bench/train_total_sec", training.get("train_total_sec")),
        "bench/eval_total_sec": benchmark.get("bench/eval_total_sec", training.get("eval_total_sec")),
    }


def render_markdown(rows: list[dict]) -> str:
    headers = [
        "model",
        "val/map50",
        "val/map50_95",
        "val/precision",
        "val/recall",
        "bench/trt_infer_latency_ms",
        "bench/trt_infer_throughput_img_s",
        "bench/trt_engine_size_mb",
        "bench/train_total_sec",
        "bench/eval_total_sec",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header)
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            elif value is None:
                values.append("n/a")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    rows = [row for row in (extract_row(model) for model in MODELS) if row is not None]
    comparison_dir = artifacts_root() / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    payload = {"backend": config.BENCHMARK_BACKEND, "rows": rows}
    write_json(comparison_dir / "latest.json", payload)
    markdown = render_markdown(rows)
    (comparison_dir / "latest.md").write_text(markdown, encoding="utf-8")
    print(comparison_dir / "latest.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
