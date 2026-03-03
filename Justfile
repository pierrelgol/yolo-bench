# Show available recipes.
default: help

# Print recipe help.
help:
    @just --list

# Create the local virtual environment once.
venv:
    @if [ -d .venv ]; then \
        echo ".venv already exists"; \
    else \
        uv venv .venv; \
    fi

# Fetch the coco128 dataset into the top-level dataset directory.
fetch:
    uv run --with ultralytics python3 src/dataset-fetcher/fetch_coco128.py

# Launch the Qt target labeler.
label:
    uv run python3 src/targets-labels/label_targets.py

# Build the synthetic target-only augmented dataset.
augment:
    uv run python3 src/dataset-augment/augment_with_targets.py

# Prepare the YOLOv7 benchmark run context and remapped dataset.
yolo7-setup:
    uv run python3 src/finetune-yolo7/setup.py

# Train YOLOv7 on the shared benchmark dataset.
yolo7-train:
    uv run python3 src/finetune-yolo7/train.py

# Evaluate the latest YOLOv7 checkpoint.
yolo7-eval:
    uv run python3 src/finetune-yolo7/eval.py

# Export YOLOv7 to ONNX and TensorRT, then run Ultralytics inference.
yolo7-infer:
    uv run python3 src/finetune-yolo7/infer.py

# Benchmark the exported YOLOv7 TensorRT runtime.
yolo7-bench:
    uv run python3 src/finetune-yolo7/bench.py

# Prepare the YOLO26 benchmark run context and remapped dataset.
yolo26-setup:
    uv run python3 src/finetune-yolo26/setup.py

# Train YOLO26 on the shared benchmark dataset.
yolo26-train:
    uv run python3 src/finetune-yolo26/train.py

# Evaluate the latest YOLO26 checkpoint.
yolo26-eval:
    uv run python3 src/finetune-yolo26/eval.py

# Export YOLO26 to ONNX and TensorRT, then run Ultralytics inference.
yolo26-infer:
    uv run python3 src/finetune-yolo26/infer.py

# Benchmark the exported YOLO26 TensorRT runtime.
yolo26-bench:
    uv run python3 src/finetune-yolo26/bench.py

# Prepare the YOLO11 benchmark run context and remapped dataset.
yolo11-setup:
    uv run python3 src/finetune-yolo11/setup.py

# Train YOLO11 on the shared benchmark dataset.
yolo11-train:
    uv run python3 src/finetune-yolo11/train.py

# Evaluate the latest YOLO11 checkpoint.
yolo11-eval:
    uv run python3 src/finetune-yolo11/eval.py

# Export YOLO11 to ONNX and TensorRT, then run Ultralytics inference.
yolo11-infer:
    uv run python3 src/finetune-yolo11/infer.py

# Benchmark the exported YOLO11 TensorRT runtime.
yolo11-bench:
    uv run python3 src/finetune-yolo11/bench.py

# Prepare the RT-DETR benchmark run context and remapped dataset.
rtdetr-setup:
    uv run python3 src/finetune-rtdetr/setup.py

# Train RT-DETR on the shared benchmark dataset.
rtdetr-train:
    uv run python3 src/finetune-rtdetr/train.py

# Evaluate the latest RT-DETR checkpoint.
rtdetr-eval:
    uv run python3 src/finetune-rtdetr/eval.py

# Export RT-DETR to ONNX and TensorRT, then run Ultralytics inference.
rtdetr-infer:
    uv run python3 src/finetune-rtdetr/infer.py

# Benchmark the exported RT-DETR TensorRT runtime.
rtdetr-bench:
    uv run python3 src/finetune-rtdetr/bench.py

# Compare the latest runs across models.
compare:
    uv run python3 src/benchmark_common/compare_models.py

# Remove Python caches and generated artifacts.
clean:
    find . -type d -name __pycache__ -prune -exec rm -rf {} +
    rm -rf .mypy_cache .pytest_cache .ruff_cache .cache artifacts wandb

# Remove all generated artifacts, environments, datasets, and downloaded weights.
fclean: clean
    rm -rf .venv dataset/augment dataset/coco128 dataset/targets dataset/class.txt
    rm -f yolo26n.pt yolo11n.pt rtdetr-l.pt
