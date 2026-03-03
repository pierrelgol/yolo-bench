from __future__ import annotations

WANDB_PROJECT = "yolo-bench"
DATASET_ROOT = "dataset/augment"
DATASET_YAML = "dataset/augment/augment.yaml"
ARTIFACT_ROOT = "artifacts"

EPOCHS = 50
EVAL_EVERY = 5
IMG_SIZE = 640
BATCH_SIZE = 16
NUM_WORKERS = 8
BENCHMARK_SUBSET_SIZE = 16
SEED = 42
CUDA_DEVICE = "0"

YOLO7_MODEL_NAME = "yolo7"
YOLO7_VARIANT = "yolov7-tiny"
YOLO7_WEIGHTS = "3rdparty/yolov7/yolov7-tiny.pt"
YOLO7_CFG = "3rdparty/yolov7/cfg/training/yolov7-tiny.yaml"
YOLO7_HYP = "3rdparty/yolov7/data/hyp.scratch.tiny.yaml"

YOLO26_MODEL_NAME = "yolo26"
YOLO26_VARIANT = "yolo26n"
YOLO26_WEIGHTS = "yolo26n.pt"

YOLO11_MODEL_NAME = "yolo11"
YOLO11_VARIANT = "yolo11n"
YOLO11_WEIGHTS = "yolo11n.pt"
