# yolo-bench
fine tuning benchmark

## Commands

- `just fetch`: download `coco128` into `dataset/coco128`
- `just label`: launch the Qt target labeler
- `just augment`: generate a synthetic target-only dataset in `dataset/augment`
- `just yolo7-setup`: prepare the YOLOv7 run and remapped dataset
- `just yolo7-train`: train YOLOv7 for the shared benchmark
- `just yolo7-eval`: evaluate the latest YOLOv7 checkpoint
- `just yolo7-infer`: export YOLOv7 to ONNX and run Ultralytics inference
- `just yolo7-bench`: benchmark the exported YOLOv7 runtime
- `just yolo26-setup`: prepare the YOLO26 run and remapped dataset
- `just yolo26-train`: train YOLO26 for the shared benchmark
- `just yolo26-eval`: evaluate the latest YOLO26 checkpoint
- `just yolo26-infer`: export YOLO26 to ONNX and run Ultralytics inference
- `just yolo26-bench`: benchmark the exported YOLO26 runtime
- `just yolo11-setup`: prepare the YOLO11 run and remapped dataset
- `just yolo11-train`: train YOLO11 for the shared benchmark
- `just yolo11-eval`: evaluate the latest YOLO11 checkpoint
- `just yolo11-infer`: export YOLO11 to ONNX and run Ultralytics inference
- `just yolo11-bench`: benchmark the exported YOLO11 runtime

## Target Labeler

The labeler reads source images from the top-level `targets/` directory.

For each image, you can:

- type a class name
- draw one bounding box
- click `Finish` to export the sample

Exports are written to:

- `dataset/class.txt`
- `dataset/targets/images/<class_name>.jpg`
- `dataset/targets/labels/<class_name>.txt`

`dataset/class.txt` stores one mapping per line as `<id> <class_name>`, with the first custom class starting at `80`.

## Dataset Augment

The augmenter reads:

- `dataset/coco128`
- `dataset/targets`
- `dataset/class.txt`

It creates a simple synthetic dataset at `dataset/augment` with:

- `images/train2017`
- `images/val2017`
- `labels/train2017`
- `labels/val2017`
- `augment.yaml`

Behavior:

- 80/20 train/val split with no overlap
- 80% probability to paste one target into each image
- original COCO labels are discarded
- images without a pasted target get an empty label file
- randomness is reproducible with `--seed`

## Fine-Tune Benchmark

The implemented model packages are `src/finetune-yolo7`, `src/finetune-yolo26`, and `src/finetune-yolo11`.

They use:

- `dataset/augment` as the shared benchmark dataset
- `artifacts/yolo7`, `artifacts/yolo26`, and `artifacts/yolo11` for generated run outputs
- W&B project `yolo-bench`
- 50 epochs with eval every 5 epochs
- the same canonical metrics and benchmark subset

The YOLOv7 pipeline is split into:

- `setup.py`: validate inputs and build a YOLOv7-compatible remapped dataset
- `train.py`: train for 50 epochs with eval every 5 epochs
- `eval.py`: evaluate a checkpoint with the shared metric contract
- `infer.py`: export ONNX and run Ultralytics inference
- `bench.py`: record final train/eval/inference benchmark metrics

The YOLO26 pipeline follows the same file layout and metric contract:

- `setup.py`: validate inputs and build a YOLO26-compatible remapped dataset
- `train.py`: fine-tune `yolo26n.pt` on the same remapped dataset
- `eval.py`: run periodic or standalone validation with the same logged metrics
- `infer.py`: export ONNX and run inference on the shared fixed subset
- `bench.py`: record the same final ONNX benchmark metrics

The YOLO11 pipeline follows the same file layout and metric contract:

- `setup.py`: validate inputs and build a YOLO11-compatible remapped dataset
- `train.py`: fine-tune `yolo11n.pt` on the same remapped dataset
- `eval.py`: run periodic or standalone validation with the same logged metrics
- `infer.py`: export ONNX and run inference on the shared fixed subset
- `bench.py`: record the same final ONNX benchmark metrics
