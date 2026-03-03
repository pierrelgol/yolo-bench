# yolo-bench
fine tuning benchmark

## Commands

- `just fetch`: download `coco128` into `dataset/coco128`
- `just label`: launch the Qt target labeler
- `just augment`: generate a synthetic target-only dataset in `dataset/augment`

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
