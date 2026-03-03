# yolo-bench
fine tuning benchmark

## Commands

- `just fetch`: download `coco128` into `dataset/coco128`
- `just label`: launch the Qt target labeler

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
