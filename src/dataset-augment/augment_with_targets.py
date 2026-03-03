#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class TargetSample:
    class_id: int
    class_name: str
    crop_image: Image.Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a simple synthetic target-only dataset from coco128 and labeled targets."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible generation.")
    parser.add_argument(
        "--paste-probability",
        type=float,
        default=0.8,
        help="Probability of pasting one target into a base coco128 image.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of images placed in the train split.",
    )
    parser.add_argument(
        "--min-scale",
        type=float,
        default=0.10,
        help="Minimum target scale as a fraction of the base image minimum dimension.",
    )
    parser.add_argument(
        "--max-scale",
        type=float,
        default=0.30,
        help="Maximum target scale as a fraction of the base image minimum dimension.",
    )
    return parser.parse_args()


def load_class_map(class_file: Path) -> dict[int, str]:
    if not class_file.exists():
        raise FileNotFoundError(f"Missing class map: {class_file}")

    mapping: dict[int, str] = {}
    for line_number, line in enumerate(class_file.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            raw_id, class_name = stripped.split(maxsplit=1)
        except ValueError as exc:
            raise ValueError(f"Malformed class mapping at line {line_number}: {line!r}") from exc
        class_id = int(raw_id)
        if class_id in mapping and mapping[class_id] != class_name:
            raise ValueError(f"Duplicate class id at line {line_number}: {class_id}")
        mapping[class_id] = class_name

    if not mapping:
        raise ValueError(f"Class map is empty: {class_file}")
    return mapping


def parse_yolo_label(label_path: Path) -> tuple[int, float, float, float, float]:
    lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"Expected exactly one label line in {label_path}, found {len(lines)}")

    parts = lines[0].split()
    if len(parts) != 5:
        raise ValueError(f"Expected 5 fields in {label_path}, found {len(parts)}")

    class_id = int(parts[0])
    x_center, y_center, width, height = (float(value) for value in parts[1:])
    return class_id, x_center, y_center, width, height


def yolo_to_pixel_box(image_size: tuple[int, int], label: tuple[int, float, float, float, float]) -> tuple[int, int, int, int]:
    _, x_center, y_center, width, height = label
    image_width, image_height = image_size
    box_width = width * image_width
    box_height = height * image_height
    left = round((x_center * image_width) - box_width / 2.0)
    top = round((y_center * image_height) - box_height / 2.0)
    right = round(left + box_width)
    bottom = round(top + box_height)
    left = max(0, min(left, image_width - 1))
    top = max(0, min(top, image_height - 1))
    right = max(left + 1, min(right, image_width))
    bottom = max(top + 1, min(bottom, image_height))
    return left, top, right, bottom


def load_target_samples(dataset_dir: Path, class_map: dict[int, str]) -> list[TargetSample]:
    images_dir = dataset_dir / "targets" / "images"
    labels_dir = dataset_dir / "targets" / "labels"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing targets images directory: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing targets labels directory: {labels_dir}")

    samples: list[TargetSample] = []
    for image_path in sorted(path for path in images_dir.iterdir() if path.suffix.lower() in SUPPORTED_SUFFIXES):
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for target image {image_path.name}: {label_path}")

        label = parse_yolo_label(label_path)
        class_id = label[0]
        class_name = class_map.get(class_id)
        if class_name is None:
            raise ValueError(f"Unknown target class id {class_id} in {label_path}")

        with Image.open(image_path) as target_image:
            source = target_image.convert("RGB")
            crop_box = yolo_to_pixel_box(source.size, label)
            crop_image = source.crop(crop_box).copy()
        if crop_image.width <= 0 or crop_image.height <= 0:
            raise ValueError(f"Invalid crop size for {image_path}")

        samples.append(TargetSample(class_id=class_id, class_name=class_name, crop_image=crop_image))

    if not samples:
        raise ValueError(f"No target samples found in {images_dir}")
    return samples


def list_coco_images(dataset_dir: Path) -> list[Path]:
    images_dir = dataset_dir / "coco128" / "images" / "train2017"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing coco128 images directory: {images_dir}")

    image_paths = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in SUPPORTED_SUFFIXES)
    if not image_paths:
        raise ValueError(f"No coco128 images found in {images_dir}")
    return image_paths


def build_split(image_paths: list[Path], train_ratio: float, rng: random.Random) -> tuple[list[Path], list[Path]]:
    shuffled = list(image_paths)
    rng.shuffle(shuffled)
    train_count = int(len(shuffled) * train_ratio)
    train_count = max(1, min(train_count, len(shuffled) - 1))
    return shuffled[:train_count], shuffled[train_count:]


def scale_patch_to_base(
    patch: Image.Image,
    base_size: tuple[int, int],
    min_scale: float,
    max_scale: float,
    rng: random.Random,
) -> Image.Image:
    base_width, base_height = base_size
    base_min_dim = min(base_width, base_height)
    target_fraction = rng.uniform(min_scale, max_scale)
    target_max_side = max(1, int(round(target_fraction * base_min_dim)))
    patch_max_side = max(patch.width, patch.height)
    scale = target_max_side / patch_max_side

    scaled_width = max(1, int(round(patch.width * scale)))
    scaled_height = max(1, int(round(patch.height * scale)))
    scaled_width = min(scaled_width, base_width)
    scaled_height = min(scaled_height, base_height)
    return patch.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)


def paste_patch(
    base_image: Image.Image,
    patch: Image.Image,
    rng: random.Random,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    base = base_image.copy()
    max_left = max(0, base.width - patch.width)
    max_top = max(0, base.height - patch.height)
    left = rng.randint(0, max_left) if max_left > 0 else 0
    top = rng.randint(0, max_top) if max_top > 0 else 0
    base.paste(patch, (left, top))
    return base, (left, top, patch.width, patch.height)


def pixel_box_to_yolo(image_size: tuple[int, int], bbox_px: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    image_width, image_height = image_size
    left, top, width, height = bbox_px
    x_center = (left + width / 2.0) / image_width
    y_center = (top + height / 2.0) / image_height
    norm_width = width / image_width
    norm_height = height / image_height
    return x_center, y_center, norm_width, norm_height


def prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "images" / "train2017").mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / "val2017").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "train2017").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "val2017").mkdir(parents=True, exist_ok=True)


def write_output_image_and_label(
    output_dir: Path,
    split: str,
    image_name: str,
    image: Image.Image,
    label_line: str,
) -> None:
    image_output = output_dir / "images" / split / image_name
    label_output = output_dir / "labels" / split / f"{Path(image_name).stem}.txt"
    image.save(image_output, "JPEG")
    label_output.write_text(label_line, encoding="utf-8")


def write_yaml(output_dir: Path, class_map: dict[int, str]) -> None:
    lines = [
        "path: dataset/augment",
        "train: images/train2017",
        "val: images/val2017",
        "names:",
    ]
    for class_id, class_name in sorted(class_map.items()):
        lines.append(f"  {class_id}: {class_name}")
    (output_dir / "augment.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_split(
    image_paths: list[Path],
    split: str,
    output_dir: Path,
    target_samples: list[TargetSample],
    rng: random.Random,
    paste_probability: float,
    min_scale: float,
    max_scale: float,
) -> None:
    for image_path in image_paths:
        with Image.open(image_path) as source_image:
            base_image = source_image.convert("RGB")

        if rng.random() < paste_probability:
            sample = rng.choice(target_samples)
            scaled_patch = scale_patch_to_base(sample.crop_image, base_image.size, min_scale, max_scale, rng)
            augmented_image, bbox_px = paste_patch(base_image, scaled_patch, rng)
            x_center, y_center, box_width, box_height = pixel_box_to_yolo(augmented_image.size, bbox_px)
            label_line = (
                f"{sample.class_id} "
                f"{x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n"
            )
            write_output_image_and_label(output_dir, split, image_path.name, augmented_image, label_line)
        else:
            write_output_image_and_label(output_dir, split, image_path.name, base_image, "")


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.paste_probability <= 1.0:
        raise ValueError("--paste-probability must be between 0 and 1")
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1")
    if not 0.0 < args.min_scale <= args.max_scale:
        raise ValueError("--min-scale must be positive and less than or equal to --max-scale")

    repo_root = Path(__file__).resolve().parents[2]
    dataset_dir = repo_root / "dataset"
    output_dir = dataset_dir / "augment"

    class_map = load_class_map(dataset_dir / "class.txt")
    target_samples = load_target_samples(dataset_dir, class_map)
    coco_image_paths = list_coco_images(dataset_dir)

    rng = random.Random(args.seed)
    train_images, val_images = build_split(coco_image_paths, args.train_ratio, rng)

    prepare_output_dir(output_dir)
    generate_split(
        image_paths=train_images,
        split="train2017",
        output_dir=output_dir,
        target_samples=target_samples,
        rng=rng,
        paste_probability=args.paste_probability,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
    )
    generate_split(
        image_paths=val_images,
        split="val2017",
        output_dir=output_dir,
        target_samples=target_samples,
        rng=rng,
        paste_probability=args.paste_probability,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
    )
    write_yaml(output_dir, class_map)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
