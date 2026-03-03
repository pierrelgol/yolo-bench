#!/usr/bin/env python3

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

from PyQt6.QtCore import QPointF, QRect, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QMouseEvent, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_FILE_HEADER_ID = 80


def load_target_images(repo_root: Path) -> list[Path]:
    targets_dir = repo_root / "targets"
    if not targets_dir.is_dir():
        return []
    return sorted(
        path
        for path in targets_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def sanitize_class_name(raw: str) -> str:
    cleaned = raw.strip().lower()
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"[^a-z0-9_-]", "", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_-")


def load_class_map(class_file: Path) -> dict[str, int]:
    if not class_file.exists():
        return {}

    mapping: dict[str, int] = {}
    for line_number, line in enumerate(class_file.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            raw_id, class_name = stripped.split(maxsplit=1)
            class_id = int(raw_id)
        except ValueError as exc:
            raise ValueError(f"Malformed class mapping at line {line_number}: {line!r}") from exc
        if class_name in mapping and mapping[class_name] != class_id:
            raise ValueError(f"Duplicate class name at line {line_number}: {class_name}")
        mapping[class_name] = class_id
    return mapping


def resolve_class_id(class_file: Path, class_name: str) -> int:
    class_file.parent.mkdir(parents=True, exist_ok=True)
    mapping = load_class_map(class_file)
    existing = mapping.get(class_name)
    if existing is not None:
        return existing

    next_id = max(mapping.values(), default=CLASS_FILE_HEADER_ID - 1) + 1
    with class_file.open("a", encoding="utf-8") as handle:
        handle.write(f"{next_id} {class_name}\n")
    return next_id


def bbox_to_yolo(image_size: QSize, bbox_px: QRectF) -> tuple[float, float, float, float]:
    width = float(image_size.width())
    height = float(image_size.height())
    x_center = (bbox_px.left() + bbox_px.width() / 2.0) / width
    y_center = (bbox_px.top() + bbox_px.height() / 2.0) / height
    box_width = bbox_px.width() / width
    box_height = bbox_px.height() / height
    return x_center, y_center, box_width, box_height


def export_sample(
    repo_root: Path,
    image: QImage,
    image_path: Path,
    class_name: str,
    class_id: int,
    bbox_px: QRectF,
) -> None:
    dataset_dir = repo_root / "dataset"
    images_dir = dataset_dir / "targets" / "images"
    labels_dir = dataset_dir / "targets" / "labels"
    class_file = dataset_dir / "class.txt"

    dataset_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    class_file.touch(exist_ok=True)

    image_output = images_dir / f"{class_name}.jpg"
    label_output = labels_dir / f"{class_name}.txt"

    if image_path.suffix.lower() in {".jpg", ".jpeg"}:
        shutil.copyfile(image_path, image_output)
    else:
        if not image.save(str(image_output), "JPEG"):
            raise ValueError(f"Failed to write image: {image_output}")

    x_center, y_center, box_width, box_height = bbox_to_yolo(image.size(), bbox_px)
    label_output.write_text(
        f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n",
        encoding="utf-8",
    )


class ImageCanvas(QWidget):
    bbox_changed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self._image = QImage()
        self._pixmap = QPixmap()
        self._draw_rect: QRectF | None = None
        self._drag_start: QPointF | None = None
        self._image_rect = QRect()
        self.setMinimumSize(640, 480)
        self.setMouseTracking(True)

    def set_image(self, image_path: Path) -> None:
        image = QImage(str(image_path))
        if image.isNull():
            raise ValueError(f"Failed to load image: {image_path}")
        self._image = image
        self._pixmap = QPixmap.fromImage(image)
        self._draw_rect = None
        self._drag_start = None
        self.update()
        self.bbox_changed.emit()

    def clear_bbox(self) -> None:
        self._draw_rect = None
        self._drag_start = None
        self.update()
        self.bbox_changed.emit()

    def has_valid_bbox(self) -> bool:
        return self._draw_rect is not None and self._draw_rect.width() > 0 and self._draw_rect.height() > 0

    def current_bbox(self) -> QRectF | None:
        return QRectF(self._draw_rect) if self.has_valid_bbox() else None

    def current_image(self) -> QImage:
        return QImage(self._image)

    def has_image(self) -> bool:
        return not self._image.isNull()

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)

        if self._pixmap.isNull():
            return

        scaled = self._pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        x_offset = (self.width() - scaled.width()) // 2
        y_offset = (self.height() - scaled.height()) // 2
        self._image_rect = QRect(x_offset, y_offset, scaled.width(), scaled.height())
        painter.drawPixmap(self._image_rect, scaled)

        if not self.has_valid_bbox():
            return

        rect = self._image_to_widget_rect(self._draw_rect)
        pen = QPen(Qt.GlobalColor.red, 2)
        painter.setPen(pen)
        painter.drawRect(rect)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton or not self._point_in_image(event.position()):
            return
        self._drag_start = self._widget_to_image_point(event.position())
        self._draw_rect = QRectF(self._drag_start, self._drag_start).normalized()
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag_start is None or self._image.isNull():
            return
        current = self._widget_to_image_point(event.position())
        self._draw_rect = QRectF(self._drag_start, current).normalized()
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton or self._drag_start is None:
            return
        current = self._widget_to_image_point(event.position())
        rect = QRectF(self._drag_start, current).normalized()
        self._drag_start = None
        if rect.width() < 1 or rect.height() < 1:
            self._draw_rect = None
        else:
            self._draw_rect = rect
        self.update()
        self.bbox_changed.emit()

    def _point_in_image(self, point) -> bool:
        return self._image_rect.contains(int(point.x()), int(point.y()))

    def _widget_to_image_point(self, point) -> QPointF:
        if self._image_rect.width() == 0 or self._image_rect.height() == 0:
            return QPointF(0.0, 0.0)
        max_x = self._image_rect.left() + self._image_rect.width() - 1
        max_y = self._image_rect.top() + self._image_rect.height() - 1
        x = min(max(point.x(), self._image_rect.left()), max_x)
        y = min(max(point.y(), self._image_rect.top()), max_y)
        x_denominator = max(1, self._image_rect.width() - 1)
        y_denominator = max(1, self._image_rect.height() - 1)
        x_ratio = (x - self._image_rect.left()) / x_denominator
        y_ratio = (y - self._image_rect.top()) / y_denominator
        image_x = x_ratio * (self._image.width() - 1)
        image_y = y_ratio * (self._image.height() - 1)
        return QPointF(image_x, image_y)

    def _image_to_widget_rect(self, rect: QRectF) -> QRectF:
        x_scale = self._image_rect.width() / self._image.width()
        y_scale = self._image_rect.height() / self._image.height()
        left = self._image_rect.left() + rect.left() * x_scale
        top = self._image_rect.top() + rect.top() * y_scale
        width = rect.width() * x_scale
        height = rect.height() * y_scale
        return QRectF(left, top, width, height)


class LabelerWindow(QMainWindow):
    def __init__(self, repo_root: Path, image_paths: list[Path]) -> None:
        super().__init__()
        self.repo_root = repo_root
        self.image_paths = image_paths
        self.index = 0
        self._dirty = False

        self.setWindowTitle("Targets Labeler")
        self.resize(1000, 800)

        self.canvas = ImageCanvas()
        self.class_input = QLineEdit()
        self.class_input.setPlaceholderText("Class name")
        self.status_label = QLabel()
        self.prev_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.clear_button = QPushButton("Clear Box")
        self.finish_button = QPushButton("Finish")

        controls = QHBoxLayout()
        controls.addWidget(self.prev_button)
        controls.addWidget(self.next_button)
        controls.addWidget(self.class_input, 1)
        controls.addWidget(self.clear_button)
        controls.addWidget(self.finish_button)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas, 1)
        layout.addLayout(controls)
        layout.addWidget(self.status_label)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.prev_button.clicked.connect(lambda: self._navigate(-1))
        self.next_button.clicked.connect(lambda: self._navigate(1))
        self.clear_button.clicked.connect(self._clear_bbox)
        self.finish_button.clicked.connect(self._finish_current)
        self.class_input.textChanged.connect(self._mark_dirty)
        self.class_input.textChanged.connect(self._update_finish_state)
        self.canvas.bbox_changed.connect(self._mark_dirty)
        self.canvas.bbox_changed.connect(self._update_finish_state)

        self._load_current_image(reset_form=True)

    def _load_current_image(self, reset_form: bool) -> None:
        self.canvas.set_image(self.image_paths[self.index])
        if reset_form:
            self.class_input.clear()
            self._dirty = False
        self._update_navigation_state()
        self._update_finish_state()
        self._update_status()

    def _update_status(self) -> None:
        current = self.image_paths[self.index]
        self.status_label.setText(f"{self.index + 1}/{len(self.image_paths)}  {current.name}")

    def _update_navigation_state(self) -> None:
        self.prev_button.setEnabled(self.index > 0)
        self.next_button.setEnabled(self.index < len(self.image_paths) - 1)

    def _update_finish_state(self) -> None:
        valid_name = bool(sanitize_class_name(self.class_input.text()))
        self.finish_button.setEnabled(valid_name and self.canvas.has_valid_bbox())

    def _mark_dirty(self) -> None:
        self._dirty = True

    def _clear_bbox(self) -> None:
        self.canvas.clear_bbox()
        self._dirty = True

    def _navigate(self, direction: int) -> None:
        next_index = self.index + direction
        if next_index < 0 or next_index >= len(self.image_paths):
            return
        if self._dirty and not self._confirm_discard_changes():
            return
        self.index = next_index
        self._load_current_image(reset_form=True)

    def _confirm_discard_changes(self) -> bool:
        reply = QMessageBox.question(
            self,
            "Discard changes",
            "Current image has unsaved changes. Discard them?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return reply == QMessageBox.StandardButton.Yes

    def _finish_current(self) -> None:
        class_name = sanitize_class_name(self.class_input.text())
        bbox = self.canvas.current_bbox()

        if not class_name:
            QMessageBox.warning(self, "Invalid class name", "Enter a valid class name before finishing.")
            return
        if bbox is None:
            QMessageBox.warning(self, "Missing bounding box", "Draw one bounding box before finishing.")
            return

        class_file = self.repo_root / "dataset" / "class.txt"
        try:
            class_id = resolve_class_id(class_file, class_name)
            export_sample(
                repo_root=self.repo_root,
                image=self.canvas.current_image(),
                image_path=self.image_paths[self.index],
                class_name=class_name,
                class_id=class_id,
                bbox_px=bbox,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return

        self._dirty = False
        QMessageBox.information(
            self,
            "Export complete",
            f"Saved {class_name}.jpg and {class_name}.txt with class id {class_id}.",
        )


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    image_paths = load_target_images(repo_root)

    app = QApplication(sys.argv)
    if not image_paths:
        QMessageBox.critical(
            None,
            "No target images",
            f"No supported images were found in {(repo_root / 'targets')}.",
        )
        return 1

    window = LabelerWindow(repo_root=repo_root, image_paths=image_paths)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
