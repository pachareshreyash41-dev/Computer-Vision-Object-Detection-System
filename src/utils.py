"""Shared helpers: configuration, file handling, detection formatting and drawing."""

from __future__ import annotations

import logging
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import cv2
import numpy as np
import pandas as pd
from dotenv import load_dotenv

if TYPE_CHECKING:
    from src.detector import Detection

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Config:
    model_path: str
    confidence: float
    iou: float
    image_size: int
    device: str
    input_dir: Path
    output_dir: Path


def load_config() -> Config:
    """Load settings from environment variables and .env."""
    load_dotenv()

    return Config(
        model_path=os.getenv("MODEL_PATH", "models/yolov8n.pt"),
        confidence=float(os.getenv("CONFIDENCE_THRESHOLD", 0.25)),
        iou=float(os.getenv("IOU_THRESHOLD", 0.45)),
        image_size=int(os.getenv("IMAGE_SIZE", 640)),
        device=os.getenv("DEVICE", "auto"),
        input_dir=Path(os.getenv("INPUT_DIR", "data/input")),
        output_dir=Path(os.getenv("OUTPUT_DIR", "data/output")),
    )


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# --------------------------------------------------------------------------- #
# Directories and files
# --------------------------------------------------------------------------- #

def ensure_dir(path: str | Path) -> Path:
    """Create one directory and return its path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_dirs(*paths: str | Path) -> None:
    """Create all supplied directories."""
    for path in paths:
        ensure_dir(path)


def is_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def is_video(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def timestamped_name(stem: str, suffix: str) -> str:
    """Create a filename with a timestamp."""
    if suffix and not suffix.startswith("."):
        suffix = f".{suffix}"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{stem}_{timestamp}{suffix}"


def list_files(directory: str | Path, extensions: set[str]) -> list[Path]:
    directory = Path(directory)

    if not directory.is_dir():
        return []

    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )


def save_uploaded_file(uploaded_file, destination_dir: str | Path) -> Path:
    """Save a Streamlit UploadedFile safely using its basename."""
    destination = ensure_dir(destination_dir)

    original_name = Path(uploaded_file.name).name
    if not original_name or original_name in {".", ".."}:
        raise ValueError("Invalid uploaded filename.")

    suffix = Path(original_name).suffix.lower()
    if suffix not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {suffix or '(none)'}")

    output_path = destination / timestamped_name(
        Path(original_name).stem,
        suffix,
    )
    output_path.write_bytes(uploaded_file.getvalue())

    return output_path


# --------------------------------------------------------------------------- #
# Detection formatting
# --------------------------------------------------------------------------- #

def detections_to_dataframe(detections: Iterable["Detection"]) -> pd.DataFrame:
    """Convert detections into a table for the Streamlit dashboard."""
    rows = []

    for detection in detections:
        x1, y1, x2, y2 = (float(value) for value in detection.bbox)

        rows.append(
            {
                "class_id": int(detection.class_id),
                "class_name": str(detection.label),
                "confidence": round(float(detection.confidence), 4),
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "class_id",
            "class_name",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
        ],
    )


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #

def get_color(class_id: int) -> tuple[int, int, int]:
    """Return a deterministic BGR color for a class ID."""
    rng = np.random.default_rng(class_id * 9973 + 7)
    b, g, r = (int(value) for value in rng.integers(60, 256, size=3))
    return b, g, r


def draw_detections(
    image: np.ndarray,
    detections: Iterable["Detection"],
    thickness: int = 2,
    show_confidence: bool = True,
) -> np.ndarray:
    """Return a copy of a BGR image with bounding boxes and labels."""
    annotated = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, min(annotated.shape[:2]) / 1000)

    for detection in detections:
        x1, y1, x2, y2 = (int(value) for value in detection.bbox)
        color = get_color(detection.class_id)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        label = (
            f"{detection.label} {detection.confidence:.2f}"
            if show_confidence
            else detection.label
        )

        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, 1
        )
        top = max(y1 - text_height - baseline - 4, 0)

        cv2.rectangle(
            annotated,
            (x1, top),
            (x1 + text_width + 6, top + text_height + baseline + 4),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 3, top + text_height + 1),
            font,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated


def summarize_detections(
    detections: Iterable["Detection"],
) -> dict[str, int]:
    """Count detections by class label, most common first."""
    return dict(Counter(item.label for item in detections).most_common())