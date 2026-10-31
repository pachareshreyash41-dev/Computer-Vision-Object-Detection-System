"""
Core object detection module using Ultralytics YOLO.

Features:
- Loads YOLO models from a local path or model name.
- Supports CPU and CUDA device selection.
- Detects objects in OpenCV images.
- Returns structured Detection objects.
- Supports custom confidence and IoU thresholds.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    """Represents one detected object."""

    class_id: int
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, Any]:
        """Convert detection into a dictionary."""
        return asdict(self)

    @property
    def area(self) -> float:
        """Calculate the bounding-box area in pixels."""
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


class ObjectDetector:
    """Load a YOLO model and use it to detect objects in images."""

    def __init__(
        self,
        model_path: str = "models/yolov8n.pt",
        confidence: float = 0.25,
        iou: float = 0.45,
        image_size: int = 640,
        device: str | None = "auto",
    ) -> None:
        """
        Initialize the detector.

        Args:
            model_path: Local model path or a YOLO model filename.
            confidence: Minimum detection confidence (0.0 to 1.0).
            iou: IoU threshold for non-maximum suppression (0.0 to 1.0).
            image_size: Image size used for inference.
            device: "auto", "cpu", "0" for GPU 0, or another supported device.
        """
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0.")

        if not 0.0 <= iou <= 1.0:
            raise ValueError("iou must be between 0.0 and 1.0.")

        if image_size <= 0:
            raise ValueError("image_size must be greater than zero.")

        self.model_path = str(model_path)
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.image_size = int(image_size)

        requested_device = (device or "auto").strip().lower()
        self.device = None if requested_device in ("", "auto") else requested_device

        self.model = self._load_model(self.model_path)
        self.class_names = self._get_class_names(self.model.names)

        logger.info(
            "YOLO detector initialized: model=%s, device=%s",
            self.model_path,
            self.device or "auto",
        )

    @staticmethod
    def _load_model(model_path: str):
        """Load the YOLO model from a local file or supported model name."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Install it using: "
                "python -m pip install ultralytics"
            ) from exc

        path = Path(model_path)

        # Use the supplied local path when it exists.
        # Otherwise, let Ultralytics resolve/download the model filename.
        resolved_path = str(path) if path.is_file() else path.name

        logger.info("Loading YOLO model: %s", resolved_path)

        try:
            return YOLO(resolved_path)
        except Exception as exc:
            logger.exception("Failed to load YOLO model: %s", resolved_path)
            raise RuntimeError(
                f"Could not load YOLO model '{resolved_path}'. "
                "Check the model filename, internet connection, and dependencies."
            ) from exc

    @staticmethod
    def _get_class_names(names: Any) -> dict[int, str]:
        """Normalize model class names into an integer-to-label dictionary."""
        if isinstance(names, dict):
            return {int(class_id): str(label) for class_id, label in names.items()}

        if isinstance(names, (list, tuple)):
            return {class_id: str(label) for class_id, label in enumerate(names)}

        return {}

    def detect(
        self,
        frame: np.ndarray,
        confidence: float | None = None,
        iou: float | None = None,
    ) -> list[Detection]:
        """
        Detect objects in an OpenCV image.

        Args:
            frame: Image as a NumPy array, normally in BGR format.
            confidence: Optional confidence threshold for this prediction.
            iou: Optional IoU threshold for this prediction.

        Returns:
            A list of Detection objects.
        """
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a NumPy ndarray.")

        if frame.size == 0:
            raise ValueError("frame is empty.")

        if frame.ndim not in (2, 3):
            raise ValueError(
                "frame must be a grayscale image or a color image."
            )

        selected_confidence = (
            self.confidence if confidence is None else float(confidence)
        )
        selected_iou = self.iou if iou is None else float(iou)

        if not 0.0 <= selected_confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0.")

        if not 0.0 <= selected_iou <= 1.0:
            raise ValueError("iou must be between 0.0 and 1.0.")

        try:
            results = self.model.predict(
                source=frame,
                conf=selected_confidence,
                iou=selected_iou,
                imgsz=self.image_size,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            logger.exception("YOLO prediction failed.")
            raise RuntimeError(
                "Object detection failed. Check the input image and model."
            ) from exc

        if not results:
            return []

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return []

        detections: list[Detection] = []

        xyxy_values = boxes.xyxy.cpu().numpy()
        confidence_values = boxes.conf.cpu().numpy()
        class_values = boxes.cls.cpu().numpy().astype(int)

        for coordinates, score, class_id in zip(
            xyxy_values,
            confidence_values,
            class_values,
        ):
            x1, y1, x2, y2 = (float(value) for value in coordinates)

            label = self.class_names.get(
                int(class_id),
                f"class_{int(class_id)}",
            )

            detections.append(
                Detection(
                    class_id=int(class_id),
                    label=label,
                    confidence=float(score),
                    bbox=(x1, y1, x2, y2),
                )
            )

        logger.debug("Detected %d objects.", len(detections))
        return detections

    def get_class_names(self) -> dict[int, str]:
        """Return the model's class ID-to-label mapping."""
        return self.class_names.copy()

    def predict(
        self,
        frame: np.ndarray,
        confidence: float | None = None,
        iou: float | None = None,
    ) -> list[Detection]:
        """Alias for detect(), useful for callers that use predict terminology."""
        return self.detect(frame, confidence=confidence, iou=iou)