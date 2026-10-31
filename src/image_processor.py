"""Image-level detection workflows."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.detector import Detection, ObjectDetector
from src.utils import (
    IMAGE_EXTENSIONS,
    draw_detections,
    ensure_dir,
    list_files,
    summarize_detections,
)

logger = logging.getLogger(__name__)


@dataclass
class ImageResult:
    """Result of processing a single image."""

    source: Path | None
    annotated: np.ndarray  # BGR image
    detections: list[Detection]
    output_path: Path | None = None
    summary: dict[str, int] = field(default_factory=dict)


class ImageProcessor:
    """Process images using an ObjectDetector."""

    def __init__(
        self,
        detector: ObjectDetector,
        output_dir: str | Path = "data/output",
    ) -> None:
        self.detector = detector
        self.output_dir = ensure_dir(output_dir)

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def process_array(
        self,
        image_bgr: np.ndarray,
        confidence: float | None = None,
        iou: float | None = None,
    ) -> ImageResult:
        """Process a BGR NumPy image and return an ImageResult."""

        if not isinstance(image_bgr, np.ndarray):
            raise TypeError("image_bgr must be a NumPy ndarray.")

        if image_bgr.size == 0:
            raise ValueError("The input image is empty.")

        if image_bgr.ndim not in (2, 3):
            raise ValueError("The input image has an unsupported shape.")

        detections = self.detector.detect(
            image_bgr,
            confidence=confidence,
            iou=iou,
        )

        annotated = draw_detections(image_bgr.copy(), detections)

        return ImageResult(
            source=None,
            annotated=annotated,
            detections=detections,
            summary=summarize_detections(detections),
        )

    def process(
        self,
        image_bgr: np.ndarray,
        confidence: float | None = None,
        iou: float | None = None,
    ) -> tuple[np.ndarray, list[Detection]]:
        """
        Process an OpenCV BGR image.

        Returns:
            A tuple containing:
            - Annotated BGR image
            - List of Detection objects

        This method is compatible with the Streamlit app.py interface.
        """

        result = self.process_array(
            image_bgr,
            confidence=confidence,
            iou=iou,
        )

        return result.annotated, result.detections

    def process_pil(
        self,
        image: Image.Image,
        confidence: float | None = None,
        iou: float | None = None,
    ) -> ImageResult:
        """Process a PIL image and return an ImageResult."""

        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL Image.")

        rgb_image = np.array(image.convert("RGB"))
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        return self.process_array(
            bgr_image,
            confidence=confidence,
            iou=iou,
        )

    def process_file(
        self,
        path: str | Path,
        save: bool = True,
        confidence: float | None = None,
        iou: float | None = None,
    ) -> ImageResult:
        """Read and process an image file, optionally saving the result."""

        path = Path(path)

        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError(f"Could not read image: {path}")

        result = self.process_array(
            image,
            confidence=confidence,
            iou=iou,
        )

        result.source = path

        if save:
            self.output_dir.mkdir(parents=True, exist_ok=True)

            output_path = (
                self.output_dir / f"{path.stem}_detected{path.suffix}"
            )

            success = cv2.imwrite(
                str(output_path),
                result.annotated,
            )

            if not success:
                raise OSError(
                    f"Could not save annotated image: {output_path}"
                )

            result.output_path = output_path

            logger.info(
                "Saved %s (%d objects)",
                output_path,
                len(result.detections),
            )

        return result

    def process_directory(
        self,
        input_dir: str | Path,
        **kwargs,
    ) -> list[ImageResult]:
        """Process all supported image files in a directory."""

        files = list_files(input_dir, IMAGE_EXTENSIONS)

        if not files:
            logger.warning("No images found in %s", input_dir)

        results: list[ImageResult] = []

        for file_path in files:
            try:
                result = self.process_file(file_path, **kwargs)
                results.append(result)
            except (ValueError, OSError, FileNotFoundError) as exc:
                logger.error("Skipping %s: %s", file_path, exc)

        return results

    # ------------------------------------------------------------------
    # Image conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def to_rgb(image_bgr: np.ndarray) -> np.ndarray:
        """Convert a BGR image to RGB for display in Streamlit."""

        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)