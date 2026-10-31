"""Video and webcam detection workflows."""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

from src.detector import Detection, ObjectDetector
from src.utils import draw_detections, ensure_dir

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]  # (frames_done, total_frames)


@dataclass
class VideoResult:
    output_path: Path
    frames_processed: int
    fps: float
    duration_seconds: float
    processing_seconds: float
    summary: dict[str, int] = field(default_factory=dict)  # total detections per class


class VideoProcessor:
    def __init__(self, detector: ObjectDetector, output_dir: str | Path = "data/output") -> None:
        self.detector = detector
        self.output_dir = ensure_dir(output_dir)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
        """Prefer H.264 (browser-playable); fall back to mp4v."""
        for codec in ("avc1", "mp4v"):
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
            if writer.isOpened():
                logger.info("Video writer using codec '%s'", codec)
                return writer
            writer.release()
        raise RuntimeError("Could not open a video writer (no usable codec).")

    # ------------------------------------------------------------------ #
    def process_video(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        frame_skip: int = 1,
        max_frames: int | None = None,
        progress_callback: ProgressCallback | None = None,
        **detect_kwargs,
    ) -> VideoResult:
        """Annotate a video file.

        frame_skip=N runs the detector on every Nth frame and reuses the
        previous detections in between (faster, slightly less accurate).
        """
        input_path = Path(input_path)
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if max_frames:
            total = min(total, max_frames) if total else max_frames

        output_path = Path(output_path) if output_path else (
            self.output_dir / f"{input_path.stem}_detected.mp4"
        )
        ensure_dir(output_path.parent)
        writer = self._open_writer(output_path, fps, (width, height))

        frame_skip = max(1, int(frame_skip))
        counts: Counter[str] = Counter()
        last: list[Detection] = []
        index = 0
        start = time.perf_counter()

        try:
            while True:
                ok, frame = cap.read()
                if not ok or (max_frames and index >= max_frames):
                    break

                if index % frame_skip == 0:
                    last = self.detector.detect(frame, **detect_kwargs)
                    counts.update(d.label for d in last)

                writer.write(draw_detections(frame, last))
                index += 1
                if progress_callback:
                    progress_callback(index, total)
        finally:
            cap.release()
            writer.release()

        elapsed = time.perf_counter() - start
        logger.info("Processed %d frames in %.1fs -> %s", index, elapsed, output_path)
        return VideoResult(
            output_path=output_path,
            frames_processed=index,
            fps=fps,
            duration_seconds=index / fps,
            processing_seconds=elapsed,
            summary=dict(counts.most_common()),
        )

    # ------------------------------------------------------------------ #
    def stream(self, source: int | str = 0, **detect_kwargs) -> Iterator[tuple[np.ndarray, list[Detection]]]:
        """Yield (annotated_frame, detections) from a webcam index or stream URL.

        Example:
            for frame, dets in processor.stream(0):
                cv2.imshow("Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                detections = self.detector.detect(frame, **detect_kwargs)
                yield draw_detections(frame, detections), detections
        finally:
            cap.release()