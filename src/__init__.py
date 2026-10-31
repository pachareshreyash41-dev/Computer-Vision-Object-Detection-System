"""Computer Vision Object Detection package."""

from src.detector import Detection, ObjectDetector
from src.image_processor import ImageProcessor, ImageResult
from src.video_processor import VideoProcessor, VideoResult

__all__ = [
    "Detection",
    "ObjectDetector",
    "ImageProcessor",
    "ImageResult",
    "VideoProcessor",
    "VideoResult",
]