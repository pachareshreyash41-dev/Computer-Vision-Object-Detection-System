"""Streamlit front end for the Computer Vision Object Detection system."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.detector import ObjectDetector
from src.image_processor import ImageProcessor
from src.utils import (
    detections_to_dataframe,
    ensure_dirs,
    save_uploaded_file,
    timestamped_name,
)
from src.video_processor import VideoProcessor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("app")


def _env_path(name: str, default: str) -> Path:
    """Resolve relative paths from the project root."""
    path = Path(os.getenv(name, default))
    return path if path.is_absolute() else BASE_DIR / path


def _env_float(name: str, default: float) -> float:
    """Read a floating-point environment setting safely."""
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        logger.warning(
            "Invalid value for %s; using default %s",
            name,
            default,
        )
        return default


MODEL_DIR = _env_path("MODEL_DIR", "models")
INPUT_DIR = _env_path("INPUT_DIR", "data/input")
OUTPUT_DIR = _env_path("OUTPUT_DIR", "data/output")

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "yolov8n.pt")
DEFAULT_DEVICE = os.getenv("DEVICE", "auto").lower()
DEFAULT_CONFIDENCE = _env_float("CONFIDENCE_THRESHOLD", 0.25)
DEFAULT_IOU = _env_float("IOU_THRESHOLD", 0.45)

BUILTIN_MODELS = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
]

DEVICE_OPTIONS = ["auto", "cpu", "cuda", "mps"]

IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]
VIDEO_TYPES = ["mp4", "avi", "mov", "mkv"]


@dataclass(frozen=True)
class Settings:
    """Options selected in the sidebar."""

    model_name: str
    device: str
    confidence: float
    iou: float
    save_outputs: bool


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def list_models() -> list[str]:
    """List local model files followed by supported built-in models."""
    local = sorted(path.name for path in MODEL_DIR.glob("*.pt"))
    remote = [name for name in BUILTIN_MODELS if name not in local]

    options = local + remote

    if DEFAULT_MODEL not in options:
        options.insert(0, DEFAULT_MODEL)

    return options


@st.cache_resource(show_spinner="Loading YOLO model...")
def load_detector(model_name: str, device: str) -> ObjectDetector:
    """
    Load and cache a YOLO detector.

    IMPORTANT:
    ObjectDetector expects model_path, not model_name or model_dir.
    """
    model_path = MODEL_DIR / model_name

    logger.info("Selected model: %s", model_name)
    logger.info("Model path: %s", model_path)
    logger.info("Selected device: %s", device)

    # For local files, pass the complete path.
    # For built-in downloadable models, ObjectDetector will resolve
    # the model filename through Ultralytics.
    return ObjectDetector(
        model_path=str(model_path),
        confidence=DEFAULT_CONFIDENCE,
        iou=DEFAULT_IOU,
        device=device,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> Settings:
    """Render configuration controls."""
    st.sidebar.header("Settings")

    models = list_models()

    model_name = st.sidebar.selectbox(
        "Model",
        models,
        index=models.index(DEFAULT_MODEL) if DEFAULT_MODEL in models else 0,
        help=(
            "Local .pt files in the models folder are listed first. "
            "Supported built-in models can be downloaded automatically."
        ),
    )

    device = st.sidebar.selectbox(
        "Device",
        DEVICE_OPTIONS,
        index=(
            DEVICE_OPTIONS.index(DEFAULT_DEVICE)
            if DEFAULT_DEVICE in DEVICE_OPTIONS
            else 0
        ),
        help=(
            "'auto' lets Ultralytics select the device. "
            "Choose CPU if CUDA or Apple MPS is unavailable."
        ),
    )

    confidence = st.sidebar.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=float(min(max(DEFAULT_CONFIDENCE, 0.05), 0.95)),
        step=0.05,
        help="Detections below this confidence score are discarded.",
    )

    iou = st.sidebar.slider(
        "IoU threshold (NMS)",
        min_value=0.05,
        max_value=0.95,
        value=float(min(max(DEFAULT_IOU, 0.05), 0.95)),
        step=0.05,
        help="Controls how overlapping detection boxes are suppressed.",
    )

    save_outputs = st.sidebar.checkbox(
        "Save results to data/output",
        value=True,
        help="Save annotated images and videos to the output directory.",
    )

    st.sidebar.divider()
    st.sidebar.caption("Computer Vision Object Detection")

    return Settings(
        model_name=model_name,
        device=device,
        confidence=confidence,
        iou=iou,
        save_outputs=save_outputs,
    )


# ---------------------------------------------------------------------------
# Image detection
# ---------------------------------------------------------------------------

def render_image_tab(
    detector: ObjectDetector,
    settings: Settings,
) -> None:
    """Upload an image, detect objects, and display the results."""
    uploaded = st.file_uploader(
        "Upload an image",
        type=IMAGE_TYPES,
        key="image_upload",
    )

    if uploaded is None:
        st.info("Upload a JPG, PNG, BMP, or WebP image to detect objects.")
        return

    raw = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    image_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)

    if image_bgr is None:
        st.error("The uploaded file could not be decoded as an image.")
        return

    processor = ImageProcessor(detector)

    try:
        started = time.perf_counter()

        with st.spinner("Detecting objects..."):
            annotated_bgr, detections = processor.process(
                image_bgr,
                confidence=settings.confidence,
                iou=settings.iou,
            )

        elapsed = time.perf_counter() - started

    except Exception as exc:
        logger.exception("Image detection failed")
        st.error(f"Image detection failed: {exc}")
        return

    left, right = st.columns(2)

    with left:
        st.subheader("Original image")
        st.image(
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
            use_container_width=True,
        )

    with right:
        st.subheader("Detected objects")
        st.image(
            cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB),
            use_container_width=True,
        )

    df = detections_to_dataframe(detections)

    metric_a, metric_b, metric_c = st.columns(3)

    metric_a.metric("Objects found", len(df))

    metric_b.metric(
        "Distinct classes",
        int(df["class_name"].nunique()) if not df.empty else 0,
    )

    metric_c.metric(
        "Inference time",
        f"{elapsed * 1000:.0f} ms",
    )

    if df.empty:
        st.warning(
            "No objects detected. Try lowering the confidence threshold."
        )
    else:
        st.subheader("Detection details")

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Objects by class")
        st.bar_chart(df["class_name"].value_counts())

    # Encode the annotated image for saving and downloading.
    ok, encoded = cv2.imencode(".png", annotated_bgr)

    if not ok:
        st.error("Could not encode the annotated image.")
        return

    image_bytes = encoded.tobytes()
    stem = Path(uploaded.name).stem

    if settings.save_outputs:
        out_path = OUTPUT_DIR / timestamped_name(
            f"{stem}_detected",
            ".png",
        )

        out_path.write_bytes(image_bytes)

        try:
            display_path = out_path.relative_to(BASE_DIR)
        except ValueError:
            display_path = out_path

        st.success(f"Result saved to: {display_path}")

    download_a, download_b = st.columns(2)

    download_a.download_button(
        "Download annotated image",
        data=image_bytes,
        file_name=f"{stem}_detected.png",
        mime="image/png",
    )

    if not df.empty:
        download_b.download_button(
            "Download detections (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"{stem}_detections.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------------
# Video detection
# ---------------------------------------------------------------------------

def render_video_tab(
    detector: ObjectDetector,
    settings: Settings,
) -> None:
    """Upload a video, process its frames, and display the result."""
    uploaded = st.file_uploader(
        "Upload a video",
        type=VIDEO_TYPES,
        key="video_upload",
    )

    if uploaded is None:
        st.info("Upload an MP4, AVI, MOV, or MKV video.")
        return

    st.video(uploaded.getvalue())

    if st.button("Run detection", type="primary", key="run_video"):
        input_path = None

        try:
            input_path = save_uploaded_file(uploaded, INPUT_DIR)

            output_path = OUTPUT_DIR / timestamped_name(
                f"{Path(uploaded.name).stem}_detected",
                ".mp4",
            )

            progress = st.progress(0.0, text="Starting video processing...")

            def on_progress(done: int, total: int) -> None:
                fraction = min(done / total, 1.0) if total else 0.0

                label = (
                    f"Processing frame {done}/{total}"
                    if total
                    else f"Processing frame {done}"
                )

                progress.progress(fraction, text=label)

            stats = VideoProcessor(detector).process_video(
                input_path=input_path,
                output_path=output_path,
                confidence=settings.confidence,
                iou=settings.iou,
                progress_callback=on_progress,
            )

            progress.progress(1.0, text="Processing complete")

            st.session_state["video_result"] = {
                "source": uploaded.name,
                "output_path": str(output_path),
                "stats": stats,
                "save_outputs": settings.save_outputs,
            }

            if settings.save_outputs:
                st.success("Video processing completed and result saved.")
            else:
                st.success("Video processing completed.")

        except Exception as exc:
            logger.exception("Video detection failed")
            st.error(f"Video processing failed: {exc}")
            return

    result = st.session_state.get("video_result")

    if not result or result["source"] != uploaded.name:
        return

    output_path = Path(result["output_path"])
    stats = result["stats"]

    if not output_path.exists():
        st.warning(
            "The processed video is no longer available. Run detection again."
        )
        return

    st.subheader("Processed video")
    st.video(str(output_path))

    metric_a, metric_b, metric_c, metric_d = st.columns(4)

    metric_a.metric("Frames", stats.get("frames", 0))
    metric_b.metric(
        "Total detections",
        stats.get("total_detections", 0),
    )
    metric_c.metric(
        "Processing time",
        f"{stats.get('processing_time', 0.0):.1f} s",
    )
    metric_d.metric(
        "Average speed",
        f"{stats.get('processing_fps', 0.0):.1f} FPS",
    )

    class_counts = stats.get("class_counts") or {}

    if class_counts:
        counts_df = pd.DataFrame(
            class_counts.items(),
            columns=["class_name", "detections"],
        ).sort_values(
            "detections",
            ascending=False,
        ).reset_index(drop=True)

        st.subheader("Detections by class")

        st.dataframe(
            counts_df,
            use_container_width=True,
            hide_index=True,
        )

        st.bar_chart(counts_df.set_index("class_name"))

    st.download_button(
        "Download annotated video",
        data=output_path.read_bytes(),
        file_name=output_path.name,
        mime="video/mp4",
    )


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Configure and launch the Streamlit application."""
    st.set_page_config(
        page_title="Computer Vision Object Detection",
        page_icon="🔍",
        layout="wide",
    )

    ensure_dirs(
        MODEL_DIR,
        INPUT_DIR,
        OUTPUT_DIR,
    )

    st.title("Computer Vision Object Detection")
    st.write(
        "Detect and label objects in images and videos using YOLO."
    )

    settings = render_sidebar()

    try:
        detector = load_detector(
            settings.model_name,
            settings.device,
        )

    except Exception as exc:
        logger.exception("Model loading failed")

        st.error(
            f"Could not load model '{settings.model_name}': {exc}"
        )

        st.info(
            "Check the selected model, internet connection if downloading "
            "a model, and the Python dependencies."
        )

        st.stop()
        return

    image_tab, video_tab = st.tabs(
        ["Image Detection", "Video Detection"]
    )

    with image_tab:
        render_image_tab(detector, settings)

    with video_tab:
        render_video_tab(detector, settings)


if __name__ == "__main__":
    main()