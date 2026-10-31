"""Tests for src/utils.py."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from src.detector import Detection
from src.utils import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    draw_detections,
    ensure_dir,
    get_color,
    is_image,
    is_video,
    list_files,
    load_config,
    summarize_detections,
    timestamped_name,
)

ENV_VARS = [
    "MODEL_PATH", "CONFIDENCE_THRESHOLD", "IOU_THRESHOLD",
    "IMAGE_SIZE", "DEVICE", "INPUT_DIR", "OUTPUT_DIR",
]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
class TestLoadConfig:
    @pytest.fixture(autouse=True)
    def isolate_env(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)  # so no real .env file is picked up
        for var in ENV_VARS:
            monkeypatch.delenv(var, raising=False)

    def test_defaults(self):
        cfg = load_config()
        assert cfg.model_path == "models/yolov8n.pt"
        assert cfg.confidence == 0.25
        assert cfg.iou == 0.45
        assert cfg.image_size == 640
        assert cfg.device == "auto"
        assert cfg.input_dir == Path("data/input")
        assert cfg.output_dir == Path("data/output")

    def test_environment_overrides(self, monkeypatch):
        monkeypatch.setenv("MODEL_PATH", "models/custom.pt")
        monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.6")
        monkeypatch.setenv("IMAGE_SIZE", "1280")
        monkeypatch.setenv("DEVICE", "cpu")
        monkeypatch.setenv("OUTPUT_DIR", "results")

        cfg = load_config()
        assert cfg.model_path == "models/custom.pt"
        assert cfg.confidence == 0.6
        assert cfg.image_size == 1280
        assert cfg.device == "cpu"
        assert cfg.output_dir == Path("results")

    def test_config_is_frozen(self):
        with pytest.raises(Exception):
            load_config().confidence = 0.9  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# File helpers
# --------------------------------------------------------------------------- #
class TestFileHelpers:
    @pytest.mark.parametrize("name", ["a.jpg", "a.JPEG", "dir/a.png", "a.webp"])
    def test_is_image_true(self, name):
        assert is_image(name)

    @pytest.mark.parametrize("name", ["a.mp4", "a.txt", "a", "a.jpg.bak"])
    def test_is_image_false(self, name):
        assert not is_image(name)

    @pytest.mark.parametrize("name", ["a.mp4", "a.AVI", "dir/a.mov", "a.mkv"])
    def test_is_video_true(self, name):
        assert is_video(name)

    @pytest.mark.parametrize("name", ["a.jpg", "a.txt", "a"])
    def test_is_video_false(self, name):
        assert not is_video(name)

    def test_extension_sets_do_not_overlap(self):
        assert IMAGE_EXTENSIONS.isdisjoint(VIDEO_EXTENSIONS)

    def test_timestamped_name_format(self):
        name = timestamped_name("cat", ".jpg")
        assert re.fullmatch(r"cat_\d{8}_\d{6}\.jpg", name)

    def test_timestamped_name_adds_missing_dot(self):
        assert timestamped_name("cat", "png").endswith(".png")

    def test_ensure_dir_creates_nested(self, tmp_path):
        target = tmp_path / "a" / "b" / "c"
        result = ensure_dir(target)
        assert result == target and target.is_dir()

    def test_ensure_dir_is_idempotent(self, tmp_path):
        ensure_dir(tmp_path / "x")
        ensure_dir(tmp_path / "x")  # must not raise

    def test_list_files_filters_and_sorts(self, tmp_path):
        for name in ["b.jpg", "a.png", "notes.txt", "clip.mp4"]:
            (tmp_path / name).touch()
        (tmp_path / "subdir.jpg").mkdir()  # directories are ignored

        found = list_files(tmp_path, IMAGE_EXTENSIONS)
        assert [p.name for p in found] == ["a.png", "b.jpg"]

    def test_list_files_missing_directory(self, tmp_path):
        assert list_files(tmp_path / "nope", IMAGE_EXTENSIONS) == []


# --------------------------------------------------------------------------- #
# Drawing and summaries
# --------------------------------------------------------------------------- #
@pytest.fixture
def detections() -> list[Detection]:
    return [
        Detection(0, "person", 0.92, (20, 30, 120, 200)),
        Detection(0, "person", 0.71, (150, 40, 220, 190)),
        Detection(2, "car", 0.60, (240, 100, 380, 180)),
    ]


class TestColors:
    def test_color_is_deterministic(self):
        assert get_color(3) == get_color(3)

    def test_color_values_are_valid_bgr(self):
        for class_id in range(80):
            color = get_color(class_id)
            assert len(color) == 3
            assert all(60 <= c <= 255 for c in color)

    def test_different_classes_usually_differ(self):
        assert len({get_color(i) for i in range(10)}) > 1


class TestDrawDetections:
    def test_returns_copy_and_leaves_input_untouched(self, detections):
        image = np.zeros((300, 400, 3), dtype=np.uint8)
        annotated = draw_detections(image, detections)

        assert annotated is not image
        assert image.sum() == 0  # original unchanged
        assert annotated.sum() > 0  # something was drawn

    def test_preserves_shape_and_dtype(self, detections):
        image = np.zeros((300, 400, 3), dtype=np.uint8)
        annotated = draw_detections(image, detections)
        assert annotated.shape == image.shape
        assert annotated.dtype == image.dtype

    def test_no_detections_returns_identical_image(self):
        image = np.full((100, 100, 3), 127, dtype=np.uint8)
        assert np.array_equal(draw_detections(image, []), image)

    def test_box_edge_uses_class_colour(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        det = Detection(1, "bicycle", 0.9, (50, 80, 150, 180))
        annotated = draw_detections(image, [det], thickness=2)
        # a pixel on the left edge, well below the label area
        assert tuple(annotated[150, 50]) == get_color(1)

    def test_box_touching_image_top_does_not_crash(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        det = Detection(0, "person", 0.9, (10, 0, 60, 50))
        draw_detections(image, [det])

    def test_label_without_confidence(self, detections):
        image = np.zeros((300, 400, 3), dtype=np.uint8)
        with_conf = draw_detections(image, detections, show_confidence=True)
        without_conf = draw_detections(image, detections, show_confidence=False)
        assert not np.array_equal(with_conf, without_conf)


class TestSummarizeDetections:
    def test_counts_per_label_most_common_first(self, detections):
        summary = summarize_detections(detections)
        assert summary == {"person": 2, "car": 1}
        assert list(summary) == ["person", "car"]

    def test_empty(self):
        assert summarize_detections([]) == {}