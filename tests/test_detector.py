"""Tests for src/detector.py.

The YOLO model is replaced with a lightweight fake, so these tests run fast
and need neither model weights nor a network connection.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.detector import Detection, ObjectDetector


# --------------------------------------------------------------------------- #
# Fakes that mimic the parts of the Ultralytics API the detector relies on
# --------------------------------------------------------------------------- #
class FakeBox:
    def __init__(self, class_id: int, conf: float, xyxy: tuple[float, float, float, float]):
        self.cls = np.array([class_id])
        self.conf = np.array([conf])
        self.xyxy = np.array([xyxy])


class FakeResult:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class FakeModel:
    names = {0: "person", 2: "car"}

    def __init__(self, boxes=None):
        self.boxes = boxes if boxes is not None else []
        self.last_kwargs: dict = {}

    def predict(self, frame, **kwargs):
        self.last_kwargs = kwargs
        return [FakeResult(self.boxes, self.names)]


@pytest.fixture
def fake_model() -> FakeModel:
    return FakeModel(
        boxes=[
            FakeBox(0, 0.91, (10, 20, 110, 220)),
            FakeBox(2, 0.55, (50, 60, 150, 100)),
        ]
    )


@pytest.fixture
def detector(monkeypatch, fake_model) -> ObjectDetector:
    monkeypatch.setattr(ObjectDetector, "_load_model", staticmethod(lambda _path: fake_model))
    return ObjectDetector(model_path="fake.pt", confidence=0.3, iou=0.5, image_size=320)


@pytest.fixture
def frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


# --------------------------------------------------------------------------- #
# Detection dataclass
# --------------------------------------------------------------------------- #
class TestDetection:
    def test_to_dict(self):
        det = Detection(1, "bicycle", 0.8, (0, 0, 10, 10))
        assert det.to_dict() == {
            "class_id": 1,
            "label": "bicycle",
            "confidence": 0.8,
            "bbox": (0, 0, 10, 10),
        }

    def test_area(self):
        assert Detection(0, "person", 0.9, (10, 20, 110, 70)).area == 100 * 50

    def test_area_is_never_negative(self):
        assert Detection(0, "person", 0.9, (50, 50, 10, 10)).area == 0.0

    def test_is_immutable(self):
        det = Detection(0, "person", 0.9, (0, 0, 1, 1))
        with pytest.raises(Exception):
            det.label = "car"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# ObjectDetector
# --------------------------------------------------------------------------- #
class TestObjectDetector:
    def test_class_names_loaded(self, detector):
        assert detector.class_names == {0: "person", 2: "car"}

    @pytest.mark.parametrize("value", [None, "", "auto"])
    def test_auto_device_maps_to_none(self, monkeypatch, fake_model, value):
        monkeypatch.setattr(ObjectDetector, "_load_model", staticmethod(lambda _p: fake_model))
        assert ObjectDetector("fake.pt", device=value).device is None

    def test_explicit_device_is_kept(self, monkeypatch, fake_model):
        monkeypatch.setattr(ObjectDetector, "_load_model", staticmethod(lambda _p: fake_model))
        assert ObjectDetector("fake.pt", device="cpu").device == "cpu"

    def test_detect_returns_detections(self, detector, frame):
        results = detector.detect(frame)

        assert len(results) == 2
        assert all(isinstance(d, Detection) for d in results)

        person, car = results
        assert (person.class_id, person.label) == (0, "person")
        assert person.confidence == pytest.approx(0.91)
        assert person.bbox == (10.0, 20.0, 110.0, 220.0)
        assert (car.class_id, car.label) == (2, "car")

    def test_detect_with_no_objects(self, monkeypatch, frame):
        empty_model = FakeModel(boxes=[])
        monkeypatch.setattr(ObjectDetector, "_load_model", staticmethod(lambda _p: empty_model))
        assert ObjectDetector("fake.pt").detect(frame) == []

    def test_default_thresholds_are_passed_to_model(self, detector, fake_model, frame):
        detector.detect(frame)
        assert fake_model.last_kwargs["conf"] == 0.3
        assert fake_model.last_kwargs["iou"] == 0.5
        assert fake_model.last_kwargs["imgsz"] == 320
        assert fake_model.last_kwargs["verbose"] is False

    def test_per_call_thresholds_override_defaults(self, detector, fake_model, frame):
        detector.detect(frame, confidence=0.7, iou=0.2)
        assert fake_model.last_kwargs["conf"] == 0.7
        assert fake_model.last_kwargs["iou"] == 0.2

    def test_zero_threshold_is_not_treated_as_missing(self, detector, fake_model, frame):
        detector.detect(frame, confidence=0.0)
        assert fake_model.last_kwargs["conf"] == 0.0

    @pytest.mark.parametrize("bad_frame", [None, np.array([])])
    def test_empty_frame_raises(self, detector, bad_frame):
        with pytest.raises(ValueError):
            detector.detect(bad_frame)