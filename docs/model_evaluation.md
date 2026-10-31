# Model Evaluation

This guide explains how to choose and evaluate a detection model for this project, and gives a template for recording your own results.

## 1. Metrics

| Metric | Meaning |
|---|---|
| **Precision** | Of all predicted boxes, the fraction that are correct. Low precision means many false alarms. |
| **Recall** | Of all real objects, the fraction that were found. Low recall means missed objects. |
| **IoU** | Overlap between a predicted box and the true box (intersection over union). |
| **AP** | Area under the precision-recall curve for one class, at a given IoU threshold. |
| **mAP@50** | Mean AP across classes at IoU 0.50 (lenient). |
| **mAP@50-95** | Mean AP averaged over IoU 0.50 to 0.95 (strict, the COCO headline metric). |
| **Latency / FPS** | Time per image, and its inverse. Measure on your target hardware. |
| **Model size** | Parameters and file size, which matter for memory and deployment. |

## 2. Choosing a model size

YOLOv8 models trade accuracy for speed. Reference figures for the COCO dataset at 640 px, as published by Ultralytics at the time of writing (check <https://docs.ultralytics.com/models/yolov8/> for current numbers):

| Model | mAP@50-95 | Params (M) | Typical use |
|---|---|---|---|
| `yolov8n` | ~37.3 | ~3.2 | Real-time on CPU, edge devices, prototypes |
| `yolov8s` | ~44.9 | ~11.2 | Good balance for CPU or light GPU |
| `yolov8m` | ~50.2 | ~25.9 | General purpose with a GPU |
| `yolov8l` | ~52.9 | ~43.7 | Accuracy-focused workloads |
| `yolov8x` | ~53.9 | ~68.2 | Maximum accuracy, offline processing |

Start with `yolov8n` for development and move up only if accuracy is not enough.

## 3. Evaluating on a dataset

The Ultralytics validator computes precision, recall, mAP and per-class results. You need a dataset in YOLO format with a `data.yaml`:

```yaml
path: datasets/my_dataset
train: images/train
val: images/val
names:
  0: person
  1: car
```

Run validation:

```python
from ultralytics import YOLO

model = YOLO("models/yolov8n.pt")
metrics = model.val(data="datasets/my_dataset/data.yaml", imgsz=640, conf=0.001, iou=0.6)

print("mAP@50-95:", metrics.box.map)
print("mAP@50   :", metrics.box.map50)
print("Precision:", metrics.box.mp)
print("Recall   :", metrics.box.mr)
```

Use a low `conf` (such as 0.001) when computing mAP so the full precision-recall curve is measured. Use your normal operating threshold (for example 0.25) only when judging how the app behaves in practice.

Validation also writes a confusion matrix, PR curves and F1 curves to `runs/detect/val*/`. Use the F1 curve to pick a confidence threshold for `CONFIDENCE_THRESHOLD`.

## 4. Benchmarking speed

Measure with this project's own `ObjectDetector`, so the numbers reflect real usage:

```python
import time
import numpy as np
from src.detector import ObjectDetector

detector = ObjectDetector("models/yolov8n.pt", image_size=640)
frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

for _ in range(5):                # warm-up
    detector.detect(frame)

runs = 50
start = time.perf_counter()
for _ in range(runs):
    detector.detect(frame)
elapsed = time.perf_counter() - start

print(f"{elapsed / runs * 1000:.1f} ms/frame  ({runs / elapsed:.1f} FPS)")
```

Tips for fair comparisons:
- Always warm up first; the first calls are slower.
- Run on the same machine with nothing else heavy running.
- Benchmark with realistic image sizes. Random noise is fine for timing, but not for accuracy.
- Report the device (CPU model or GPU) along with the numbers.

## 5. Results template

Fill this in for each model you try.

**Setup:** dataset: `________` · images: `____` · device: `________` · image size: `____`

| Model | mAP@50 | mAP@50-95 | Precision | Recall | ms/frame | FPS | Notes |
|---|---|---|---|---|---|---|---|
| yolov8n | | | | | | | |
| yolov8s | | | | | | | |
| yolov8m | | | | | | | |
| custom | | | | | | | |

## 6. Qualitative review

Numbers do not tell the whole story. Also check:

- **Failure cases**: save frames with missed or wrong detections and look for patterns (lighting, small objects, occlusion, motion blur).
- **Per-class results**: a good overall mAP can hide a weak class you care about.
- **Threshold behaviour**: try thresholds from 0.1 to 0.6 in the app and see how false positives and misses change.
- **Video stability**: watch whether boxes flicker between frames. If so, consider a slightly lower confidence threshold, a larger model, or adding tracking.

## 7. Improving results

| Symptom | Things to try |
|---|---|
| Misses small objects | Increase `IMAGE_SIZE` (for example 960 or 1280), use a larger model |
| Many false positives | Raise `CONFIDENCE_THRESHOLD`, add negative examples when fine-tuning |
| Duplicate boxes on one object | Lower `IOU_THRESHOLD` |
| Overlapping objects get merged | Raise `IOU_THRESHOLD` |
| Poor accuracy on your domain | Fine-tune on your own labelled data |
| Too slow | Smaller model, smaller `IMAGE_SIZE`, GPU, `frame_skip`, or export to ONNX/TensorRT |

### Fine-tuning example

```python
from ultralytics import YOLO

model = YOLO("models/yolov8n.pt")
model.train(data="datasets/my_dataset/data.yaml", epochs=50, imgsz=640, batch=16)
# Best weights are saved to runs/detect/train*/weights/best.pt
```

Copy `best.pt` into `models/` and set `MODEL_PATH=models/best.pt`.

## 8. Reproducibility checklist

- Record the Ultralytics, PyTorch and Python versions (`pip freeze`).
- Record the dataset version and split.
- Record all thresholds and the image size.
- Keep the exact weights file (or its hash) used for each reported result.