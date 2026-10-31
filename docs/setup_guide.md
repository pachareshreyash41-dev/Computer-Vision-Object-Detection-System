# Setup Guide

## Requirements

- Python 3.9 or newer (3.10+ recommended)
- About 2 GB of free disk space (PyTorch and dependencies)
- Optional: an NVIDIA GPU with CUDA for faster inference
- Internet access on first run to download model weights (unless you supply them)

## 1. Get the project

```bash
git clone <your-repo-url> Computer-Vision-Object-Detection
cd Computer-Vision-Object-Detection
```

## 2. Create a virtual environment

**macOS / Linux**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

## 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### GPU support (optional)
`pip install` may give you a CPU-only PyTorch build. For NVIDIA GPUs, install PyTorch first using the command from <https://pytorch.org/get-started/locally/> for your CUDA version, then run `pip install -r requirements.txt`.

Check that PyTorch sees your GPU:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 4. Configure

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

Edit `.env` if needed:

| Variable | Default | Notes |
|---|---|---|
| `MODEL_PATH` | `models/yolov8n.pt` | Use `yolov8s.pt`, `yolov8m.pt`, ... for more accuracy |
| `CONFIDENCE_THRESHOLD` | `0.25` | Higher = fewer, more certain detections |
| `IOU_THRESHOLD` | `0.45` | Lower = more aggressive removal of overlapping boxes |
| `IMAGE_SIZE` | `640` | Larger = more accurate on small objects, but slower |
| `DEVICE` | `auto` | `cpu`, `cuda`, `cuda:0`, `mps` (Apple Silicon) |
| `INPUT_DIR` / `OUTPUT_DIR` | `data/input` / `data/output` | |

## 5. Model weights

You have two options:

1. **Automatic**: leave `models/` empty. If `MODEL_PATH` points to a missing official file such as `models/yolov8n.pt`, the code requests `yolov8n.pt` from Ultralytics, which downloads it into the **current working directory**. Move it into `models/` afterwards to keep things tidy.
2. **Manual**: download weights yourself from the Ultralytics releases and place them in `models/`.

Custom-trained weights (`best.pt`) work too. Point `MODEL_PATH` at the file.

## 6. Run

### Web app
```bash
streamlit run app.py
```
Open the URL printed in the terminal (usually <http://localhost:8501>).

### Python API
```python
from src.detector import ObjectDetector
from src.image_processor import ImageProcessor

detector = ObjectDetector("models/yolov8n.pt")
result = ImageProcessor(detector).process_file("data/input/photo.jpg")
print(result.summary)
```

### Webcam
```python
import cv2
from src.detector import ObjectDetector
from src.video_processor import VideoProcessor

processor = VideoProcessor(ObjectDetector())
for frame, detections in processor.stream(0):
    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
```
Webcam streaming opens an OpenCV window, so it needs a desktop environment (it will not work over SSH or in most containers).

## 7. Run the tests

```bash
pytest -v
```
The tests use a fake model, so they need no weights or internet.

## Troubleshooting

| Problem | Likely cause and fix |
|---|---|
| `ModuleNotFoundError: src` | Run commands from the project root, not from inside `src/` or `tests/` |
| Model download fails | No internet or a proxy is blocking it. Download the weights manually into `models/` |
| `CUDA out of memory` | Lower `IMAGE_SIZE`, use a smaller model, or set `DEVICE=cpu` |
| Very slow video processing | Use a GPU, a smaller model, a lower `IMAGE_SIZE`, or raise "Process every Nth frame" |
| Output video won't play in the browser | OpenCV fell back to `mp4v`. Download and play it locally, or install an OpenCV build with H.264 support, or re-encode with `ffmpeg -i in.mp4 -vcodec libx264 out.mp4` |
| `Could not open video` | Unsupported codec or corrupted file. Try re-encoding with ffmpeg |
| Webcam window does not open | No desktop session, or wrong camera index (try `1` instead of `0`) |
| `libGL.so.1` error on Linux servers | Use `opencv-python-headless` instead of `opencv-python` |
| Streamlit upload size error | Raise it with `streamlit run app.py --server.maxUploadSize 500` |