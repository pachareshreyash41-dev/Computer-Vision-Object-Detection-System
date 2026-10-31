# Computer Vision Object Detection System

A Python-based computer vision application that detects common objects
in images and videos using **YOLOv8n**, **Ultralytics**, **OpenCV**, and
**Streamlit**. The application can draw bounding boxes around detected
objects and display their labels and confidence scores.

> **Runtime note:** This guide uses Windows Command Prompt (`cmd`) and
> the project's existing `.venv` virtual environment. The application is
> configured for CPU inference, so a CUDA-enabled GPU is not required.

## Features

-   Detect multiple objects in an image.
-   Display object labels, bounding boxes, and confidence scores.
-   Process video files if the video-processing module is configured in
    the project.
-   Adjust detection confidence and IoU settings in the app, where
    available.
-   View detection results in the Streamlit interface.
-   Export detection results as CSV if the export option is enabled.
-   Use the pretrained YOLOv8n model for common object categories.

## Technology Stack

-   **Python 3.11**
-   **Streamlit** --- web interface
-   **Ultralytics YOLOv8n** --- object detection
-   **OpenCV** --- image/video processing
-   **NumPy and Pillow** --- image handling
-   **Pandas** --- tabular detection results, where used

## Project Structure

The exact structure can vary slightly depending on the files in your
local copy. A typical layout is:

``` text
Computer-Vision-Object-Detection/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
│
├── models/
│   └── .gitkeep
│
├── data/
│   ├── input/
│   │   └── .gitkeep
│   └── output/
│       └── .gitkeep
│
├── src/
│   ├── __init__.py
│   ├── detector.py
│   ├── image_processor.py
│   ├── video_processor.py
│   └── utils.py
│
├── tests/
│   ├── __init__.py
│   ├── test_detector.py
│   └── test_utils.py
│
└── docs/
    ├── architecture.md
    ├── setup_guide.md
    └── model_evaluation.md
```

## Requirements

Before building the project, install or verify:

1.  **Python 3.11.x** (recommended for this project setup).
2.  **pip**, which is included with Python.
3.  Internet access for installing dependencies and downloading model
    weights on the first run.
4.  A modern web browser.
5.  Enough disk space for Python packages and the model file.

CPU inference is supported. Detection may be slower than on a compatible
GPU.

## Step-by-Step Build Guide (Windows CMD)

### Step 1 --- Open Command Prompt

Press `Win + R`, type `cmd`, and press Enter.

### Step 2 --- Go to the project directory

Use the path where you saved the project:

``` cmd
cd /d C:\Users\pacha\Computer-Vision-Object-Detection
```

If your project is saved somewhere else, replace the path with your
actual folder path.

### Step 3 --- Verify Python

``` cmd
py -3.11 --version
python --version
```

The project environment described here uses Python 3.11. If `py -3.11`
is not recognized, install Python 3.11 and make sure the Python Launcher
is available.

### Step 4 --- Create a virtual environment (only if `.venv` does not already exist)

``` cmd
py -3.11 -m venv .venv
```

If `.venv` already exists, do not recreate it unless you intentionally
want a fresh environment.

### Step 5 --- Activate the virtual environment

``` cmd
.venv\Scripts\activate
```

After activation, the command prompt should begin with `(.venv)`.

### Step 6 --- Upgrade pip

``` cmd
python -m pip install --upgrade pip
```

### Step 7 --- Install dependencies

Make sure `requirements.txt` exists in the project root, then run:

``` cmd
python -m pip install -r requirements.txt
```

If this command fails, read the error carefully and resolve the
dependency or Python-version issue before continuing. Do not install
random package versions on top of a partially broken environment.

### Step 8 --- Check the important files

``` cmd
dir
dir src
dir models
```

Confirm that `app.py`, `src\detector.py`, and `src\image_processor.py`
exist. If the project uses a local model file, check whether
`models\yolov8n.pt` exists.

### Step 9 --- Compile-check the Python files

``` cmd
python -m py_compile app.py src\detector.py src\image_processor.py
```

No output normally means Python found no syntax errors in these files.

### Step 10 --- Verify that the detector and image processor can be imported

``` cmd
python -c "from src.detector import ObjectDetector; print('ObjectDetector import OK')"
python -c "from src.image_processor import ImageProcessor; print('ImageProcessor import OK'); print('process method:', hasattr(ImageProcessor, 'process'))"
```

The second command should show `process method: True` if the application
uses the `ImageProcessor.process()` interface.

### Step 11 --- Start the application

``` cmd
python -m streamlit run app.py
```

Streamlit normally prints a local address such as
`http://localhost:8501`. Open the address shown in the terminal in your
browser. Keep the terminal open while using the application.

On the first run, Ultralytics may download the YOLOv8n weights if they
are not already available. Allow the download to finish.

### Step 12 --- Test image detection

1.  Open the application in your browser.
2.  Select the image-detection section or tab.
3.  Upload a clear JPG, JPEG, or PNG image, depending on the formats
    enabled by the app.
4.  Choose the available confidence/IoU settings if shown.
5.  Run detection.
6.  Check the annotated image, object labels, and confidence scores.
7.  If available, test the CSV/download control.

Start with an image containing common objects such as a person, car,
dog, cat, bottle, chair, laptop, or fruit.

### Step 13 --- Test video detection (if enabled)

1.  Open the video-detection section or tab.
2.  Upload a supported video file.
3.  Start processing and wait for it to finish.
4.  Check the output video and any detection summary the app provides.

Video formats and output locations depend on the project's
implementation and installed OpenCV codecs.

### Step 14 --- Stop the application

In the Command Prompt window running Streamlit, press:

``` text
Ctrl + C
```

To leave the virtual environment:

``` cmd
deactivate
```

## What Can the Model Detect?

The standard pretrained `yolov8n.pt` model is trained on the COCO
dataset and supports 80 common categories. Examples include:

-   **People:** person
-   **Vehicles:** bicycle, car, motorcycle, airplane, bus, train, truck,
    boat
-   **Animals:** bird, cat, dog, horse, sheep, cow, elephant, bear,
    zebra, giraffe
-   **Furniture:** chair, couch, bed, dining table
-   **Electronics:** TV, laptop, mouse, remote, keyboard, cell phone
-   **Food:** banana, apple, sandwich, orange, broccoli, carrot, hot
    dog, pizza, donut, cake
-   **Everyday objects:** backpack, umbrella, handbag, suitcase, bottle,
    cup, book, clock, scissors, teddy bear, toothbrush, vase, potted
    plant
-   **Sports items:** sports ball, skateboard, surfboard, tennis racket,
    skis, snowboard

This is not a general-purpose detector for every possible object. It may
miss small, blurry, hidden, unusual, or unsupported objects. For
project-specific objects, collect and label a dataset and train or
fine-tune a suitable custom model.

To print the model's actual class list from the project root (when the
local weights file exists):

``` cmd
python -c "from ultralytics import YOLO; m=YOLO('models/yolov8n.pt'); print('\n'.join(f'{i}: {name}' for i, name in m.names.items()))"
```

If the model weights are stored elsewhere, adjust the path to match your
configuration.

## Troubleshooting

### `ModuleNotFoundError: No module named ...`

Activate the virtual environment and reinstall the project's
requirements:

``` cmd
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

### `streamlit` is not recognized

Run Streamlit through Python instead:

``` cmd
python -m streamlit run app.py
```

### Model weights are missing

Check the configured model path and the `models` folder. If using
Ultralytics' automatic download, make sure internet access is available
and restart the application. Do not download model files from untrusted
sources.

### `ImageProcessor` has no attribute `process`

Check that `src\image_processor.py` defines `process()` and that it
returns the values expected by `app.py`. Then compile-check the file and
restart Streamlit.

### Image upload fails

-   Try a clear JPG or PNG file.
-   Check that the uploaded file is not corrupted.
-   Confirm that the app's file uploader accepts that extension.
-   Read the traceback in the Command Prompt; the final lines usually
    identify the failing file and method.

### Detection finds no objects

-   Use a clearer, well-lit image.
-   Make sure the object is one of the model's supported categories.
-   Try a larger or closer view of the object.
-   If available, lower the confidence threshold slightly.
-   Remember that a pretrained model may not recognize custom or
    uncommon objects.

### Port 8501 is already in use

Stop the other Streamlit process, or start this app on another port:

``` cmd
python -m streamlit run app.py --server.port 8502
```

Then open the address printed by Streamlit.

## Security and Good Practices

-   Do not upload private or confidential images to an environment that
    is accessible to other people.
-   Do not commit API keys, tokens, private data, virtual environments,
    or large generated files to source control.
-   Use trusted sources for Python packages and model weights.
-   Keep dependencies updated in a controlled way and test after
    upgrades.
-   Review model predictions before relying on them for important
    decisions.

## License and Model Notes

Check the licenses and terms for the project code, Python packages, and
YOLO model weights before redistributing the application. This README
does not grant rights to third-party components.

## Quick Start Summary

After Python 3.11 and the project files are in place, the usual commands
are:

``` cmd
cd /d C:\Users\pacha\Computer-Vision-Object-Detection
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed in the terminal and upload an image to begin.
