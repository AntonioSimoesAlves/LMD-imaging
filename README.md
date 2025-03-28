# Laser Metal Deposition

This is a Python application that, with the aid of machine learning, identifies melt pool regions in a given dataset.

This is an academic project.

## Requirements

* Python 3.12

### Development requirements

* [uv](https://docs.astral.sh/uv/)

## Setup

### Windows

Setup:

```pwsh
py -3 -m venv venv
.\venv\Scripts\pip install . # without GPU support
.\venv\Scripts\pip install .[gpu] --extra-index-url "https://download.pytorch.org/whl/cu126" # with GPU support
```

Running:

```pwsh
.\venv\Scripts\lmd_image.exe
```

### Unix

Setup:

```sh
python3 -m venv venv
venv/bin/pip install . # without GPU support
venv/bin/pip install .[gpu] --extra-index-url "https://download.pytorch.org/whl/cu126" # with GPU support
```

Running:

```sh
venv/bin/lmd_imaging
```

## Normal operation
Running the program will fetch the images in /dataset/, create segmentation masks according to predefined melting and maximum temperatures.

The masks are then split into training and testing folders inside the /data/ directory.

YOLOv8n-seg will then train and validate the model on said images.

Upon successful completion of the model, the directory with the model data will open automatically.
