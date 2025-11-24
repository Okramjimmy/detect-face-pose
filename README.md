# Face Detection and Pose Estimation

## Setup Instructions

Follow these steps to set up the environment for running the script.

### 1. Create Conda Environment

First, create a Conda environment with Python 3.7.

```bash
conda create -n face_venv_3.7 python=3.7
```

### 2. Activate Environment

Activate the newly created environment.

```bash
conda activate face_venv_3.7
```

### 3. Install Dependencies

Install the required packages using the provided requirements file.

*Note: The requirements file is named `requiremens.txt`. It is recommended to rename it to `requirements.txt`.*

```bash
pip install -r requiremens.txt
```

## How to Run

### Command-Line Options

The script supports the following command-line arguments:

-   `-p`, `--path`: Path to a local input image.
-   `-u`, `--url`: URL of an input image to process.
-   `-c`, `--camera`: The ID of the webcam to use (default is `0`).
-   `-d`, `--device`: The device to run the model on. Options are `auto`, `cpu`, or `cuda` (default is `auto`).

### Examples

**Process a local image:**

```bash
python DetectFacePose.py -p /path/to/your/image.jpg
```

**Process an image from a URL:**

```bash
python DetectFacePose.py -u https://example.com/image.jpg
```

**Run from webcam:**

```bash
python DetectFacePose.py
```

**Use a specific webcam (e.g., camera 1):**

```bash
python DetectFacePose.py -c 1
```
