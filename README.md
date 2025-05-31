# Helmet Detection System

This project detects **motorcycle riders without helmets** using a custom-trained **YOLOv8** model. When a violation is detected, the system saves:

- The entire **bounding box** region of the **rider** (not just the helmet or head)
- The entire **bounding box** region of the **number plate**
- Both are saved in a **timestamped folder** for easy tracking and record-keeping

---

## Features

- Detects riders **with and without helmets**
- Identifies and saves **number plates** of violators
- Generates a **CSV file** with paths to each violation image (optional)
- Organized **folder structure by date and time**

---

## Workflow

### Step 1: Dataset

We used a custom dataset from Kaggle:

> [Rider with Helmet / Without Helmet / Number Plate Dataset](https://www.kaggle.com/datasets/aneesarom/rider-with-helmet-without-helmet-number-plate/data)

---

### Step 2: Training

The training logic is implemented in [`train_model.py`](./train_model.py), which uses **YOLOv8** to train on the custom dataset.

---

### Step 3: Inference

Inference is handled by the [`rider_safety_check.py`](./rider_safety_check.py) script.

**What the script does:**

- Loads the trained YOLOv8 model
- Applies it to a video to detect:
  - **Riders**
  - **Helmet status** (with helmet / without helmet)
  - **Number plates**
- Draws **bounding boxes** on the video
- Saves the **full bounding box** region (not cropped) of:
  - The **rider** (without a helmet)
  - The **number plate**
- Stores each detection in a folder named using the exact timestamp:  
  `track/YYYY-MM-DD_HH-MM-SS`
- Automatically stops after saving **5 valid violations** to avoid redundancy

---

### Step 4: CSV Records

A CSV file named `records.csv` is generated containing:

| rider_paths       | number_plate_paths |
|-------------------|--------------------|
| track/...         | track/...          |

> This file stores only the **file paths** of the detected rider and number plate images.  
> It is useful for referencing, automation, or further data processing.



---

## Contents of `track.zip`

- Output video showing detected violations
- Rider images (**full bounding box**)
- Number plate images (**full bounding box**)
- `records.csv` with file paths of saved images

---



## Video Credits

The video used for inference was sourced from **[Pexels](https://pexels.com)**.  
All credit goes to the original video creators on Pexels for providing high-quality content for testing and demonstration purposes.

---

## Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt

