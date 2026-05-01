# AI-Generated Image Detector

A hybrid deep learning framework that detects and attributes AI-generated images using metadata forensics combined with a ResNet-50 + CLIP fusion architecture.

## What It Does

- Classifies images into 6 categories: **Real, DALL·E 3, Midjourney, SD2.1, SDXL, SD3**
- **Stage 1:** Inspects EXIF, XMP, and C2PA metadata for AI generation signatures
- **Stage 2:** If metadata is inconclusive, runs a dual-branch deep learning model — ResNet-50 for texture/artifact features and CLIP ViT-B/32 for semantic features — fused for final classification
- Achieves **99.16% accuracy** on internal test set and **90.13%** on the VCT2 COCO AI external benchmark

## How It Works

```
Input Image
    │
    ▼
Metadata Inspection (EXIF / XMP / C2PA)
    │
    ├── Conclusive → Direct Classification
    │
    └── Inconclusive ↓
         │
    ┌────┴────┐
 ResNet-50  CLIP ViT-B/32
 (textures) (semantics)
    └────┬────┘
         │ Fused → Classification Head
         ▼
 Real / DALL·E / Midjourney / SD2.1 / SDXL / SD3
```

## How to Run

### 1. Clone the Repository
```bash
git clone <repo-url>
cd ai_detector_project3
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv env
env\Scripts\activate

# Linux/Mac
python3 -m venv env
source env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Dataset Setup
Dataset is **not included**. Set the path before training:

```bash
# Windows
set DATASET_PATH=D:\your\dataset

# Linux/Mac
export DATASET_PATH=/your/dataset
```

Or edit `ROOT_DIR` directly in the training script. Expected structure:
```
dataset/
├── Real/
├── DALLE3/
├── Midjourney/
├── SD21/
├── SDXL/
└── SD3/
```

### 5. Train
```bash
python training/train.py
```
Outputs: `best_model.pth`, `final_model.pth`, training graphs

### 6. Benchmark
```bash
python benchmark/benchmark2.py
```
Outputs: Accuracy, Precision, Recall, F1, Confusion Matrix

### 7. Predict
```bash
python inference/predict.py
```
Supports single image, folder of images, and metadata + model prediction

## Notes
- Dataset and model weights not included (size constraints)
- GPU strongly recommended for training
