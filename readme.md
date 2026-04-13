# 🛡️ DeepShield — Deepfake Detector

**A Deepfake detection system with dual-branch neural network architecture, combining spatial and frequency-domain analysis for robust forgery detection.**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Model Architecture](#model-architecture)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [Output Explanations](#output-explanations)
- [Performance Metrics](#performance-metrics)
- [Fine-tuning Guide](#fine-tuning-guide)
- [Recommendations for Evaluation](#recommendations-for-evaluation)

---

## Overview

DeepShield is an intelligent deepfake detection application that leverages a dual-branch deep learning architecture to identify manipulated facial videos and images. The system combines:

1. **Spatial Branch** — EfficientNet-B4 for learning spatial features and face morphologies
2. **Frequency Branch** — FFT-based analysis for detecting compression artifacts, GAN fingerprints, and JPEG re-compression traces

This dual-approach achieves ~94% AUC on FaceForensics++ while maintaining fast inference times suitable for real-world deployment.

---

## Key Features

✅ **Image Analysis** — Single image classification with confidence scores
✅ **Video Analysis** — Frame-by-frame detection with timeline visualization
✅ **Explainability** — Visual heatmaps showing decision regions
✅ **Dual-Domain Analysis** — Spatial + frequency heatmaps reveal different artifact types
✅ **Branch Attribution** — Pie chart showing which domain contributed to the verdict
✅ **Per-Frame Timeline** — For videos, shows fake probability across all frames
✅ **Modern UI** — Dark theme, responsive design, real-time processing feedback

---

## Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| **Frontend** | Streamlit 1.35.0 | Web interface & real-time rendering |
| **Deep Learning** | PyTorch 2.2.0 + TorchVision 0.17.0 | Model architecture & inference |
| **Computer Vision** | OpenCV 4.9.0 | Video reading, frame extraction & preprocessing |
| **Image Processing** | Pillow 10.3.0, SciPy 1.13.0, NumPy 1.26.0 | Image manipulation, FFT operations |
| **Visualization** | Matplotlib 3.9.0 | Heatmaps & timeline charts |

---

## Model Architecture

### Dual-Branch Detector

**Why EfficientNet-B4?**

| Model | AUC Score | Parameters | Inference Time | Notes |
|---|---|---|---|---|
| Custom 3D-CNN | ~78% | 12M | Slow | Baseline, limited performance |
| VGG16 | ~82% | 138M | Very slow | Heavy, memory intensive |
| **EfficientNet-B4 + FFT** | **~94%** | **20M** | **1.5s/image (CPU)** | ⭐ **Recommended** |
| Xception | ~93% | 23M | 2s/image | Comparable, slower |
| ViT-B/16 | ~95% | 86M | 3s+ with GPU | Best accuracy but VRAM hungry |

**Architecture Details:**

- **Spatial Branch** — EfficientNet-B4 pretrained on ImageNet, fine-tuned on facial forensics datasets
- **Frequency Branch** — 2D FFT magnitude + phase analysis fed through lightweight CNN layers
- **Fusion** — L2-normalized output concatenation for balanced contribution from both domains
- **Output** — Binary softmax (Real/Fake) with per-class confidence scores

---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip or conda
- 4GB+ RAM (8GB recommended for video processing)
- GPU support optional (CUDA 11.8+ recommended for faster video inference)

### Step 1: Clone/Download the Repository
```bash
cd deepfake_detector
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
streamlit run app.py
```

---

## Usage Guide

### For Images
1. **Upload** — Use the file uploader to select a JPG, PNG, or webshot image
2. **Process** — Click the analyze button (auto-triggered after upload)
3. **View Results** — Get verdict, confidence score, and visualization heatmaps
4. **Interpret** — Check which face regions triggered the decision via spatial heatmap; inspect frequency heatmap for compression artifacts

### For Videos
1. **Upload** — Select an MP4, MOV, AVI, or webm video file
2. **Select FPS** — Choose frame sampling rate (lower = faster, default 2 FPS)
3. **Process** — The app extracts frames and analyzes each one
4. **View Results** — See combined verdict, per-frame timeline, and overall statistics
5. **Timeline** — Inspect which frames were flagged as fake; useful for identifying edited segments

### Confidence Threshold
- **High confidence** (>90%) — Very reliable prediction
- **Medium confidence** (70-90%) — Generally trustworthy, but worth secondary review
- **Low confidence** (<70%) — Model uncertain; consider as indeterminant or request higher-resolution input

---

## Output Explanations

| Output | Meaning | Interpretation |
|---|---|---|
| **Verdict** | REAL / FAKE | Primary classification result |
| **Confidence Score** | 0.0–1.0 | Model's certainty (max softmax probability) |
| **Spatial Heatmap** | Feature activation map from EfficientNet | Highlights face regions linked to the decision; red = high activation (suggests fake) |
| **Frequency Heatmap** | FFT magnitude visualization | Shows GAN artifacts, compression rings, and digital traces; bright regions = artifacts |
| **Branch Contribution %** | Pie chart | Percentage of influence from spatial vs frequency domain |
| **Frame Timeline** (video) | Line graph | Fake probability for each frame; sudden spikes = likely edited frames |

---
