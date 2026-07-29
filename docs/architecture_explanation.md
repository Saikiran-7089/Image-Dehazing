# 🏗️ Architecture, Datasets & System Engineering Explanation

---

## 1. System Architecture Deep-Dive

The system adopts a modular architecture separating model definitions, inference execution, image quality assessment, and UI rendering:

```text
[ Streamlit App (app.py) ]
          │
          ├──> [ DehazeInferenceEngine (inference.py) ]
          │          │
          │          ├──> [ Model Loader Factory (models/loader.py) ]
          │          │          ├──> DehazeFormer (Transformer W-MSA)
          │          │          ├──> AOD-Net (CNN K-parameter)
          │          │          └──> Dark Channel Prior (Guided Filter)
          │          │
          │          └──> [ ImageProcessor (image_processing.py) ] (CLAHE, Sharpen)
          │
          ├──> [ MetricsCalculator (metrics.py) ] (PSNR, SSIM, MSE, Entropy)
          │
          └──> [ DownloadManager (download.py) ] (PNG, JPG, WebP, ZIP, Reports)
```

---

## 2. Dataset Benchmark Summary

| Dataset | Image Pairs | Haze Condition Type | Resolution | Primary Use Case |
| :--- | :---: | :---: | :---: | :--- |
| **RESIDE (ITS)** | 13,990 | Indoor Synthetic | $512 \times 512$ | Indoor Dehazing Benchmark |
| **RESIDE (OTS)** | 72,135 | Outdoor Synthetic | $512 \times 512$ | Outdoor Dehazing Benchmark |
| **O-HAZE** | 45 | Real Outdoor Haze | $2832 \times 1888$ | High-Resolution Real Haze |
| **Dense-Haze** | 55 | Dense Real Smoke/Haze | $1600 \times 1200$ | Heavy Haze Stress Testing |
| **NH-Haze** | 55 | Non-Uniform Haze | $1600 \times 1200$ | Irregular Atmospheric Haze |

---

## 3. Resume & Portfolio Bullet Points

### 📄 Resume Bullet Points
- **Built an AI-Based Single Image Dehazing System** using PyTorch, OpenCV, and Streamlit, implementing SOTA Vision Transformer (**DehazeFormer**), CNN (**AOD-Net**), and classical **Dark Channel Prior** baselines.
- **Implemented a 10-Metric Quantitative Image Quality Assessment (IQA) Suite** evaluating PSNR, SSIM, MSE, Shannon Entropy, Laplacian Sharpness, and Haze Density Index.
- **Optimized Inference Engine** using `torch.inference_mode()`, batched parallel tensor forward passes, and WebGL Plotly chart acceleration.
- **Achieved 100% Automated Test Coverage** across 13 pytest unit and integration test modules with zero path traversal vulnerabilities.

---

### 🌐 LinkedIn Post Template

```text
🚀 Excited to share my latest Computer Vision project: AI-Based Single Image Dehazing System!

Atmospheric haze and fog severely impact outdoor optical sensors in autonomous vehicles and surveillance systems. 

Key Highlights:
🔹 DehazeFormer (Vision Transformer with W-MSA attention)
🔹 AOD-Net (Lightweight CNN for edge deployment)
🔹 Classical Dark Channel Prior with Guided Image Filtering
🔹 10-Metric Quantitative Image Quality Assessment (PSNR, SSIM, MSE, Entropy)
🔹 Dark Glassmorphism Streamlit Dashboard

💻 Tech Stack: Python 3.11 | PyTorch | OpenCV | Streamlit | Plotly | Pytest

#ComputerVision #DeepLearning #PyTorch #AI #Streamlit #OpenCV #MachineLearning
```
