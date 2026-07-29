# 🎓 Comprehensive Viva Voce Questions & Answers (30 Questions)

---

## 🏛️ Section 1: Fundamental Physics & Atmospheric Science

### Q1: What is single image dehazing?
**Answer:** Single image dehazing is a computer vision task aimed at recovering clear scene radiance $\mathbf{J}(x)$ from a single degraded hazy input image $\mathbf{I}(x)$ affected by atmospheric scattering (fog, haze, smoke).

### Q2: What is the Atmospheric Scattering Model equation?
**Answer:** $\mathbf{I}(x) = \mathbf{J}(x) \cdot t(x) + \mathbf{A}(1 - t(x))$, where $\mathbf{I}(x)$ is observed intensity, $\mathbf{J}(x)$ is clear radiance, $t(x)$ is medium transmission map, and $\mathbf{A}$ is atmospheric light vector.

### Q3: How is medium transmission $t(x)$ defined physically?
**Answer:** $t(x) = e^{-\beta d(x)}$, where $\beta$ is the atmospheric scattering coefficient and $d(x)$ is scene depth.

### Q4: Why is single image dehazing an ill-posed inverse problem?
**Answer:** Because for a single RGB input $\mathbf{I}(x)$ (3 knowns at each pixel), we must estimate scene radiance $\mathbf{J}(x)$ (3 unknowns), transmission $t(x)$ (1 unknown), and atmospheric light $\mathbf{A}$ (3 unknowns), yielding 7 unknowns per pixel.

### Q5: What is the Dark Channel Prior (DCP)?
**Answer:** An empirical discovery by Kaiming He et al. stating that in most non-sky outdoor clear images, at least one color channel has very low intensity (near zero) in local spatial patches.

### Q6: Write the mathematical expression for Dark Channel Prior.
**Answer:** $J^{dark}(x) = \min_{y \in \Omega(x)} \left( \min_{c \in \{R,G,B\}} I^c(y) \right) \approx 0$.

### Q7: Why is Guided Filtering necessary in DCP?
**Answer:** Raw transmission maps derived from local patch min-filtering exhibit halo artifacts at depth discontinuities. Guided filtering smoothes transmission maps while preserving edge boundaries.

### Q8: What parameter prevents division by zero during scene radiance recovery?
**Answer:** Transmission lower bound parameter $t_0 = 0.1$. Formula: $\mathbf{J}(x) = \frac{\mathbf{I}(x) - \mathbf{A}}{\max(t(x), t_0)} + \mathbf{A}$.

---

## 🤖 Section 2: Deep Learning & Model Architectures

### Q9: How does AOD-Net differ from traditional two-stage dehazing methods?
**Answer:** Traditional methods separately estimate $t(x)$ and $\mathbf{A}$, accumulating errors. AOD-Net reformulates the scattering equation into a single unified $K(x)$ parameter map: $\mathbf{J}(x) = K(x)\mathbf{I}(x) - K(x) + b$.

### Q10: What is the primary architecture of DehazeFormer?
**Answer:** A Vision Transformer (ViT) incorporating Window Multi-Head Self-Attention (W-MSA), residual learning, and resblock feature fusion.

### Q11: What is Window Multi-Head Self-Attention (W-MSA)?
**Answer:** Self-attention calculated locally inside non-overlapping $8 \times 8$ pixel windows to reduce computational complexity from quadratic $\mathcal{O}(H^2 W^2)$ to linear $\mathcal{O}(HW)$.

### Q12: Why does DehazeFormer use residual learning $J(x) = \text{clamp}(I(x) + R(x))$?
**Answer:** Hazy and clear images share underlying structure. Predicting additive residual radiance $R(x)$ simplifies network learning and stabilizes training gradients.

### Q13: How are spatial dimensions handled for arbitrary resolution images in PyTorch?
**Answer:** The image tensor is padded to dimensions divisible by 16 using `pad_to_multiple()` before inference, then cropped back to original resolution.

### Q14: What is `@torch.inference_mode()` and why is it preferred over `@torch.no_grad()`?
**Answer:** `torch.inference_mode()` disables autograd tracking and version counting entirely, yielding faster execution and lower memory usage than `no_grad()`.

---

## 📊 Section 3: Metrics & Performance Evaluation

### Q15: What is PSNR?
**Answer:** Peak Signal-to-Noise Ratio measuring image reconstruction fidelity in dB. Higher values (>28 dB) indicate superior restoration.

### Q16: What is SSIM?
**Answer:** Structural Similarity Index Measure (0 to 1) evaluating perceptual similarity across luminance, contrast, and structural information.

### Q17: What is Shannon Entropy in image processing?
**Answer:** $H(I) = -\sum p_i \log_2(p_i)$, measuring texture detail richness and information content in bits per pixel.

### Q18: What is Laplacian Variance used for?
**Answer:** Estimating high-frequency edge definition / sharpness. Higher values indicate sharper edges.

### Q19: What is the RESIDE dataset?
**Answer:** Realistic Single Image Dehazing Dataset containing Synthetic (ITS/OTS) and real-world hazy/ground-truth image pairs.

---

## 💻 Section 4: System Implementation & Software Architecture

### Q20: What framework powers the interactive dashboard?
**Answer:** Streamlit with custom CSS dark glassmorphism styling.

### Q21: How are Plotly RGB histograms optimized?
**Answer:** Uses `go.Scattergl` (WebGL rendering) and `@st.cache_data` caching to prevent recalculations on tab navigation.

### Q22: How does `DehazeInferenceEngine` handle model loading?
**Answer:** Uses a Model Factory pattern (`models/loader.py`) with memory caching and automatic CPU/GPU placement.

### Q23: How does the system handle missing pretrained `.pth` files?
**Answer:** Uses smart Kaiming Normal initialization fallbacks and logs warnings without crashing.

### Q24: What security protections exist against Path Traversal?
**Answer:** Filename sanitization via `Path.name` stripping and `is_relative_to()` boundary checks.

### Q25: How does batch inference work?
**Answer:** Stacks image tensors into `(B, 3, H, W)` batch tensors for parallel PyTorch forward pass execution.

### Q26: What file formats are supported for export?
**Answer:** PNG, JPEG, WebP, comparison image grids, ZIP archives, CSV, JSON, and Markdown reports.

### Q27: How are automated reports generated?
**Answer:** `generate_docs.py` uses `python-docx` for `.docx` reports and `python-pptx` for `.pptx` presentations.

### Q28: How many tests exist in the test suite?
**Answer:** 13 pytest unit and integration tests with 100% pass rate.

### Q29: What is CLAHE?
**Answer:** Contrast Limited Adaptive Histogram Equalization, preventing over-amplification of noise in homogeneous regions.

### Q30: What is the main entry point to launch the web dashboard?
**Answer:** `streamlit run app.py`.
