# Changelog

All notable changes to the **AI-Based Single Image Dehazing System** project will be documented in this file.

## [1.0.0] - 2026-07-29

### Added
- **DehazeFormer Architecture**: Vision Transformer (W-MSA) with residual skip-connections.
- **AOD-Net Architecture**: Lightweight 5-layer CNN for K-parameter estimation.
- **Dark Channel Prior**: Classical computer vision baseline with Guided Image Filtering.
- **Streamlit Web Dashboard**: 10 interactive navigation pages with dark glassmorphism aesthetic theme.
- **10-Metric IQA Suite**: PSNR, SSIM, MSE, Brightness, Contrast, Sharpness, Entropy, Visibility Score, Haze Density Index, and Overall Composite Quality Score.
- **Document & Presentation Generators**: Automated Word report (`.docx`), Markdown report (`.md`), PowerPoint presentation (`.pptx`), and Mermaid UML diagrams.
- **Evaluation Pipeline**: Benchmark runner evaluating models across RESIDE, O-HAZE, Dense-Haze, and NH-Haze datasets.
- **Automated Pytest Suite**: 13 unit and integration tests with 100% pass rate.
