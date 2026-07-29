"""
===============================================================================
AI-Based Single Image Dehazing System
Configuration Management Module (config.py) [UI/UX Refined Edition]
===============================================================================

This module provides centralized configuration settings, system paths, UI theme
constants, default model parameters, and environment settings for the Image
Dehazing System.

Author: Senior Computer Vision & AI Engineer / Senior UI/UX Designer
License: MIT
===============================================================================
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import torch

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger: logging.Logger = logging.getLogger("DehazeConfig")


# =============================================================================
# SYSTEM PATHS CONFIGURATION
# =============================================================================
BASE_DIR: Path = Path(__file__).resolve().parent

MODELS_DIR: Path = BASE_DIR / "models"
WEIGHTS_DIR: Path = BASE_DIR / "weights"
OUTPUTS_DIR: Path = BASE_DIR / "outputs"
REPORTS_DIR: Path = BASE_DIR / "reports"
PRESENTATION_DIR: Path = BASE_DIR / "presentation"
ASSETS_DIR: Path = BASE_DIR / "assets"
STATIC_DIR: Path = BASE_DIR / "static"
TESTS_DIR: Path = BASE_DIR / "tests"
DATASET_DIR: Path = BASE_DIR / "dataset"

# List of required system directories
REQUIRED_DIRECTORIES: List[Path] = [
    MODELS_DIR,
    WEIGHTS_DIR,
    OUTPUTS_DIR,
    REPORTS_DIR,
    PRESENTATION_DIR,
    ASSETS_DIR,
    STATIC_DIR,
    TESTS_DIR,
    DATASET_DIR,
]


def ensure_directories() -> None:
    """
    Creates all necessary project directories if they do not exist.
    """
    for folder in REQUIRED_DIRECTORIES:
        folder.mkdir(parents=True, exist_ok=True)
    logger.info("Project directory structure verified and initialized.")


# Initialize directories on configuration import
ensure_directories()


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
MODEL_DEHAZEFORMER: str = "DehazeFormer"
MODEL_AODNET: str = "AOD-Net"
MODEL_DCP: str = "Dark Channel Prior"

SUPPORTED_MODELS: List[str] = [
    MODEL_DEHAZEFORMER,
    MODEL_AODNET,
    MODEL_DCP
]

DEFAULT_MODEL: str = MODEL_DEHAZEFORMER

WEIGHT_FILES: Dict[str, str] = {
    MODEL_DEHAZEFORMER: "dehazeformer.pth",
    MODEL_AODNET: "aodnet.pth",
}

# =============================================================================
# HARDWARE & EXECUTION SETTINGS
# =============================================================================
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
USE_HALF_PRECISION: bool = False  # FP16 acceleration if CUDA is enabled
NUM_WORKERS: int = 4
BATCH_SIZE: int = 16

logger.info("Execution hardware detected: %s (CUDA Available: %s)", DEVICE, torch.cuda.is_available())


# =============================================================================
# IMAGE PROCESSING & ENHANCEMENT DEFAULTS
# =============================================================================
MAX_IMAGE_SIZE: Tuple[int, int] = (1920, 1080)
PATCH_SIZE: int = 256  # Sliding window patch size for transformer inference

SUPPORTED_IMAGE_EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

DEFAULT_CLAHE_CLIP_LIMIT: float = 2.0
DEFAULT_CLAHE_TILE_GRID: Tuple[int, int] = (8, 8)
DEFAULT_SHARPNESS_FACTOR: float = 1.5
DEFAULT_DENOISE_STRENGTH: int = 10
DEFAULT_BRIGHTNESS_FACTOR: float = 1.0
DEFAULT_CONTRAST_FACTOR: float = 1.0

# =============================================================================
# UI THEME CONFIGURATION (Senior UI/UX Glassmorphism)
# =============================================================================
THEME_PRIMARY_COLOR: str = "#0f172a"      # Deep Navy Dark
THEME_SECONDARY_COLOR: str = "#1e293b"    # Slate Card BG
THEME_ACCENT_COLOR: str = "#38bdf8"       # Neon Light Blue
THEME_TEXT_COLOR: str = "#f8fafc"         # Off-white Text
THEME_SUCCESS_COLOR: str = "#10b981"      # Emerald Green
THEME_WARNING_COLOR: str = "#f59e0b"      # Amber Warning
THEME_CARD_BG: str = "rgba(30, 41, 59, 0.75)"

CUSTOM_CSS: str = """
<style>
    /* Google Fonts import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global Dark Glassmorphism App Canvas */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #1e293b 0%, #0f172a 60%, #080c14 100%);
        color: #f8fafc;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    /* Typography & Headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        letter-spacing: -0.025em;
        color: #f8fafc;
    }

    /* Keyframe Animations */
    @keyframes pulseGlow {
        0% { box-shadow: 0 0 15px rgba(56, 189, 248, 0.2); }
        50% { box-shadow: 0 0 25px rgba(56, 189, 248, 0.4); }
        100% { box-shadow: 0 0 15px rgba(56, 189, 248, 0.2); }
    }

    /* Glassmorphic Metric Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), border-color 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 16px;
    }

    .glass-card:hover {
        transform: translateY(-4px);
        border-color: rgba(56, 189, 248, 0.5);
        box-shadow: 0 12px 35px 0 rgba(56, 189, 248, 0.25);
    }

    /* Metric Header and Value Styling */
    .metric-title {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94a3b8;
        margin-bottom: 6px;
    }

    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #38bdf8;
        letter-spacing: -0.03em;
    }

    .metric-sub {
        font-size: 0.8rem;
        font-weight: 500;
        color: #10b981;
        margin-top: 4px;
    }

    /* Gradient Interactive Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #0284c7 0%, #38bdf8 100%);
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 1.02rem !important;
        padding: 12px 24px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 15px rgba(2, 132, 199, 0.4) !important;
        width: 100%;
    }

    .stButton>button:hover {
        transform: translateY(-2px) scale(1.01) !important;
        box-shadow: 0 8px 25px rgba(56, 189, 248, 0.6) !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95);
        border-right: 1px solid rgba(56, 189, 248, 0.15);
    }

    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #0284c7 0%, #38bdf8 100%);
        border-radius: 10px;
    }

    /* Scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0f172a;
    }
    ::-webkit-scrollbar-thumb {
        background: #1e293b;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #38bdf8;
    }
</style>
"""

# =============================================================================
# QUALITY ASSESSMENT TARGET RANGES & METADATA
# =============================================================================
METRIC_TARGETS: Dict[str, Dict[str, Any]] = {
    "PSNR": {
        "unit": "dB",
        "ideal_range": "> 28 dB",
        "description": "Peak Signal-to-Noise Ratio measuring image reconstruction accuracy.",
        "formula": r"PSNR = 10 \cdot \log_{10}\left(\frac{MAX_I^2}{MSE}\right)"
    },
    "SSIM": {
        "unit": "Score (0 to 1)",
        "ideal_range": "> 0.85",
        "description": "Structural Similarity Index evaluating perceptual similarity in luminance, contrast, and structure.",
        "formula": r"SSIM(x,y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}"
    },
    "MSE": {
        "unit": "Pixel Error",
        "ideal_range": "< 0.01",
        "description": "Mean Squared Error calculating pixel-wise squared differences.",
        "formula": r"MSE = \frac{1}{m \cdot n} \sum_{i=0}^{m-1} \sum_{j=0}^{n-1} [I(i,j) - K(i,j)]^2"
    },
    "Brightness": {
        "unit": "Luminance (0-255)",
        "ideal_range": "100 - 160",
        "description": "Mean pixel luminance across the RGB spectrum.",
        "formula": r"\text{Brightness} = \frac{1}{N} \sum_{i=1}^{N} I_i"
    },
    "Contrast": {
        "unit": "RMS Value",
        "ideal_range": "40 - 80",
        "description": "Root Mean Square (RMS) standard deviation of pixel intensities.",
        "formula": r"\text{Contrast} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (I_i - \bar{I})^2}"
    },
    "Sharpness": {
        "unit": "Variance",
        "ideal_range": "> 100",
        "description": "Laplacian filter variance quantifying high-frequency edge definition.",
        "formula": r"\text{Sharpness} = \text{Var}(\nabla^2 I)"
    },
    "Entropy": {
        "unit": "Bits per pixel",
        "ideal_range": "> 7.0",
        "description": "Shannon information entropy representing detail richness and texture complexity.",
        "formula": r"H(I) = -\sum_{i=0}^{255} p_i \log_2(p_i)"
    }
}

# =============================================================================
# DATASET CONFIGURATIONS
# =============================================================================
SUPPORTED_DATASETS: List[str] = ["RESIDE (ITS/OTS)", "O-HAZE", "Dense-Haze", "NH-Haze"]

# =============================================================================
# PROJECT METADATA
# =============================================================================
PROJECT_TITLE: str = "AI-Based Single Image Dehazing System"
PROJECT_SUBTITLE: str = "Transformer-Based Deep Learning and Image Quality Assessment"
PROJECT_VERSION: str = "1.0.0"
PROJECT_AUTHOR: str = "Senior CV & AI Research Team"
