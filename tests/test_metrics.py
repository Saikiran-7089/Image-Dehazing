"""
===============================================================================
Unit Tests: Image Quality Assessment Metrics (tests/test_metrics.py)
===============================================================================
"""

import pytest
import numpy as np

from metrics import (
    calculate_psnr,
    calculate_ssim,
    calculate_mse,
    calculate_brightness,
    calculate_contrast,
    calculate_sharpness,
    calculate_entropy,
    MetricsCalculator
)


def test_identical_image_metrics():
    """Test PSNR, SSIM, MSE for identical images."""
    img = (np.ones((100, 100, 3), dtype=np.uint8) * 128)
    psnr = calculate_psnr(img, img)
    ssim = calculate_ssim(img, img)
    mse = calculate_mse(img, img)

    assert psnr >= 99.0, "PSNR for identical images should be ~100 dB"
    assert ssim == 1.0, "SSIM for identical images should be 1.0"
    assert mse == 0.0, "MSE for identical images should be 0.0"


def test_individual_metrics_calculation():
    """Test individual brightness, contrast, sharpness, and entropy functions."""
    img = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)

    brightness = calculate_brightness(img)
    contrast = calculate_contrast(img)
    sharpness = calculate_sharpness(img)
    entropy = calculate_entropy(img)

    assert 0.0 <= brightness <= 255.0
    assert contrast >= 0.0
    assert sharpness >= 0.0
    assert 0.0 <= entropy <= 8.0


def test_metrics_calculator_all_metrics():
    """Test full MetricsCalculator dictionary output."""
    img1 = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
    img2 = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)

    calc = MetricsCalculator()
    m_dict = calc.calculate_all_metrics(img1, img2)

    required_keys = [
        "psnr", "ssim", "mse", "brightness", "contrast",
        "sharpness", "entropy", "visibility_score",
        "haze_density_score", "overall_quality_score"
    ]

    for key in required_keys:
        assert key in m_dict, f"Missing required metric key: {key}"
        assert isinstance(m_dict[key], (int, float)), f"Metric {key} must be numeric"
