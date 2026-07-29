"""
===============================================================================
Unit Tests: Image Processor Filters & Histograms (tests/test_image_processing.py)
===============================================================================
"""

import pytest
import numpy as np
from image_processing import ImageProcessor


def test_image_processor_filters():
    """Test CLAHE, sharpening, blur, and brightness adjustments."""
    processor = ImageProcessor()
    dummy_img = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)

    # 1. CLAHE
    res_clahe = processor.apply_clahe(dummy_img, clip_limit=2.0)
    assert res_clahe.shape == dummy_img.shape

    # 2. Histogram Equalization
    res_eq = processor.apply_histogram_equalization(dummy_img)
    assert res_eq.shape == dummy_img.shape

    # 3. Gamma Correction
    res_gamma = processor.apply_gamma_correction(dummy_img, gamma=1.2)
    assert res_gamma.shape == dummy_img.shape

    # 4. Brightness & Contrast
    res_bc = processor.adjust_brightness_contrast(dummy_img, brightness=1.1, contrast=1.2)
    assert res_bc.shape == dummy_img.shape

    # 5. Unsharp Masking
    res_sharp = processor.apply_unsharp_mask(dummy_img, amount=1.5)
    assert res_sharp.shape == dummy_img.shape


def test_histogram_and_compression_estimation():
    """Test Plotly histogram generation and compression estimation."""
    processor = ImageProcessor()
    dummy_img = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)

    # Histogram data
    hist_data = processor.generate_plotly_histogram_data(dummy_img)
    assert "x" in hist_data and "r" in hist_data and "g" in hist_data and "b" in hist_data
    assert len(hist_data["x"]) == 256

    # Compression estimation
    comp_info = processor.estimate_compression_ratio(dummy_img, format_ext=".jpg", quality=85)
    assert "compression_ratio" in comp_info
    assert "raw_bytes" in comp_info and "compressed_bytes" in comp_info
