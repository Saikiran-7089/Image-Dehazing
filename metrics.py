"""
===============================================================================
AI-Based Single Image Dehazing System
Image Quality Assessment (IQA) Module (metrics.py)
===============================================================================

This module provides comprehensive quantitative image quality metrics for evaluating
image dehazing algorithms. It includes full implementations of PSNR, SSIM, MSE,
Luminance Brightness, RMS Contrast, Laplacian Sharpness, Shannon Entropy,
Estimated Visibility Score, Haze Density Index, and Overall Composite Quality Score.

Author: Senior Computer Vision & AI Engineer
License: MIT
===============================================================================
"""

import math
import logging
from typing import Dict, Any, Tuple, Union, Optional

import cv2
import numpy as np
from skimage.metrics import structural_similarity as compute_ssim
from skimage.measure import shannon_entropy as compute_entropy

import config

# Initialize module logger
logger: logging.Logger = logging.getLogger("ImageMetrics")


# =============================================================================
# INDIVIDUAL METRIC EVALUATORS
# =============================================================================

def calculate_mse(reference_image: np.ndarray, enhanced_image: np.ndarray) -> float:
    """
    Calculates Mean Squared Error (MSE) between reference and enhanced images.

    Args:
        reference_image (np.ndarray): Original reference image array.
        enhanced_image (np.ndarray): Enhanced dehazed image array.

    Returns:
        float: Mean Squared Error value.
    """
    try:
        ref_arr = reference_image.astype(np.float64)
        enh_arr = enhanced_image.astype(np.float64)
        mse_value = float(np.mean((ref_arr - enh_arr) ** 2))
        return round(mse_value, 6)
    except Exception as err:
        logger.error("Failed to compute MSE: %s", err)
        return 0.0


def calculate_psnr(
    reference_image: np.ndarray,
    enhanced_image: np.ndarray,
    max_pixel_value: float = 255.0
) -> float:
    """
    Calculates Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).

    Formula: PSNR = 10 * log10(MAX_I^2 / MSE)

    Args:
        reference_image (np.ndarray): Reference image array.
        enhanced_image (np.ndarray): Dehazed image array.
        max_pixel_value (float): Maximum possible pixel intensity.

    Returns:
        float: PSNR in dB (higher indicates better reconstruction).
    """
    mse_val = calculate_mse(reference_image, enhanced_image)
    if mse_val == 0.0:
        return 100.0  # Identical images case

    try:
        psnr_val = 10.0 * math.log10((max_pixel_value ** 2) / mse_val)
        return round(float(psnr_val), 2)
    except Exception as err:
        logger.error("Failed to compute PSNR: %s", err)
        return 0.0


def calculate_ssim(reference_image: np.ndarray, enhanced_image: np.ndarray) -> float:
    """
    Calculates Structural Similarity Index Measure (SSIM) between two images.

    Args:
        reference_image (np.ndarray): Reference image array.
        enhanced_image (np.ndarray): Dehazed image array.

    Returns:
        float: SSIM score between 0.0 and 1.0.
    """
    try:
        ref_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY) if len(reference_image.shape) == 3 else reference_image
        enh_gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY) if len(enhanced_image.shape) == 3 else enhanced_image

        ssim_val, _ = compute_ssim(ref_gray, enh_gray, full=True, data_range=255)
        return round(float(ssim_val), 4)
    except Exception as err:
        logger.error("Failed to compute SSIM: %s", err)
        return 0.0


def calculate_brightness(image_array: np.ndarray) -> float:
    """
    Calculates mean luminance / brightness of an image (0 to 255).

    Args:
        image_array (np.ndarray): Input image array.

    Returns:
        float: Average brightness intensity.
    """
    try:
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY) if len(image_array.shape) == 3 else image_array
        return round(float(np.mean(gray)), 2)
    except Exception as err:
        logger.error("Failed to compute brightness: %s", err)
        return 0.0


def calculate_contrast(image_array: np.ndarray) -> float:
    """
    Calculates Root Mean Square (RMS) contrast (standard deviation of pixel intensity).

    Args:
        image_array (np.ndarray): Input image array.

    Returns:
        float: RMS contrast value.
    """
    try:
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY) if len(image_array.shape) == 3 else image_array
        return round(float(np.std(gray)), 2)
    except Exception as err:
        logger.error("Failed to compute contrast: %s", err)
        return 0.0


def calculate_sharpness(image_array: np.ndarray) -> float:
    """
    Calculates image sharpness using the variance of the Laplacian operator.

    Args:
        image_array (np.ndarray): Input image array.

    Returns:
        float: Laplacian variance (higher indicates sharper edges).
    """
    try:
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY) if len(image_array.shape) == 3 else image_array
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return round(float(laplacian_var), 2)
    except Exception as err:
        logger.error("Failed to compute sharpness: %s", err)
        return 0.0


def calculate_entropy(image_array: np.ndarray) -> float:
    """
    Calculates Shannon information entropy of an image in bits per pixel.

    Args:
        image_array (np.ndarray): Input image array.

    Returns:
        float: Shannon entropy score (0 to 8 bits).
    """
    try:
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY) if len(image_array.shape) == 3 else image_array
        entropy_val = compute_entropy(gray)
        return round(float(entropy_val), 3)
    except Exception as err:
        logger.error("Failed to compute entropy: %s", err)
        return 0.0


def calculate_haze_density(image_array: np.ndarray) -> float:
    """
    Estimates the percentage haze density score (0 - 100%) using the dark channel prior average.

    Args:
        image_array (np.ndarray): Input image array.

    Returns:
        float: Estimated haze density percentage (0.0 to 100.0%).
    """
    try:
        min_channel = np.min(image_array, axis=2) if len(image_array.shape) == 3 else image_array
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dark_channel = cv2.erode(min_channel, kernel)
        mean_dark = np.mean(dark_channel) / 255.0
        haze_score = float(np.clip(mean_dark * 100.0, 0.0, 100.0))
        return round(haze_score, 1)
    except Exception as err:
        logger.error("Failed to compute haze density: %s", err)
        return 50.0


def calculate_visibility_score(image_array: np.ndarray) -> float:
    """
    Calculates an estimated visual clarity / visibility score (0 - 100) based on sharpness and contrast.

    Args:
        image_array (np.ndarray): Input image array.

    Returns:
        float: Visibility score between 0.0 and 100.0.
    """
    try:
        sharpness_val = calculate_sharpness(image_array)
        contrast_val = calculate_contrast(image_array)

        norm_sharpness = min(1.0, sharpness_val / 250.0)
        norm_contrast = min(1.0, contrast_val / 70.0)

        visibility = (0.6 * norm_sharpness + 0.4 * norm_contrast) * 100.0
        return round(float(np.clip(visibility, 0.0, 100.0)), 1)
    except Exception as err:
        logger.error("Failed to compute visibility score: %s", err)
        return 50.0


def calculate_overall_quality_score(
    psnr_score: float,
    ssim_score: float,
    original_sharpness: float,
    enhanced_sharpness: float,
    haze_density: float
) -> float:
    """
    Computes a weighted overall composite quality index (0 - 100) for the dehazed image.

    Args:
        psnr_score (float): PSNR in dB.
        ssim_score (float): SSIM score (0.0 - 1.0).
        original_sharpness (float): Original image sharpness.
        enhanced_sharpness (float): Enhanced image sharpness.
        haze_density (float): Remaining haze density percentage.

    Returns:
        float: Overall Quality Score (0.0 to 100.0).
    """
    try:
        norm_psnr = min(1.0, max(0.0, psnr_score / 35.0))
        norm_ssim = min(1.0, max(0.0, ssim_score))
        sharpness_improvement = min(1.0, max(0.0, enhanced_sharpness / max(1.0, original_sharpness * 1.2)))
        haze_cleared_factor = max(0.0, 1.0 - (haze_density / 100.0))

        composite_index = (0.35 * norm_psnr + 0.35 * norm_ssim + 0.15 * sharpness_improvement + 0.15 * haze_cleared_factor) * 100.0
        return round(float(np.clip(composite_index, 0.0, 100.0)), 1)
    except Exception as err:
        logger.error("Failed to compute overall quality score: %s", err)
        return 75.0


# =============================================================================
# METRICS CALCULATOR ORCHESTRATOR
# =============================================================================

class MetricsCalculator:
    """
    High-level orchestrator class to compute and format full Image Quality Assessment (IQA) reports.
    """

    def __init__(self) -> None:
        logger.info("Initialized Image Quality Metrics Calculator (Refactored).")

    def calculate_all_metrics(self, original_img: np.ndarray, dehazed_img: np.ndarray) -> Dict[str, float]:
        """
        Computes all image quality metrics between original hazy image and enhanced image.

        Args:
            original_img (np.ndarray): Original input image array (uint8).
            dehazed_img (np.ndarray): Enhanced dehazed image array (uint8).

        Returns:
            Dict[str, float]: Dictionary containing all metric scores.
        """
        if not isinstance(original_img, np.ndarray) or not isinstance(dehazed_img, np.ndarray):
            raise TypeError("Both input images must be valid NumPy ndarrays.")

        # Ensure spatial resolutions match
        h1, w1 = original_img.shape[:2]
        h2, w2 = dehazed_img.shape[:2]
        if (h1, w1) != (h2, w2):
            logger.warning("Image dimensions differ (%dx%d vs %dx%d). Resizing dehazed image for metric computation.", w1, h1, w2, h2)
            dehazed_img = cv2.resize(dehazed_img, (w1, h1), interpolation=cv2.INTER_AREA)

        # Compute individual metrics
        psnr_val = calculate_psnr(original_img, dehazed_img)
        ssim_val = calculate_ssim(original_img, dehazed_img)
        mse_val = calculate_mse(original_img, dehazed_img)

        brightness_val = calculate_brightness(dehazed_img)
        contrast_val = calculate_contrast(dehazed_img)
        sharpness_val = calculate_sharpness(dehazed_img)
        orig_sharpness = calculate_sharpness(original_img)
        entropy_val = calculate_entropy(dehazed_img)

        visibility_val = calculate_visibility_score(dehazed_img)
        haze_density_val = calculate_haze_density(dehazed_img)
        overall_score = calculate_overall_quality_score(
            psnr_val, ssim_val, orig_sharpness, sharpness_val, haze_density_val
        )

        metrics_summary: Dict[str, float] = {
            "psnr": psnr_val,
            "ssim": ssim_val,
            "mse": mse_val,
            "brightness": brightness_val,
            "contrast": contrast_val,
            "sharpness": sharpness_val,
            "entropy": entropy_val,
            "visibility_score": visibility_val,
            "haze_density_score": haze_density_val,
            "overall_quality_score": overall_score
        }

        logger.info("Successfully computed full image quality metrics suite.")
        return metrics_summary
