"""
===============================================================================
AI-Based Single Image Dehazing System
Image Processing & Computer Vision Module (image_processing.py)
===============================================================================

This module provides classical image enhancement algorithms, spatial filters,
adaptive histogram transforms, noise reduction filters, background isolation tools,
and RGB histogram data generators for visualization.

Author: Senior Computer Vision & AI Engineer
License: MIT
===============================================================================
"""

import io
import logging
from typing import Tuple, Dict, Any, Optional, List, Union

import cv2
import numpy as np
from PIL import Image

import config

# Initialize module logger
logger: logging.Logger = logging.getLogger("ImageProcessing")


class ImageProcessor:
    """
    Production-grade Image Processing engine providing classical Computer Vision
    transforms, enhancement filters, histogram generators, and noise reduction.
    """

    def __init__(self) -> None:
        logger.info("Initialized ImageProcessor engine.")

    # =========================================================================
    # HISTOGRAM & CONTRAST ENHANCEMENTS
    # =========================================================================

    def apply_clahe(
        self,
        img: np.ndarray,
        clip_limit: float = config.DEFAULT_CLAHE_CLIP_LIMIT,
        tile_grid_size: Tuple[int, int] = config.DEFAULT_CLAHE_TILE_GRID
    ) -> np.ndarray:
        """
        Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) in LAB color space.

        Args:
            img (np.ndarray): Input image array (BGR uint8).
            clip_limit (float): Threshold for contrast limiting.
            tile_grid_size (Tuple[int, int]): Size of grid for histogram equalization.

        Returns:
            np.ndarray: Enhanced image array.
        """
        try:
            if len(img.shape) == 2:
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
                return clahe.apply(img)

            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            cl = clahe.apply(l)
            merged_lab = cv2.merge((cl, a, b))
            return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
        except Exception as err:
            logger.error("Error applying CLAHE: %s", err)
            return img

    def apply_histogram_equalization(self, img: np.ndarray) -> np.ndarray:
        """
        Applies global Histogram Equalization to the luminance channel.

        Args:
            img (np.ndarray): Input image array (BGR uint8).

        Returns:
            np.ndarray: Equalized image array.
        """
        try:
            if len(img.shape) == 2:
                return cv2.equalizeHist(img)

            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            y_eq = cv2.equalizeHist(y)
            return cv2.cvtColor(cv2.merge((y_eq, cr, cb)), cv2.COLOR_YCrCb2BGR)
        except Exception as err:
            logger.error("Error applying histogram equalization: %s", err)
            return img

    def apply_gamma_correction(self, img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """
        Applies non-linear Gamma Correction to adjust luminance curve.

        Args:
            img (np.ndarray): Input image array.
            gamma (float): Gamma value (>1.0 brightens shadows, <1.0 darkens).

        Returns:
            np.ndarray: Gamma corrected image.
        """
        try:
            if gamma <= 0:
                gamma = 1.0
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
            return cv2.LUT(img, table)
        except Exception as err:
            logger.error("Error applying gamma correction: %s", err)
            return img

    def adjust_brightness_contrast(
        self,
        img: np.ndarray,
        brightness: float = 1.0,
        contrast: float = 1.0
    ) -> np.ndarray:
        """
        Adjusts image brightness and contrast scale factors.

        Args:
            img (np.ndarray): Input image array (uint8).
            brightness (float): Brightness multiplier factor (1.0 = no change).
            contrast (float): Contrast multiplier factor (1.0 = no change).

        Returns:
            np.ndarray: Adjusted image array.
        """
        try:
            # Formula: Output = contrast * Input + (brightness - 1.0) * 128
            beta = (brightness - 1.0) * 128.0
            adjusted = cv2.convertScaleAbs(img, alpha=contrast, beta=beta)
            return adjusted
        except Exception as err:
            logger.error("Error adjusting brightness/contrast: %s", err)
            return img

    # =========================================================================
    # SHARPNESS & SPATIAL FILTERS
    # =========================================================================

    def apply_unsharp_mask(
        self,
        img: np.ndarray,
        kernel_size: Tuple[int, int] = (5, 5),
        sigma: float = 1.0,
        amount: float = config.DEFAULT_SHARPNESS_FACTOR,
        threshold: int = 0
    ) -> np.ndarray:
        """
        Applies Unsharp Masking filter for edge sharpening.

        Args:
            img (np.ndarray): Input image array.
            kernel_size (Tuple[int, int]): Gaussian blur kernel size.
            sigma (float): Gaussian blur standard deviation.
            amount (float): Sharpening strength factor.
            threshold (int): Minimum intensity difference threshold.

        Returns:
            np.ndarray: Sharpened image.
        """
        try:
            blurred = cv2.GaussianBlur(img, kernel_size, sigma)
            sharpened = float(amount + 1.0) * img.astype(np.float32) - float(amount) * blurred.astype(np.float32)
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

            if threshold > 0:
                low_contrast_mask = np.abs(img.astype(np.int32) - blurred.astype(np.int32)) < threshold
                np.copyto(sharpened, img, where=low_contrast_mask)
            return sharpened
        except Exception as err:
            logger.error("Error applying unsharp mask: %s", err)
            return img

    def apply_laplacian_sharpening(self, img: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Applies Laplacian high-pass filter sharpening.

        Args:
            img (np.ndarray): Input image array.
            alpha (float): Sharpening blend factor.

        Returns:
            np.ndarray: Sharpened image.
        """
        try:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
            sharpened = cv2.filter2D(img, -1, kernel)
            blended = cv2.addWeighted(img, 1.0 - alpha, sharpened, alpha, 0)
            return blended
        except Exception as err:
            logger.error("Error applying Laplacian sharpening: %s", err)
            return img

    # =========================================================================
    # NOISE REDUCTION & BLUR FILTERS
    # =========================================================================

    def apply_gaussian_blur(
        self,
        img: np.ndarray,
        kernel_size: Tuple[int, int] = (5, 5),
        sigma: float = 1.0
    ) -> np.ndarray:
        """Applies Gaussian Smoothing Filter."""
        try:
            return cv2.GaussianBlur(img, kernel_size, sigma)
        except Exception as err:
            logger.error("Error applying Gaussian blur: %s", err)
            return img

    def apply_median_blur(self, img: np.ndarray, ksize: int = 5) -> np.ndarray:
        """Applies Median Filter for salt-and-pepper noise removal."""
        try:
            k = ksize if ksize % 2 == 1 else ksize + 1
            return cv2.medianBlur(img, k)
        except Exception as err:
            logger.error("Error applying median blur: %s", err)
            return img

    def apply_bilateral_filter(
        self,
        img: np.ndarray,
        d: int = 9,
        sigma_color: float = 75.0,
        sigma_space: float = 75.0
    ) -> np.ndarray:
        """Applies Edge-Preserving Bilateral Filter."""
        try:
            return cv2.bilateralFilter(img, d, sigma_color, sigma_space)
        except Exception as err:
            logger.error("Error applying bilateral filter: %s", err)
            return img

    def apply_denoising_nlm(
        self,
        img: np.ndarray,
        h: float = config.DEFAULT_DENOISE_STRENGTH,
        h_color: float = config.DEFAULT_DENOISE_STRENGTH
    ) -> np.ndarray:
        """Applies Fast Non-Local Means Denoising."""
        try:
            if len(img.shape) == 2:
                return cv2.fastNlMeansDenoising(img, None, h, 7, 21)
            return cv2.fastNlMeansDenoisingColored(img, None, h, h_color, 7, 21)
        except Exception as err:
            logger.error("Error applying NLM denoising: %s", err)
            return img

    # =========================================================================
    # BACKGROUND ISOLATION & HISTOGRAM GENERATION
    # =========================================================================

    def isolate_background_grabcut(
        self,
        img: np.ndarray,
        rect: Optional[Tuple[int, int, int, int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies OpenCV GrabCut interactive foreground/background segmentation.

        Args:
            img (np.ndarray): Input BGR image array.
            rect (Optional[Tuple[int, int, int, int]]): Foreground bounding box (x, y, w, h).

        Returns:
            Tuple[np.ndarray, np.ndarray]: (Segmented Foreground Image, Binary Mask).
        """
        try:
            h, w = img.shape[:2]
            if rect is None:
                # Default centered bounding rectangle
                margin_w, margin_h = int(w * 0.05), int(h * 0.05)
                rect = (margin_w, margin_h, w - 2 * margin_w, h - 2 * margin_h)

            mask = np.zeros((h, w), np.uint8)
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)

            cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            binary_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
            fg_segmented = img * binary_mask[:, :, np.newaxis]

            return fg_segmented, binary_mask * 255
        except Exception as err:
            logger.error("Error performing GrabCut background isolation: %s", err)
            return img, np.ones(img.shape[:2], dtype=np.uint8) * 255

    def generate_plotly_histogram_data(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Computes 256-bin RGB pixel intensity frequencies for interactive Plotly charts.

        Args:
            img_bgr (np.ndarray): Input BGR/RGB image array.

        Returns:
            Dict[str, Any]: Dictionary containing x-axis bins [0..255] and r, g, b channel lists.
        """
        try:
            x_bins = list(range(256))
            if len(img_bgr.shape) == 2:
                hist_g = cv2.calcHist([img_bgr], [0], None, [256], [0, 256]).flatten().tolist()
                return {"x": x_bins, "r": hist_g, "g": hist_g, "b": hist_g}

            # Convert BGR to RGB for correct channel mapping
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            hist_r = cv2.calcHist([img_rgb], [0], None, [256], [0, 256]).flatten().tolist()
            hist_g = cv2.calcHist([img_rgb], [1], None, [256], [0, 256]).flatten().tolist()
            hist_b = cv2.calcHist([img_rgb], [2], None, [256], [0, 256]).flatten().tolist()

            return {
                "x": x_bins,
                "r": hist_r,
                "g": hist_g,
                "b": hist_b
            }
        except Exception as err:
            logger.error("Error generating Plotly histogram data: %s", err)
            dummy = [0] * 256
            return {"x": list(range(256)), "r": dummy, "g": dummy, "b": dummy}

    def estimate_compression_ratio(
        self,
        img_bgr: np.ndarray,
        format_ext: str = ".jpg",
        quality: int = 85
    ) -> Dict[str, Any]:
        """
        Estimates file size savings when compressing the enhanced image.

        Args:
            img_bgr (np.ndarray): Input image array.
            format_ext (str): Target format extension ('.jpg', '.png', '.webp').
            quality (int): Compression quality (1-100).

        Returns:
            Dict[str, Any]: Dictionary with raw size, compressed size, and compression ratio.
        """
        try:
            raw_bytes = img_bgr.nbytes
            is_success, buffer = cv2.imencode(format_ext, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not is_success:
                raise ValueError("cv2.imencode failed.")

            compressed_bytes = len(buffer)
            ratio = round(raw_bytes / max(1, compressed_bytes), 2)
            saving_pct = round((1.0 - (compressed_bytes / raw_bytes)) * 100.0, 1)

            return {
                "raw_bytes": raw_bytes,
                "compressed_bytes": compressed_bytes,
                "compression_ratio": f"{ratio}:1",
                "savings_percent": f"{saving_pct}%"
            }
        except Exception as err:
            logger.error("Error estimating compression ratio: %s", err)
            return {"raw_bytes": 0, "compressed_bytes": 0, "compression_ratio": "1:1", "savings_percent": "0%"}
