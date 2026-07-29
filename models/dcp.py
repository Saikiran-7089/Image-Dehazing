"""
===============================================================================
AI-Based Single Image Dehazing System
Dark Channel Prior Dehazing Module (models/dcp.py) [Research-Grade Edition]
===============================================================================

This module implements the classical Dark Channel Prior (DCP) single image dehazing
algorithm by He et al. (IEEE TPAMI 2010) with physical atmospheric scattering physics:
- Dark Channel: J_dark(x) = min_{y in Omega(x)} (min_{c in {r,g,b}} I_c(y))
- Atmospheric Light: A estimated from the top 0.1% brightest dark channel pixels
- Transmission Map: t(x) = 1 - omega * min_c (I_c(x) / A_c)
- Edge-Preserving Guided Filter refinement
- Scene Radiance Recovery: J(x) = (I(x) - A) / max(t(x), t0) + A

Author: Senior Computer Vision & AI Engineer / CV Researcher
License: MIT
===============================================================================
"""

import logging
from typing import Tuple, Optional

import cv2
import numpy as np

# Configure logger for Dark Channel Prior module
logger: logging.Logger = logging.getLogger("DarkChannelPrior")


class DarkChannelPrior:
    """
    Classical Dark Channel Prior (DCP) Single Image Dehazing Pipeline.

    Hyperparameters:
        patch_size (int): Local spatial window Omega(x) size (default: 15).
        omega (float): Amount of haze retained for depth perception (default: 0.95).
        t0 (float): Lower bound for transmission map to prevent division by zero (default: 0.1).
        radius (int): Radius of guided filter for edge-preserving smoothing (default: 60).
        eps (float): Regularization parameter for guided filter (default: 0.001).
    """

    def __init__(
        self,
        patch_size: int = 15,
        omega: float = 0.95,
        t0: float = 0.1,
        radius: int = 60,
        eps: float = 0.001
    ) -> None:
        """Initializes Dark Channel Prior algorithm parameters and pre-allocates kernels."""
        self.patch_size: int = patch_size
        self.omega: float = omega
        self.t0: float = t0
        self.radius: int = radius
        self.eps: float = eps

        # Pre-allocate structuring element for spatial minimum erosion
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))

    def compute_dark_channel(self, img_float: np.ndarray, patch_size: Optional[int] = None) -> np.ndarray:
        """
        Computes dark channel map: J_dark(x) = min_{y in Omega(x)} (min_{c} I_c(y)).

        Args:
            img_float (np.ndarray): Image array normalized in range [0.0, 1.0].
            patch_size (Optional[int]): Patch radius.

        Returns:
            np.ndarray: 2D dark channel map (H, W).
        """
        if patch_size is None or patch_size == self.patch_size:
            kernel = self._kernel
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))

        min_channel = np.min(img_float, axis=2)
        dark_channel = cv2.erode(min_channel, kernel)
        return dark_channel

    def estimate_atmospheric_light(
        self,
        img_float: np.ndarray,
        dark_channel: np.ndarray,
        top_percent: float = 0.001
    ) -> np.ndarray:
        """
        Estimates atmospheric light vector A (R, G, B) from top 0.1% brightest dark channel pixels.

        Args:
            img_float (np.ndarray): Image array normalized in range [0.0, 1.0].
            dark_channel (np.ndarray): Dark channel map (H, W).
            top_percent (float): Sampling ratio.

        Returns:
            np.ndarray: Atmospheric light vector A (3,) bounded in [0.1, 1.0].
        """
        h, w = dark_channel.shape
        num_pixels = h * w
        num_brightest = max(1, int(num_pixels * top_percent))

        dark_flat = dark_channel.reshape(-1)
        img_flat = img_float.reshape(-1, 3)

        indices = np.argpartition(dark_flat, -num_brightest)[-num_brightest:]
        candidate_pixels = img_flat[indices]

        brightest_idx = np.argmax(np.sum(candidate_pixels, axis=1))
        A = candidate_pixels[brightest_idx]

        # Bound atmospheric light to prevent zero scaling
        A = np.clip(A, 0.1, 1.0)
        return A

    def estimate_transmission(
        self,
        img_float: np.ndarray,
        A: np.ndarray,
        omega: Optional[float] = None,
        patch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Estimates raw transmission map: t~(x) = 1 - omega * dark_channel(I(x) / A).

        Args:
            img_float (np.ndarray): Hazy image [0.0 - 1.0].
            A (np.ndarray): Atmospheric light vector (3,).
            omega (Optional[float]): Haze retention factor.
            patch_size (Optional[int]): Patch size.

        Returns:
            np.ndarray: Raw transmission map (H, W).
        """
        omg = omega if omega is not None else self.omega

        inv_A = 1.0 / np.maximum(A, 0.001)
        normalized_img = img_float * inv_A

        dark_norm = self.compute_dark_channel(normalized_img, patch_size=patch_size)
        raw_transmission = 1.0 - omg * dark_norm
        return np.clip(raw_transmission, 0.0, 1.0)

    def refine_transmission(
        self,
        img_gray: np.ndarray,
        transmission: np.ndarray,
        radius: Optional[int] = None,
        eps: Optional[float] = None
    ) -> np.ndarray:
        """Refines transmission map using Guided Image Filtering."""
        r = radius if radius is not None else self.radius
        e = eps if eps is not None else self.eps

        try:
            if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
                guidance_uint8 = (img_gray * 255.0).astype(np.uint8)
                transmission_float32 = transmission.astype(np.float32)
                refined = cv2.ximgproc.guidedFilter(
                    guide=guidance_uint8,
                    src=transmission_float32,
                    radius=r,
                    eps=e
                )
                return np.clip(refined, 0.0, 1.0)
        except Exception:
            pass

        return self._custom_guided_filter(guide=img_gray, src=transmission, radius=r, eps=e)

    def _custom_guided_filter(
        self,
        guide: np.ndarray,
        src: np.ndarray,
        radius: int,
        eps: float
    ) -> np.ndarray:
        """Fast implementation of He et al. Guided Filter."""
        ksize = (2 * radius + 1, 2 * radius + 1)
        mean_I = cv2.boxFilter(guide, cv2.CV_64F, ksize)
        mean_p = cv2.boxFilter(src, cv2.CV_64F, ksize)
        mean_Ip = cv2.boxFilter(guide * src, cv2.CV_64F, ksize)
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = cv2.boxFilter(guide * guide, cv2.CV_64F, ksize)
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = cv2.boxFilter(a, cv2.CV_64F, ksize)
        mean_b = cv2.boxFilter(b, cv2.CV_64F, ksize)

        q = mean_a * guide + mean_b
        return np.clip(q, 0.0, 1.0)

    def recover_scene_radiance(
        self,
        img_float: np.ndarray,
        transmission: np.ndarray,
        A: np.ndarray,
        t0: Optional[float] = None
    ) -> np.ndarray:
        """Recovers clear scene radiance: J(x) = (I(x) - A) / max(t(x), t0) + A."""
        t_min = t0 if t0 is not None else self.t0
        t_bounded = np.maximum(transmission, t_min)
        t_3ch = np.stack([t_bounded] * 3, axis=-1)

        radiance = (img_float - A) / t_3ch + A
        return np.clip(radiance, 0.0, 1.0)

    def dehaze(self, image: np.ndarray) -> np.ndarray:
        """Main public dehazing pipeline."""
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Input image array is invalid or empty.")

        is_grayscale = len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1)

        if is_grayscale:
            img_3ch = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            img_3ch = image

        if img_3ch.dtype == np.uint8:
            img_float = img_3ch.astype(np.float32) / 255.0
        else:
            img_float = np.clip(img_3ch.astype(np.float32), 0.0, 1.0)

        dark_channel = self.compute_dark_channel(img_float, patch_size=self.patch_size)
        A = self.estimate_atmospheric_light(img_float, dark_channel, top_percent=0.001)
        raw_transmission = self.estimate_transmission(img_float, A, omega=self.omega)

        guidance_gray = cv2.cvtColor((img_float * 255.0).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        refined_transmission = self.refine_transmission(guidance_gray, raw_transmission)

        clear_float = self.recover_scene_radiance(img_float, refined_transmission, A, t0=self.t0)
        clear_uint8 = (clear_float * 255.0).round().astype(np.uint8)

        if is_grayscale:
            clear_uint8 = cv2.cvtColor(clear_uint8, cv2.COLOR_BGR2GRAY)

        return clear_uint8
