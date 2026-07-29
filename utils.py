"""
===============================================================================
AI-Based Single Image Dehazing System
Utility Helpers & Image Processing Conversions (utils.py)
===============================================================================

This module provides essential image conversion utilities, PyTorch tensor mapping,
image metadata extraction, synthetic haze simulation using the Atmospheric Scattering
Model, sample test image generation, and file validation helpers.

Author: Senior Computer Vision & AI Engineer
License: MIT
===============================================================================
"""

import os
import io
import math
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Union, Optional

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

import config

# Initialize logger for utilities
logger: logging.Logger = logging.getLogger("DehazeUtils")


# =============================================================================
# FORMAT CONVERSIONS (PIL <-> OpenCV <-> PyTorch Tensor)
# =============================================================================

def pil_to_opencv(pil_img: Image.Image) -> np.ndarray:
    """
    Converts a PIL Image object to an OpenCV BGR NumPy array.

    Args:
        pil_img (Image.Image): Input PIL Image.

    Returns:
        np.ndarray: OpenCV BGR image array (uint8).
    """
    try:
        rgb_array = np.array(pil_img.convert("RGB"))
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        return bgr_array
    except Exception as err:
        logger.error("Error converting PIL Image to OpenCV: %s", err)
        raise ValueError(f"Failed to convert PIL image: {err}") from err


def opencv_to_pil(cv_img: np.ndarray) -> Image.Image:
    """
    Converts an OpenCV BGR NumPy array to a PIL Image object.

    Args:
        cv_img (np.ndarray): Input OpenCV BGR image array (uint8).

    Returns:
        Image.Image: Converted RGB PIL Image.
    """
    try:
        if len(cv_img.shape) == 2:
            # Grayscale image
            rgb_array = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        else:
            rgb_array = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_array)
    except Exception as err:
        logger.error("Error converting OpenCV image to PIL: %s", err)
        raise ValueError(f"Failed to convert OpenCV image: {err}") from err


def numpy_to_tensor(img_np: np.ndarray, device: str = config.DEVICE) -> torch.Tensor:
    """
    Converts an image NumPy array (H, W, C) [0-255] or [0.0-1.0] to a PyTorch Tensor (1, C, H, W).

    Args:
        img_np (np.ndarray): Image array in HWC format (RGB or BGR).
        device (str): PyTorch device ('cpu' or 'cuda').

    Returns:
        torch.Tensor: Normalized PyTorch tensor in range [0.0, 1.0] with shape (1, C, H, W).
    """
    try:
        if img_np.dtype == np.uint8:
            img_float = img_np.astype(np.float32) / 255.0
        else:
            img_float = img_np.astype(np.float32)

        # Handle grayscale images
        if len(img_float.shape) == 2:
            img_float = np.expand_dims(img_float, axis=-1)
            img_float = np.repeat(img_float, 3, axis=-1)

        # Transpose from (H, W, C) to (C, H, W)
        tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(device)
        return tensor
    except Exception as err:
        logger.error("Error converting NumPy array to PyTorch Tensor: %s", err)
        raise ValueError(f"Failed to convert NumPy array to tensor: {err}") from err


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a PyTorch Tensor (1, C, H, W) or (C, H, W) in range [0.0, 1.0] to a NumPy array (H, W, C) in [0, 255] uint8.

    Args:
        tensor (torch.Tensor): PyTorch image tensor.

    Returns:
        np.ndarray: OpenCV-compatible uint8 NumPy array in HWC format.
    """
    try:
        if tensor.ndim == 4:
            tensor = tensor.squeeze(0)
        tensor = tensor.detach().cpu().clamp(0.0, 1.0)
        img_np = tensor.permute(1, 2, 0).numpy()
        img_uint8 = (img_np * 255.0).round().astype(np.uint8)
        return img_uint8
    except Exception as err:
        logger.error("Error converting PyTorch Tensor to NumPy: %s", err)
        raise ValueError(f"Failed to convert tensor to NumPy array: {err}") from err


# =============================================================================
# METADATA & RESIZING UTILITIES
# =============================================================================

def get_file_size_str(file_bytes: int) -> str:
    """
    Formats a raw byte count into human-readable size string (KB/MB).

    Args:
        file_bytes (int): File size in bytes.

    Returns:
        str: Formatted string representation (e.g. "1.45 MB").
    """
    if file_bytes <= 0:
        return "0 Bytes"
    units = ["Bytes", "KB", "MB", "GB"]
    i = int(math.floor(math.log(file_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(file_bytes / p, 2)
    return f"{s} {units[i]}"


def extract_image_metadata(img: Union[Image.Image, np.ndarray], file_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Extracts essential spatial and file metadata from an image.

    Args:
        img (Union[Image.Image, np.ndarray]): Input image.
        file_path (Optional[Path]): Path to the original image file (optional).

    Returns:
        Dict[str, Any]: Dictionary containing resolution, aspect ratio, channels, file size, format.
    """
    try:
        if isinstance(img, Image.Image):
            width, height = img.size
            channels = len(img.getbands())
            color_space = img.mode
        elif isinstance(img, np.ndarray):
            height, width = img.shape[:2]
            channels = img.shape[2] if len(img.shape) == 3 else 1
            color_space = "RGB/BGR" if channels == 3 else "Grayscale"
        else:
            raise TypeError("Unsupported image type for metadata extraction.")

        aspect_ratio = round(width / max(1, height), 2)
        file_size_str = "N/A (In Memory)"
        if file_path and os.path.exists(file_path):
            file_size_str = get_file_size_str(os.path.getsize(file_path))

        metadata = {
            "width": width,
            "height": height,
            "resolution": f"{width} x {height}",
            "aspect_ratio": f"{aspect_ratio}:1",
            "channels": channels,
            "color_space": color_space,
            "file_size": file_size_str
        }
        return metadata
    except Exception as err:
        logger.error("Failed to extract metadata: %s", err)
        return {
            "width": 0,
            "height": 0,
            "resolution": "Unknown",
            "aspect_ratio": "1:1",
            "channels": 3,
            "color_space": "RGB",
            "file_size": "Unknown"
        }


def resize_image_aspect_ratio(img_np: np.ndarray, max_dim: int = 1920) -> np.ndarray:
    """
    Resizes an image array maintaining aspect ratio if its max dimension exceeds max_dim.

    Args:
        img_np (np.ndarray): OpenCV input image.
        max_dim (int): Maximum allowable dimension.

    Returns:
        np.ndarray: Resized image array.
    """
    h, w = img_np.shape[:2]
    if max(h, w) <= max_dim:
        return img_np

    if h > w:
        new_h = max_dim
        new_w = int(w * (max_dim / h))
    else:
        new_w = max_dim
        new_h = int(h * (max_dim / w))

    logger.info("Resizing image from (%d, %d) to (%d, %d)", w, h, new_w, new_h)
    return cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_AREA)


def pad_to_multiple(img_tensor: torch.Tensor, multiple: int = 16) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Pads a PyTorch tensor (1, C, H, W) so that height and width are divisible by `multiple`.

    Args:
        img_tensor (torch.Tensor): Tensor with shape (1, C, H, W).
        multiple (int): Divisibility factor (e.g., 16 for CNN/Transformer blocks).

    Returns:
        Tuple[torch.Tensor, Tuple[int, int]]: (Padded Tensor, (Original H, Original W)).
    """
    _, _, h, w = img_tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    if pad_h > 0 or pad_w > 0:
        padded_tensor = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode="reflect")
        return padded_tensor, (h, w)
    return img_tensor, (h, w)


# =============================================================================
# SYNTHETIC HAZE GENERATOR & SAMPLE ASSET INITIALIZER
# =============================================================================

def apply_atmospheric_scattering(
    clean_img_np: np.ndarray,
    beta: float = 1.0,
    atmospheric_light: float = 0.85
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates physical haze on a clean RGB image using the Atmospheric Scattering Model:
        I(x) = J(x) * t(x) + A * (1 - t(x))
    where t(x) = exp(-beta * d(x)) is the medium transmission map.

    Args:
        clean_img_np (np.ndarray): Clean RGB uint8 image array.
        beta (float): Scattering coefficient (controls haze density).
        atmospheric_light (float): Global atmospheric light intensity A [0.0 - 1.0].

    Returns:
        Tuple[np.ndarray, np.ndarray]: (Synthesized Hazy Image uint8, Transmission Map float32).
    """
    try:
        img_float = clean_img_np.astype(np.float32) / 255.0
        h, w = clean_img_np.shape[:2]

        # Generate a synthetic depth map d(x) normalized between 0.1 and 1.0
        x_grid, y_grid = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
        depth_map = 0.5 + 0.5 * np.sin(x_grid * np.pi) * np.cos(y_grid * np.pi * 0.5)

        # Transmission map t(x)
        transmission = np.exp(-beta * depth_map)
        transmission = np.clip(transmission, 0.1, 1.0)
        transmission_3ch = np.stack([transmission] * 3, axis=-1)

        # Global Atmospheric Light A
        A = atmospheric_light

        # Atmospheric Scattering Equation: I(x) = J(x)t(x) + A(1 - t(x))
        hazy_float = img_float * transmission_3ch + A * (1.0 - transmission_3ch)
        hazy_uint8 = (np.clip(hazy_float, 0.0, 1.0) * 255.0).astype(np.uint8)

        return hazy_uint8, transmission
    except Exception as err:
        logger.error("Failed to generate synthetic haze: %s", err)
        return clean_img_np, np.ones(clean_img_np.shape[:2], dtype=np.float32)


def generate_sample_hazy_image(save_path: Path) -> Path:
    """
    Generates a high-quality synthetic landscape image with haze and saves it to disk.

    Args:
        save_path (Path): Path to save the sample image.

    Returns:
        Path: Path where sample image was written.
    """
    h, w = 480, 640
    # Generate synthetic landscape background (sky + mountains)
    x_grid, y_grid = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))

    # Sky gradient (blue to orange-golden)
    r_chan = np.clip((1.0 - y_grid) * 0.2 + 0.7 * y_grid, 0, 1)
    g_chan = np.clip((1.0 - y_grid) * 0.4 + 0.5 * y_grid, 0, 1)
    b_chan = np.clip((1.0 - y_grid) * 0.8 + 0.2 * y_grid, 0, 1)

    landscape = np.stack([r_chan, g_chan, b_chan], axis=-1)

    # Add mountain silhouette
    mountain_line = 0.5 + 0.15 * np.sin(x_grid * 8.0) + 0.05 * np.cos(x_grid * 20.0)
    mountain_mask = y_grid > mountain_line
    landscape[mountain_mask] = [0.15, 0.25, 0.15]  # Dark green forest/mountain

    clean_uint8 = (landscape * 255.0).astype(np.uint8)

    # Apply synthetic haze
    hazy_uint8, _ = apply_atmospheric_scattering(clean_uint8, beta=1.2, atmospheric_light=0.9)

    # Save to disk
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(hazy_uint8).save(save_path, "JPEG", quality=95)
    logger.info("Sample hazy image created at %s", save_path)
    return save_path


def initialize_sample_assets() -> None:
    """
    Ensures sample hazy test images exist in the assets directory for immediate UI demonstration.
    """
    sample_file = config.ASSETS_DIR / "sample_hazy_landscape.jpg"
    if not sample_file.exists():
        generate_sample_hazy_image(sample_file)


# Initialize sample assets on module import
initialize_sample_assets()


# =============================================================================
# FILE VALIDATION HELPERS
# =============================================================================

def validate_image_file(file_obj: Union[bytes, io.BytesIO, Path, str]) -> Tuple[bool, str]:
    """
    Validates if an uploaded or given file object is a valid, uncorrupted image.

    Args:
        file_obj (Union[bytes, io.BytesIO, Path, str]): Image file object or path.

    Returns:
        Tuple[bool, str]: (isValid, Error message if invalid else "OK").
    """
    try:
        if isinstance(file_obj, (str, Path)):
            with Image.open(file_obj) as img:
                img.verify()
        else:
            if isinstance(file_obj, bytes):
                file_obj = io.BytesIO(file_obj)
            with Image.open(file_obj) as img:
                img.verify()
        return True, "OK"
    except Exception as err:
        logger.warning("Image validation failed: %s", err)
        return False, f"Invalid or corrupted image file: {err}"
