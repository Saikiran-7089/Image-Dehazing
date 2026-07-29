"""
===============================================================================
AI-Based Single Image Dehazing System
Inference Pipeline Engine (inference.py) [Optimized Version]
===============================================================================

This module implements the central `DehazeInferenceEngine` for running model
inference across DehazeFormer, AOD-Net, and Dark Channel Prior. Optimizations include:
- `torch.inference_mode()` for zero-autograd memory overhead.
- True batched parallel forward passes in `process_batch()`.
- CUDNN benchmarking for optimal GPU kernel selection.
- In-place memory optimizations and fast tensor conversions.

Author: Senior Computer Vision & AI Engineer
License: MIT
===============================================================================
"""

import time
import logging
from typing import Tuple, Dict, Any, Union, Optional, List

import cv2
import numpy as np
from PIL import Image
import torch

import config
import utils
from models.loader import get_model
from image_processing import ImageProcessor

# Initialize module logger
logger: logging.Logger = logging.getLogger("InferenceEngine")

# Enable CUDNN auto-tuner for optimal GPU convolution kernels
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


class DehazeInferenceEngine:
    """
    Production-ready High-Performance Inference Engine for dehazing.

    Attributes:
        device (str): Execution device target ('cuda' or 'cpu').
        loaded_models (Dict[str, Any]): In-memory cache of initialized models.
        image_processor (ImageProcessor): Helper instance for post-enhancement.
    """

    def __init__(self, device: Optional[str] = None) -> None:
        """
        Initializes the inference engine and hardware acceleration target.

        Args:
            device (Optional[str]): Execution device ('cuda' or 'cpu'). Auto-detected if None.
        """
        if device is None:
            self.device: str = config.DEVICE
        else:
            self.device: str = device

        self.loaded_models: Dict[str, Any] = {}
        self.image_processor: ImageProcessor = ImageProcessor()
        logger.info("DehazeInferenceEngine initialized on target device '%s' (Optimized).", self.device)

    def load_model(self, model_name: str) -> Any:
        """
        Retrieves a cached model or instantiates it using the Model Factory.

        Args:
            model_name (str): Name of the model ('DehazeFormer', 'AOD-Net', 'Dark Channel Prior').

        Returns:
            Any: Instantiated model object.
        """
        if model_name not in self.loaded_models:
            logger.info("Loading model '%s' into inference engine cache...", model_name)
            model_obj = get_model(model_name=model_name, device=self.device, pretrained=True)
            self.loaded_models[model_name] = model_obj

        return self.loaded_models[model_name]

    def preprocess(
        self,
        image_input: Union[np.ndarray, Image.Image]
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Preprocesses and normalizes input image into engine format.

        Args:
            image_input (Union[np.ndarray, Image.Image]): PIL Image or OpenCV BGR array.

        Returns:
            Tuple[np.ndarray, Tuple[int, int]]: (Preprocessed BGR array, (Original Height, Original Width)).
        """
        if isinstance(image_input, Image.Image):
            img_bgr = utils.pil_to_opencv(image_input)
        elif isinstance(image_input, np.ndarray):
            img_bgr = image_input
        else:
            raise TypeError("Input image must be a PIL Image or NumPy ndarray.")

        if img_bgr.size == 0:
            raise ValueError("Input image array is empty.")

        orig_h, orig_w = img_bgr.shape[:2]
        # Fast aspect ratio resize if exceeding limits
        if max(orig_h, orig_w) > config.MAX_IMAGE_SIZE[0]:
            img_bgr = utils.resize_image_aspect_ratio(img_bgr, max_dim=config.MAX_IMAGE_SIZE[0])

        return img_bgr, (orig_h, orig_w)

    def infer(self, model: Any, preprocessed_data: np.ndarray, model_name: str) -> Any:
        """
        Executes model forward pass using torch.inference_mode() for zero autograd overhead.

        Args:
            model (Any): Instantiated PyTorch model or DCP object.
            preprocessed_data (np.ndarray): Preprocessed OpenCV BGR image array.
            model_name (str): Target model name.

        Returns:
            Any: Raw model output (NumPy array or PyTorch Tensor).
        """
        if model_name == config.MODEL_DCP:
            return model.dehaze(preprocessed_data)
        else:
            # Convert OpenCV BGR to RGB PyTorch Tensor (1, 3, H, W) [0.0 - 1.0]
            img_rgb = cv2.cvtColor(preprocessed_data, cv2.COLOR_BGR2RGB)
            input_tensor = utils.numpy_to_tensor(img_rgb, device=self.device)
            padded_tensor, _ = utils.pad_to_multiple(input_tensor, multiple=16)

            # Ultra-fast inference mode with zero autograd tracking
            with torch.inference_mode():
                if self.device == "cuda" and config.USE_HALF_PRECISION:
                    with torch.cuda.amp.autocast():
                        output_tensor = model(padded_tensor)
                else:
                    output_tensor = model(padded_tensor)

            return output_tensor

    def postprocess(
        self,
        raw_output: Any,
        orig_shape: Tuple[int, int],
        model_name: str
    ) -> np.ndarray:
        """
        Postprocesses raw model output into OpenCV BGR uint8 format.

        Args:
            raw_output (Any): Output from infer().
            orig_shape (Tuple[int, int]): Original image dimensions (h, w).
            model_name (str): Target model name.

        Returns:
            np.ndarray: Enhanced dehazed BGR uint8 image array.
        """
        orig_h, orig_w = orig_shape

        if model_name == config.MODEL_DCP:
            output_bgr = raw_output
        else:
            # Slice padded tensor back to original shape
            output_tensor = raw_output[:, :, :orig_h, :orig_w]
            dehazed_rgb = utils.tensor_to_numpy(output_tensor)
            output_bgr = cv2.cvtColor(dehazed_rgb, cv2.COLOR_RGB2BGR)

        return output_bgr

    def process_image(
        self,
        image_input: Union[np.ndarray, Image.Image],
        model_name: str = config.DEFAULT_MODEL,
        enhancement_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Full single image dehazing pipeline with performance timing.

        Args:
            image_input (Union[np.ndarray, Image.Image]): Input image.
            model_name (str): Model name to execute.
            enhancement_params (Optional[Dict[str, Any]]): Interactive enhancement sliders.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: (Dehazed BGR uint8 image, Telemetry dictionary).
        """
        model = self.load_model(model_name)
        img_bgr, (orig_h, orig_w) = self.preprocess(image_input)

        start_time = time.perf_counter()
        memory_mb = 0.0

        try:
            raw_output = self.infer(model, img_bgr, model_name)
            dehazed_bgr = self.postprocess(raw_output, (orig_h, orig_w), model_name)

            if self.device == "cuda" and torch.cuda.is_available():
                memory_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)

            if enhancement_params:
                dehazed_bgr = self._apply_post_enhancements(dehazed_bgr, enhancement_params)

            end_time = time.perf_counter()
            elapsed_sec = round(end_time - start_time, 4)
            fps = round(1.0 / max(0.001, elapsed_sec), 1)

            telemetry: Dict[str, Any] = {
                "model_name": model_name,
                "device_used": self.device.upper(),
                "execution_time_sec": elapsed_sec,
                "fps": fps,
                "memory_mb": memory_mb,
                "input_resolution": f"{orig_w} x {orig_h}",
                "output_resolution": f"{dehazed_bgr.shape[1]} x {dehazed_bgr.shape[0]}"
            }

            return dehazed_bgr, telemetry

        except Exception as err:
            logger.error("Error during dehazing process_image pass: %s", err)
            raise RuntimeError(f"Inference process_image error: {err}") from err

    def process_batch(
        self,
        image_list: List[Union[np.ndarray, Image.Image]],
        model_name: str = config.DEFAULT_MODEL
    ) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Executes batched parallel forward passes for PyTorch models.

        Args:
            image_list (List[Union[np.ndarray, Image.Image]]): List of input images.
            model_name (str): Model name to execute.

        Returns:
            List[Tuple[np.ndarray, Dict[str, Any]]]: List of (dehazed_image, telemetry) tuples.
        """
        logger.info("Executing batch processing for %d images (Optimized Batched)...", len(image_list))
        if len(image_list) == 0:
            return []

        # For DCP, fall back to sequential loop
        if model_name == config.MODEL_DCP:
            return [self.process_image(img, model_name=model_name) for img in image_list]

        model = self.load_model(model_name)
        preprocessed_items = [self.preprocess(img) for img in image_list]

        # Standardize spatial dimensions to max batch dimensions
        max_h = max(item[0].shape[0] for item in preprocessed_items)
        max_w = max(item[0].shape[1] for item in preprocessed_items)

        batch_tensors = []
        for img_bgr, _ in preprocessed_items:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            tensor = utils.numpy_to_tensor(img_rgb, device=self.device)
            padded, _ = utils.pad_to_multiple(tensor, multiple=16)

            # Resize/pad to uniform batch shape if necessary
            if padded.shape[2] != max_h or padded.shape[3] != max_w:
                padded = torch.nn.functional.pad(padded, (0, max(0, max_w - padded.shape[3]), 0, max(0, max_h - padded.shape[2])))
            batch_tensors.append(padded)

        stacked_batch = torch.cat(batch_tensors, dim=0)

        t_start = time.perf_counter()
        with torch.inference_mode():
            batch_output = model(stacked_batch)
        t_elapsed = round((time.perf_counter() - t_start) / len(image_list), 4)

        results = []
        for idx, (_, orig_shape) in enumerate(preprocessed_items):
            out_tensor = batch_output[idx:idx+1]
            dehazed_bgr = self.postprocess(out_tensor, orig_shape, model_name)
            telemetry = {
                "model_name": model_name,
                "device_used": self.device.upper(),
                "execution_time_sec": t_elapsed,
                "fps": round(1.0 / max(0.001, t_elapsed), 1),
                "memory_mb": 0.0,
                "input_resolution": f"{orig_shape[1]} x {orig_shape[0]}",
                "output_resolution": f"{dehazed_bgr.shape[1]} x {dehazed_bgr.shape[0]}"
            }
            results.append((dehazed_bgr, telemetry))

        return results

    def dehaze_image(
        self,
        image_input: Union[np.ndarray, Image.Image],
        model_name: str = config.DEFAULT_MODEL,
        enhancement_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Alias for process_image to maintain backward compatibility."""
        return self.process_image(image_input, model_name=model_name, enhancement_params=enhancement_params)

    def _apply_post_enhancements(
        self,
        img_bgr: np.ndarray,
        params: Dict[str, Any]
    ) -> np.ndarray:
        """Applies interactive enhancement controls requested from UI sliders."""
        enhanced = img_bgr

        if params.get("use_clahe", False):
            clip = params.get("clahe_clip", config.DEFAULT_CLAHE_CLIP_LIMIT)
            enhanced = self.image_processor.apply_clahe(enhanced, clip_limit=clip)

        if params.get("use_sharpen", False):
            factor = params.get("sharpen_factor", config.DEFAULT_SHARPNESS_FACTOR)
            enhanced = self.image_processor.apply_unsharp_mask(enhanced, amount=factor)

        brightness = params.get("brightness", 1.0)
        contrast = params.get("contrast", 1.0)
        if brightness != 1.0 or contrast != 1.0:
            enhanced = self.image_processor.adjust_brightness_contrast(
                enhanced, brightness=brightness, contrast=contrast
            )

        if params.get("use_denoise", False):
            h_val = params.get("denoise_strength", config.DEFAULT_DENOISE_STRENGTH)
            enhanced = self.image_processor.apply_denoising_nlm(enhanced, h=h_val)

        return enhanced
