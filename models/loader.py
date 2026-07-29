"""
===============================================================================
AI-Based Single Image Dehazing System
Model Factory & Loader Module (models/loader.py)
===============================================================================

This module implements the unified model factory `get_model(model_name)`.
It dynamically instantiates DehazeFormer, AOD-Net, or Dark Channel Prior, handles
automatic pretrained weight loading from the weights/ directory, provides GPU/CPU
device placement, and falls back safely to initial weights if checkpoint files are missing.

Author: Senior Computer Vision & AI Engineer
License: MIT
===============================================================================
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union, Any

import torch
import torch.nn as nn

import config

# Initialize module logger
logger: logging.Logger = logging.getLogger("ModelLoader")


def load_checkpoint_weights(
    model: nn.Module,
    weights_path: Path,
    device: str = "cpu"
) -> bool:
    """
    Safely loads pretrained state dictionary weights into a PyTorch network.

    Args:
        model (nn.Module): Target PyTorch model instance.
        weights_path (Path): Absolute or relative path to the .pth weights checkpoint.
        device (str): Execution device target ('cpu' or 'cuda').

    Returns:
        bool: True if weights loaded cleanly, False if missing or corrupted.
    """
    if not weights_path.exists():
        logger.warning(
            "Pretrained weights file not found at '%s'. Using fallback model initialization.",
            weights_path
        )
        return False

    try:
        checkpoint = torch.load(weights_path, map_location=device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
        logger.info("Successfully loaded pretrained weights from '%s'", weights_path.name)
        return True
    except Exception as err:
        logger.error(
            "Failed to load weights from '%s': %s. Proceeding with fallback weights.",
            weights_path, err
        )
        return False


def get_model(
    model_name: str,
    device: str = "auto",
    pretrained: bool = True,
    weights_path: str | Path | None = None
) -> Any:
    """
    Factory function to retrieve initialized AI dehazing model instances.

    Supported Models:
    - 'DehazeFormer': Transformer-based W-MSA Deep Learning Network
    - 'AOD-Net': All-in-One Dehazing CNN Network
    - 'Dark Channel Prior': Classical Computer Vision baseline algorithm

    Args:
        model_name (str): Exact name of the model to instantiate.
        device (str): Hardware target ('auto', 'cuda', or 'cpu').
        pretrained (bool): Whether to attempt loading pretrained .pth weights.
        weights_path (str | Path | None): Custom path to weight file. If None, checks default weights/.

    Returns:
        Any: Instantiated PyTorch model in eval mode or classical DCP algorithm object.

    Raises:
        ValueError: If model_name is invalid or unsupported.
    """
    # Validate model name
    if model_name not in config.SUPPORTED_MODELS:
        supported_str = ", ".join(config.SUPPORTED_MODELS)
        logger.error("Invalid model name '%s'. Supported models: [%s]", model_name, supported_str)
        raise ValueError(f"Invalid model name '{model_name}'. Must be one of: [{supported_str}]")

    # Resolve execution device
    if device == "auto":
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        target_device = device

    logger.info("Initializing model '%s' on device '%s'...", model_name, target_device)

    # 1. DehazeFormer Model
    if model_name == config.MODEL_DEHAZEFORMER:
        try:
            from .dehazeformer import DehazeFormer
        except ImportError as err:
            logger.error("Could not import DehazeFormer model architecture: %s", err)
            raise ImportError("DehazeFormer model module is unavailable.") from err

        model = DehazeFormer()
        model.to(target_device)
        model.eval()

        if pretrained:
            resolved_path = Path(weights_path) if weights_path else config.WEIGHTS_DIR / config.WEIGHT_FILES[config.MODEL_DEHAZEFORMER]
            load_checkpoint_weights(model, resolved_path, target_device)

        return model

    # 2. AOD-Net Model
    elif model_name == config.MODEL_AODNET:
        try:
            from .aodnet import AODNet
        except ImportError as err:
            logger.error("Could not import AODNet model architecture: %s", err)
            raise ImportError("AODNet model module is unavailable.") from err

        model = AODNet()
        model.to(target_device)
        model.eval()

        if pretrained:
            resolved_path = Path(weights_path) if weights_path else config.WEIGHTS_DIR / config.WEIGHT_FILES[config.MODEL_AODNET]
            load_checkpoint_weights(model, resolved_path, target_device)

        return model

    # 3. Dark Channel Prior Classical Model
    elif model_name == config.MODEL_DCP:
        try:
            from .dcp import DarkChannelPrior
        except ImportError as err:
            logger.error("Could not import DarkChannelPrior algorithm module: %s", err)
            raise ImportError("DarkChannelPrior module is unavailable.") from err

        dcp_alg = DarkChannelPrior(patch_size=15, omega=0.95, radius=60, eps=0.001)
        logger.info("Initialized Dark Channel Prior classical computer vision pipeline.")
        return dcp_alg

    else:
        raise ValueError(f"Unhandled model name '{model_name}'.")
