"""
===============================================================================
AI-Based Single Image Dehazing System
Models Package Initializer (models/__init__.py)
===============================================================================

This package contains state-of-the-art Deep Learning models (DehazeFormer, AOD-Net)
and traditional Computer Vision algorithms (Dark Channel Prior) for single image dehazing.

Exposed API Functions and Classes:
- get_model: Unified factory function to retrieve initialized model instances.
- DehazeFormer: Transformer-based image dehazing network architecture.
- AODNet: All-in-One Dehazing CNN network architecture.
- DarkChannelPrior: Classical Dark Channel Prior (DCP) dehazing pipeline.

Author: Senior Computer Vision & AI Engineer
License: MIT
===============================================================================
"""

import logging
from typing import List

# Configure logger for models package
logger: logging.Logger = logging.getLogger("DehazeModels")

# Import model loader factory and model classes
try:
    from .loader import get_model
except ImportError as _loader_err:
    logger.debug("Loader module deferred load: %s", _loader_err)

try:
    from .dehazeformer import DehazeFormer
except ImportError as _dehazeformer_err:
    logger.debug("DehazeFormer module deferred load: %s", _dehazeformer_err)

try:
    from .aodnet import AODNet
except ImportError as _aodnet_err:
    logger.debug("AODNet module deferred load: %s", _aodnet_err)

try:
    from .dcp import DarkChannelPrior
except ImportError as _dcp_err:
    logger.debug("DarkChannelPrior module deferred load: %s", _dcp_err)


__all__: List[str] = [
    "get_model",
    "DehazeFormer",
    "AODNet",
    "DarkChannelPrior"
]
