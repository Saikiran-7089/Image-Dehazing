"""
===============================================================================
AI-Based Single Image Dehazing System
Dataset Package Initializer (dataset/__init__.py)
===============================================================================

This package contains PyTorch Dataset implementations for RESIDE, O-HAZE,
Dense-Haze, and NH-Haze benchmark datasets.

Exposed Exports:
- DehazeDataset: PyTorch Dataset for paired hazy and clear image samples.

Author: Senior Computer Vision & AI Engineer
License: MIT
===============================================================================
"""

from .dehaze_dataset import DehazeDataset

__all__ = ["DehazeDataset"]
