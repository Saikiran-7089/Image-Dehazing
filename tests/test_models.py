"""
===============================================================================
Unit Tests: AI Dehazing Models (tests/test_models.py)
===============================================================================
"""

import pytest
import torch
import numpy as np

import config
from models.loader import get_model
from models.dehazeformer import DehazeFormer
from models.aodnet import AODNet
from models.dcp import DarkChannelPrior


def test_dehazeformer_forward():
    """Test DehazeFormer forward pass and shape preservation."""
    model = DehazeFormer(embed_dim=36)
    model.eval()
    dummy_input = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == dummy_input.shape, f"Expected {dummy_input.shape}, got {output.shape}"
    assert torch.all(output >= 0.0) and torch.all(output <= 1.0), "Output tensor values must be bounded in [0, 1]"


def test_aodnet_forward():
    """Test AOD-Net forward pass and shape preservation."""
    model = AODNet()
    model.eval()
    dummy_input = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == dummy_input.shape, f"Expected {dummy_input.shape}, got {output.shape}"
    assert torch.all(output >= 0.0) and torch.all(output <= 1.0), "Output tensor values must be bounded in [0, 1]"


def test_dark_channel_prior_dehaze():
    """Test Dark Channel Prior dehaze method on dummy image."""
    dcp = DarkChannelPrior(patch_size=15, omega=0.95)
    dummy_img = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
    output = dcp.dehaze(dummy_img)

    assert output.shape == dummy_img.shape, "DCP output shape must match input shape"
    assert output.dtype == np.uint8, "DCP output dtype must be uint8"


def test_model_loader_factory():
    """Test get_model factory with all supported model names."""
    m1 = get_model(config.MODEL_DEHAZEFORMER, device="cpu", pretrained=False)
    assert isinstance(m1, DehazeFormer)

    m2 = get_model(config.MODEL_AODNET, device="cpu", pretrained=False)
    assert isinstance(m2, AODNet)

    m3 = get_model(config.MODEL_DCP, device="cpu", pretrained=False)
    assert isinstance(m3, DarkChannelPrior)


def test_invalid_model_name():
    """Test that invalid model name raises ValueError."""
    with pytest.raises(ValueError):
        get_model("NonExistentModel")
