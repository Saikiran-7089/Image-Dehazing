"""
===============================================================================
AI-Based Single Image Dehazing System
AOD-Net Architecture Module (models/aodnet.py)
===============================================================================

This module implements the All-in-One Dehazing Network (AOD-Net) proposed by
Li et al. (ICCV 2017). AOD-Net reformulates the atmospheric scattering model to
directly estimate a unified K-parameter matrix K(x), combining transmission map
and atmospheric light estimation into a single end-to-end trainable CNN.

Mathematical Formulation:
    J(x) = K(x) * I(x) - K(x) + b
where b is a constant bias (default: 1.0) and K(x) is estimated via concatenated
multi-scale convolutional layers.

Author: Senior Computer Vision & AI Engineer
License: MIT
===============================================================================
"""

import logging
from typing import Tuple

torch_import_error = None
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as err:
    torch_import_error = err

# Configure logger for AOD-Net module
logger: logging.Logger = logging.getLogger("AODNet")


class AODNet(nn.Module):
    """
    All-in-One Dehazing Network (AOD-Net) for Single Image Dehazing.

    Architecture:
        - Conv1: 3 -> 3 channels, 1x1 kernel
        - Conv2: 3 -> 3 channels, 3x3 kernel (pad 1)
        - Conv3: 6 -> 3 channels (cat Conv1, Conv2), 5x5 kernel (pad 2)
        - Conv4: 6 -> 3 channels (cat Conv2, Conv3), 7x7 kernel (pad 3)
        - Conv5: 12 -> 3 channels (cat Conv1, Conv2, Conv3, Conv4), 3x3 kernel (pad 1)
        - Output Reconstruction: J(x) = ReLU(K(x) * I(x) - K(x) + b)

    Attributes:
        b (float): Constant atmospheric scattering bias parameter (default: 1.0).
    """

    def __init__(self, b: float = 1.0) -> None:
        """
        Initializes AOD-Net convolutional layers and weight initializations.

        Args:
            b (float): Atmospheric scattering bias constant.
        """
        super(AODNet, self).__init__()
        self.b: float = b

        # Convolutional layers for multi-scale K-estimation
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=7, stride=1, padding=3)
        self.conv5 = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU(inplace=True)

        # Initialize network weights
        self._init_weights()
        logger.info("Initialized AOD-Net CNN architecture (Parameters: ~1.8K).")

    def _init_weights(self) -> None:
        """
        Applies Kaiming / MSRA normal weight initialization across convolutional layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for AOD-Net single image dehazing.

        Args:
            x (torch.Tensor): Hazy input image tensor (B, 3, H, W) normalized to [0.0, 1.0].

        Returns:
            torch.Tensor: Dehazed output image tensor (B, 3, H, W) in range [0.0, 1.0].
        """
        # Feature extraction and multi-scale concatenation
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))

        # Concatenate Conv1 and Conv2 outputs
        cat1 = torch.cat((x1, x2), dim=1)
        x3 = self.relu(self.conv3(cat1))

        # Concatenate Conv2 and Conv3 outputs
        cat2 = torch.cat((x2, x3), dim=1)
        x4 = self.relu(self.conv4(cat2))

        # Concatenate Conv1, Conv2, Conv3, Conv4 outputs
        cat3 = torch.cat((x1, x2, x3, x4), dim=1)
        K = self.relu(self.conv5(cat3))

        # Atmospheric model formula: J(x) = K(x) * I(x) - K(x) + b
        output = K * x - K + self.b
        output = torch.clamp(output, 0.0, 1.0)
        return output


if __name__ == "__main__":
    # Self-test block to verify forward pass
    if torch_import_error is None:
        model = AODNet()
        dummy_input = torch.randn(1, 3, 256, 256)
        dummy_output = model(dummy_input)
        print(f"AODNet test successful. Input shape: {dummy_input.shape}, Output shape: {dummy_output.shape}")
