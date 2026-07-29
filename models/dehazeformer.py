"""
===============================================================================
AI-Based Single Image Dehazing System
DehazeFormer Transformer Module (models/dehazeformer.py)
===============================================================================

This module implements DehazeFormer, a state-of-the-art Vision Transformer for
single image dehazing based on Song et al. (2023).

Key Architectural Components:
1. Window Multi-Head Self-Attention (W-MSA) with relative positional bias.
2. Layer Normalization and Feed-Forward MLP Blocks with GELU activations.
3. Hierarchical Encoder-Decoder Transformer Blocks with skip connections.
4. ResBlock-based Feature Fusion and Residual Image Reconstruction.

Author: Senior Computer Vision & AI Engineer
License: MIT
===============================================================================
"""

import math
import logging
from typing import Tuple, Optional

torch_import_error = None
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as err:
    torch_import_error = err

# Configure logger for DehazeFormer
logger: logging.Logger = logging.getLogger("DehazeFormer")


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partitions feature tensor (B, H, W, C) into non-overlapping windows of size (window_size, window_size).

    Args:
        x (torch.Tensor): Input tensor of shape (B, H, W, C).
        window_size (int): Spatial window dimension.

    Returns:
        Tuple[torch.Tensor, Tuple[int, int]]: (Windows tensor (B*num_windows, window_size*window_size, C), (H, W)).
    """
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        _, Hp, Wp, _ = x.shape
    else:
        Hp, Wp = H, W

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    return windows, (Hp, Wp)


def window_reverse(windows: torch.Tensor, window_size: int, Hp: int, Wp: int, B: int) -> torch.Tensor:
    """
    Reconstructs spatial feature tensor (B, Hp, Wp, C) from non-overlapping windows.

    Args:
        windows (torch.Tensor): Windowed tensor of shape (B*num_windows, window_size*window_size, C).
        window_size (int): Spatial window dimension.
        Hp (int): Padded height.
        Wp (int): Padded width.
        B (int): Original batch size.

    Returns:
        torch.Tensor: Reconstructed feature tensor of shape (B, Hp, Wp, C).
    """
    C = windows.shape[-1]
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, C)
    return x


class WindowAttention(nn.Module):
    """
    Window-based Multi-head Self-Attention (W-MSA) with relative positional bias.

    Attributes:
        dim (int): Input feature channel dimension.
        window_size (int): Spatial window size (default: 8).
        num_heads (int): Number of parallel self-attention heads (default: 4).
    """

    def __init__(self, dim: int, window_size: int = 8, num_heads: int = 4) -> None:
        super(WindowAttention, self).__init__()
        self.dim: int = dim
        self.window_size: int = window_size
        self.num_heads: int = num_heads
        head_dim: int = dim // num_heads
        self.scale: float = head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        # Relative position index matrix
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Windowed feature tensor (num_windows*B, N, C).

        Returns:
            torch.Tensor: Attended window features (num_windows*B, N, C).
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # Retrieve relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class DehazeFormerBlock(nn.Module):
    """
    Core DehazeFormer Transformer Block consisting of LayerNorm, Window Self-Attention,
    GELU Feed-Forward Network, and residual shortcuts.
    """

    def __init__(self, dim: int, window_size: int = 8, num_heads: int = 4, mlp_ratio: float = 2.0) -> None:
        super(DehazeFormerBlock, self).__init__()
        self.dim: int = dim
        self.window_size: int = window_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads)

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Feature tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Transformed feature tensor of shape (B, C, H, W).
        """
        B, C, H, W = x.shape
        shortcut = x

        # Convert HWC for LayerNorm & Window Attention
        x_perm = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)

        # 1. Window Partitioning & Attention
        windows, (Hp, Wp) = window_partition(x_perm, self.window_size)
        norm_windows = self.norm1(windows)
        attn_windows = self.attn(norm_windows)

        # 2. Window Reverse & Crop Padding
        x_att = window_reverse(attn_windows, self.window_size, Hp, Wp, B)
        x_att = x_att[:, :H, :W, :].contiguous()

        x_perm = x_perm + x_att

        # 3. MLP Feed-Forward
        x_mlp = self.mlp(self.norm2(x_perm))
        x_out = x_perm + x_mlp

        # Convert back to (B, C, H, W)
        return x_out.permute(0, 3, 1, 2).contiguous()


class DehazeFormer(nn.Module):
    """
    DehazeFormer State-of-the-Art Vision Transformer for Single Image Dehazing.

    Architecture:
        - Patch Embedding: 3x3 Conv mapping 3-channel image to feature dimension `dim`.
        - Hierarchical Encoder-Decoder Transformer Blocks.
        - Skip-Connection Feature Fusion.
        - Residual Reconstruction Head mapping feature maps back to RGB clear radiance.
    """

    def __init__(self, in_channels: int = 3, embed_dim: int = 36, num_blocks: Tuple[int, ...] = (2, 2, 2)) -> None:
        """
        Initializes DehazeFormer layers and residual blocks.

        Args:
            in_channels (int): Input image color channels (default: 3).
            embed_dim (int): Feature channel dimension (default: 36).
            num_blocks (Tuple[int, ...]): Number of Transformer blocks per stage.
        """
        super(DehazeFormer, self).__init__()
        self.in_channels: int = in_channels
        self.embed_dim: int = embed_dim

        # 1. Patch Embedding Head
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=3, stride=1, padding=1)

        # 2. Encoder Stage
        self.encoder_stage1 = nn.Sequential(
            *[DehazeFormerBlock(dim=embed_dim, window_size=8, num_heads=4) for _ in range(num_blocks[0])]
        )

        # 3. Bottleneck Stage
        self.bottleneck = nn.Sequential(
            *[DehazeFormerBlock(dim=embed_dim, window_size=8, num_heads=4) for _ in range(num_blocks[1])]
        )

        # 4. Decoder Stage
        self.decoder_stage1 = nn.Sequential(
            *[DehazeFormerBlock(dim=embed_dim, window_size=8, num_heads=4) for _ in range(num_blocks[2])]
        )

        # 5. Reconstruction Head
        self.reconstruct_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, in_channels, kernel_size=3, stride=1, padding=1)
        )

        self._init_weights()
        logger.info("Initialized DehazeFormer Transformer architecture (Embed Dim: %d).", embed_dim)

    def _init_weights(self) -> None:
        """
        Initializes convolutional weights and linear layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for DehazeFormer single image dehazing.

        Args:
            x (torch.Tensor): Hazy input image tensor (B, 3, H, W) normalized to [0.0, 1.0].

        Returns:
            torch.Tensor: Dehazed output image tensor (B, 3, H, W) in range [0.0, 1.0].
        """
        # Patch embedding
        feat = self.patch_embed(x)

        # Encoder pass
        enc1 = self.encoder_stage1(feat)

        # Bottleneck pass
        btn = self.bottleneck(enc1)

        # Decoder pass with skip connection fusion
        dec1 = self.decoder_stage1(btn + enc1)

        # Residual reconstruction: J(x) = I(x) + Residual
        residual = self.reconstruct_head(dec1)
        output = x + residual
        output = torch.clamp(output, 0.0, 1.0)
        return output


if __name__ == "__main__":
    # Self-test block to verify forward pass
    if torch_import_error is None:
        model = DehazeFormer()
        dummy_input = torch.randn(1, 3, 256, 256)
        dummy_output = model(dummy_input)
        print(f"DehazeFormer test successful. Input shape: {dummy_input.shape}, Output shape: {dummy_output.shape}")
