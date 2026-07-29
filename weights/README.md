# Model Pretrained Weights Store

This directory stores pretrained PyTorch checkpoint files (`.pth`):

- `dehazeformer.pth`: Pretrained weights for DehazeFormer Transformer architecture.
- `aodnet.pth`: Pretrained weights for AOD-Net CNN architecture.

## Automatic Fallback System
If checkpoint files are not found in this directory, the system automatically uses smart default initialization (Kaiming Normal weights) and logs an informative warning. The web application and evaluation scripts will **never crash** even in the absence of external checkpoint files.

To use custom trained weights, place your `.pth` file here or specify the `--weights` parameter.
