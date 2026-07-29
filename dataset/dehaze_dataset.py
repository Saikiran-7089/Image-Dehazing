"""
===============================================================================
AI-Based Single Image Dehazing System
PyTorch Dataset Loader Module (dataset/dehaze_dataset.py)
===============================================================================

This module implements the `DehazeDataset` PyTorch Dataset class for loading paired
hazy and clean ground-truth images from standard Computer Vision benchmarks:
RESIDE (ITS/OTS), O-HAZE, Dense-Haze, and NH-Haze.

Author: Senior Computer Vision & AI Engineer
License: MIT
===============================================================================
"""

import os
import random
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import config
import utils

# Configure logger for DehazeDataset
logger: logging.Logger = logging.getLogger("DehazeDataset")


class DehazeDataset(Dataset):
    """
    PyTorch Dataset for paired Hazy and Clear Ground-Truth Image Datasets.

    Supported Dataset Types:
    - RESIDE (Indoor Training Set ITS / Outdoor Training Set OTS)
    - O-HAZE
    - Dense-Haze
    - NH-Haze

    Attributes:
        dataset_dir (Path): Base directory containing dataset files.
        dataset_type (str): Name of the dataset benchmark.
        crop_size (int): Spatial dimensions for random cropping during training.
        is_train (bool): True for training mode with data augmentations, False for validation/test.
        image_pairs (List[Tuple[Path, Path]]): List of (hazy_image_path, clear_image_path) pairs.
    """

    def __init__(
        self,
        dataset_dir: Optional[Union[str, Path]] = None,
        dataset_type: str = "RESIDE",
        crop_size: int = config.PATCH_SIZE,
        is_train: bool = True
    ) -> None:
        """
        Initializes the dataset, pairs hazy/clean files, and configures transformations.

        Args:
            dataset_dir (Optional[Union[str, Path]]): Root dataset folder path.
            dataset_type (str): Name of the dataset (e.g. "RESIDE", "O-HAZE", "Dense-Haze", "NH-Haze").
            crop_size (int): Size of square patch to crop.
            is_train (bool): Mode flag controlling data augmentations.
        """
        super(DehazeDataset, self).__init__()
        self.dataset_type: str = dataset_type
        self.crop_size: int = crop_size
        self.is_train: bool = is_train

        if dataset_dir is None:
            self.dataset_dir: Path = config.DATASET_DIR / dataset_type.lower()
        else:
            self.dataset_dir: Path = Path(dataset_dir)

        # Scan folder for image pairs
        self.image_pairs: List[Tuple[Path, Path]] = self._scan_and_pair_images()

        # If no dataset exists on disk, initialize synthetic benchmark images
        if len(self.image_pairs) == 0:
            logger.warning(
                "No dataset files found at '%s'. Initializing synthetic benchmark samples for testing...",
                self.dataset_dir
            )
            self._generate_synthetic_benchmark()
            self.image_pairs = self._scan_and_pair_images()

        logger.info(
            "DehazeDataset initialized (%s - %d paired samples, Mode: %s).",
            self.dataset_type, len(self.image_pairs), "Train" if is_train else "Eval"
        )

    def _scan_and_pair_images(self) -> List[Tuple[Path, Path]]:
        """
        Scans dataset directory and pairs hazy images with their corresponding clear GT images.

        Returns:
            List[Tuple[Path, Path]]: List of (hazy_path, clear_path) tuples.
        """
        pairs: List[Tuple[Path, Path]] = []
        if not self.dataset_dir.exists():
            return pairs

        hazy_dir = self.dataset_dir / "hazy"
        clear_dir = self.dataset_dir / "clear"

        if not (hazy_dir.exists() and clear_dir.exists()):
            # Alternative layout: search subfolders directly
            hazy_dir = self.dataset_dir
            clear_dir = self.dataset_dir

        # Match files by basename or index
        hazy_files = sorted([
            f for f in hazy_dir.glob("*") if f.suffix.lower() in config.SUPPORTED_IMAGE_EXTENSIONS and "clear" not in f.name.lower()
        ])
        clear_files = sorted([
            f for f in clear_dir.glob("*") if f.suffix.lower() in config.SUPPORTED_IMAGE_EXTENSIONS and ("clear" in f.name.lower() or "gt" in f.name.lower())
        ])

        if len(clear_files) > 0 and len(hazy_files) > 0:
            # Map by matching stem or index
            clear_dict = {f.stem.replace("_clear", "").replace("_GT", ""): f for f in clear_files}
            for h_file in hazy_files:
                stem_clean = h_file.stem.split("_")[0]
                if stem_clean in clear_dict:
                    pairs.append((h_file, clear_dict[stem_clean]))
                elif len(clear_files) == 1:
                    pairs.append((h_file, clear_files[0]))
                elif len(clear_files) == len(hazy_files):
                    idx = hazy_files.index(h_file)
                    pairs.append((h_file, clear_files[idx]))

        return pairs

    def _generate_synthetic_benchmark(self) -> None:
        """
        Generates sample paired hazy/clear images if no dataset files are present on disk.
        """
        hazy_dir = self.dataset_dir / "hazy"
        clear_dir = self.dataset_dir / "clear"
        hazy_dir.mkdir(parents=True, exist_ok=True)
        clear_dir.mkdir(parents=True, exist_ok=True)

        for i in range(1, 6):
            clear_path = clear_dir / f"sample_{i}_clear.jpg"
            hazy_path = hazy_dir / f"sample_{i}_hazy.jpg"

            # Create synthetic clean image
            utils.generate_sample_hazy_image(hazy_path)
            # Create clear counterpart
            clean_img = Image.new("RGB", (640, 480), color=(100 + i * 20, 150, 200))
            clean_img.save(clear_path, "JPEG")

    def _apply_transforms(
        self,
        hazy_img: Image.Image,
        clear_img: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies paired spatial augmentations (Random Crop, Random Flips) and converts to Tensors.

        Args:
            hazy_img (Image.Image): Hazy image input.
            clear_img (Image.Image): Clear ground truth image input.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (Hazy Tensor (3, H, W), Clear Tensor (3, H, W)).
        """
        # Ensure images match spatial dimensions
        if hazy_img.size != clear_img.size:
            clear_img = clear_img.resize(hazy_img.size, Image.BILINEAR)

        w, h = hazy_img.size

        if self.is_train:
            # Random Crop
            if w >= self.crop_size and h >= self.crop_size:
                i, j, th, tw = transforms.RandomCrop.get_params(
                    hazy_img, output_size=(self.crop_size, self.crop_size)
                )
                hazy_img = TF.crop(hazy_img, i, j, th, tw)
                clear_img = TF.crop(clear_img, i, j, th, tw)

            # Random Horizontal Flip
            if random.random() > 0.5:
                hazy_img = TF.hflip(hazy_img)
                clear_img = TF.hflip(clear_img)

            # Random Vertical Flip
            if random.random() > 0.5:
                hazy_img = TF.vflip(hazy_img)
                clear_img = TF.vflip(clear_img)

        # Convert to PyTorch Tensor [0.0 - 1.0]
        hazy_tensor = TF.to_tensor(hazy_img)
        clear_tensor = TF.to_tensor(clear_img)

        return hazy_tensor, clear_tensor

    def __len__(self) -> int:
        """Returns total number of paired samples in dataset."""
        return len(self.image_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves paired hazy and clear image tensors by index.

        Args:
            idx (int): Sample index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (Hazy Tensor, Clear Tensor).
        """
        hazy_path, clear_path = self.image_pairs[idx]
        try:
            hazy_img = Image.open(hazy_path).convert("RGB")
            clear_img = Image.open(clear_path).convert("RGB")

            hazy_tensor, clear_tensor = self._apply_transforms(hazy_img, clear_img)
            return hazy_tensor, clear_tensor
        except Exception as err:
            logger.error("Error loading image pair at index %d (%s): %s", idx, hazy_path, err)
            # Return zero tensor fallback
            dummy = torch.zeros(3, self.crop_size, self.crop_size)
            return dummy, dummy

    def get_stats(self) -> Dict[str, Any]:
        """Returns dataset statistics and metadata."""
        return {
            "dataset_type": self.dataset_type,
            "total_pairs": len(self.image_pairs),
            "dataset_dir": str(self.dataset_dir),
            "is_train": self.is_train,
            "crop_size": self.crop_size
        }
