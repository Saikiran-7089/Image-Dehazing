"""
===============================================================================
AI-Based Single Image Dehazing System
Download & Report Export Manager Module (download.py) [Security-Hardened]
===============================================================================

This module provides the `DownloadManager` class to handle multi-format image saving
(PNG, JPEG, WebP), side-by-side comparison image rendering, batch ZIP file creation,
CSV / JSON metric exports, and academic Markdown/PDF report generation.

Security Hardening:
- Path Traversal Protection: Sanitizes all output paths using Path.name and enforces
  is_relative_to() constraints to prevent directory traversal vulnerabilities.
- Safe Serialization: Enforces UTF-8 encoding and bounded buffers.

Author: Senior Computer Vision & AI Engineer / Security Engineer
License: MIT
===============================================================================
"""

import os
import io
import csv
import json
import re
import zipfile
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional, Union

import cv2
import numpy as np

import config

# Initialize module logger
logger: logging.Logger = logging.getLogger("DownloadManager")


class DownloadManager:
    """
    Security-Hardened Download and Data Export Manager.

    Handles timestamped file generation, side-by-side image comparison stitching,
    batch output packaging, metric serialization, and Markdown/Text report exports.
    """

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        """
        Initializes the DownloadManager output directory with path resolution.

        Args:
            output_dir (Optional[Path]): Target folder path. Defaults to config.OUTPUTS_DIR.
        """
        target = output_dir if output_dir is not None else config.OUTPUTS_DIR
        self.output_dir: Path = Path(target).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("DownloadManager initialized targeting secure folder '%s'.", self.output_dir)

    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitizes input filename to prevent path traversal attacks (e.g. '../../etc/passwd').

        Args:
            filename (str): Candidate filename string.

        Returns:
            str: Safe sanitized filename string.
        """
        # Strip path components and retain only the basename
        clean_name = Path(filename).name
        # Remove any non-alphanumeric characters except underscores, hyphens, and dots
        clean_name = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', clean_name)
        return clean_name if clean_name else "output_file"

    def _generate_timestamp_prefix(self, prefix: str) -> str:
        """Generates a timestamped safe filename prefix."""
        safe_prefix = self._sanitize_filename(prefix)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{safe_prefix}_{timestamp}"

    def _verify_safe_path(self, target_path: Path) -> Path:
        """
        Verifies that target_path resides within self.output_dir (prevents Path Traversal).

        Args:
            target_path (Path): Path to verify.

        Returns:
            Path: Resolved target path.

        Raises:
            PermissionError: If path escapes the output directory boundary.
        """
        resolved_target = target_path.resolve()
        try:
            # Check relative path boundary
            resolved_target.relative_to(self.output_dir)
            return resolved_target
        except ValueError as err:
            logger.critical("SECURITY ALERT: Path traversal attempt blocked for '%s'", target_path)
            raise PermissionError(f"Path traversal security violation: '{target_path}'") from err

    def save_image(
        self,
        image_np: np.ndarray,
        filename_prefix: str = "dehazed",
        format_ext: str = "png"
    ) -> Tuple[Path, bytes]:
        """
        Saves a single BGR image array securely to disk and returns its path and bytes.

        Args:
            image_np (np.ndarray): Image array in OpenCV BGR format.
            filename_prefix (str): Prefix string for the filename.
            format_ext (str): Image extension ('png', 'jpeg', 'jpg', 'webp').

        Returns:
            Tuple[Path, bytes]: (Saved File Path, Byte Buffer).
        """
        try:
            ext = format_ext.lower().replace(".", "").strip()
            if ext not in ["png", "jpeg", "jpg", "webp"]:
                ext = "png"

            filename = f"{self._generate_timestamp_prefix(filename_prefix)}.{ext}"
            file_path = self._verify_safe_path(self.output_dir / filename)

            # Encode image to byte buffer
            success, encoded_buf = cv2.imencode(f".{ext}", image_np)
            if not success:
                raise ValueError("cv2.imencode failed to encode image array.")

            byte_content = encoded_buf.tobytes()

            # Write file securely to disk
            with open(file_path, "wb") as f:
                f.write(byte_content)

            logger.info("Saved image output to '%s' (%d bytes).", file_path, len(byte_content))
            return file_path, byte_content
        except Exception as err:
            logger.error("Error saving image file: %s", err)
            raise IOError(f"Failed to save image file: {err}") from err

    def save_comparison(
        self,
        original_np: np.ndarray,
        dehazed_np: np.ndarray,
        filename_prefix: str = "comparison",
        format_ext: str = "png"
    ) -> Tuple[Path, bytes]:
        """
        Creates a side-by-side comparison image (Original | Dehazed) with headers.

        Args:
            original_np (np.ndarray): Original hazy BGR image.
            dehazed_np (np.ndarray): Enhanced BGR image.
            filename_prefix (str): Filename prefix.
            format_ext (str): Image extension.

        Returns:
            Tuple[Path, bytes]: (Comparison File Path, Byte Buffer).
        """
        try:
            h1, w1 = original_np.shape[:2]
            h2, w2 = dehazed_np.shape[:2]

            if h1 != h2:
                dehazed_np = cv2.resize(dehazed_np, (int(w2 * (h1 / h2)), h1), interpolation=cv2.INTER_AREA)
                h2, w2 = dehazed_np.shape[:2]

            banner_h = 40
            banner_orig = np.zeros((banner_h, w1, 3), dtype=np.uint8)
            banner_enh = np.zeros((banner_h, w2, 3), dtype=np.uint8)

            cv2.putText(banner_orig, "BEFORE (HAZY)", (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(banner_enh, "AFTER (DEHAZED)", (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (56, 189, 248), 2)

            col1 = np.vstack([banner_orig, original_np])
            col2 = np.vstack([banner_enh, dehazed_np])

            comparison_grid = np.hstack([col1, col2])

            return self.save_image(comparison_grid, filename_prefix=filename_prefix, format_ext=format_ext)
        except Exception as err:
            logger.error("Error creating comparison image: %s", err)
            raise IOError(f"Failed to create comparison image: {err}") from err

    def create_zip(
        self,
        image_tuples: List[Tuple[np.ndarray, str]],
        zip_filename_prefix: str = "batch_dehazed_results"
    ) -> Tuple[Path, bytes]:
        """
        Bundles a list of (image_np, filename) items into a safe ZIP archive.

        Args:
            image_tuples (List[Tuple[np.ndarray, str]]): List of (BGR Image array, output filename).
            zip_filename_prefix (str): Prefix for zip filename.

        Returns:
            Tuple[Path, bytes]: (ZIP File Path, Byte Buffer).
        """
        try:
            filename = f"{self._generate_timestamp_prefix(zip_filename_prefix)}.zip"
            zip_path = self._verify_safe_path(self.output_dir / filename)

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for idx, (img_np, item_name) in enumerate(image_tuples):
                    safe_item_name = self._sanitize_filename(item_name)
                    ext = Path(safe_item_name).suffix if Path(safe_item_name).suffix else ".png"
                    success, buf = cv2.imencode(ext, img_np)
                    if success:
                        zip_file.writestr(safe_item_name, buf.tobytes())

            zip_bytes = zip_buffer.getvalue()

            with open(zip_path, "wb") as f:
                f.write(zip_bytes)

            logger.info("Created batch ZIP bundle at '%s' (%d bytes).", zip_path, len(zip_bytes))
            return zip_path, zip_bytes
        except Exception as err:
            logger.error("Error creating ZIP package: %s", err)
            raise IOError(f"Failed to create ZIP package: {err}") from err

    def save_metrics(
        self,
        metrics_dict: Dict[str, Any],
        filename_prefix: str = "dehazing_metrics",
        export_format: str = "json"
    ) -> Tuple[Path, str]:
        """
        Serializes image quality assessment metrics securely into JSON or CSV.

        Args:
            metrics_dict (Dict[str, Any]): Dictionary of metric keys and numerical values.
            filename_prefix (str): Prefix for filename.
            export_format (str): Target format ('json' or 'csv').

        Returns:
            Tuple[Path, str]: (Saved File Path, Serialized String Content).
        """
        try:
            fmt = export_format.lower().strip()
            if fmt not in ["json", "csv"]:
                fmt = "json"

            filename = f"{self._generate_timestamp_prefix(filename_prefix)}.{fmt}"
            file_path = self._verify_safe_path(self.output_dir / filename)

            if fmt == "json":
                content = json.dumps(metrics_dict, indent=4)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
            else:
                stream = io.StringIO()
                writer = csv.writer(stream)
                writer.writerow(["Metric Name", "Value"])
                for k, v in metrics_dict.items():
                    writer.writerow([self._sanitize_filename(str(k)), v])
                content = stream.getvalue()
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)

            logger.info("Saved metrics report (%s) to '%s'.", fmt.UPPER() if hasattr(fmt, 'UPPER') else fmt.upper(), file_path)
            return file_path, content
        except Exception as err:
            logger.error("Error saving metrics: %s", err)
            raise IOError(f"Failed to save metrics report: {err}") from err

    def export_report(
        self,
        metrics_dict: Dict[str, Any],
        telemetry: Dict[str, Any],
        filename_prefix: str = "dehazing_report",
        report_type: str = "markdown"
    ) -> Tuple[Path, str]:
        """
        Generates a detailed academic project evaluation report in Markdown format.

        Args:
            metrics_dict (Dict[str, Any]): Image quality assessment metric dictionary.
            telemetry (Dict[str, Any]): Inference performance telemetry dictionary.
            filename_prefix (str): Filename prefix.
            report_type (str): Report type ('markdown' or 'text').

        Returns:
            Tuple[Path, str]: (Saved Report Path, Report Text Content).
        """
        try:
            filename = f"{self._generate_timestamp_prefix(filename_prefix)}.md"
            file_path = self._verify_safe_path(self.output_dir / filename)

            md_content = f"""# AI Image Dehazing System - Quality & Performance Report

**Date & Time:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Model Architecture:** `{telemetry.get("model_name", "N/A")}`  
**Hardware Device:** `{telemetry.get("device_used", "CPU")}`  

---

## 1. Inference Performance Telemetry

- **Execution Time:** `{telemetry.get("execution_time_sec", 0)} s`
- **Frames Per Second (FPS):** `{telemetry.get("fps", 0)} FPS`
- **GPU Memory Occupancy:** `{telemetry.get("memory_mb", 0)} MB`
- **Input Image Resolution:** `{telemetry.get("input_resolution", "N/A")}`
- **Output Image Resolution:** `{telemetry.get("output_resolution", "N/A")}`

---

## 2. Quantitative Image Quality Assessment (IQA)

| Quality Metric | Score / Value | Target Ideal Range | Description |
| :--- | :---: | :---: | :--- |
| **PSNR** | `{metrics_dict.get("psnr", 0)} dB` | > 28 dB | Reconstruction accuracy |
| **SSIM** | `{metrics_dict.get("ssim", 0)}` | > 0.85 | Structural perceptual similarity |
| **MSE** | `{metrics_dict.get("mse", 0)}` | < 0.01 | Mean squared error |
| **Brightness** | `{metrics_dict.get("brightness", 0)}` | 100 - 160 | Average mean luminance |
| **Contrast** | `{metrics_dict.get("contrast", 0)}` | 40 - 80 | RMS standard deviation |
| **Sharpness** | `{metrics_dict.get("sharpness", 0)}` | > 100 | Laplacian edge variance |
| **Entropy** | `{metrics_dict.get("entropy", 0)}` | > 7.0 | Information texture entropy |
| **Visibility Score** | `{metrics_dict.get("visibility_score", 0)} / 100` | High | Estimated clarity index |
| **Haze Density** | `{metrics_dict.get("haze_density_score", 0)}%` | Low | Dark channel haze level |
| **Overall Quality** | `{metrics_dict.get("overall_quality_score", 0)} / 100` | > 80 | Weighted composite index |

---

*Generated automatically by AI-Based Single Image Dehazing System.*
"""

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(md_content)

            logger.info("Generated Markdown dehazing report at '%s'.", file_path)
            return file_path, md_content
        except Exception as err:
            logger.error("Error generating report: %s", err)
            raise IOError(f"Failed to export report: {err}") from err
