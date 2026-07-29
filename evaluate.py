"""
===============================================================================
AI-Based Single Image Dehazing System
Model Benchmarking & Evaluation Pipeline (evaluate.py)
===============================================================================

This script provides a standalone benchmark pipeline for evaluating DehazeFormer,
AOD-Net, and Dark Channel Prior across standard dehazing datasets (RESIDE, O-HAZE,
Dense-Haze, NH-Haze).

Computed Benchmark Metrics:
- Average PSNR (dB), SSIM, MSE
- Average Brightness, Contrast, Sharpness, Entropy
- Average Visibility Score, Haze Density Score
- Average Execution Time (seconds/image) and Frames Per Second (FPS)

Exports comparison tables in CSV, JSON, and Markdown formats.

Author: Senior Computer Vision & AI Engineer
License: MIT
===============================================================================
"""

import os
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import config
import utils
from inference import DehazeInferenceEngine
from metrics import MetricsCalculator
from dataset.dehaze_dataset import DehazeDataset

# Initialize module logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger: logging.Logger = logging.getLogger("Evaluator")


class DehazeEvaluator:
    """
    Evaluator pipeline for running quantitative model benchmarks.
    """

    def __init__(self, device: Optional[str] = None) -> None:
        self.engine: DehazeInferenceEngine = DehazeInferenceEngine(device=device)
        self.metrics_calculator: MetricsCalculator = MetricsCalculator()

    def evaluate_model_on_dataset(
        self,
        model_name: str,
        dataset: DehazeDataset,
        max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluates a single model on a dataset instance.

        Args:
            model_name (str): Name of model to evaluate.
            dataset (DehazeDataset): Instantiated dataset object.
            max_samples (Optional[int]): Maximum number of samples to process.

        Returns:
            Dict[str, Any]: Aggregated benchmark metrics and timing stats.
        """
        logger.info("Evaluating model '%s' on %d samples...", model_name, min(len(dataset), max_samples or len(dataset)))

        sample_count = min(len(dataset), max_samples) if max_samples else len(dataset)
        metrics_accumulator: Dict[str, List[float]] = {
            "psnr": [], "ssim": [], "mse": [], "brightness": [], "contrast": [],
            "sharpness": [], "entropy": [], "visibility_score": [], "haze_density_score": [],
            "overall_quality_score": [], "execution_time": []
        }

        pbar = tqdm(range(sample_count), desc=f"Evaluating {model_name}")
        for idx in pbar:
            hazy_tensor, clear_tensor = dataset[idx]

            # Convert PyTorch Tensors to OpenCV BGR uint8
            hazy_np = cv2.cvtColor(utils.tensor_to_numpy(hazy_tensor), cv2.COLOR_RGB2BGR)
            clear_np = cv2.cvtColor(utils.tensor_to_numpy(clear_tensor), cv2.COLOR_RGB2BGR)

            # Dehaze image with timing
            t_start = time.perf_counter()
            dehazed_bgr, telemetry = self.engine.dehaze_image(hazy_np, model_name=model_name)
            t_elapsed = time.perf_counter() - t_start

            # Calculate IQA metrics against Ground Truth clear image
            m = self.metrics_calculator.calculate_all_metrics(clear_np, dehazed_bgr)

            for key in m:
                if key in metrics_accumulator:
                    metrics_accumulator[key].append(m[key])
            metrics_accumulator["execution_time"].append(t_elapsed)

        # Calculate averages
        avg_results: Dict[str, Any] = {
            "model_name": model_name,
            "samples_evaluated": sample_count,
            "avg_psnr": round(float(np.mean(metrics_accumulator["psnr"])), 2),
            "avg_ssim": round(float(np.mean(metrics_accumulator["ssim"])), 4),
            "avg_mse": round(float(np.mean(metrics_accumulator["mse"])), 6),
            "avg_brightness": round(float(np.mean(metrics_accumulator["brightness"])), 2),
            "avg_contrast": round(float(np.mean(metrics_accumulator["contrast"])), 2),
            "avg_sharpness": round(float(np.mean(metrics_accumulator["sharpness"])), 2),
            "avg_entropy": round(float(np.mean(metrics_accumulator["entropy"])), 3),
            "avg_visibility_score": round(float(np.mean(metrics_accumulator["visibility_score"])), 1),
            "avg_haze_density_score": round(float(np.mean(metrics_accumulator["haze_density_score"])), 1),
            "avg_overall_quality_score": round(float(np.mean(metrics_accumulator["overall_quality_score"])), 1),
            "avg_time_sec": round(float(np.mean(metrics_accumulator["execution_time"])), 4),
            "avg_fps": round(1.0 / max(0.001, float(np.mean(metrics_accumulator["execution_time"]))), 1)
        }

        logger.info("Evaluation complete for %s: PSNR = %.2f dB, SSIM = %.4f, FPS = %.1f",
                    model_name, avg_results["avg_psnr"], avg_results["avg_ssim"], avg_results["avg_fps"])
        return avg_results

    def run_full_benchmark(
        self,
        models_to_eval: List[str],
        dataset_type: str = "RESIDE",
        max_samples: int = 10,
        output_dir: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Runs evaluation across all requested models and generates comparison tables.

        Args:
            models_to_eval (List[str]): List of model names.
            dataset_type (str): Dataset benchmark name.
            max_samples (int): Sample count.
            output_dir (Optional[Path]): Folder to save CSV/JSON/MD outputs.

        Returns:
            pd.DataFrame: Summary comparison table DataFrame.
        """
        out_dir = output_dir if output_dir else config.REPORTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        dataset = DehazeDataset(dataset_type=dataset_type, is_train=False)

        all_results = []
        for m_name in models_to_eval:
            try:
                res = self.evaluate_model_on_dataset(m_name, dataset, max_samples=max_samples)
                all_results.append(res)
            except Exception as err:
                logger.error("Failed to evaluate model '%s': %s", m_name, err)

        df_summary = pd.DataFrame(all_results)

        # Export CSV
        csv_path = out_dir / "evaluation_results.csv"
        df_summary.to_csv(csv_path, index=False)

        # Export JSON
        json_path = out_dir / "evaluation_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=4)

        # Export Markdown Summary Table
        md_path = out_dir / "evaluation_summary.md"
        md_content = f"# Model Evaluation Summary ({dataset_type} Dataset)\n\n"
        try:
            md_content += df_summary.to_markdown(index=False)
        except Exception:
            md_content += df_summary.to_string(index=False)

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info("Saved evaluation benchmark results to '%s'.", out_dir)
        return df_summary


def main() -> None:
    """CLI entrypoint for evaluate.py."""
    parser = argparse.ArgumentParser(description="Evaluate Dehazing Models on Benchmarks.")
    parser.add_argument("--model", type=str, default="all", choices=["DehazeFormer", "AOD-Net", "Dark Channel Prior", "all"])
    parser.add_argument("--dataset", type=str, default="RESIDE", help="Dataset benchmark name.")
    parser.add_argument("--samples", type=int, default=5, help="Number of test samples.")
    args = parser.parse_args()

    models = config.SUPPORTED_MODELS if args.model == "all" else [args.model]

    evaluator = DehazeEvaluator()
    df = evaluator.run_full_benchmark(models_to_eval=models, dataset_type=args.dataset, max_samples=args.samples)
    print("\n================ EVALUATION BENCHMARK RESULTS ================")
    print(df[["model_name", "avg_psnr", "avg_ssim", "avg_time_sec", "avg_fps"]].to_string(index=False))


if __name__ == "__main__":
    main()
