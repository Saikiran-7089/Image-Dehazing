"""
===============================================================================
Integration Tests: End-to-End Dehazing Pipeline & Exporter (tests/test_pipeline.py)
===============================================================================
"""

import pytest
import numpy as np
from PIL import Image

import config
from inference import DehazeInferenceEngine
from download import DownloadManager


def test_inference_engine_single_image():
    """Test full single image dehazing pipeline across models."""
    engine = DehazeInferenceEngine(device="cpu")
    dummy_pil = Image.fromarray((np.random.rand(128, 128, 3) * 255).astype(np.uint8))

    for m_name in config.SUPPORTED_MODELS:
        res_bgr, tel = engine.process_image(dummy_pil, model_name=m_name)

        assert isinstance(res_bgr, np.ndarray), f"Model {m_name} must return NumPy ndarray"
        assert res_bgr.shape[:2] == (128, 128), f"Model {m_name} spatial dimensions must match input"
        assert "execution_time_sec" in tel and "fps" in tel, "Telemetry must include timing metrics"


def test_inference_engine_batch_processing():
    """Test batch dehazing pipeline."""
    engine = DehazeInferenceEngine(device="cpu")
    dummy_img1 = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
    dummy_img2 = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)

    batch_results = engine.process_batch([dummy_img1, dummy_img2], model_name=config.MODEL_DCP)

    assert len(batch_results) == 2, "Batch processing must return 2 results for 2 input images"
    assert isinstance(batch_results[0][0], np.ndarray)


def test_download_manager_exports(tmp_path):
    """Test file, comparison, ZIP, and report exports."""
    dl_mgr = DownloadManager(output_dir=tmp_path)
    dummy_img1 = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
    dummy_img2 = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)

    # 1. Single Image Export
    file_path, buf = dl_mgr.save_image(dummy_img1, filename_prefix="test_dehazed", format_ext="png")
    assert file_path.exists() and len(buf) > 0

    # 2. Comparison Image Export
    comp_path, comp_buf = dl_mgr.save_comparison(dummy_img1, dummy_img2)
    assert comp_path.exists() and len(comp_buf) > 0

    # 3. ZIP Export
    zip_path, zip_buf = dl_mgr.create_zip([(dummy_img1, "img1.png"), (dummy_img2, "img2.png")])
    assert zip_path.exists() and len(zip_buf) > 0

    # 4. Metrics Export
    metrics_path, content = dl_mgr.save_metrics({"psnr": 30.5, "ssim": 0.95}, export_format="json")
    assert metrics_path.exists() and "psnr" in content

    # 5. Report Export
    report_path, r_text = dl_mgr.export_report({"psnr": 30.5}, {"model_name": "DehazeFormer"})
    assert report_path.exists() and "DehazeFormer" in r_text
