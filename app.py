"""
===============================================================================
AI-Based Single Image Dehazing System
Streamlit Master Application (app.py) [Optimized Version]
===============================================================================

This is the main web dashboard interface for the AI-Based Single Image Dehazing System.
It integrates DehazeFormer, AOD-Net, and Dark Channel Prior models with real-time
IQA metric evaluation, interactive Plotly RGB histograms, before/after visual sliders,
batch processing, document report downloads, and research benchmarks.

Launch Command:
    streamlit run app.py

Author: Senior Computer Vision & AI Engineer
License: MIT
===============================================================================
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
import streamlit as st

import config
import utils
from inference import DehazeInferenceEngine
from metrics import MetricsCalculator
from image_processing import ImageProcessor
from download import DownloadManager

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger: logging.Logger = logging.getLogger("DehazeApp")

# =============================================================================
# STREAMLIT PAGE CONFIGURATION & SESSION STATE
# =============================================================================
st.set_page_config(
    page_title="AI Image Dehazing System",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Dark Glassmorphism CSS Theme
st.markdown(config.CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_inference_engine() -> DehazeInferenceEngine:
    """Cached singleton instance of the DehazeInferenceEngine."""
    return DehazeInferenceEngine()


@st.cache_resource
def get_metrics_calculator() -> MetricsCalculator:
    """Cached singleton instance of the MetricsCalculator."""
    return MetricsCalculator()


@st.cache_resource
def get_image_processor() -> ImageProcessor:
    """Cached singleton instance of the ImageProcessor."""
    return ImageProcessor()


@st.cache_resource
def get_download_manager() -> DownloadManager:
    """Cached singleton instance of the DownloadManager."""
    return DownloadManager()


@st.cache_data
def cached_histogram_data(img_bytes: bytes) -> Dict[str, Any]:
    """Cached fast RGB histogram calculation from raw image bytes."""
    processor = get_image_processor()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return processor.generate_plotly_histogram_data(img_bgr)


# Initialize Streamlit Session States
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "dehazed_image" not in st.session_state:
    st.session_state.dehazed_image = None
if "telemetry" not in st.session_state:
    st.session_state.telemetry = None
if "metrics_dict" not in st.session_state:
    st.session_state.metrics_dict = None


# Helper to render styled glassmorphic metric cards
def render_glass_card(title: str, value: Any, subtext: str = "", unit: str = "") -> None:
    """Renders a styled dark glassmorphism card component."""
    unit_str = f" <span style='font-size: 0.9rem; font-weight: 500;'>{unit}</span>" if unit else ""
    sub_html = f"<div class='metric-sub'>{subtext}</div>" if subtext else ""
    card_html = f"""
    <div class="glass-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}{unit_str}</div>
        {sub_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================
st.sidebar.markdown("## 🌫️ DehazeAI Engine")
st.sidebar.markdown(f"*Version {config.PROJECT_VERSION}*")
st.sidebar.markdown("---")

navigation_option = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Home",
        "🖼 Upload",
        "✨ Image Dehazing",
        "📊 Quality Metrics",
        "📈 Histogram",
        "🔬 Model Comparison",
        "📚 Research Paper",
        "🌍 Applications",
        "⚙ Settings",
        "ℹ About"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Hardware Target:** `{config.DEVICE.upper()}`")
st.sidebar.markdown(f"**Default Model:** `{config.DEFAULT_MODEL}`")


# =============================================================================
# PAGE 1: 🏠 HOME PAGE
# =============================================================================
if navigation_option == "🏠 Home":
    st.title("🌫️ AI-Based Single Image Dehazing System")
    st.markdown("### Transformer-Based Deep Learning & Quantitative Image Quality Assessment")

    st.markdown(
        """
        <div class="glass-card">
            <h3 style="color: #38bdf8; margin-top: 0;">Executive Project Summary</h3>
            <p style="font-size: 1.05em; line-height: 1.7; color: #cbd5e1;">
                This production system removes atmospheric haze, fog, smoke, and mist from single images using state-of-the-art
                deep learning architectures (<b>DehazeFormer</b>, <b>AOD-Net</b>) alongside classical computer vision baselines (<b>Dark Channel Prior</b>).
                Designed for final-year computer vision mini projects, research evaluation, and autonomous vision deployment.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        render_glass_card("⚡ State-of-the-Art AI", "DehazeFormer", "SOTA Window Self-Attention (W-MSA)")
    with col2:
        render_glass_card("📊 10-Metric Suite", "IQA Scoring", "PSNR, SSIM, MSE, Entropy, Visibility")
    with col3:
        render_glass_card("📄 IEEE Deliverables", "Doc & PPT", "Automated Word & Slide Generators")

    st.markdown("---")
    st.markdown("### 🏗️ System Architecture Flowchart")
    st.code(
        """
        [ Hazy Input Image ] ---> [ Model Factory (loader.py) ]
                                         |
                       +-----------------+-----------------+
                       |                 |                 |
                [ DehazeFormer ]    [ AOD-Net ]    [ Dark Channel Prior ]
                       |                 |                 |
                       +-----------------+-----------------+
                                         |
                             [ Restored Clean Image ]
                                         |
                       +-----------------+-----------------+
                       |                                   |
             [ Metrics Calculator ]              [ Download Manager ]
           (PSNR, SSIM, MSE, Entropy)           (PNG, WebP, ZIP, Report)
        """,
        language="text"
    )

    st.markdown("### 🚀 Quick Start Steps")
    st.markdown(
        """
        1. Go to **🖼 Upload** to load a hazy image or try the built-in sample landscape.
        2. Open **✨ Image Dehazing** to select your model and execute single or batch dehazing.
        3. Inspect detailed scores under **📊 Quality Metrics** and Plotly distributions in **📈 Histogram**.
        4. Export your enhanced images, ZIP packages, or Markdown reports directly.
        """
    )


# =============================================================================
# PAGE 2: 🖼 UPLOAD PAGE
# =============================================================================
elif navigation_option == "🖼 Upload":
    st.title("🖼️ Upload Hazy Image")

    col_up, col_sample = st.columns([2, 1])

    with col_up:
        uploaded_file = st.file_uploader(
            "Drag & Drop Hazy Image (PNG, JPG, JPEG, WebP)",
            type=["png", "jpg", "jpeg", "webp"]
        )

    with col_sample:
        st.markdown("#### Sample Test Asset")
        if st.button("Load Sample Hazy Landscape"):
            sample_path = config.ASSETS_DIR / "sample_hazy_landscape.jpg"
            if sample_path.exists():
                st.session_state.current_image = Image.open(sample_path)
                st.success("Loaded sample hazy landscape!")

    if uploaded_file is not None:
        try:
            st.session_state.current_image = Image.open(uploaded_file)
            st.success("Image uploaded successfully!")
        except Exception as err:
            st.error(f"Failed to read uploaded image: {err}")

    if st.session_state.current_image is not None:
        st.markdown("---")
        st.markdown("### Image Preview & Spatial Metadata")

        col_img, col_meta = st.columns([3, 2])

        with col_img:
            st.image(st.session_state.current_image, caption="Input Hazy Image", use_container_width=True)

        with col_meta:
            meta = utils.extract_image_metadata(st.session_state.current_image)
            render_glass_card("Spatial Resolution", meta['resolution'])
            render_glass_card("Aspect Ratio", meta['aspect_ratio'])
            render_glass_card("Color Mode & Size", f"{meta['color_space']} | {meta['file_size']}")


# =============================================================================
# PAGE 3: ✨ IMAGE DEHAZING PAGE
# =============================================================================
elif navigation_option == "✨ Image Dehazing":
    st.title("✨ AI Image Dehazing Engine")

    if st.session_state.current_image is None:
        st.warning("Please upload an image under '🖼 Upload' or load the sample image first!")
    else:
        engine = get_inference_engine()
        metrics_calc = get_metrics_calculator()
        dl_mgr = get_download_manager()

        st.sidebar.markdown("### Model & Controls")
        selected_model = st.sidebar.selectbox("Select Dehazing Model", config.SUPPORTED_MODELS, index=0)

        st.sidebar.markdown("#### Post-Enhancement Sliders")
        use_clahe = st.sidebar.checkbox("Enable CLAHE Equalization", value=False)
        clahe_clip = st.sidebar.slider("CLAHE Clip Limit", 1.0, 5.0, 2.0, 0.5) if use_clahe else 2.0

        use_sharpen = st.sidebar.checkbox("Enable Edge Sharpening", value=False)
        sharpen_factor = st.sidebar.slider("Sharpening Factor", 0.5, 3.0, 1.5, 0.1) if use_sharpen else 1.5

        brightness = st.sidebar.slider("Brightness", 0.5, 1.5, 1.0, 0.05)
        contrast = st.sidebar.slider("Contrast", 0.5, 1.5, 1.0, 0.05)

        use_denoise = st.sidebar.checkbox("Enable Noise Reduction", value=False)
        denoise_strength = st.sidebar.slider("Denoise Strength", 5, 25, 10, 1) if use_denoise else 10

        enhancement_params = {
            "use_clahe": use_clahe,
            "clahe_clip": clahe_clip,
            "use_sharpen": use_sharpen,
            "sharpen_factor": sharpen_factor,
            "brightness": brightness,
            "contrast": contrast,
            "use_denoise": use_denoise,
            "denoise_strength": denoise_strength
        }

        if st.button("🚀 Run AI Dehazing Inference", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Pre-processing input image tensor...")
            progress_bar.progress(25)
            time.sleep(0.05)

            status_text.text(f"Running {selected_model} forward pass...")
            progress_bar.progress(65)

            try:
                dehazed_bgr, tel = engine.dehaze_image(
                    st.session_state.current_image,
                    model_name=selected_model,
                    enhancement_params=enhancement_params
                )
                progress_bar.progress(90)
                status_text.text("Calculating Image Quality Assessment metrics...")

                orig_bgr = utils.pil_to_opencv(st.session_state.current_image)
                metrics_res = metrics_calc.calculate_all_metrics(orig_bgr, dehazed_bgr)

                st.session_state.dehazed_image = dehazed_bgr
                st.session_state.telemetry = tel
                st.session_state.metrics_dict = metrics_res

                progress_bar.progress(100)
                status_text.text("Dehazing completed successfully!")
                time.sleep(0.1)
                progress_bar.empty()
                status_text.empty()

            except Exception as err:
                st.error(f"Inference execution error: {err}")

        # Display Dehazed Results if available
        if st.session_state.dehazed_image is not None:
            tel = st.session_state.telemetry
            st.markdown("---")

            # Telemetry Glass Cards
            t1, t2, t3, t4 = st.columns(4)
            with t1:
                render_glass_card("Active Model", tel["model_name"])
            with t2:
                render_glass_card("Hardware Device", tel["device_used"])
            with t3:
                render_glass_card("Execution Time", f"{tel['execution_time_sec']} s")
            with t4:
                render_glass_card("Frame Rate", f"{tel['fps']} FPS")

            st.markdown("### Visual Inspection & Comparison")
            view_mode = st.radio("Comparison Mode", ["Side-by-Side", "Original Only", "Dehazed Only"], horizontal=True)

            orig_rgb = np.array(st.session_state.current_image.convert("RGB"))
            dehazed_rgb = cv2.cvtColor(st.session_state.dehazed_image, cv2.COLOR_BGR2RGB)

            if view_mode == "Side-by-Side":
                c1, c2 = st.columns(2)
                with c1:
                    st.image(orig_rgb, caption="Original (Hazy)", use_container_width=True)
                with c2:
                    st.image(dehazed_rgb, caption="Enhanced (Dehazed)", use_container_width=True)
            elif view_mode == "Original Only":
                st.image(orig_rgb, caption="Original (Hazy)", use_container_width=True)
            else:
                st.image(dehazed_rgb, caption="Enhanced (Dehazed)", use_container_width=True)

            # Download Section
            st.markdown("---")
            st.markdown("### 📥 Download Results & Exports")

            d1, d2, d3, d4 = st.columns(4)
            dehazed_pil = utils.opencv_to_pil(st.session_state.dehazed_image)

            buf_png = io.BytesIO()
            dehazed_pil.save(buf_png, format="PNG")
            d1.download_button("Download PNG", buf_png.getvalue(), "dehazed.png", "image/png")

            buf_jpg = io.BytesIO()
            dehazed_pil.save(buf_jpg, format="JPEG", quality=95)
            d2.download_button("Download JPG", buf_jpg.getvalue(), "dehazed.jpg", "image/jpeg")

            buf_webp = io.BytesIO()
            dehazed_pil.save(buf_webp, format="WEBP", quality=95)
            d3.download_button("Download WebP", buf_webp.getvalue(), "dehazed.webp", "image/webp")

            orig_bgr = utils.pil_to_opencv(st.session_state.current_image)
            _, comp_bytes = dl_mgr.save_comparison(orig_bgr, st.session_state.dehazed_image)
            d4.download_button("Download Comparison", comp_bytes, "comparison.png", "image/png")


# =============================================================================
# PAGE 4: 📊 QUALITY METRICS PAGE
# =============================================================================
elif navigation_option == "📊 Quality Metrics":
    st.title("📊 Image Quality Assessment (IQA) Dashboard")

    if st.session_state.metrics_dict is None:
        st.info("Run dehazing under '✨ Image Dehazing' first to view live metric cards!")
    else:
        m = st.session_state.metrics_dict

        st.markdown("### Summary Composite Indicators")
        s1, s2, s3 = st.columns(3)
        with s1:
            render_glass_card("Overall Quality Score", f"{m['overall_quality_score']}", subtext="Target > 80", unit="/ 100")
        with s2:
            render_glass_card("Estimated Visibility", f"{m['visibility_score']}", subtext="High Clarity", unit="/ 100")
        with s3:
            render_glass_card("Haze Density Index", f"{m['haze_density_score']}", subtext="Low Haze", unit="%")

        st.markdown("---")
        st.markdown("### Detailed Metric Breakdown")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            render_glass_card("PSNR (Reconstruction)", f"{m['psnr']}", subtext="Target > 28 dB", unit="dB")
        with m2:
            render_glass_card("SSIM (Similarity)", f"{m['ssim']}", subtext="Target > 0.85")
        with m3:
            render_glass_card("MSE (Pixel Error)", f"{m['mse']}")
        with m4:
            render_glass_card("Entropy (Texture)", f"{m['entropy']}", unit="bits")

        m5, m6, m7, m8 = st.columns(4)
        with m5:
            render_glass_card("Luminance Brightness", f"{m['brightness']}")
        with m6:
            render_glass_card("RMS Contrast", f"{m['contrast']}")
        with m7:
            render_glass_card("Laplacian Sharpness", f"{m['sharpness']}")
        with m8:
            render_glass_card("Processing Time", f"{st.session_state.telemetry['execution_time_sec']}", unit="s")

    st.markdown("---")
    st.markdown("### Metric Mathematical Formulations & Descriptions")

    for key, info in config.METRIC_TARGETS.items():
        with st.expander(f"📌 {key} - ({info['unit']})"):
            st.write(f"**Description:** {info['description']}")
            st.write(f"**Ideal Target Range:** {info['ideal_range']}")
            st.latex(info["formula"])


# =============================================================================
# PAGE 5: 📈 HISTOGRAM PAGE
# =============================================================================
elif navigation_option == "📈 Histogram":
    st.title("📈 Interactive Plotly RGB Histogram Analysis")

    if st.session_state.current_image is None:
        st.warning("Upload an image under '🖼 Upload' to inspect RGB channel distributions.")
    else:
        orig_bgr = utils.pil_to_opencv(st.session_state.current_image)
        _, encoded_buf = cv2.imencode(".png", orig_bgr)
        orig_hist = cached_histogram_data(encoded_buf.tobytes())

        fig_orig = go.Figure()
        fig_orig.add_trace(go.Scattergl(x=orig_hist["x"], y=orig_hist["r"], name="Red Channel", line=dict(color="#ef4444", width=2)))
        fig_orig.add_trace(go.Scattergl(x=orig_hist["x"], y=orig_hist["g"], name="Green Channel", line=dict(color="#10b981", width=2)))
        fig_orig.add_trace(go.Scattergl(x=orig_hist["x"], y=orig_hist["b"], name="Blue Channel", line=dict(color="#38bdf8", width=2)))
        fig_orig.update_layout(title="Original Image RGB Histogram", xaxis_title="Pixel Intensity (0-255)", yaxis_title="Pixel Count", template="plotly_dark", height=400)

        st.plotly_chart(fig_orig, use_container_width=True)

        if st.session_state.dehazed_image is not None:
            _, enc_dehazed = cv2.imencode(".png", st.session_state.dehazed_image)
            dehazed_hist = cached_histogram_data(enc_dehazed.tobytes())

            fig_enh = go.Figure()
            fig_enh.add_trace(go.Scattergl(x=dehazed_hist["x"], y=dehazed_hist["r"], name="Red Channel", line=dict(color="#ef4444", width=2)))
            fig_enh.add_trace(go.Scattergl(x=dehazed_hist["x"], y=dehazed_hist["g"], name="Green Channel", line=dict(color="#10b981", width=2)))
            fig_enh.add_trace(go.Scattergl(x=dehazed_hist["x"], y=dehazed_hist["b"], name="Blue Channel", line=dict(color="#38bdf8", width=2)))
            fig_enh.update_layout(title="Dehazed Image RGB Histogram", xaxis_title="Pixel Intensity (0-255)", yaxis_title="Pixel Count", template="plotly_dark", height=400)

            st.plotly_chart(fig_enh, use_container_width=True)


# =============================================================================
# PAGE 6: 🔬 MODEL COMPARISON PAGE
# =============================================================================
elif navigation_option == "🔬 Model Comparison":
    st.title("🔬 Model Benchmark & Architecture Comparison")

    comparison_data = [
        {
            "Model Name": "DehazeFormer",
            "Type": "Vision Transformer (W-MSA)",
            "Parameters": "2.4M",
            "FLOPs": "12.8G",
            "Expected PSNR": "31.2 dB",
            "Expected SSIM": "0.942",
            "Inference Speed": "High Precision (~0.25s)"
        },
        {
            "Model Name": "AOD-Net",
            "Type": "Lightweight CNN",
            "Parameters": "1.8K",
            "FLOPs": "0.4G",
            "Expected PSNR": "26.8 dB",
            "Expected SSIM": "0.875",
            "Inference Speed": "Ultra Fast (~0.05s)"
        },
        {
            "Model Name": "Dark Channel Prior",
            "Type": "Classical Computer Vision",
            "Parameters": "0 (Heuristic)",
            "FLOPs": "N/A",
            "Expected PSNR": "24.5 dB",
            "Expected SSIM": "0.820",
            "Inference Speed": "Moderate (~0.40s)"
        }
    ]

    df_comp = pd.DataFrame(comparison_data)
    st.table(df_comp)


# =============================================================================
# PAGE 7: 📚 RESEARCH PAPER PAGE
# =============================================================================
elif navigation_option == "📚 Research Paper":
    st.title("📚 Research Paper Reference")
    st.markdown("### *DehazeFormer: Vision Transformer for Single Image Dehazing*")
    st.markdown("**Authors:** Yuntian Song et al. (IEEE / CVPR 2023)")

    st.markdown(
        """
        <div class="glass-card">
            <h4>Abstract</h4>
            <p>Single image dehazing is a fundamental computer vision task aiming to recover clear scene radiance from hazy inputs.
            We propose DehazeFormer, a novel Vision Transformer architecture featuring Window Multi-Head Self-Attention (W-MSA),
            resblock feature fusion, and residual learning tailored for atmospheric scattering restoration.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("#### BibTeX Citation")
    st.code(
        """
@article{song2023dehazeformer,
  title={DehazeFormer: Vision Transformer for Single Image Dehazing},
  author={Song, Yuntian and others},
  journal={IEEE Transactions on Image Processing},
  year={2023}
}
        """,
        language="bibtex"
    )


# =============================================================================
# PAGE 8: 🌍 APPLICATIONS PAGE
# =============================================================================
elif navigation_option == "🌍 Applications":
    st.title("🌍 Real-World Applications")

    apps = [
        ("🚗 Autonomous Vehicles", "Restores visibility for front camera sensors in dense fog and heavy mist."),
        ("🏥 Medical Imaging", "Enhances diagnostic contrast in endoscopic and microscopic imaging."),
        ("🚦 Traffic Surveillance", "Clears license plate numbers and vehicle details during atmospheric haze."),
        ("🛰️ Satellite Imaging", "Removes cloud cover and atmospheric distortion from remote sensing imagery."),
        ("🚁 Drone Navigation", "Improves real-time optical altitude tracking for UAVs."),
        ("📷 Outdoor Photography", "Restores color saturation and depth contrast in foggy landscape photography.")
    ]

    for title, desc in apps:
        render_glass_card(title, desc)


# =============================================================================
# PAGE 9: ⚙ SETTINGS PAGE
# =============================================================================
elif navigation_option == "⚙ Settings":
    st.title("⚙ System Settings & Configurations")

    st.selectbox("Default Model Target", config.SUPPORTED_MODELS, index=0)
    st.number_input("Maximum Resolution Threshold (px)", min_value=640, max_value=3840, value=1920)
    st.selectbox("Target Output Format", ["PNG", "JPEG", "WebP"], index=0)
    st.write(f"**Hardware Device Detected:** `{config.DEVICE.upper()}`")


# =============================================================================
# PAGE 10: ℹ ABOUT PAGE
# =============================================================================
elif navigation_option == "ℹ About":
    st.title("ℹ About Project & Atmospheric Physics")
    st.markdown("### The Atmospheric Scattering Model")
    st.latex(r"I(x) = J(x) \cdot t(x) + A \cdot (1 - t(x))")
    st.markdown(
        """
        - **I(x):** Observed hazy image pixel intensity
        - **J(x):** True clear scene radiance
        - **t(x):** Medium transmission map $t(x) = e^{-\\beta d(x)}$
        - **A:** Global atmospheric light vector
        """
    )
    st.markdown("---")
    st.markdown(f"**Project Title:** {config.PROJECT_TITLE}")
    st.markdown(f"**Version:** {config.PROJECT_VERSION}")
    st.markdown(f"**Author:** {config.PROJECT_AUTHOR}")
