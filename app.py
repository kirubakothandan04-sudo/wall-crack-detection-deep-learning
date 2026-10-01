import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import json
import os
import base64
import glob
import pandas as pd
import random
import torch

from predict_csaf import load_model, preprocess, analyze_crack, to_base64
from gradcam import (
    GradCAM, get_target_layer, get_fine_target_layer,
    overlay_heatmap, combined_crack_mask, mask_linearity_score, CNN_BRANCH_VARIANTS,
)
from sliding_window_detect import sliding_window_detect, ensemble_sliding_window_detect, DEFAULT_ENSEMBLE_VARIANTS

st.set_page_config(
    page_title="CSAF-Net: Wall Crack Detection & Analytics",
    page_icon="🏗️",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

.hero-banner {
    background: linear-gradient(120deg, #1E3A8A 0%, #2563EB 55%, #0EA5E9 100%);
    padding: 2.2rem 2rem;
    border-radius: 14px;
    margin-bottom: 1.6rem;
    box-shadow: 0 8px 24px rgba(37, 99, 235, 0.25);
}
.hero-banner h1 {
    color: #FFFFFF;
    font-size: 2.1rem;
    font-weight: 800;
    margin: 0 0 0.35rem 0;
}
.hero-banner p {
    color: #DBEAFE;
    font-size: 1.02rem;
    margin: 0;
}

.metric-card {
    background: #161B22;
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 12px;
    padding: 1rem 1.1rem;
    text-align: center;
}
.metric-card .metric-label {
    color: #94A3B8;
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
.metric-card .metric-value {
    color: #F1F5F9;
    font-size: 1.9rem;
    font-weight: 800;
    margin-top: 0.2rem;
}

.finding-card {
    background: linear-gradient(135deg, rgba(37, 99, 235, 0.14), rgba(14, 165, 233, 0.08));
    border-left: 4px solid #2563EB;
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    margin: 0.6rem 0 1.2rem 0;
}
.finding-card h4 {
    margin: 0 0 0.5rem 0;
    color: #93C5FD;
    font-size: 1.02rem;
}
.finding-card p {
    margin: 0;
    color: #E2E8F0;
    font-size: 0.95rem;
    line-height: 1.5;
}

div[data-baseweb="tab-list"] {
    gap: 4px;
}
button[data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-banner">
    <h1>🏗️ CSAF-Net</h1>
    <p>Gated Cross-Scale Attention Fusion of DenseNet-121 &amp; ViT for structural wall-crack detection.
    Explore live predictions, training analytics, and the full ablation study behind the model's design.</p>
</div>
""", unsafe_allow_html=True)


def metric_card(col, label, value):
    """Renders one styled metric card inside a given Streamlit column."""
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def finding_card(title, body_html):
    """Renders a styled callout card for a 'Key Finding' section."""
    st.markdown(f"""
    <div class="finding-card">
        <h4>🔑 {title}</h4>
        <p>{body_html}</p>
    </div>
    """, unsafe_allow_html=True)


# PIL-based lightweight image augmentation to prevent macOS Keras threading conflicts
def augment_image_pil(image):
    angle = random.uniform(-30, 30)
    img = image.rotate(angle)

    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    enhancer = ImageEnhance.Brightness(img)
    brightness = random.uniform(0.7, 1.3)
    img = enhancer.enhance(brightness)

    zoom_factor = random.uniform(0.8, 1.0)
    w, h = img.size
    new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    img = img.crop((left, top, left + new_w, top + new_h))
    img = img.resize((w, h), Image.Resampling.LANCZOS)

    return img


def load_training_logs():
    """Load all logs/<variant>_log.json files saved by train.py."""
    logs = {}
    for path in sorted(glob.glob("logs/*.json")):
        variant = os.path.basename(path).replace("_log.json", "")
        with open(path) as f:
            logs[variant] = json.load(f)
    return logs


VARIANT_LABELS = {
    "cnn_only": "CNN Only (DenseNet-121)",
    "vit_only": "ViT Only",
    "concat": "Concat Fusion",
    "ungated_cross_attn": "Cross-Attn (No Gate)",
    "scalar_gate": "Scalar Gate",
    "full_csaf": "CSAF-Net (Full, Gated)",
}
VARIANT_ORDER = ["cnn_only", "vit_only", "concat", "ungated_cross_attn", "scalar_gate", "full_csaf"]


# ---------------------------------------------------------------------------
# Cached model loading -- runs ONCE per variant per session instead of once
# per prediction. Streamlit keys the cache by the function's arguments, so
# switching the variant dropdown loads a fresh model only when needed.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def get_cached_model(variant):
    model, device = load_model(variant=variant)
    return model, device


@st.cache_resource(show_spinner="Preparing Grad-CAM...")
def get_cached_gradcam_engine(variant):
    """
    Builds (once per variant per session) a GradCAM engine hooked onto the
    variant's DenseNet-121 branch, reusing the already-cached model so we
    don't load weights twice. Returns (None, None) for variants with no
    CNN branch (vit_only).
    """
    if variant not in CNN_BRANCH_VARIANTS:
        return None, None
    model, device = get_cached_model(variant)
    target_layer = get_target_layer(model, variant)
    engine = GradCAM(model, target_layer)
    return engine, device


@st.cache_resource(show_spinner="Preparing fine-resolution Grad-CAM...")
def get_cached_fine_gradcam_engine(variant):
    """
    Same idea as get_cached_gradcam_engine, but hooked onto an EARLIER,
    higher-resolution DenseNet block (14x14 instead of 7x7). Used only for
    the combined Grad-CAM + Canny segmentation, where spatial precision
    matters more than the semantic strength of the final layer -- see
    get_fine_target_layer's docstring in gradcam.py.
    """
    if variant not in CNN_BRANCH_VARIANTS:
        return None, None
    model, device = get_cached_model(variant)
    target_layer = get_fine_target_layer(model, variant)
    engine = GradCAM(model, target_layer)
    return engine, device


def run_prediction(image, variant):
    """
    Runs prediction in-process using the cached model -- no subprocess,
    no temp files, no JSON parsing. Mirrors the dict shape that
    predict_csaf.py used to print, so the rest of app.py stays unchanged.
    """
    model, device = get_cached_model(variant)

    # preprocess() in predict_csaf.py takes a path; reproduce it here for
    # an in-memory PIL image instead of round-tripping through disk.
    from torchvision import transforms
    IMAGE_SIZE = 224
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _ = model(img_tensor)
        prediction = torch.sigmoid(logits).item()

    result = {
        "prediction": prediction,
        "variant": variant,
        "crack_type": None,
        "canny_img": None,
        "annotated_img": None,
    }

    crack_type, edges, annotated_img = analyze_crack(image)
    result["canny_img"] = to_base64(edges)
    result["annotated_img"] = to_base64(annotated_img)

    if prediction > 0.5:
        result["crack_type"] = crack_type

    return result


# Tabs structure
tab1, tab2, tab3 = st.tabs([
    "🔍 Live Prediction", "📊 Model Training Analytics", "⚔️ Ablation Comparison"
])

with tab1:
    st.title("🏗️ Wall Crack Identification — CSAF-Net")
    st.write("Upload or drag & drop an image to detect wall cracks using the gated cross-scale "
             "attention fusion model (DenseNet-121 + ViT).")

    available_variants = [v for v in VARIANT_ORDER
                           if os.path.exists(f"checkpoints/{v}_best.pth")]
    if not available_variants:
        st.warning("No trained checkpoints found in checkpoints/. Run train.py first.")
    else:
        selected_variant = st.selectbox(
            "Model variant",
            options=available_variants,
            format_func=lambda v: VARIANT_LABELS.get(v, v),
            index=available_variants.index("full_csaf") if "full_csaf" in available_variants else 0,
        )

        scan_mode = st.radio(
            "Scan mode",
            ["Single Crop (pre-cropped wall patch)", "Wide-Photo Scan (multi-region)"],
            horizontal=True,
        )

        # ------------------------------------------------------------------
        # MODE 1: Single Crop -- same behavior as before (prediction,
        # Canny/Hough, Grad-CAM, augmentation retry) on one 224x224 patch.
        # ------------------------------------------------------------------
        if scan_mode == "Single Crop (pre-cropped wall patch)":
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"], key="single_uploader")

            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")

                col1, col2 = st.columns([1, 1])

                with col1:
                    st.subheader("Uploaded Image")
                    st.image(image, caption="Original Image", width="stretch")

                with col2:
                    st.subheader("Analysis Results")

                    with st.spinner("Analyzing image..."):
                        try:
                            data = run_prediction(image, selected_variant)

                            if "error" in data:
                                st.error(f"Error during prediction: {data['error']}")
                            else:
                                prediction = data.get("prediction", 0)
                                crack_type = data.get("crack_type")

                                is_crack = prediction > 0.5
                                confidence = prediction if is_crack else (1 - prediction)

                                if is_crack:
                                    st.error(f"🚨 **Crack Detected** ({crack_type})")
                                    st.progress(float(confidence))
                                    st.write(f"**Confidence:** {confidence*100:.2f}%")
                                else:
                                    st.success("✅ **No Crack Detected**")
                                    st.progress(float(confidence))
                                    st.write(f"**Confidence:** {confidence*100:.2f}%")

                                st.caption(f"Model used: {VARIANT_LABELS.get(selected_variant, selected_variant)}")

                                st.markdown("---")
                                st.subheader("Image Feature Visualizations")
                                vis_col1, vis_col2 = st.columns(2)

                                if data.get("canny_img") and data.get("annotated_img"):
                                    canny_bytes = base64.b64decode(data["canny_img"])
                                    annotated_bytes = base64.b64decode(data["annotated_img"])

                                    with vis_col1:
                                        st.caption("Canny Edge Detection Map")
                                        st.image(canny_bytes, width="stretch")
                                    with vis_col2:
                                        st.caption("Detected Hough Crack Paths (Red)")
                                        st.image(annotated_bytes, width="stretch")

                                st.markdown("---")
                                st.subheader("Grad-CAM: Where the CNN Branch Looks")
                                if selected_variant not in CNN_BRANCH_VARIANTS:
                                    st.caption(
                                        "Grad-CAM isn't available for ViT Only -- it has no "
                                        "DenseNet-121 branch to visualize."
                                    )
                                else:
                                    with st.spinner("Computing Grad-CAM..."):
                                        try:
                                            gc_engine, gc_device = get_cached_gradcam_engine(selected_variant)
                                            from torchvision import transforms as _t
                                            _IMAGE_SIZE = 224
                                            _MEAN = [0.485, 0.456, 0.406]
                                            _STD = [0.229, 0.224, 0.225]
                                            gc_transform = _t.Compose([
                                                _t.Resize((_IMAGE_SIZE, _IMAGE_SIZE)),
                                                _t.ToTensor(),
                                                _t.Normalize(_MEAN, _STD),
                                            ])
                                            gc_tensor = gc_transform(image).unsqueeze(0)
                                            cam, gc_confidence = gc_engine.generate(gc_tensor, gc_device)
                                            overlay = overlay_heatmap(cam, image)
                                            st.image(
                                                overlay,
                                                caption=f"CNN branch attention (confidence: {gc_confidence*100:.1f}%)",
                                                width="stretch",
                                            )
                                            st.caption(
                                                "Highlights the region the DenseNet-121 branch focused on for "
                                                "this prediction. The final decision also incorporates the "
                                                "ViT branch and the learned gate, so this shows only one "
                                                "branch's contribution, not the full model's reasoning."
                                                )
                                        except Exception as e_cam:
                                            st.error(f"Grad-CAM failed: {e_cam}")

                                st.markdown("---")
                                if st.button("Apply Augmentation & Predict Again"):
                                    aug_image = augment_image_pil(image)
                                    st.subheader("Augmented Prediction")
                                    st.image(aug_image, caption="Augmented Image", width="stretch")

                                    with st.spinner("Analyzing augmented image..."):
                                        try:
                                            data_aug = run_prediction(aug_image, selected_variant)

                                            if "error" in data_aug:
                                                st.error(f"Error during augmented prediction: {data_aug['error']}")
                                            else:
                                                aug_pred = data_aug.get("prediction", 0)
                                                aug_crack_type = data_aug.get("crack_type")

                                                if aug_pred > 0.5:
                                                    st.error("🚨 **Augmented Prediction: Crack Detected**")
                                                    st.write(f"**Crack Type:** {aug_crack_type}")
                                                    st.write(f"**Confidence:** {aug_pred*100:.2f}%")
                                                else:
                                                    st.success("✅ **Augmented Prediction: No Crack**")
                                                    st.write(f"**Confidence:** {(1-aug_pred)*100:.2f}%")

                                        except Exception as e_aug:
                                            st.error(f"Failed to run prediction on augmented image. Error: {e_aug}")

                        except Exception as e:
                            st.error(f"Failed to run prediction. Error: {e}")

        # ------------------------------------------------------------------
        # MODE 2: Wide-Photo Scan -- localization stage (sliding window)
        # + dual verification (Canny/Hough + Grad-CAM) per detected region.
        # Reuses the same trained classifier selected above; no retraining.
        # ------------------------------------------------------------------
        else:
            st.write(
                "CSAF-Net was trained on close-range, pre-cropped 224×224 wall patches "
                "(SDNET2018). A wide-angle photo shrinks any crack down to a handful of "
                "pixels after resizing, and the model has never seen full-scene context "
                "(furniture, text, fixtures). This mode adds a **localization stage** in "
                "front of the same trained classifier selected above: it slides a 224×224 "
                "window across the photo, classifies each window, and draws boxes around "
                "regions predicted as cracks."
            )
            st.info(
                "⚠️ Known limitation: since the classifier has only ever seen wall texture "
                "(never text, objects, or fixtures), it can mistake high-contrast edges in "
                "printed text, screws, or fittings for cracks. This is a dataset-coverage "
                "limitation, not a bug -- see the Verify section below and the project's "
                "Future Work discussion."
            )

            sw_use_ensemble = st.checkbox(
                "Use ensemble mode (require agreement across multiple trained models)",
                value=True, key="sw_use_ensemble",
            )
            if sw_use_ensemble:
                st.caption(
                    f"Ensemble mode polls **{', '.join(VARIANT_LABELS.get(v, v) for v in DEFAULT_ENSEMBLE_VARIANTS)}** "
                    "on every window (ignoring the model dropdown above) and only keeps a "
                    "detection when enough of them independently agree it's a crack. "
                    "Different architectures tend to be fooled by different things -- a "
                    "window frame that fools the fused model may not fool the single-backbone "
                    "models -- so requiring agreement can filter out some false positives a "
                    "single model would keep. This runs 3 full models per window, so scanning "
                    "is noticeably slower."
                )
                sw_min_agree = st.slider(
                    "Minimum models that must agree", 1, len(DEFAULT_ENSEMBLE_VARIANTS),
                    2, 1, key="sw_min_agree",
                )

            sw_col1, sw_col2 = st.columns(2)
            with sw_col1:
                sw_threshold = st.slider(
                    "Confidence threshold (per model)" if sw_use_ensemble else "Confidence threshold",
                    0.50, 0.99,
                    0.90 if sw_use_ensemble else 0.99,
                    0.01,
                    key="sw_threshold_ens" if sw_use_ensemble else "sw_threshold_single",
                )
            with sw_col2:
                sw_stride = st.slider(
                    "Window stride (px) -- smaller = more thorough, slower",
                    56, 224, 112, 8, key="sw_stride"
                )

            sw_use_wall_filter = st.checkbox(
                "Filter out non-wall backgrounds (sky, paper, flat surfaces)",
                value=True, key="sw_use_wall_filter",
            )
            sw_wall_strictness = st.slider(
                "Wall-texture strictness -- higher rejects more borderline surfaces",
                0.0, 30.0, 0.0, 1.0, key="sw_wall_strictness",
                disabled=not sw_use_wall_filter,
            )
            st.caption(
                "Heuristic pre-filter: rejects a detection if the region's background lacks "
                "wall-like micro-texture (e.g. a wire against open sky, glass, or a smooth "
                "painted surface), regardless of classifier confidence. Not a trained "
                "segmentation model -- a simple texture-variance check. Raise strictness if "
                "you still see false positives on smooth/non-wall surfaces; lower it if real "
                "cracks on very smooth walls are being rejected."
            )

            sw_uploaded = st.file_uploader(
                "Upload a wide-angle photo to scan", type=["jpg", "png", "jpeg"], key="sw_uploader"
            )
            st.caption(
                "No photo handy? Use the button below to generate a synthetic wide-shot demo "
                "-- real crack and no-crack crops from the test set tiled into a larger canvas."
            )
            use_synthetic = st.button("Generate synthetic demo instead")

            sw_image = None
            sw_true_box = None
            if sw_uploaded is not None:
                sw_image = Image.open(sw_uploaded).convert("RGB")
            elif use_synthetic:
                from sliding_window_detect import make_synthetic_wide_demo
                sw_image, sw_true_box = make_synthetic_wide_demo()

            if sw_image is not None and sw_true_box is None:
                # Only offer manual cropping for real uploaded photos, not the
                # synthetic demo (which is already a clean wall-only canvas).
                st.markdown("---")
                st.write(
                    "**Optional: narrow the scan to just the wall surface.** The "
                    "classifier was trained only on close-up wall texture (never "
                    "windows, glass, sky, or text) -- cropping those out before "
                    "scanning gives a much cleaner result on real photos that "
                    "include non-wall content."
                )
                orig_w, orig_h = sw_image.size
                crop_on = st.checkbox("Crop before scanning", value=False, key="sw_crop_on")
                if crop_on:
                    cc1, cc2 = st.columns(2)
                    with cc1:
                        left_pct = st.slider("Left edge (%)", 0, 95, 0, key="sw_crop_left")
                        top_pct = st.slider("Top edge (%)", 0, 95, 0, key="sw_crop_top")
                    with cc2:
                        right_pct = st.slider("Right edge (%)", 5, 100, 100, key="sw_crop_right")
                        bottom_pct = st.slider("Bottom edge (%)", 5, 100, 100, key="sw_crop_bottom")

                    left_px = int(orig_w * left_pct / 100)
                    top_px = int(orig_h * top_pct / 100)
                    right_px = int(orig_w * right_pct / 100)
                    bottom_px = int(orig_h * bottom_pct / 100)

                    if right_px > left_px + 20 and bottom_px > top_px + 20:
                        sw_image = sw_image.crop((left_px, top_px, right_px, bottom_px))
                        st.image(sw_image, caption="Cropped region to scan", width="stretch")
                    else:
                        st.warning("Crop region too small -- adjust the sliders above.")

            if sw_image is not None:
                if not sw_use_ensemble and selected_variant not in CNN_BRANCH_VARIANTS:
                    st.warning(
                        "Wide-Photo Scan needs a variant with a DenseNet-121 branch -- "
                        "ViT Only isn't supported here. Choose a different model variant above, "
                        "or enable ensemble mode."
                    )
                else:
                    scan_label = "ensemble" if sw_use_ensemble else selected_variant
                    with st.spinner(f"Scanning {sw_image.size[0]}x{sw_image.size[1]} image ({scan_label})..."):
                        if sw_use_ensemble:
                            annotated, detections = ensemble_sliding_window_detect(
                                sw_image,
                                variants=DEFAULT_ENSEMBLE_VARIANTS,
                                stride=sw_stride,
                                threshold=sw_threshold,
                                min_agree=sw_min_agree,
                                use_wall_filter=sw_use_wall_filter,
                                wall_texture_std_threshold=sw_wall_strictness,
                            )
                        else:
                            annotated, detections = sliding_window_detect(
                                sw_image,
                                variant=selected_variant,
                                stride=sw_stride,
                                threshold=sw_threshold,
                                use_wall_filter=sw_use_wall_filter,
                                wall_texture_std_threshold=sw_wall_strictness,
                            )

                    if sw_true_box:
                        x1, y1, x2, y2 = sw_true_box
                        annotated_display = annotated.copy()
                        cv2.rectangle(annotated_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        annotated = annotated_display

                    st.image(annotated, caption=f"{len(detections)} region(s) detected", width="stretch")

                    if sw_use_ensemble and detections:
                        st.write("**Per-model vote breakdown:**")
                        for i, d in enumerate(detections, start=1):
                            vote_str = ", ".join(
                                f"{VARIANT_LABELS.get(v, v)}: {c*100:.0f}%" for v, c in d["votes"].items()
                            )
                            st.write(f"{i}. {vote_str}")

                    def describe_location(box, img_w, img_h):
                        """Coarse, honest location description from box position --
                        NOT a floor/story estimate (that needs camera geometry we
                        don't have), just where in the FRAME the region sits."""
                        x1, y1, x2, y2 = box
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        vert = "top" if cy < img_h / 3 else ("bottom" if cy > 2 * img_h / 3 else "middle")
                        horiz = "left" if cx < img_w / 3 else ("right" if cx > 2 * img_w / 3 else "center")
                        if vert == "middle" and horiz == "center":
                            return "center of frame"
                        return f"{vert}-{horiz} of frame"

                    def risk_level(confidence, is_linear):
                        """
                        Confidence banding, DOWNGRADED when the segmentation mask
                        found no clearly line-like (elongated) structure -- i.e.
                        the classifier is confident, but nothing in the region
                        actually looks like a thin crack line rather than a
                        blocky architectural feature (window surround, stonework).
                        Still not a calibrated structural risk score -- a
                        transparent, explainable summary tier only.
                        """
                        if not is_linear:
                            return "⚪ Unclear (no linear shape found)"
                        if confidence >= 0.90:
                            return "🔴 High"
                        elif confidence >= 0.75:
                            return "🟠 Medium"
                        return "🟡 Low"

                    # Precompute Grad-CAM + segmentation + linearity ONCE per
                    # detection here, so both the summary table below and the
                    # per-region verification panels reuse the same result
                    # instead of recomputing it twice. In ensemble mode, use
                    # full_csaf for this stage regardless of the dropdown,
                    # since it's guaranteed to have a CNN branch.
                    gradcam_variant = "full_csaf" if sw_use_ensemble else selected_variant
                    gc_engine_sw, gc_device_sw = get_cached_gradcam_engine(gradcam_variant)
                    gc_fine_engine_sw, gc_fine_device_sw = get_cached_fine_gradcam_engine(gradcam_variant)

                    from torchvision import transforms as _sw_t
                    _sw_transform = _sw_t.Compose([
                        _sw_t.Resize((224, 224)),
                        _sw_t.ToTensor(),
                        _sw_t.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ])

                    LINEARITY_ASPECT_THRESHOLD = 2.5  # long/short side ratio to count as "line-like"

                    per_detection = []  # cached results, one dict per detection, aligned by index
                    if detections:
                        for d in detections:
                            x1, y1, x2, y2 = d["box"]
                            crop = sw_image.crop((x1, y1, x2, y2))
                            entry = {"crop": crop, "cam": None, "gc_conf": None,
                                     "fine_cam": None, "seg_overlay": None, "seg_mask": None,
                                     "aspect": 0.0, "is_linear": False, "error": None}
                            try:
                                if gc_engine_sw is not None:
                                    gc_tensor = _sw_transform(crop).unsqueeze(0)
                                    cam_result, gc_conf = gc_engine_sw.generate(gc_tensor, gc_device_sw)
                                    entry["cam"] = cam_result
                                    entry["gc_conf"] = gc_conf
                                if gc_fine_engine_sw is not None:
                                    fine_tensor = _sw_transform(crop).unsqueeze(0)
                                    fine_cam, _ = gc_fine_engine_sw.generate(fine_tensor, gc_fine_device_sw)
                                    seg_overlay, seg_mask = combined_crack_mask(fine_cam, crop)
                                    entry["fine_cam"] = fine_cam
                                    entry["seg_overlay"] = seg_overlay
                                    entry["seg_mask"] = seg_mask
                                    aspect = mask_linearity_score(seg_mask)
                                    entry["aspect"] = aspect
                                    entry["is_linear"] = aspect >= LINEARITY_ASPECT_THRESHOLD
                            except Exception as e_pre:
                                entry["error"] = str(e_pre)
                            per_detection.append(entry)

                    if detections:
                        img_w, img_h = sw_image.size
                        st.write("**Detection summary:**")
                        summary_rows = []
                        for i, (d, pre) in enumerate(zip(detections, per_detection), start=1):
                            summary_rows.append({
                                "Region": i,
                                "Location": describe_location(d["box"], img_w, img_h),
                                "Confidence": f"{d['confidence']*100:.1f}%",
                                "Shape": "📏 Linear" if pre["is_linear"] else "⬛ Blocky/unclear",
                                "Risk": risk_level(d["confidence"], pre["is_linear"]),
                            })
                        st.table(pd.DataFrame(summary_rows).set_index("Region"))
                        st.caption(
                            "Location is the region's coarse position within the photo frame "
                            "(not a floor/story estimate, which would need camera geometry this "
                            "app doesn't have). Shape checks whether the segmentation mask (Grad-CAM "
                            "∩ Canny) contains an elongated, line-like structure versus a blocky "
                            "region typical of window surrounds or stonework; Risk is downgraded to "
                            "'Unclear' when no linear shape is found, regardless of classifier "
                            "confidence. None of this is a calibrated structural risk score -- treat "
                            "it as a prioritization aid for manual inspection, not an engineering "
                            "assessment."
                        )
                    else:
                        st.info("No crack regions detected above the current threshold.")

                    if detections:
                        st.markdown("---")
                        st.subheader("🔬 Verify Each Detected Region")
                        st.write(
                            "Each box is cross-checked: **Canny/Hough crack-type analysis** (the "
                            "same geometric check used above in Single Crop mode), **Grad-CAM** "
                            "(where the CNN branch focused), and a **Shape check** (does the "
                            "segmentation mask form a line, or a blocky region?). Regions where "
                            "all three agree are more trustworthy; disagreement -- e.g. high "
                            "confidence but a blocky, non-linear mask -- is a useful signal to "
                            "flag for human review rather than trust automatically."
                        )

                        for i, (d, pre) in enumerate(zip(detections, per_detection), start=1):
                            crop = pre["crop"]
                            loc = describe_location(d["box"], img_w, img_h)
                            risk = risk_level(d["confidence"], pre["is_linear"])
                            with st.expander(
                                f"Region {i} — {risk} — {d['confidence']*100:.1f}% confidence — {loc}"
                            ):
                                vcol1, vcol2, vcol3, vcol4 = st.columns(4)

                                with vcol1:
                                    st.caption("Cropped region")
                                    st.image(crop, width="stretch")

                                try:
                                    crack_type, edges, annotated_crop = analyze_crack(crop)
                                    with vcol2:
                                        st.caption(f"Canny/Hough: {crack_type}")
                                        st.image(annotated_crop, width="stretch")
                                except Exception as e_geo:
                                    with vcol2:
                                        st.caption("Canny/Hough analysis unavailable")
                                        st.error(str(e_geo))

                                if pre["cam"] is not None:
                                    overlay_crop = overlay_heatmap(pre["cam"], crop)
                                    with vcol3:
                                        st.caption(f"Grad-CAM ({pre['gc_conf']*100:.1f}% crack)")
                                        st.image(overlay_crop, width="stretch")
                                else:
                                    with vcol3:
                                        st.caption("Grad-CAM not available for this variant")

                                if pre["seg_overlay"] is not None:
                                    with vcol4:
                                        shape_label = "📏 Linear" if pre["is_linear"] else "⬛ Blocky/unclear"
                                        st.caption(f"Segmentation -- {shape_label} (aspect {pre['aspect']:.1f}x)")
                                        st.image(pre["seg_overlay"], width="stretch")
                                else:
                                    with vcol4:
                                        st.caption("Segmentation needs Grad-CAM (unavailable above)")
                                        if pre["error"]:
                                            st.error(pre["error"])

                        st.caption(
                            "**Segmentation column**: a heuristic pixel mask, not a trained "
                            "segmentation model (SDNET2018 has no pixel-level crack annotations "
                            "to train one). It keeps only the Canny edge pixels that also fall "
                            "inside the model's high-attention region -- computed from an "
                            "earlier DenseNet-121 block (14×14 resolution) rather than the "
                            "final 7×7 layer used for the Grad-CAM column, since finer spatial "
                            "resolution reduces conflating a thin crack with a nearby window "
                            "frame in the same attention cell. **Shape** fits a rotated bounding "
                            "box to the mask and reports its long/short side ratio -- a real "
                            "crack tends to be thin and elongated, while a window frame or "
                            "texture patch tends to be blocky. Both are heuristic combinations "
                            "of existing signals, not a learned segmentation or shape model; a "
                            "model trained on real pixel-mask annotations would be more precise "
                            "and is noted as Future Work."
                        )

with tab2:
    st.title("📊 CSAF-Net Training Performance")
    logs = load_training_logs()

    if "full_csaf" not in logs:
        st.warning("No training log found for full_csaf. Run train.py --variant full_csaf first.")
    else:
        history = logs["full_csaf"]["history"]
        epochs = [h["epoch"] for h in history]
        history_df = pd.DataFrame({
            "Epoch": epochs,
            "Training Accuracy": [h["train"]["accuracy"] for h in history],
            "Validation Accuracy": [h["val"]["accuracy"] for h in history],
            "Training Loss": [h["train"]["loss"] for h in history],
            "Validation Loss": [h["val"]["loss"] for h in history],
        })

        col_metric1, col_metric2 = st.columns(2)

        with col_metric1:
            st.subheader("📈 Model Accuracy History")
            st.line_chart(history_df.set_index("Epoch")[["Training Accuracy", "Validation Accuracy"]])
            final_val_acc = history_df["Validation Accuracy"].iloc[-1]
            st.caption(f"Validation accuracy reaches {final_val_acc*100:.1f}% by the final epoch.")

        with col_metric2:
            st.subheader("📉 Model Loss History")
            st.line_chart(history_df.set_index("Epoch")[["Training Loss", "Validation Loss"]])
            final_val_loss = history_df["Validation Loss"].iloc[-1]
            st.caption(f"Validation loss reaches {final_val_loss:.3f} by the final epoch.")

        st.markdown("---")
        st.subheader("📋 Evaluation Summary (Test Dataset) — CSAF-Net (Full, Gated)")

        test_metrics = logs["full_csaf"]["test"]
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        metric_card(mcol1, "Accuracy", f"{test_metrics['accuracy']*100:.2f}%")
        metric_card(mcol2, "F1-Score", f"{test_metrics['f1']*100:.2f}%")
        metric_card(mcol3, "Precision", f"{test_metrics['precision']*100:.2f}%")
        metric_card(mcol4, "Recall", f"{test_metrics['recall']*100:.2f}%")

with tab3:
    st.title("⚔️ Ablation Comparison")
    st.write("Comparing all fusion strategies tested in this project. This is the core evidence "
             "for CSAF-Net's novelty: the gated fusion module outperforms every simpler alternative.")

    logs = load_training_logs()
    available = [v for v in VARIANT_ORDER if v in logs]

    if not available:
        st.warning("No training logs found. Run train.py for each variant first.")
    else:
        if "full_csaf" in available:
            hc1, hc2, hc3, hc4 = st.columns(4)
            fc_metrics = logs["full_csaf"]["test"]
            metric_card(hc1, "Best Accuracy", f"{fc_metrics['accuracy']*100:.2f}%")
            metric_card(hc2, "Best F1-Score", f"{fc_metrics['f1']*100:.2f}%")
            metric_card(hc3, "Best Recall", f"{fc_metrics['recall']*100:.2f}%")
            if "ungated_cross_attn" in available:
                fn_reduction = (1 - (1 - fc_metrics["recall"]) /
                                (1 - logs["ungated_cross_attn"]["test"]["recall"])) * 100
                metric_card(hc4, "Fewer Missed Cracks", f"-{fn_reduction:.0f}%")
            st.markdown("<br>", unsafe_allow_html=True)

        comparison_data = {
            "Model Variant": [VARIANT_LABELS[v] for v in available],
            "Accuracy": [logs[v]["test"]["accuracy"] for v in available],
            "F1-Score": [logs[v]["test"]["f1"] for v in available],
            "Precision": [logs[v]["test"]["precision"] for v in available],
            "Recall": [logs[v]["test"]["recall"] for v in available],
        }
        comp_df = pd.DataFrame(comparison_data)

        display_df = comp_df.copy()
        for col in ["Accuracy", "F1-Score", "Precision", "Recall"]:
            display_df[col] = (display_df[col] * 100).round(2).astype(str) + "%"
        st.table(display_df.set_index("Model Variant"))

        st.markdown("---")

        col_comp1, col_comp2 = st.columns(2)

        with col_comp1:
            st.subheader("📊 Accuracy & F1 Comparison")
            chart_df = comp_df.set_index("Model Variant")[["Accuracy", "F1-Score"]]
            st.bar_chart(chart_df)

        with col_comp2:
            st.subheader("📊 Precision & Recall Comparison")
            chart_df2 = comp_df.set_index("Model Variant")[["Precision", "Recall"]]
            st.bar_chart(chart_df2)

        st.markdown("---")
        st.subheader("🎯 Confusion Matrices")
        if os.path.exists("figures/confusion_matrices.png"):
            st.image("figures/confusion_matrices.png", width="stretch")

        st.subheader("🌡️ Gate Activation Behavior")
        if os.path.exists("figures/gate_distribution.png"):
            st.image("figures/gate_distribution.png", width="stretch")

        st.subheader("📈 Precision-Recall Curves")
        if os.path.exists("figures/pr_curves.png"):
            st.image("figures/pr_curves.png", width="stretch")
            st.caption(
                "Average Precision is comparable across all attention-based variants, "
                "confirming the gate's benefit is a shift in operating point (precision/recall "
                "trade-off at the deployed threshold) rather than a general ranking improvement."
            )

        if "full_csaf" in available and "ungated_cross_attn" in available:
            gate_gain_f1 = (logs["full_csaf"]["test"]["f1"] - logs["ungated_cross_attn"]["test"]["f1"]) * 100
            gate_gain_recall = (logs["full_csaf"]["test"]["recall"] - logs["ungated_cross_attn"]["test"]["recall"]) * 100
            st.markdown("---")
            finding_card(
                "Effect of the Gate",
                f"Adding the learned gate on top of bidirectional cross-attention improves F1 by "
                f"<strong>{gate_gain_f1:+.2f} points</strong> and recall by "
                f"<strong>{gate_gain_recall:+.2f} points</strong> compared to ungated cross-attention "
                f"fusion — isolating the exact contribution of CSAF-Net's novelty."
            )

        if "scalar_gate" in available and "full_csaf" in available and "ungated_cross_attn" in available:
            scalar_vs_ungated_f1 = (logs["scalar_gate"]["test"]["f1"] - logs["ungated_cross_attn"]["test"]["f1"]) * 100
            full_vs_scalar_f1 = (logs["full_csaf"]["test"]["f1"] - logs["scalar_gate"]["test"]["f1"]) * 100
            finding_card(
                "Why an Adaptive (Per-Token) Gate?",
                f"A single, input-independent scalar gate already improves F1 by "
                f"<strong>{scalar_vs_ungated_f1:+.2f} points</strong> over no gate at all, confirming "
                f"that <em>having</em> a learned mixing weight helps. But making that gate "
                f"<strong>adaptive per input token</strong> (CSAF-Net's actual design) adds a further "
                f"<strong>{full_vs_scalar_f1:+.2f} points</strong> of F1 on top of the scalar gate — "
                f"showing the benefit comes specifically from the gate's input-conditioned flexibility, "
                f"not merely its presence."
            )