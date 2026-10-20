import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import json
import os
import base64
import glob
import hashlib
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


def load_multi_run_stats(variants):
    """
    For each variant, collects the test metrics of the original run (logs/) plus
    any seeded reruns saved by run_seeds.py under ~/csaf_seeds/logs/seed*/.
    Returns {variant: [test_metrics_dict, ...]} only for variants with 2+ runs.
    """
    root = os.path.expanduser("~/csaf_seeds")
    out = {}
    for v in variants:
        paths = [f"logs/{v}_log.json"] + sorted(
            glob.glob(os.path.join(root, "logs", "seed*", f"{v}_log.json"))
        )
        runs = []
        for p in paths:
            if os.path.exists(p):
                with open(p) as f:
                    runs.append(json.load(f)["test"])
        if len(runs) > 1:
            out[v] = runs
    return out


def run_mean(runs, key):
    return float(np.mean([r[key] for r in runs]))


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


def compute_occlusion_heat(image, variant, win=32, stride=16):
    """
    Whole-model occlusion map: how much the crack logit drops when a square of the
    (224x224-resized) image is replaced by the dataset-mean colour. Works for every
    variant because it only needs the model's output. Cached per image and variant.
    """
    key = (variant, hashlib.md5(image.tobytes()).hexdigest(), win, stride)
    cache = st.session_state.setdefault("occ_cache", {})
    if key in cache:
        return cache[key]

    from torchvision import transforms
    model, device = get_cached_model(variant)
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    x = tf(image).unsqueeze(0).to(device)
    pos = [(i, j) for i in range(0, 224 - win + 1, stride) for j in range(0, 224 - win + 1, stride)]
    heat = np.zeros((224, 224))
    cnt = np.zeros((224, 224))
    with torch.no_grad():
        base = model(x)[0].item()
        for s in range(0, len(pos), 32):
            chunk = pos[s:s + 32]
            xb = x.repeat(len(chunk), 1, 1, 1)
            for k, (i, j) in enumerate(chunk):
                xb[k, :, i:i + win, j:j + win] = 0.0
            drops = (base - model(xb)[0]).cpu().numpy()
            for (i, j), d in zip(chunk, drops):
                heat[i:i + win, j:j + win] += d
                cnt[i:i + win, j:j + win] += 1
    heat = np.maximum(heat / np.maximum(cnt, 1), 0)
    if heat.max() > 0:
        heat = heat / heat.max()
    cache[key] = heat
    return heat


def render_occlusion(image, variant):
    with st.spinner("Computing occlusion map..."):
        heat = compute_occlusion_heat(image, variant)
    st.image(
        overlay_heatmap(heat, image),
        caption=f"Whole-model occlusion map ({VARIANT_LABELS.get(variant, variant)})",
        width="stretch",
    )
    st.caption(
        "Shows how much the crack score drops when each square is hidden, using the full model "
        "(CNN branch, ViT branch and gate). On 25 annotated images its strongest point fell near the "
        "drawn crack in 84% of cases, versus 36% for the CNN-branch Grad-CAM above. It shows where the "
        "model's output is sensitive, not proof of how it reasons."
    )


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
        # MODE 1: Single Crop -- prediction, Canny/Hough, Grad-CAM, occlusion,
        # augmentation retry on one 224x224 patch.
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
                                    st.caption(
                                        "Canny/Hough marks any strong straight edge (window frames, door edges, "
                                        "wires as well as cracks), so treat it as auxiliary analysis only."
                                    )

                                st.markdown("---")
                                st.subheader("Grad-CAM: CNN Branch Only (Coarse)")
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
                                                "Shows where the DenseNet-121 branch is sensitive, on a coarse 7x7 grid. "
                                                "The final decision also uses the ViT branch and the learned gate, and "
                                                "this map is image-dependent: it does not always coincide with the crack "
                                                "(compare with the whole-model occlusion map below)."
                                            )
                                        except Exception as e_cam:
                                            st.error(f"Grad-CAM failed: {e_cam}")

                                st.markdown("---")
                                st.subheader("Pixel-Level Crack Localization")
                                st.info(
                                    "ℹ️ **Important**: CSAF-Net is a **classification model** trained on image-level labels "
                                    "(crack/no-crack), not pixel-level crack annotations. It classifies whether a crop contains "
                                    "a crack but **does not provide validated pixel-level segmentation**."
                                )
                                st.write(
                                    "**Why pixel-level localization is not available:**\n"
                                    "- The model was trained on SDNET2018, which has image-level labels only (no pixel masks)\n"
                                    "- CNN attention (shown in Grad-CAM above) indicates decision-relevant regions — texture, "
                                    "context, edges — not precise crack boundaries\n"
                                    "- Attention is typically diffuse across ~20% of the image, not focal on crack pixels\n"
                                    "- Intersection of attention and edge detection produces blocky patches, not thin crack lines"
                                )
                                st.write(
                                    "**For reliable pixel-level crack localization:**\n"
                                    "A dedicated segmentation model (U-Net, Mask R-CNN, DeepLabV3+) trained on pixel-annotated "
                                    "crack masks would be necessary. This requires a dataset with ground-truth crack boundaries "
                                    "drawn at the pixel level, not just image-level crack/no-crack labels."
                                )

                                st.subheader("Occlusion: Where the Whole Model Is Sensitive")
                                try:
                                    render_occlusion(image, selected_variant)
                                except Exception as e_occ:
                                    st.error(f"Occlusion map failed: {e_occ}")

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
        # + verification (Canny/Hough + Grad-CAM + shape check) per region.
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
                "printed text, screws, window frames or fittings for cracks. This is a "
                "dataset-coverage limitation, not a bug -- see the Verify section below and "
                "the project's Future Work discussion."
            )

            sw_use_ensemble = st.checkbox(
                "Use ensemble mode (require agreement across multiple trained models)",
                value=False, key="sw_use_ensemble",
            )
            if sw_use_ensemble:
                st.caption(
                    f"Ensemble mode polls **{', '.join(VARIANT_LABELS.get(v, v) for v in DEFAULT_ENSEMBLE_VARIANTS)}** "
                    "on every window (ignoring the model dropdown above) and only keeps a "
                    "detection when enough of them independently agree it's a crack. "
                    "On our synthetic benchmark this did not improve crack localization over the "
                    "single model, so it is off by default. It runs 3 full models per window, "
                    "so scanning is noticeably slower."
                )
                sw_min_agree = st.slider(
                    "Minimum models that must agree", 1, len(DEFAULT_ENSEMBLE_VARIANTS),
                    2, 1, key="sw_min_agree",
                )

            sw_col1, sw_col2 = st.columns(2)
            with sw_col1:
                sw_threshold = st.slider(
                    "Confidence threshold (per model)" if sw_use_ensemble else "Confidence threshold",
                    0.50, 0.999,
                    0.93 if sw_use_ensemble else 0.995,
                    0.005,
                    format="%.3f",
                    key="sw_threshold_ens" if sw_use_ensemble else "sw_threshold_single",
                )
            with sw_col2:
                sw_stride = st.slider(
                    "Window stride (px) -- smaller = more thorough, slower",
                    56, 224, 112, 8, key="sw_stride"
                )
            st.caption(
                "Defaults come from a validation-tuned benchmark: 0.995 for the single model and 0.93 for "
                "the ensemble. Lower thresholds find more cracks but add many false boxes."
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
                            "confidence. Shape and Risk depend on the CNN-branch attention map, which "
                            "in our 25-image check fell near the drawn crack only about 20% of the time "
                            "at this resolution, so treat them as prioritization hints for manual "
                            "inspection, not as crack evidence or an engineering assessment."
                        )
                    else:
                        st.info("No crack regions detected above the current threshold.")

                    if detections:
                        st.markdown("---")
                        st.subheader("🔬 Verify Each Detected Region")
                        st.write(
                            "Each box is shown with three auxiliary checks: **Canny/Hough crack-type "
                            "analysis** (the same geometric check used in Single Crop mode), **Grad-CAM** "
                            "(where the CNN branch is sensitive) and a **Shape check** (does the "
                            "segmentation mask form a line, or a blocky region?). These checks can "
                            "disagree with each other and with the classifier -- e.g. high "
                            "confidence but a blocky, non-linear mask -- which is a useful signal to "
                            "flag a region for human review rather than trust it automatically."
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
                            "inside the CNN branch's high-attention region, computed from an "
                            "earlier DenseNet-121 block (14×14 resolution). **Shape** fits a rotated "
                            "bounding box to the mask and reports its long/short side ratio -- a real "
                            "crack tends to be thin and elongated, while a window frame or "
                            "texture patch tends to be blocky. Both are heuristic combinations "
                            "of existing signals, not a learned segmentation or shape model, and "
                            "the CNN-branch attention they rely on is not always on the crack; a "
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
        st.caption("Original training run; see the Ablation Comparison tab for mean ± std over repeated runs.")

        test_metrics = logs["full_csaf"]["test"]
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        metric_card(mcol1, "Accuracy", f"{test_metrics['accuracy']*100:.2f}%")
        metric_card(mcol2, "F1-Score", f"{test_metrics['f1']*100:.2f}%")
        metric_card(mcol3, "Precision", f"{test_metrics['precision']*100:.2f}%")
        metric_card(mcol4, "Recall", f"{test_metrics['recall']*100:.2f}%")

with tab3:
    st.title("⚔️ Ablation Comparison")
    st.write("Comparing all fusion strategies tested in this project. Fusing both backbones clearly beats "
             "the CNN alone. Over repeated runs, the gate raises recall over ungated cross-attention at "
             "lower precision, while F1 and accuracy are not meaningfully different (see the mean ± std "
             "table below).")

    logs = load_training_logs()
    available = [v for v in VARIANT_ORDER if v in logs]
    multi = load_multi_run_stats(["ungated_cross_attn", "scalar_gate", "full_csaf"])

    if not available:
        st.warning("No training logs found. Run train.py for each variant first.")
    else:
        if "full_csaf" in available:
            hc1, hc2, hc3, hc4 = st.columns(4)
            if "full_csaf" in multi:
                fm = multi["full_csaf"]
                n_f = len(fm)
                metric_card(hc1, f"Mean Accuracy ({n_f} runs)", f"{run_mean(fm, 'accuracy')*100:.2f}%")
                metric_card(hc2, f"Mean F1-Score ({n_f} runs)", f"{run_mean(fm, 'f1')*100:.2f}%")
                metric_card(hc3, f"Mean Recall ({n_f} runs)", f"{run_mean(fm, 'recall')*100:.2f}%")
                if "ungated_cross_attn" in multi:
                    miss_f = 1 - run_mean(fm, "recall")
                    miss_u = 1 - run_mean(multi["ungated_cross_attn"], "recall")
                    fn_reduction = (1 - miss_f / miss_u) * 100
                    metric_card(hc4, "Missed Cracks vs Ungated (mean)", f"{-fn_reduction:+.0f}%")
            else:
                fc_metrics = logs["full_csaf"]["test"]
                metric_card(hc1, "Accuracy (original run)", f"{fc_metrics['accuracy']*100:.2f}%")
                metric_card(hc2, "F1-Score (original run)", f"{fc_metrics['f1']*100:.2f}%")
                metric_card(hc3, "Recall (original run)", f"{fc_metrics['recall']*100:.2f}%")
                if "ungated_cross_attn" in available:
                    fn_reduction = (1 - (1 - fc_metrics["recall"]) /
                                    (1 - logs["ungated_cross_attn"]["test"]["recall"])) * 100
                    metric_card(hc4, "Missed Cracks vs Ungated (1 run)", f"{-fn_reduction:+.0f}%")
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
        st.subheader("Original training run (one run per variant)")
        st.table(display_df.set_index("Model Variant"))

        if multi:
            ms_rows = []
            for v in ["ungated_cross_attn", "scalar_gate", "full_csaf"]:
                if v in multi:
                    runs = multi[v]
                    row = {"Model Variant": VARIANT_LABELS[v], "Runs": len(runs)}
                    for key, name in (("accuracy", "Accuracy"), ("precision", "Precision"),
                                      ("recall", "Recall"), ("f1", "F1-Score")):
                        vals = np.array([r[key] for r in runs]) * 100
                        row[name] = f"{vals.mean():.2f} ± {vals.std(ddof=1):.2f}"
                    ms_rows.append(row)
            st.subheader("Mean ± std over repeated runs")
            st.table(pd.DataFrame(ms_rows).set_index("Model Variant"))
            st.caption(
                "Original run plus seeded reruns (saved in ~/csaf_seeds). CNN-only, ViT-only and "
                "concatenation were trained once. With only a few runs per variant, treat differences "
                "smaller than the ± values as noise."
            )

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
        if os.path.exists("figures/gate_analysis.png"):
            st.image("figures/gate_analysis.png", width="stretch")
            st.caption(
                "Learned gate on the 600 test images (1 = CNN-attended branch, 0 = ViT-attended branch). "
                "The mean gate is higher for cracked images, but it is nearly uniform across spatial "
                "position: the gate is input-conditioned and channel-wise."
            )

        st.subheader("📈 Precision-Recall Curves")
        if os.path.exists("figures/pr_curves.png"):
            st.image("figures/pr_curves.png", width="stretch")
            st.caption(
                "Average Precision is comparable across all attention-based variants, "
                "confirming the gate's benefit is a shift in operating point (precision/recall "
                "trade-off at the deployed threshold) rather than a general ranking improvement."
            )

        if "full_csaf" in multi and "ungated_cross_attn" in multi:
            fm, um = multi["full_csaf"], multi["ungated_cross_attn"]
            st.markdown("---")
            finding_card(
                "Effect of the Gate",
                f"Over {len(fm)} runs, the gate changes mean recall from "
                f"<strong>{run_mean(um, 'recall')*100:.1f}%</strong> (ungated cross-attention) to "
                f"<strong>{run_mean(fm, 'recall')*100:.1f}%</strong> and mean precision from "
                f"<strong>{run_mean(um, 'precision')*100:.1f}%</strong> to "
                f"<strong>{run_mean(fm, 'precision')*100:.1f}%</strong>. Mean F1 is "
                f"{run_mean(fm, 'f1')*100:.1f} vs {run_mean(um, 'f1')*100:.1f}, within run-to-run "
                f"variation, so the gate shifts the operating point toward fewer missed cracks "
                f"rather than improving overall accuracy."
            )
        elif "full_csaf" in available and "ungated_cross_attn" in available:
            gate_gain_f1 = (logs["full_csaf"]["test"]["f1"] - logs["ungated_cross_attn"]["test"]["f1"]) * 100
            gate_gain_recall = (logs["full_csaf"]["test"]["recall"] - logs["ungated_cross_attn"]["test"]["recall"]) * 100
            st.markdown("---")
            finding_card(
                "Effect of the Gate (single run)",
                f"In the original run the gate changes recall by <strong>{gate_gain_recall:+.2f} points</strong> "
                f"and F1 by <strong>{gate_gain_f1:+.2f} points</strong> versus ungated cross-attention. "
                f"One run per variant cannot separate a real effect from run-to-run variation; "
                f"repeat training with different seeds before drawing conclusions."
            )

        if "full_csaf" in multi and "scalar_gate" in multi and "ungated_cross_attn" in multi:
            finding_card(
                "Is an Input-Conditioned Channel-wise Gate Needed?",
                f"Mean recall: full gate <strong>{run_mean(multi['full_csaf'], 'recall')*100:.1f}%</strong>, "
                f"scalar gate <strong>{run_mean(multi['scalar_gate'], 'recall')*100:.1f}%</strong>, "
                f"no gate <strong>{run_mean(multi['ungated_cross_attn'], 'recall')*100:.1f}%</strong>. "
                f"Compare the differences with the ± values in the table above before reading anything "
                f"into the ordering: the full-vs-scalar gap is within run-to-run variation, so the benefit "
                f"of making the gate adaptive is not established. The learned gate varies across channels "
                f"and responds to crack presence, but not across spatial position."
            )