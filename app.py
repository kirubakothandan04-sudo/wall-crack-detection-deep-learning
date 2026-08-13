import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance
import json
import os
import base64
import glob
import pandas as pd
import random
import torch

from predict_csaf import load_model, preprocess, analyze_crack, to_base64

st.set_page_config(
    page_title="CSAF-Net: Wall Crack Detection & Analytics",
    page_icon="🏗️",
    layout="wide"
)


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
    "full_csaf": "CSAF-Net (Full, Gated)",
}
VARIANT_ORDER = ["cnn_only", "vit_only", "concat", "ungated_cross_attn", "full_csaf"]


# ---------------------------------------------------------------------------
# Cached model loading -- runs ONCE per variant per session instead of once
# per prediction. Streamlit keys the cache by the function's arguments, so
# switching the variant dropdown loads a fresh model only when needed.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def get_cached_model(variant):
    model, device = load_model(variant=variant)
    return model, device


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
tab1, tab2, tab3 = st.tabs(["🔍 Live Prediction", "📊 Model Training Analytics", "⚔️ Ablation Comparison"])

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

        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

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
        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "F1-Score", "Precision", "Recall"],
            "Value": [
                f"{test_metrics['accuracy']*100:.2f}%",
                f"{test_metrics['f1']*100:.2f}%",
                f"{test_metrics['precision']*100:.2f}%",
                f"{test_metrics['recall']*100:.2f}%",
            ]
        })
        st.table(metrics_df.set_index("Metric"))

with tab3:
    st.title("⚔️ Ablation Comparison")
    st.write("Comparing all fusion strategies tested in this project. This is the core evidence "
             "for CSAF-Net's novelty: the gated fusion module outperforms every simpler alternative.")

    logs = load_training_logs()
    available = [v for v in VARIANT_ORDER if v in logs]

    if not available:
        st.warning("No training logs found. Run train.py for each variant first.")
    else:
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

        if "full_csaf" in available and "ungated_cross_attn" in available:
            gate_gain_f1 = (logs["full_csaf"]["test"]["f1"] - logs["ungated_cross_attn"]["test"]["f1"]) * 100
            gate_gain_recall = (logs["full_csaf"]["test"]["recall"] - logs["ungated_cross_attn"]["test"]["recall"]) * 100
            st.markdown("---")
            st.subheader("🔑 Key Finding: Effect of the Gate")
            st.write(
                f"Adding the learned gate on top of bidirectional cross-attention improves F1 by "
                f"**{gate_gain_f1:+.2f} points** and recall by **{gate_gain_recall:+.2f} points** "
                f"compared to ungated cross-attention fusion — isolating the exact contribution of "
                f"CSAF-Net's novelty."
            )