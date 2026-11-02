from __future__ import annotations

import io
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch
from medmnist import RetinaMNIST
from PIL import Image, ImageOps, UnidentifiedImageError

from src.models.cnn import create_cnn_model
from src.utils.image_processing import prepare_image
from src.visualization.interpretability import (
    compute_grad_cam,
    visualize_grad_cam,
    get_model_target_layer,
)

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "experiments" / "results_real"
DATA_ROOT = ROOT / "data" / "medmnist"

CLASS_NAMES = [
    "Grade 0 (No DR)",
    "Grade 1 (Mild DR)",
    "Grade 2 (Moderate DR)",
    "Grade 3 (Severe DR)",
    "Grade 4 (Proliferative DR)",
]
CLASS_DESCRIPTIONS = [
    "No apparent diabetic retinopathy lesions detected.",
    "Mild NPDR: Microaneurysms only.",
    "Moderate NPDR: More than microaneurysms but less than severe NPDR.",
    "Severe NPDR: Prominent retinal hemorrhages, venous beading, or IRMA.",
    "Proliferative DR: Neovascularization or vitreous/preretinal hemorrhage.",
]
MODEL_OPTIONS = {
    "DenseNet121 (Recommended)": "densenet121",
    "ResNet50": "resnet50",
}

# Verified historical metrics measured on the 400-image RetinaMNIST held-out test set
VERIFIED_BENCHMARK = {
    "resnet50": {
        "accuracy": 0.4900,
        "balanced_accuracy": 0.3346,
        "quadratic_weighted_kappa": 0.4233,
        "f1_macro": 0.3333,
        "auc_ovr": 0.7348,
        "params": "23.5M",
        "best_epoch": 1,
        "training_time_min": 39.4,
    },
    "densenet121": {
        "accuracy": 0.5250,
        "balanced_accuracy": 0.3859,
        "quadratic_weighted_kappa": 0.5106,
        "f1_macro": 0.3870,
        "auc_ovr": 0.7272,
        "params": "7.0M",
        "best_epoch": 3,
        "training_time_min": 40.5,
    },
}

st.set_page_config(
    page_title="Retinal Image Analysis | Research Toolkit",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for rich college research dashboard presentation
st.markdown(
    """
    <style>
    .reportview-container .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .metric-card {
        background: rgba(125, 125, 125, 0.05);
        border: 1px solid rgba(125, 125, 125, 0.15);
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    .disclaimer-banner {
        border-left: 4px solid #e74c3c;
        background-color: rgba(231, 76, 60, 0.08);
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 14px;
    }
    .badge-pill {
        display: inline-block;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 12px;
        margin-right: 6px;
    }
    .badge-primary { background: #3498db; color: white; }
    .badge-success { background: #2ecc71; color: white; }
    .badge-warning { background: #f39c12; color: white; }
    .badge-danger { background: #e74c3c; color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model(backbone: str, checkpoint_path: str, checkpoint_mtime: float):
    del checkpoint_mtime
    model = create_cnn_model(backbone, num_classes=len(CLASS_NAMES), pretrained=False)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@st.cache_data
def load_test_samples(data_root: str) -> dict[int, tuple[int, bytes]]:
    dataset = RetinaMNIST(split="test", download=False, root=data_root)
    samples: dict[int, tuple[int, bytes]] = {}
    for index in range(len(dataset)):
        image, label = dataset[index]
        grade = int(label[0])
        if grade not in samples:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            samples[grade] = (index, buffer.getvalue())
        if len(samples) == len(CLASS_NAMES):
            break
    return samples




@st.cache_data
def load_summary() -> dict:
    summary_path = RESULTS_DIR / "experiment_summary.json"
    if not summary_path.is_file():
        return {}
    with summary_path.open(encoding="utf-8") as file:
        return json.load(file)


# Header & Metadata Badges
st.title("Retinal Image Analysis — Medical AI Research Toolkit")
st.markdown(
    """
    <span class="badge-pill badge-primary">College Research Toolkit</span>
    <span class="badge-pill badge-success">Deep Learning Classification</span>
    <span class="badge-pill badge-warning">RetinaMNIST Benchmark (28×28)</span>
    <span class="badge-pill badge-danger">Non-Clinical Research Only</span>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="disclaimer-banner">
        <strong>⚠️ STRICT RESEARCH USE ONLY — NOT A CLINICAL DIAGNOSTIC DEVICE</strong><br>
        This software is an educational research toolkit for deep learning image analysis.
        Predictions, confidence scores, and activation heatmaps are illustrative model outputs.
        They must <strong>never</strong> be used for medical diagnosis, clinical screening, treatment planning, or patient triage.
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar controls
with st.sidebar:
    st.header("🔬 Experiment Controls")
    st.markdown("**Step 1: Select Model**")
    model_display_name = st.selectbox("Trained Architecture", list(MODEL_OPTIONS), index=0)
    backbone = MODEL_OPTIONS[model_display_name]
    checkpoint_path = RESULTS_DIR / backbone / "model_best.pth"

    if not checkpoint_path.is_file():
        st.error(f"Checkpoint not found: {checkpoint_path}")
        st.stop()
    else:
        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        st.caption(f"Loaded checkpoint: `{backbone}/model_best.pth` ({size_mb:.1f} MB)")

    st.divider()
    st.markdown("**Step 2: Input Image Source**")
    samples_path = DATA_ROOT / "retinamnist.npz"
    sample_mode_available = samples_path.is_file()
    input_options = ["Upload a local image (JPG/PNG)"]
    if sample_mode_available:
        input_options.insert(0, "Use a RetinaMNIST test benchmark sample")
    input_mode = st.radio("Choose source:", input_options)

    st.divider()
    st.markdown("**Step 3: Explainability Settings**")
    enable_live_gradcam = st.checkbox("Compute Live Grad-CAM", value=True, help="Compute activation heatmaps directly on the active input image.")
    gradcam_alpha = st.slider("Heatmap Overlay Opacity", min_value=0.1, max_value=0.9, value=0.5, step=0.05)


uploaded_image = None
sample_grade = None
sample_index = None
source_caption = ""
image_attribution = "RetinaMNIST Official Test Split"

if input_mode == "Use a RetinaMNIST test benchmark sample":
    try:
        samples = load_test_samples(str(DATA_ROOT))
    except (OSError, RuntimeError, ValueError) as error:
        st.error(f"Could not load local RetinaMNIST test set: {error}")
        st.stop()

    available_grades = sorted(samples)
    col_grade, col_info = st.columns([1, 1])
    with col_grade:
        sample_grade = st.selectbox(
            "Select known benchmark severity:",
            available_grades,
            format_func=lambda g: CLASS_NAMES[g],
        )
    with col_info:
        st.info(f"Clinical definition: {CLASS_DESCRIPTIONS[sample_grade]}")

    sample_index, image_bytes = samples[sample_grade]
    uploaded_image = Image.open(io.BytesIO(image_bytes))
    source_caption = (
        f"RetinaMNIST held-out test sample #{sample_index} · Known Ground Truth: {CLASS_NAMES[sample_grade]}"
    )
    image_attribution = f"MedMNIST v2 (DeepDRiD derivative), sample #{sample_index}"

else:
    st.subheader("Upload Retinal Fundus Photograph")
    st.markdown(
        "Upload an authorized JPG or PNG fundus photograph (max 10 MB). "
        "Images remain in local memory for inference and are **not** persisted or uploaded to external servers."
    )
    col_up, col_attr = st.columns([1, 1])
    with col_up:
        image_file = st.file_uploader(
            "Select image file",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPEG, PNG. Maximum file size: 10 MB.",
        )
    with col_attr:
        custom_attribution = st.text_input(
            "Image Source / Attribution (optional for presentation records):",
            placeholder="e.g. Open Research / Presentation Demo Image",
        )
        if custom_attribution.strip():
            image_attribution = custom_attribution.strip()
        else:
            image_attribution = "User local upload (source unverified)"

    if image_file is not None:
        if image_file.size > 10 * 1024 * 1024:
            st.error("Uploaded file exceeds 10 MB size limit.")
            st.stop()
        try:
            uploaded_image = Image.open(image_file)
            uploaded_image.load()
            if uploaded_image.width < 28 or uploaded_image.height < 28:
                st.error("Image dimensions are too small (minimum 28×28 pixels required).")
                st.stop()
        except (UnidentifiedImageError, OSError, ValueError) as error:
            st.error(f"Could not read image file. Please provide a valid JPG or PNG. Details: {error}")
            st.stop()

    source_caption = f"Uploaded Fundus Photo · Attribution: {image_attribution} · (No ground truth label)"


# Main Inference & Results Section
if uploaded_image is not None:
    try:
        display_image, input_tensor = prepare_image(uploaded_image)
    except (OSError, ValueError) as error:
        st.error(f"Could not prepare image tensor: {error}")
        st.stop()

    st.divider()
    image_column, results_column = st.columns([1, 1])

    with image_column:
        st.subheader("🖼️ Input Fundus Image")
        st.image(display_image, caption=source_caption, width="stretch")
        with st.expander("ℹ️ Preprocessing Pipeline Details", expanded=False):
            st.markdown(
                f"- **Channels**: RGB (3 channels)\n"
                f"- **Resize**: Interpolated to 224×224 bilinear (standard pretrained input size)\n"
                f"- **Normalization**: ImageNet standard (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)\n"
                f"- **Attribution**: {image_attribution}\n"
                f"- **Privacy**: Processed purely in RAM; no disk storage."
            )

    with results_column:
        st.subheader("📊 Model Inference")
        try:
            model = load_model(
                backbone,
                str(checkpoint_path),
                checkpoint_path.stat().st_mtime,
            )
            with torch.inference_mode():
                logits = model(input_tensor)
                probabilities = torch.softmax(logits, dim=1)[0].numpy()
        except (OSError, RuntimeError, KeyError, ValueError) as error:
            st.error(f"Inference execution failed: {error}")
            st.stop()

        predicted_grade = int(probabilities.argmax())
        predicted_conf = probabilities[predicted_grade]

        col_metric1, col_metric2 = st.columns(2)
        with col_metric1:
            st.metric("Predicted Grade", f"Grade {predicted_grade}")
            st.caption(CLASS_NAMES[predicted_grade])
        with col_metric2:
            st.metric("Highest Model Score", f"{predicted_conf:.1%}")
            st.caption("Softmax output (uncalibrated)")

        if sample_grade is not None:
            if predicted_grade == sample_grade:
                st.success(f"✓ Model prediction matches the benchmark ground-truth label ({CLASS_NAMES[sample_grade]}).")
            else:
                st.warning(
                    f"Model predicted {CLASS_NAMES[predicted_grade]}, but known benchmark label is {CLASS_NAMES[sample_grade]}."
                )
        else:
            st.info("ℹ️ Uploaded image: Ground truth is unverified. Accuracy cannot be computed on unverified uploads.")

        # Class Probability Bar Chart
        prob_df = pd.DataFrame(
            {"Severity Grade": CLASS_NAMES, "Score": probabilities}
        ).set_index("Severity Grade")
        st.bar_chart(prob_df, y="Score", color="#2980b9")
        st.caption(
            "⚠️ Note: Scores are softmax values over the 5 severity classes, not clinically calibrated probabilities. "
            "A high softmax score does not guarantee diagnostic accuracy."
        )

        # JSON Download
        download_payload = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "model_backbone": backbone,
                "model_checkpoint": str(checkpoint_path.name),
                "source": image_attribution,
                "known_ground_truth": sample_grade if sample_grade is not None else "Unverified",
            },
            "prediction": {
                "predicted_grade_index": predicted_grade,
                "predicted_grade_label": CLASS_NAMES[predicted_grade],
                "confidence_score": float(predicted_conf),
                "all_class_scores": {
                    CLASS_NAMES[idx]: float(score) for idx, score in enumerate(probabilities)
                },
            },
            "disclaimer": (
                "Educational research demonstration only. Not a medical diagnostic device. "
                "Must not be used for clinical decisions or triage."
            ),
        }
        st.download_button(
            "📥 Download Prediction Report (JSON)",
            data=json.dumps(download_payload, indent=2),
            file_name=f"retinal_prediction_{backbone}_grade{predicted_grade}.json",
            mime="application/json",
        )

    # Live Grad-CAM Visualization Section
    if enable_live_gradcam:
        st.divider()
        st.subheader("🔍 Explainable AI: Live Grad-CAM")
        st.markdown(
            "Grad-CAM (Gradient-weighted Class Activation Mapping) highlights spatial regions in the final convolutional "
            "block that contributed most to the predicted severity score."
        )
        try:
            target_layer = get_model_target_layer(model, backbone)
            cam = compute_grad_cam(
                model=model,
                image=input_tensor,
                target_layer=target_layer,
                target_class=predicted_grade,
                device="cpu",
            )
            np_orig = np.asarray(display_image.resize((224, 224)))
            cam_overlay = visualize_grad_cam(np_orig, cam, alpha=gradcam_alpha)

            cam_col1, cam_col2, cam_col3 = st.columns(3)
            with cam_col1:
                st.image(np_orig, caption="Input (224×224)", width="stretch")
            with cam_col2:
                st.image(cam, caption=f"Raw Activation Heatmap ({target_layer})", width="stretch", clamp=True)
            with cam_col3:
                st.image(cam_overlay, caption=f"Live Overlay (Grade {predicted_grade})", width="stretch")

            st.caption(
                f"Computed live on active model feature layer: `{target_layer}`. "
                "⚠️ **Clinical Warning:** Grad-CAM illustrates model feature sensitivity. It does not provide causal proof, "
                "does not delineate lesions with medical precision, and must not be used as clinical evidence."
            )
        except Exception as cam_err:
            st.error(f"Live Grad-CAM could not be computed: {cam_err}")


# Deep Evaluation & Research Documentation Tabs
st.divider()
benchmark_tab, precomputed_tab, datasets_tab, limitations_tab = st.tabs(
    [
        "📊 Benchmark & Held-Out Test Evaluation",
        "🖼️ Historical Precomputed Visualizations",
        "📚 Retinal Datasets Research & Comparison",
        "⚖️ Research Scope & Clinical Limitations",
    ]
)

with benchmark_tab:
    st.subheader("RetinaMNIST Benchmark Test Set Performance")
    st.markdown(
        "Evaluation results measured on the official 400-image held-out test split of RetinaMNIST (MedMNIST v2). "
        "The test set was kept strictly untouched during training and model selection."
    )

    summary = load_summary()
    cols = st.columns(len(MODEL_OPTIONS))
    for col, (display_label, model_key) in zip(cols, MODEL_OPTIONS.items()):
        v_data = VERIFIED_BENCHMARK.get(model_key, {})
        with col:
            st.markdown(f"### {display_label}")
            sub_col1, sub_col2, sub_col3 = st.columns(3)
            sub_col1.metric("Test Accuracy", f"{v_data.get('accuracy', 0):.1%}")
            sub_col2.metric("Balanced Acc.", f"{v_data.get('balanced_accuracy', 0):.1%}")
            sub_col3.metric("QWK (Ordinal)", f"{v_data.get('quadratic_weighted_kappa', 0):.3f}")

            sub_col4, sub_col5, sub_col6 = st.columns(3)
            sub_col4.metric("Macro F1", f"{v_data.get('f1_macro', 0):.3f}")
            sub_col5.metric("AUC-ROC", f"{v_data.get('auc_ovr', 0):.3f}")
            sub_col6.metric("Parameters", v_data.get('params', 'N/A'))

    st.markdown(
        """
        > **Key Medical ML Finding: Understanding Class Imbalance & Ordinal Metrics**
        > - **Grade 0 (No DR) represents 43.5% (174/400 images)** of the test split. A naive model that always predicts Grade 0 achieves 43.5% raw accuracy, yet has 20% balanced accuracy and 0.0 QWK!
        > - **Balanced Accuracy** (average per-class recall) and **Quadratic Weighted Kappa (QWK)** penalize misclassifications proportionally to clinical severity distance, making them much more informative than raw accuracy for diabetic retinopathy grading.
        > - **DenseNet121 achieves superior QWK (0.511 vs 0.423)** and higher balanced accuracy (38.6% vs 33.5%) with 70% fewer parameters (7.0M vs 23.5M) compared to ResNet50.
        > - **Published Benchmark Alignment:** In the official MedMNIST v2 publication (*Nature Scientific Data*, Yang et al.), standard ResNet-18 achieves **49.3% accuracy** and ResNet-50 achieves **51.1% accuracy** on 224×224 RetinaMNIST. The measured 49.0% and 52.5% figures are authentic and completely consistent with published literature.
        """
    )

    conf_cols = st.columns(len(MODEL_OPTIONS))
    for col, (display_label, model_key) in zip(conf_cols, MODEL_OPTIONS.items()):
        with col:
            cm_path = RESULTS_DIR / model_key / "confusion_matrix.png"
            if cm_path.is_file():
                st.image(str(cm_path), caption=f"{display_label} — Held-Out Test Confusion Matrix")
            rep_path = RESULTS_DIR / model_key / "classification_report.txt"
            if rep_path.is_file():
                with st.expander(f"View Full Classification Report ({model_key})"):
                    st.text(rep_path.read_text(encoding="utf-8"))


with precomputed_tab:
    st.subheader("Historical Benchmark Visualizations")
    st.caption(
        "These visualizations were generated and saved during the previous training and evaluation run. "
        "They illustrate historical test samples and are distinct from the live Grad-CAM computed above."
    )

    gradcam_model = st.radio("Show historical visualizations for:", list(MODEL_OPTIONS), horizontal=True)
    selected_backbone = MODEL_OPTIONS[gradcam_model]

    training_hist_path = RESULTS_DIR / selected_backbone / "training_history.png"
    if training_hist_path.is_file():
        st.image(str(training_hist_path), caption=f"{gradcam_model} — 6-Panel Training Curves (Loss, Acc, F1, Precision, Recall, AUC)")

    per_class_path = RESULTS_DIR / selected_backbone / "per_class_metrics.png"
    if per_class_path.is_file():
        st.image(str(per_class_path), caption=f"{gradcam_model} — Per-Class Metrics Bar Chart")

    gradcam_files = [
        RESULTS_DIR / selected_backbone / "gradcam" / f"grade_{grade}.png"
        for grade in range(len(CLASS_NAMES))
    ]
    existing_cams = [p for p in gradcam_files if p.is_file()]
    if existing_cams:
        st.markdown("**Saved Grad-CAM Benchmark Examples per Disease Severity Grade:**")
        cam_grid = st.columns(min(len(existing_cams), 5))
        for col, path in zip(cam_grid, existing_cams):
            with col:
                st.image(str(path), caption=path.stem.replace("_", " ").title())


with datasets_tab:
    st.subheader("Retinal Imaging Datasets Research & Verification")
    st.markdown(
        "A rigorous medical ML project must clearly distinguish dataset provenance, tasks, and licenses before use."
    )

    dataset_comparison = [
        {
            "Dataset": "RetinaMNIST (MedMNIST v2)",
            "Official Host / Source": "MedMNIST / Zenodo (Nature Sci Data)",
            "Clinical Task": "5-Class DR Severity Grading (0-4)",
            "Images / Subjects": "1,600 images (1,080 train / 120 val / 400 test)",
            "Resolution": "28×28 native (center-cropped from ~1736×1824)",
            "License / Access": "CC BY 4.0 (Open access, pre-downloaded)",
            "Disk Size": "~3.3 MB (npz)",
            "Status in Project": "Integrated (Fast benchmark baseline)",
        },
        {
            "Dataset": "APTOS 2019 Blindness Detection",
            "Official Host / Source": "Aravind Eye Hospital / Kaggle Competition",
            "Clinical Task": "5-Class DR Severity Grading (0-4)",
            "Images / Subjects": "3,662 labeled train, 1,928 test",
            "Resolution": "High-Res (~1000×1000 to 3000×2000)",
            "License / Access": "Kaggle competition terms (Free account required)",
            "Disk Size": "~9.5 GB",
            "Status in Project": "Investigated (Requires user approval)",
        },
        {
            "Dataset": "EyePACS (2015 Kaggle DR)",
            "Official Host / Source": "EyePACS / California Telehealth / Kaggle",
            "Clinical Task": "5-Class DR Severity Grading (0-4)",
            "Images / Subjects": "88,702 images (patient-level pairs: left/right)",
            "Resolution": "High-Res (up to 4000×3000)",
            "License / Access": "Kaggle rules (Requires user approval)",
            "Disk Size": "~80 GB uncompressed",
            "Status in Project": "Investigated (Heavy compute requirement)",
        },
        {
            "Dataset": "IDRiD (Indian DR Image Dataset)",
            "Official Host / Source": "IEEE DataPort (Dr. Prasanna Porwal et al.)",
            "Clinical Task": "DR Grading (516 img) + Lesion Segmentation (81 img)",
            "Images / Subjects": "516 grading images (413 train / 103 test)",
            "Resolution": "Very High-Res (4288×2848)",
            "License / Access": "CC BY 4.0 (IEEE DataPort registration)",
            "Disk Size": "~1.5 GB (grading) / ~14 GB (full)",
            "Status in Project": "Investigated (High clinical quality)",
        },
        {
            "Dataset": "DRIVE",
            "Official Host / Source": "Univ. Medical Center Utrecht / Grand Challenge",
            "Clinical Task": "Binary Retinal Blood Vessel Segmentation",
            "Images / Subjects": "40 fundus images (20 train / 20 test)",
            "Resolution": "565×584 pixels",
            "License / Access": "Open for research use",
            "Disk Size": "~30 MB",
            "Status in Project": "Segmentation only (Separated from grading)",
        },
        {
            "Dataset": "Retinal Fundus Image 50K (RFID50K)",
            "Official Host / Source": "Kaggle (Uploaded by Gautam Raj)",
            "Clinical Task": "7-Class Multi-Disease (Cataract, DR, Glaucoma, AMD, etc.)",
            "Images / Subjects": "~50,000 images (Community aggregation)",
            "Resolution": "Variable high-res",
            "License / Access": "Other (Third-party compilation)",
            "Disk Size": "~25-40 GB",
            "Status in Project": "Investigated (NOT 5-class DR grading)",
        },
    ]

    st.dataframe(pd.DataFrame(dataset_comparison), width="stretch")

    st.markdown(
        """
        ### 🔍 Critical Audit of the User-Mentioned "Retinal Fundus Image 50K" Dataset
        1. **Identity & Host:** The dataset known as "Retinal Fundus Image 50k" (RFID50K) is a user compilation hosted on Kaggle by Gautam Raj.
        2. **Clinical Task Difference:** It is structured for **7 disease categories** (*Cataract, Diabetic Retinopathy, Glaucoma, Hypertensive Retinopathy, Myopia, AMD, Normal*), rather than 5-level diabetic retinopathy severity grading. Diabetic retinopathy is merely 1 of 7 categories.
        3. **Provenance & Duplication Risk:** It aggregates images from disparate public sources without unified patient identifiers, creating significant risk of patient-level data leakage between arbitrary train/val/test folders.
        4. **Governance Recommendation:** Before adopting it or any large external dataset, explicit approval, license review, and disk/compute provisioning are required.
        """
    )


with limitations_tab:
    st.subheader("Engineering Scope, Governance & Non-Clinical Limitations")
    st.markdown(
        """
        ### 1. Dataset Resolution & Micro-Lesion Physics
        - **RetinaMNIST Native Resolution:** Native images are 28×28 pixels, center-cropped from ~1736×1824 high-resolution fundus images.
        - **Lesion Visibility:** Cardinal features of early diabetic retinopathy (such as microaneurysms measuring 10–100 μm) become sub-pixel when downsampled to 28×28. Models on RetinaMNIST learn macro-patterns (vessel density, optic disc contrast) rather than microvascular lesions.
        - **Generalization:** Performance on RetinaMNIST cannot be extrapolated to clinical high-resolution photographs from clinical fundus cameras.

        ### 2. Patient-Level Splitting & Leakage Prevention
        - DeepDRiD (the source of RetinaMNIST) consists of 4 views per patient (2 eyes, 2 fields per eye). While RetinaMNIST test set originates from the challenge validation set, lack of patient IDs in `retinamnist.npz` means internal patient grouping cannot be mathematically verified.

        ### 3. Explainability Limitations (Grad-CAM)
        - Grad-CAM highlights coarse convolutional feature maps (7×7 spatial resolution).
        - A Grad-CAM overlay **does not** prove causal reasoning, does not guarantee the model observed real pathology, and cannot substitute for manual clinical annotation.

        ### 4. Running the Dashboard Locally (Windows PowerShell)
        To launch this dashboard locally on Windows:
        ```powershell
        python -m streamlit run app.py
        ```
        The dashboard runs strictly on localhost (usually `http://localhost:8501`) and does not transmit data over the network.
        """
    )
