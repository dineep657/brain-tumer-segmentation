import os
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

from inference_2d import preprocess_2d, predict_2d
from classifier_2d import classify_tumor_2d, CLASSES
from metrics_2d import analyze_tumor_metrics
from image_processing import render_segmentation_figure, export_overlay_as_png
from pdf_generator import generate_2d_pdf_report
from confidence_display import resolve_tumor_type_label, LOW_CONFIDENCE_WARNING_TEXT
from patient_records import init_db, validate_patient_fields, insert_patient_record, query_patient_history, delete_all_records
from gradcam import generate_gradcam, resize_activation_map, blend_heatmap_with_image, get_gradcam_panel_title
from classifier_2d import load_trained_classifier

# Initialise SQLite DB on startup
init_db()

# 1. Page Configuration & Custom Styling
st.set_page_config(
    page_title="Academic 2D Brain Tumor Segmentation & Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: #E0E6ED;
    }
    .stMetric {
        background: linear-gradient(135deg, #1E2640 0%, #0F172A 100%);
        border: 1px solid #334155;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .stMetric label {
        color: #94A3B8 !important;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .stMetric .metric-value {
        color: #38BDF8 !important;
        font-weight: 700;
    }
    div[data-testid="stSidebar"] {
        background-color: #1E293B;
        border-right: 1px solid #334155;
    }
    h1, h2, h3 {
        color: #F8FAFC !important;
    }
    .status-badge-yes {
        background-color: rgba(239, 68, 68, 0.2);
        color: #F87171;
        border: 1px solid #EF4444;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .status-badge-no {
        background-color: rgba(34, 197, 94, 0.2);
        color: #4ADE80;
        border: 1px solid #22C55E;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .status-badge-awaiting {
        background-color: rgba(59, 130, 246, 0.2);
        color: #60A5FA;
        border: 1px solid #3B82F6;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .status-badge-invalid {
        background-color: rgba(234, 179, 8, 0.2);
        color: #FACC15;
        border: 1px solid #EAB308;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .status-badge-lowconf {
        background-color: rgba(234, 179, 8, 0.2);
        color: #FACC15;
        border: 1px solid #EAB308;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .placeholder-card {
        background-color: #1E293B;
        border: 2px dashed #3B82F6;
        border-radius: 12px;
        padding: 40px;
        text-align: center;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Mandatory Academic Research Disclaimer Banner
st.warning("⚠️ **Academic Research Notice:** For academic/research demonstration only. This system is not a clinical diagnostic tool.")

st.title("🧠 Academic 2D Brain Tumor Segmentation & Classifier System")
st.caption("B.Tech Final Project | Genuine PyTorch UNet Segmentation & PyTorch CNN Multi-Class Classifier")

# 2. Sidebar Controls
st.sidebar.header("📁 MRI Input Options")

uploaded_file = st.sidebar.file_uploader("Upload 2D MRI (.png, .jpg, .jpeg)", type=["png", "jpg", "jpeg"])

sample_glioma_path = os.path.join("data", "sample_glioma.png")
sample_meningioma_path = os.path.join("data", "sample_meningioma.png")
sample_pituitary_path = os.path.join("data", "sample_pituitary.png")
sample_normal_path = os.path.join("data", "sample_normal_mri.png")
sample_bottle_path = os.path.join("data", "sample_non_mri_bottle.png")

sample_choice = st.sidebar.selectbox(
    "Or Select Preset Sample Scan:",
    options=[
        "None (Awaiting File Upload)",
        "Glioma Tumor MRI Sample",
        "Meningioma Tumor MRI Sample",
        "Pituitary Tumor MRI Sample",
        "Healthy Normal Brain MRI Sample",
        "Non-Brain Image Test (Plastic Bottle)"
    ],
    index=0
)

# Determine Input Source
image_source = None
display_filename = None

if uploaded_file is not None:
    display_filename = uploaded_file.name
    image_source = uploaded_file
    st.sidebar.success(f"File loaded: {display_filename}")
elif sample_choice == "Glioma Tumor MRI Sample" and os.path.exists(sample_glioma_path):
    display_filename = "sample_glioma.png"
    image_source = sample_glioma_path
    st.sidebar.info("Preset: Glioma Tumor Brain MRI.")
elif sample_choice == "Meningioma Tumor MRI Sample" and os.path.exists(sample_meningioma_path):
    display_filename = "sample_meningioma.png"
    image_source = sample_meningioma_path
    st.sidebar.info("Preset: Meningioma Tumor Brain MRI.")
elif sample_choice == "Pituitary Tumor MRI Sample" and os.path.exists(sample_pituitary_path):
    display_filename = "sample_pituitary.png"
    image_source = sample_pituitary_path
    st.sidebar.info("Preset: Pituitary Tumor Brain MRI.")
elif sample_choice == "Healthy Normal Brain MRI Sample" and os.path.exists(sample_normal_path):
    display_filename = "sample_normal_mri.png"
    image_source = sample_normal_path
    st.sidebar.info("Preset: Healthy Normal Brain MRI.")
elif sample_choice == "Non-Brain Image Test (Plastic Bottle)" and os.path.exists(sample_bottle_path):
    display_filename = "sample_non_mri_bottle.png"
    image_source = sample_bottle_path
    st.sidebar.warning("Preset: Non-Brain Image Test.")

# Display Controls
st.sidebar.subheader("👁️ Display & Contrast Controls")
show_overlay = st.sidebar.checkbox("Show Tumor Mask Overlay", value=True)
high_contrast = st.sidebar.checkbox("High Contrast Mode (2%-98% Scaling)", value=False)
show_contours = st.sidebar.checkbox("Highlight Perimeter Contours", value=True)
overlay_alpha = st.sidebar.slider("Overlay Opacity (Alpha)", min_value=0.1, max_value=1.0, value=0.4, step=0.05)

# NEUTRAL AWAITING STATE (No file uploaded & No preset selected)
if image_source is None:
    st.subheader("📊 Quantitative Clinical & Classification Metrics")
    card_col1, card_col2, card_col3, card_col4, card_col5 = st.columns(5)
    
    with card_col1:
        st.markdown("**Tumor Status**<br><span class='status-badge-awaiting'>ℹ️ AWAITING INPUT</span>", unsafe_allow_html=True)
    with card_col2:
        st.metric(label="Tumor Type", value="-", delta="Awaiting upload")
    with card_col3:
        st.metric(label="Classifier Confidence", value="-", delta="Awaiting upload")
    with card_col4:
        st.metric(label="Tumor Area", value="-", delta="Awaiting upload")
    with card_col5:
        st.metric(label="Brain Coverage Ratio", value="-", delta="Awaiting upload")
        
    st.markdown("---")
    
    st.markdown("""
        <div class='placeholder-card'>
            <h2 style='color:#38BDF8;'>📁 Ready for MRI Image Upload</h2>
            <p style='color:#94A3B8; font-size:1.1rem;'>
                Please upload a 2D Brain MRI scan (<strong>.png, .jpg, .jpeg</strong>) using the sidebar file uploader,<br>
                or select a preset sample scan to execute PyTorch UNet segmentation & PyTorch CNN multi-class classification.
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

# Real-Time Dynamic Inference Pipeline
def run_dynamic_pipeline(src):
    tensor, raw_original, orig_dims = preprocess_2d(src)
    
    # 1. UNet Segmentation
    pred_info = predict_2d(tensor, raw_original)
    mask = pred_info["mask"]
    tumor_detected = pred_info["tumor_detected"]
    
    # 2. Multi-Class Classifier (Only executed if valid MRI scan)
    if pred_info["is_valid_mri"]:
        cls_info = classify_tumor_2d(tensor, tumor_detected=tumor_detected)
    else:
        cls_info = {
            "tumor_type": "Invalid Input",
            "confidence": None,
            "confidence_display": "N/A",
            "raw_probabilities": {c: 0.0 for c in CLASSES},
            "classifier_called": False,
            "execution_time_ms": 0.0
        }
        
    metrics = analyze_tumor_metrics(
        mask, raw_original,
        seg_confidence=pred_info["confidence"],
        cls_result=cls_info
    )
    
    os.makedirs("data", exist_ok=True)
    overlay_img_path = os.path.join("data", "2d_tumor_overlay.png")
    
    fig_pdf = render_segmentation_figure(
        raw_original, mask, alpha=0.5, show_overlay=pred_info["is_valid_mri"],
        high_contrast=False, show_contours=pred_info["is_valid_mri"], title="2D MRI Segmentation Result"
    )
    fig_pdf.savefig(overlay_img_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig_pdf)
    
    png_bytes = export_overlay_as_png(raw_original, mask, alpha=0.5)

    # 3. Grad-CAM (only when tumor detected and classifier was called)
    gradcam_img = None
    if pred_info["is_valid_mri"] and pred_info["tumor_detected"] and cls_info.get("classifier_called"):
        try:
            classifier_model = load_trained_classifier()
            from train_2d_classifier import CLASSES as CLS
            top_idx = CLS.index(cls_info["tumor_type"]) if cls_info["tumor_type"] in CLS else 0
            # Resize tensor to 128x128 for classifier input
            import torch.nn.functional as F
            tensor_128 = F.interpolate(tensor, size=(128, 128), mode="bilinear", align_corners=False)
            activation = generate_gradcam(classifier_model, tensor_128, top_idx)
            activation_resized = resize_activation_map(activation, raw_original.shape[0], raw_original.shape[1])
            gradcam_img = blend_heatmap_with_image(raw_original, activation_resized, alpha=0.5)
        except Exception as e:
            gradcam_img = None
    
    return raw_original, mask, metrics, pred_info, cls_info, overlay_img_path, png_bytes, gradcam_img

try:
    raw_original, mask, metrics, pred_info, cls_info, overlay_img_path, png_bytes, gradcam_img = run_dynamic_pipeline(image_source)
except Exception as e:
    st.error(f"Error processing image: {e}")
    st.stop()

# Store scan result in session state for patient record saving
st.session_state["scan_result"] = {"metrics": metrics, "cls_info": cls_info}

# Resolve confidence label (Task 8.1)
display_label, is_low_confidence = resolve_tumor_type_label(
    cls_info["tumor_type"],
    cls_info["confidence"],
    pred_info["is_valid_mri"],
    pred_info["tumor_detected"]
)

# Handle Invalid Non-MRI Input Warning Banner
if not pred_info["is_valid_mri"]:
    st.error("Invalid input: Please upload a brain MRI image.")
    st.caption(f"Reason: {pred_info['validation_error']}")

# 3. Sidebar Exports
st.sidebar.subheader("📥 Export & Report Options")

try:
    pdf_bytes = generate_2d_pdf_report(
        filename=display_filename,
        metrics=metrics,
        slice_image_path=overlay_img_path
    )
    
    st.sidebar.download_button(
        label="📄 Download PDF Summary Report",
        data=pdf_bytes,
        file_name=f"Brain_Tumor_Report_{display_filename}.pdf",
        mime="application/pdf"
    )
except Exception as pdf_err:
    st.sidebar.error(f"Unable to generate PDF report: {pdf_err}")

st.sidebar.download_button(
    label="🖼️ Download Segmented Image (.png)",
    data=png_bytes,
    file_name=f"Segmented_{display_filename}.png",
    mime="image/png"
)

# Task 8.2 — Patient Record System sidebar sections
st.sidebar.markdown("---")
st.sidebar.subheader("🏥 Patient Record System")
patient_id_input = st.sidebar.text_input("Patient ID", key="patient_id")
patient_name_input = st.sidebar.text_input("Patient Name", key="patient_name")

# Save Record button — disabled if no scan result in session state
scan_done = "scan_result" in st.session_state
if st.sidebar.button("💾 Save Record", disabled=not scan_done, help="Run inference first to enable"):
    err = validate_patient_fields(patient_id_input, patient_name_input)
    if err:
        st.sidebar.error(err)
    else:
        try:
            insert_patient_record(
                patient_id_input,
                patient_name_input,
                st.session_state["scan_result"]["metrics"],
                st.session_state["scan_result"]["cls_info"]
            )
            st.sidebar.success("Record saved!")
        except Exception as e:
            st.sidebar.error(f"Failed to save: {e}")

st.sidebar.markdown("---")
st.sidebar.subheader("📋 View Patient History")
history_pid = st.sidebar.text_input("Patient ID to lookup", key="history_pid")
if st.sidebar.button("🔍 Load History"):
    st.session_state["show_history"] = history_pid

st.sidebar.markdown("---")
st.sidebar.subheader("⚠️ Danger Zone")
if st.sidebar.button("🗑️ Clear All Records"):
    st.session_state["confirm_clear"] = True
if st.session_state.get("confirm_clear"):
    st.sidebar.warning("This will delete ALL patient records. Are you sure?")
    if st.sidebar.button("✅ Confirm Delete"):
        delete_all_records()
        st.session_state["confirm_clear"] = False
        st.sidebar.success("All records deleted.")
    if st.sidebar.button("❌ Cancel"):
        st.session_state["confirm_clear"] = False

# 4. Results Metrics Cards
st.subheader("📊 Quantitative Clinical & Classification Metrics")

card_col1, card_col2, card_col3, card_col4, card_col5 = st.columns(5)

with card_col1:
    if not pred_info["is_valid_mri"]:
        status_html = "<span class='status-badge-invalid'>⚠️ INVALID INPUT</span>"
    elif metrics["tumor_detected"]:
        status_html = "<span class='status-badge-yes'>🔴 TUMOR DETECTED</span>"
    else:
        status_html = "<span class='status-badge-no'>🟢 NO TUMOR DETECTED</span>"
        
    st.markdown(f"**Tumor Status**<br>{status_html}", unsafe_allow_html=True)

with card_col2:
    if is_low_confidence:
        st.markdown(
            f"**Tumor Type**<br><span class='status-badge-lowconf'>{LOW_CONFIDENCE_WARNING_TEXT}</span>",
            unsafe_allow_html=True
        )
    else:
        st.metric(
            label="Tumor Type",
            value=display_label,
            delta="PyTorch CNN Classifier"
        )

with card_col3:
    st.metric(
        label="Classification Confidence",
        value=metrics['classification_confidence'],
        delta="Max Softmax Probability"
    )

with card_col4:
    st.metric(
        label="Tumor Area",
        value=f"{metrics['tumor_area_pixels']:,} Pixels",
        delta="UNet Segmentation Mask"
    )

with card_col5:
    st.metric(
        label="Brain Coverage Ratio",
        value=f"{metrics['brain_coverage_pct']:.2f} %",
        delta="Relative to Brain Area"
    )

st.markdown("---")

# Task 8.2 — Show patient history if requested
if st.session_state.get("show_history"):
    pid = st.session_state["show_history"]
    records = query_patient_history(pid)
    st.subheader(f"📋 Patient History: {pid}")
    if records:
        import pandas as pd
        st.dataframe(pd.DataFrame(records))
    else:
        st.info(f"No records found for patient ID: {pid}")

# 5. Visualizations — 3-column layout (Task 8.3)
col_left, col_mid, col_right = st.columns(3)

with col_left:
    st.subheader("📷 Input Image Matrix")
    fig_raw = render_segmentation_figure(
        raw_original, mask, show_overlay=False, high_contrast=high_contrast,
        title=f"Input Grayscale Image Matrix ({raw_original.shape[1]}x{raw_original.shape[0]})"
    )
    st.pyplot(fig_raw, width="stretch")
    plt.close(fig_raw)

with col_mid:
    st.subheader("🎯 PyTorch UNet Predicted Mask Overlay")
    fig_segmented = render_segmentation_figure(
        raw_original, mask, alpha=overlay_alpha, show_overlay=show_overlay and pred_info["is_valid_mri"],
        high_contrast=high_contrast, show_contours=show_contours and pred_info["is_valid_mri"],
        title="Predicted Tumor Segmentation Overlay" if pred_info["is_valid_mri"] else "Segmentation Skipped (Non-MRI)"
    )
    st.pyplot(fig_segmented, width="stretch")
    plt.close(fig_segmented)

with col_right:
    # Use the raw tumor_type for Grad-CAM title (not the warning text)
    gradcam_title = get_gradcam_panel_title(
        cls_info.get("tumor_type", "Unknown") if is_low_confidence else display_label,
        is_low_confidence
    )
    st.subheader(gradcam_title)
    if gradcam_img is not None:
        st.image(gradcam_img, use_container_width=True)
    elif not pred_info["is_valid_mri"]:
        st.info("Grad-CAM not available — Invalid input")
    else:
        st.info("Grad-CAM not available — No tumor detected")

# 6. Viva Presentation Diagnostics Panel
st.markdown("---")
with st.expander("🔬 Model Information & Architecture Telemetry (Viva Panel)", expanded=True):
    v_col1, v_col2 = st.columns(2)
    
    with v_col1:
        st.markdown(f"**Segmentation Model:** `LightweightUNet2D ({pred_info['checkpoint_path']})`")
        st.markdown(f"**Classification Model:** `LightweightClassifier2D (models/brain_tumor_classifier_2d.pth)`")
        st.markdown(f"**Classification Classes:** `{CLASSES}`")
        st.markdown(f"**Input Dimensions:** `{pred_info['tensor_shape']}`")
        st.markdown(f"**Preprocessing:** `Grayscale -> Resize (256x256) -> Min-Max Normalization (0.0 to 1.0)`")
        
    with v_col2:
        st.markdown(f"**Segmentation Threshold:** `Sigmoid Probability >= {pred_info['prob_threshold']} (Min {pred_info['min_pixel_threshold']} px)`")
        st.markdown(f"**Segmentation Inference Time:** `⚡ {pred_info['execution_time_ms']} ms`")
        st.markdown(f"**Classifier Inference Time:** `⚡ {cls_info.get('execution_time_ms', cls_info.get('classifier_time_ms', 0.0))} ms`")
        st.markdown(f"**Predicted Tumor Voxels:** `{pred_info['tumor_pixel_count']} pixels`")
        st.markdown(f"**Pipeline Forward Pass:** `{'UNet + Classifier Executed' if cls_info.get('classifier_called', cls_info.get('classifier_executed', False)) else 'Skipped / No Tumor'}`")
