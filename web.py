import io
import os
import cv2
import base64
import gdown
import torch
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from ultralytics import YOLO
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage
)

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Renal Stone Detection Demo",
    layout="wide"
)

# =========================================================
# SESSION STATE
# =========================================================
if "results_ready" not in st.session_state:
    st.session_state.results_ready = False

if "results_data" not in st.session_state:
    st.session_state.results_data = []

if "report_visibility" not in st.session_state:
    st.session_state.report_visibility = {}

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# =========================================================
# HELPERS FOR STATE
# =========================================================
def clear_all():
    st.session_state.results_ready = False
    st.session_state.results_data = []
    st.session_state.report_visibility = {}
    st.session_state.uploader_key += 1

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #f6f9fc 0%, #eef4f9 100%);
    }

    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    .hero-box {
        margin-top: 0.8rem;
        background: linear-gradient(135deg, #dfeeff 0%, #edf5ff 60%, #d9e9fa 100%);
        padding: 2rem 1.8rem;
        border-radius: 24px;
        text-align: center;
        margin-bottom: 1.4rem;
        border: 1px solid #d4e2f1;
        box-shadow: 0 8px 24px rgba(25, 55, 109, 0.07);
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        color: #17365f;
        line-height: 1.3;
        margin-bottom: 0.8rem;
        letter-spacing: 0.2px;
    }

    .hero-subtitle {
        font-size: 0.96rem;
        color: #425c7b;
        max-width: 930px;
        margin: auto;
        line-height: 1.85;
        font-weight: 500;
    }

    .focus-wrap {
        position: relative;
        margin-bottom: 1.2rem;
        transition: all 0.35s ease;
    }

    .focus-wrap::before {
        content: "";
        position: absolute;
        inset: 0;
        border-radius: 22px;
        background:
            radial-gradient(circle at 20% 20%, rgba(84, 141, 255, 0.12), transparent 30%),
            radial-gradient(circle at 80% 20%, rgba(84, 141, 255, 0.10), transparent 25%);
        filter: blur(8px);
        opacity: 0.55;
        transition: all 0.4s ease;
        z-index: 0;
    }

    .focus-wrap:hover::before {
        opacity: 1;
        filter: blur(14px);
        background:
            radial-gradient(circle at 20% 20%, rgba(84, 141, 255, 0.25), transparent 35%),
            radial-gradient(circle at 80% 20%, rgba(84, 141, 255, 0.22), transparent 30%),
            radial-gradient(circle at 50% 80%, rgba(84, 141, 255, 0.15), transparent 35%);
    }

    .focus-wrap:hover {
        transform: translateY(-4px) scale(1.01);
    }

    .focus-card {
        position: relative;
        z-index: 1;
        background: rgba(255,255,255,0.95);
        padding: 1.35rem 1.35rem;
        border-radius: 22px;
        border: 1px solid #dce7f2;
        box-shadow: 0 10px 24px rgba(14, 30, 62, 0.05);
        backdrop-filter: blur(4px);
    }

    .section-title {
        font-size: 1.3rem;
        font-weight: 800;
        color: #17365f;
        margin-bottom: 0.5rem;
    }

    .section-text {
        color: #4b6380;
        font-size: 0.95rem;
        line-height: 1.85;
        font-weight: 500;
    }

    .step-panel {
        background: rgba(248, 252, 255, 0.92);
        border: 1px solid #d7e6f5;
        border-radius: 18px;
        padding: 1rem 0.95rem 1rem 0.95rem;
        box-shadow: 0 8px 20px rgba(18, 48, 84, 0.05);
        min-height: 190px;
        transition: all 0.3s ease;
    }

    .step-panel:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(18, 48, 84, 0.08);
    }

    .step-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: linear-gradient(135deg, #2f6fed, #5f95ff);
        color: white;
        font-weight: 800;
        font-size: 0.98rem;
        margin-bottom: 0.65rem;
        box-shadow: 0 6px 12px rgba(47, 111, 237, 0.20);
    }

    .step-title {
        font-size: 1.02rem;
        font-weight: 800;
        color: #17365f;
        margin-bottom: 0.35rem;
    }

    .step-text {
        color: #536b87;
        font-size: 0.93rem;
        line-height: 1.75;
        font-weight: 500;
    }

    .upload-card {
        background: rgba(255,255,255,0.95);
        border: 1px solid #dce7f2;
        border-radius: 20px;
        padding: 1.15rem 1.2rem;
        box-shadow: 0 8px 20px rgba(18, 48, 84, 0.04);
        margin-top: 0.65rem;
        margin-bottom: 1rem;
    }

    .upload-title {
        font-size: 1.28rem;
        font-weight: 800;
        color: #17365f;
        margin-bottom: 0.3rem;
    }

    .upload-note {
        font-size: 0.96rem;
        color: #61758e;
        line-height: 1.7;
        font-weight: 500;
    }

    .result-title {
        font-size: 1.28rem;
        font-weight: 800;
        color: #17365f;
        margin-bottom: 0.2rem;
    }

    .small-scan-label {
        font-size: 0.82rem;
        color: #7b8a9d;
        font-weight: 500;
        margin-bottom: 0.35rem;
    }

    .summary-box-ui {
        background: rgba(255,255,255,0.95);
        border: 1px solid #dce7f2;
        border-radius: 18px;
        padding: 1rem 1.15rem;
        box-shadow: 0 8px 18px rgba(16, 42, 74, 0.04);
        line-height: 1.95;
        color: #17365f;
        font-size: 0.98rem;
    }

    .summary-section-title {
        font-size: 1.05rem;
        font-weight: 800;
        color: #17365f;
        margin-bottom: 0.75rem;
    }

    .note-card {
        background: linear-gradient(135deg, #ffffff 0%, #f9fcff 100%);
        border: 1px solid #dce7f2;
        border-radius: 18px;
        padding: 1.05rem 1.15rem;
        margin-top: 1rem;
        box-shadow: 0 8px 18px rgba(16, 42, 74, 0.04);
    }

    .badge {
        display: inline-block;
        padding: 0.4rem 0.85rem;
        background: #e7f4ea;
        color: #1e6d4b;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.88rem;
        margin-bottom: 0.65rem;
    }

    .note-text {
        font-size: 0.95rem;
        color: #5d7289;
        line-height: 1.8;
        font-weight: 500;
    }

    .footer-note {
        text-align: center;
        color: #74849a;
        font-size: 0.9rem;
        margin-top: 1rem;
        font-weight: 500;
    }

    .report-shell {
        position: relative;
        background: rgba(255,255,255,0.95);
        border: 1px solid #dce7f2;
        border-radius: 20px;
        padding: 1.4rem;
        box-shadow: 0 8px 20px rgba(18, 48, 84, 0.04);
        overflow: hidden;
        min-height: 280px;
    }

    .report-blur {
        filter: blur(7px);
        opacity: 0.52;
        pointer-events: none;
        user-select: none;
    }

    .report-preview-card {
        background: white;
        border: 1px solid #dce7f2;
        border-radius: 18px;
        padding: 1rem;
        box-shadow: inset 0 0 0 1px #eef3f8;
    }

    .report-preview-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #17365f;
        margin-bottom: 0.8rem;
    }

    .clear-wrap {
        margin-top: 1rem;
    }

    div.stButton > button {
        background: linear-gradient(135deg, #2f6fed, #4d87fa);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.5rem;
        font-weight: 700;
        font-size: 1rem;
        box-shadow: 0 8px 16px rgba(47, 111, 237, 0.18);
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #295fd0, #4478df);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# MODEL / DEVICE
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best.pt"
GDRIVE_FILE_ID = "12FxZPRakPMVpDJ6yRQ9Wmcgy70lPfNC0"

def ensure_model_downloaded():
    if not os.path.exists(MODEL_PATH):
        download_url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        with st.spinner("Downloading model file for first-time setup..."):
            gdown.download(download_url, MODEL_PATH, quiet=False)

@st.cache_resource
def load_models():
    ensure_model_downloaded()

    yolo_inf = YOLO(MODEL_PATH)

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model = model.to(DEVICE).float().eval()

    for p in model.parameters():
        p.requires_grad_(True)

    return yolo_inf, model

with st.spinner("Loading model and interface..."):
    yolo_inf, raw_model = load_models()

# =========================================================
# IMAGE / MODEL HELPERS
# =========================================================
def read_uploaded_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)

def preprocess_image(img_rgb, img_size=640, crop_top=35):
    if img_rgb.shape[0] > crop_top:
        img_rgb = img_rgb[crop_top:, :, :]
    img_resized = cv2.resize(img_rgb, (img_size, img_size))
    img_float = img_resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    tensor = tensor.float().requires_grad_(True)
    return img_resized, img_float, tensor

def draw_box(image_rgb, box_xyxy, color=(255, 0, 0), thickness=2):
    img = image_rgb.copy()
    if box_xyxy is not None:
        x1, y1, x2, y2 = box_xyxy.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img

def get_ultrasound_mask(img_rgb_float):
    gray = cv2.cvtColor((img_rgb_float * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 8, 255, cv2.THRESH_BINARY)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    return mask.astype(np.float32) / 255.0

def make_roi_weight_map(h, w, box_xyxy, sigma_scale=0.55):
    x1, y1, x2, y2 = box_xyxy.astype(np.float32)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    sigma_x = bw * sigma_scale
    sigma_y = bh * sigma_scale

    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)

    gaussian = np.exp(-(((xx - cx) ** 2) / (2 * sigma_x ** 2) +
                        ((yy - cy) ** 2) / (2 * sigma_y ** 2)))

    rect = np.zeros((h, w), dtype=np.float32)
    x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
    x1i = max(0, min(x1i, w - 1))
    x2i = max(x1i + 1, min(x2i, w))
    y1i = max(0, min(y1i, h - 1))
    y2i = max(y1i + 1, min(y2i, h))
    rect[y1i:y2i, x1i:x2i] = 1.0
    rect = cv2.GaussianBlur(rect, (31, 31), 0)

    weight = 0.65 * gaussian + 0.35 * rect
    weight -= weight.min()
    if weight.max() > 0:
        weight /= weight.max()
    return weight

def overlay_heatmap(img_float, cam):
    cam_resized = cv2.resize(cam, (img_float.shape[1], img_float.shape[0]))
    heatmap = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = 0.56 * (heatmap.astype(np.float32) / 255.0) + 0.44 * img_float
    overlay = np.clip(overlay, 0, 1)
    return cam_resized, overlay

def gradcam_from_box(model, input_tensor, target_layer, box_xyxy, input_hw=(640, 640)):
    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        if isinstance(out, (list, tuple)):
            out = out[0]
        activations.clear()
        gradients.clear()
        activations.append(out)

        def grad_hook(grad):
            gradients.append(grad)

        out.register_hook(grad_hook)

    hook_handle = target_layer.register_forward_hook(forward_hook)

    try:
        model.zero_grad(set_to_none=True)
        _ = model(input_tensor)

        if len(activations) == 0:
            raise RuntimeError("No activations captured.")

        act = activations[0]
        _, _, h, w = act.shape
        in_h, in_w = input_hw
        x1, y1, x2, y2 = box_xyxy.astype(np.float32)

        fx1 = int(np.floor(x1 * w / in_w))
        fx2 = int(np.ceil(x2 * w / in_w))
        fy1 = int(np.floor(y1 * h / in_h))
        fy2 = int(np.ceil(y2 * h / in_h))

        fx1 = max(0, min(fx1, w - 1))
        fx2 = max(fx1 + 1, min(fx2, w))
        fy1 = max(0, min(fy1, h - 1))
        fy2 = max(fy1 + 1, min(fy2, h))

        roi = act[:, :, fy1:fy2, fx1:fx2]
        if roi.numel() == 0:
            raise RuntimeError("ROI is empty.")

        target_score = (roi ** 2).mean()
        target_score.backward()

        if len(gradients) == 0:
            raise RuntimeError("No gradients captured.")

        grad = gradients[0]
        grad_roi = grad[:, :, fy1:fy2, fx1:fx2]
        weights = grad_roi.mean(dim=(2, 3), keepdim=True)

        cam = (weights * act).sum(dim=1)
        cam = torch.relu(cam)[0].detach().cpu().numpy()

        if np.max(cam) <= 1e-10:
            cam = np.abs(((weights * act).sum(dim=1))[0].detach().cpu().numpy())

        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        cam_roi = cam[fy1:fy2, fx1:fx2]
        inside_mean = float(cam_roi.mean()) if cam_roi.size > 0 else 0.0
        outside_mean = float(cam.mean())
        layer_score = inside_mean - outside_mean

        return cam, layer_score

    finally:
        hook_handle.remove()

def get_size_label(diagonal_px):
    if diagonal_px < 60:
        return "Small"
    elif diagonal_px < 120:
        return "Medium"
    return "Large"

def image_to_pil(image_rgb_or_float):
    if image_rgb_or_float.dtype != np.uint8:
        img = (np.clip(image_rgb_or_float, 0, 1) * 255).astype(np.uint8)
    else:
        img = image_rgb_or_float.copy()
    return Image.fromarray(img)

def image_to_png_bytes(image_rgb_or_float):
    pil_img = image_to_pil(image_rgb_or_float)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

def image_to_base64(image_rgb_or_float):
    return base64.b64encode(image_to_png_bytes(image_rgb_or_float)).decode("utf-8")

def pil_to_rl_image(pil_image, width=2.8*inch, height=2.2*inch):
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    buf.seek(0)
    img = RLImage(buf, width=width, height=height)
    return img

# =========================================================
# REPORT HELPERS
# =========================================================
def make_report_html(report_data):
    now = datetime.now()
    date_str = now.strftime("%d-%m-%Y")
    time_str = now.strftime("%H:%M:%S")

    detection_b64 = image_to_base64(report_data["detection_img"])
    overlay_b64 = image_to_base64(report_data["gradcam_overlay"])

    html = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 28px;
                color: #17365f;
            }}
            .title {{
                font-size: 28px;
                font-weight: 700;
                text-align: center;
                margin-bottom: 6px;
            }}
            .subtitle {{
                text-align: center;
                font-size: 14px;
                color: #4b6380;
                margin-bottom: 26px;
            }}
            .section {{
                border: 1px solid #dce7f2;
                border-radius: 16px;
                padding: 18px;
                margin-bottom: 18px;
            }}
            .section-title {{
                font-size: 20px;
                font-weight: 700;
                margin-bottom: 12px;
            }}
            .grid {{
                display: flex;
                gap: 18px;
                align-items: flex-start;
            }}
            .col {{
                flex: 1;
            }}
            .imgbox {{
                border: 1px solid #dce7f2;
                border-radius: 14px;
                padding: 10px;
                text-align: center;
                background: #fafcff;
            }}
            img {{
                width: 100%;
                max-width: 380px;
                border-radius: 10px;
            }}
            .label {{
                margin-top: 8px;
                font-size: 13px;
                color: #5d7289;
            }}
            .summary-box {{
                border: 1px solid #dce7f2;
                border-radius: 14px;
                padding: 16px;
                background: #fafcff;
                line-height: 1.9;
                font-size: 15px;
            }}
            .note {{
                background: #f9fcff;
                border: 1px solid #dce7f2;
                border-radius: 14px;
                padding: 14px;
                font-size: 14px;
                line-height: 1.8;
                color: #5d7289;
            }}
        </style>
    </head>
    <body>
        <div class="title">Renal Stone Detection Report</div>
        <div class="subtitle">Deep Learning-Based Analysis of Ultrasonography Image</div>

        <div class="section">
            <div class="section-title">Case Information</div>
            <div class="summary-box">
                <b>Case ID / Scan ID:</b> {report_data["scan_name"]}<br>
                <b>Date:</b> {date_str}<br>
                <b>Time:</b> {time_str}
            </div>
        </div>

        <div class="section">
            <div class="section-title">Examination Details</div>
            <div class="summary-box">
                <b>Examination Type:</b> Renal Ultrasonography<br>
                <b>Analysis Type:</b> Automated Deep Learning-Based Assessment<br>
                <b>Input Modality:</b> Ultrasound Image
            </div>
        </div>

        <div class="section">
            <div class="section-title">AI Detection Summary</div>
            <div class="summary-box">
                <b>Stone Detection Status:</b> Detected<br>
                <b>Approximate Size Category:</b> {report_data["size_label"]}
            </div>
        </div>

        <div class="section">
            <div class="section-title">Bounding Box Measurements</div>
            <div class="summary-box">
                Width: {report_data["width_px"]:.1f} px<br>
                Height: {report_data["height_px"]:.1f} px<br>
                Diagonal: {report_data["diagonal_px"]:.1f} px
            </div>
        </div>

        <div class="section">
            <div class="section-title">Visual Findings</div>
            <div class="grid">
                <div class="col">
                    <div class="imgbox">
                        <img src="data:image/png;base64,{detection_b64}" />
                        <div class="label">Detection Output</div>
                    </div>
                </div>
                <div class="col">
                    <div class="imgbox">
                        <img src="data:image/png;base64,{overlay_b64}" />
                        <div class="label">Focused Grad-CAM Overlay</div>
                    </div>
                </div>
            </div>
            <div style="margin-top:14px; color:#5d7289; line-height:1.8; font-size:14px;">
                <b>These visualizations highlight:</b><br>
                • The localized region where the stone is detected<br>
                • The area where the model focused during prediction
            </div>
        </div>

        <div class="section">
            <div class="section-title">Impression</div>
            <div class="summary-box">
                Automated analysis suggests the presence of a renal stone in the provided ultrasonography image.
                The highlighted region and attention visualization support the model’s prediction.
            </div>
        </div>

        <div class="section">
            <div class="section-title">Size Estimation Note</div>
            <div class="summary-box">
                The approximate size is derived from the detected bounding box in pixel space and should be interpreted
                as a visual estimate rather than an exact clinical measurement.
            </div>
        </div>

        <div class="note">
            <b>Disclaimer:</b><br>
            This report is generated by a deep learning-based decision-support system for research and demonstration purposes only.
            It is not a substitute for clinical diagnosis or expert radiological interpretation.
        </div>
    </body>
    </html>
    """
    return html

def make_report_pdf(report_data):
    now = datetime.now()
    date_str = now.strftime("%d-%m-%Y")
    time_str = now.strftime("%H:%M:%S")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=32,
        rightMargin=32,
        topMargin=28,
        bottomMargin=28
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        textColor=colors.HexColor("#17365f"),
        alignment=1,
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        "SubtitleStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        textColor=colors.HexColor("#4b6380"),
        alignment=1,
        spaceAfter=16
    )
    section_style = ParagraphStyle(
        "SectionStyle",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        textColor=colors.HexColor("#17365f"),
        spaceAfter=8,
        spaceBefore=6
    )
    normal_style = ParagraphStyle(
        "NormalStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=16,
        textColor=colors.HexColor("#4b6380")
    )

    story = []

    story.append(Paragraph("Renal Stone Detection Report", title_style))
    story.append(Paragraph("Deep Learning-Based Analysis of Ultrasonography Image", subtitle_style))
    story.append(Spacer(1, 8))

    def section_box(title, content_paragraphs):
        story.append(Paragraph(title, section_style))
        for p in content_paragraphs:
            story.append(p)
        story.append(Spacer(1, 10))

    section_box("Case Information", [
        Paragraph(f"<b>Case ID / Scan ID:</b> {report_data['scan_name']}", normal_style),
        Paragraph(f"<b>Date:</b> {date_str}", normal_style),
        Paragraph(f"<b>Time:</b> {time_str}", normal_style),
    ])

    section_box("Examination Details", [
        Paragraph("<b>Examination Type:</b> Renal Ultrasonography", normal_style),
        Paragraph("<b>Analysis Type:</b> Automated Deep Learning-Based Assessment", normal_style),
        Paragraph("<b>Input Modality:</b> Ultrasound Image", normal_style),
    ])

    section_box("AI Detection Summary", [
        Paragraph("<b>Stone Detection Status:</b> Detected", normal_style),
        Paragraph(f"<b>Approximate Size Category:</b> {report_data['size_label']}", normal_style),
    ])

    section_box("Bounding Box Measurements", [
        Paragraph(f"Width: {report_data['width_px']:.1f} px", normal_style),
        Paragraph(f"Height: {report_data['height_px']:.1f} px", normal_style),
        Paragraph(f"Diagonal: {report_data['diagonal_px']:.1f} px", normal_style),
    ])

    story.append(Paragraph("Visual Findings", section_style))
    det_img = pil_to_rl_image(image_to_pil(report_data["detection_img"]), width=2.5*inch, height=2.3*inch)
    cam_img = pil_to_rl_image(image_to_pil(report_data["gradcam_overlay"]), width=2.5*inch, height=2.3*inch)

    visual_table = Table(
        [[det_img, cam_img],
         [Paragraph("Detection Output", normal_style), Paragraph("Focused Grad-CAM Overlay", normal_style)]],
        colWidths=[250, 250]
    )
    visual_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
    ]))
    story.append(visual_table)
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "<b>These visualizations highlight:</b><br/>"
        "• The localized region where the stone is detected<br/>"
        "• The area where the model focused during prediction",
        normal_style
    ))
    story.append(Spacer(1, 12))

    section_box("Impression", [
        Paragraph(
            "Automated analysis suggests the presence of a renal stone in the provided ultrasonography image. "
            "The highlighted region and attention visualization support the model’s prediction.",
            normal_style
        )
    ])

    section_box("Size Estimation Note", [
        Paragraph(
            "The approximate size is derived from the detected bounding box in pixel space and should be interpreted "
            "as a visual estimate rather than an exact clinical measurement.",
            normal_style
        )
    ])

    section_box("Disclaimer", [
        Paragraph(
            "This report is generated by a deep learning-based decision-support system for research and demonstration "
            "purposes only. It is not a substitute for clinical diagnosis or expert radiological interpretation.",
            normal_style
        )
    ])

    doc.build(story)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf

# =========================================================
# MAIN DETECTION
# =========================================================
def run_detection_and_gradcam(uploaded_file):
    img_rgb = read_uploaded_image(uploaded_file)
    img_resized, img_float, input_tensor = preprocess_image(img_rgb, img_size=640, crop_top=35)

    results = yolo_inf.predict(
        source=img_resized,
        imgsz=640,
        conf=0.05,
        verbose=False
    )

    if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
        return {
            "detected": False,
            "original": img_resized
        }

    boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()

    best_idx = int(np.argmax(confs))
    best_box = boxes_xyxy[best_idx]
    best_conf = float(confs[best_idx])

    x1, y1, x2, y2 = best_box
    width_px = float(max(0, x2 - x1))
    height_px = float(max(0, y2 - y1))
    diagonal_px = float(np.sqrt(width_px**2 + height_px**2))
    size_label = get_size_label(diagonal_px)

    detection_img = draw_box(img_resized, best_box)

    candidate_idxs = [-5, -4, -3, -2]
    best_cam = None
    best_layer_score = -1e9

    for idx in candidate_idxs:
        try:
            cam, layer_score = gradcam_from_box(
                model=raw_model,
                input_tensor=input_tensor,
                target_layer=raw_model.model[idx],
                box_xyxy=best_box,
                input_hw=(640, 640)
            )
            if layer_score > best_layer_score:
                best_layer_score = layer_score
                best_cam = cam
        except Exception:
            pass

    if best_cam is None:
        overlay = img_float.copy()
    else:
        cam_resized = cv2.resize(best_cam, (img_float.shape[1], img_float.shape[0]))
        fan_mask = get_ultrasound_mask(img_float)
        roi_weight = make_roi_weight_map(
            h=img_float.shape[0],
            w=img_float.shape[1],
            box_xyxy=best_box,
            sigma_scale=0.55
        )

        focused_cam = cam_resized * fan_mask * roi_weight
        focused_cam -= focused_cam.min()
        if focused_cam.max() > 0:
            focused_cam /= focused_cam.max()

        focused_cam = cv2.GaussianBlur(focused_cam, (25, 25), 0)
        focused_cam -= focused_cam.min()
        if focused_cam.max() > 0:
            focused_cam /= focused_cam.max()

        _, overlay = overlay_heatmap(img_float, focused_cam)

    return {
        "detected": True,
        "original": img_resized,
        "detection_img": detection_img,
        "gradcam_overlay": overlay,
        "confidence": best_conf,
        "width_px": width_px,
        "height_px": height_px,
        "diagonal_px": diagonal_px,
        "size_label": size_label
    }

# =========================================================
# CALLBACKS
# =========================================================
def trigger_detection():
    if uploaded_files:
        st.session_state.results_data = []
        st.session_state.report_visibility = {}

        for idx, file in enumerate(uploaded_files, start=1):
            result = run_detection_and_gradcam(file)
            result["scan_index"] = idx
            result["scan_name"] = f"Scan {idx}"
            st.session_state.results_data.append(result)
            st.session_state.report_visibility[idx] = False

        st.session_state.results_ready = True

def show_report_for_scan(scan_idx):
    st.session_state.report_visibility[scan_idx] = True

# =========================================================
# HERO
# =========================================================
st.markdown("""
<div class="hero-box">
    <div class="hero-title">Deep Learning - Based Detection of Renal Stones in Ultrasonography Images</div>
    <div class="hero-subtitle">
        This webpage is developed to demonstrate the project workflow, including renal stone detection from ultrasound scans,
        focused visual explanation using Grad-CAM, and approximate bounding-box-based size estimation with report generation.
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# ABOUT
# =========================================================
st.markdown("""
<div class="focus-wrap">
    <div class="focus-card">
        <div class="section-title">About the System</div>
        <div class="section-text">
            This system is designed to support the automated analysis of renal ultrasonography images using deep learning.
            It identifies the likely presence of renal stones, localizes the region of interest, and highlights the model’s
            attention using focused Grad-CAM visualization.
            <br><br>
            In addition, the system provides an approximate size indication derived from the detected bounding box, making the
            final output more informative and easier to interpret during project demonstration.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# HOW TO USE
# =========================================================
st.markdown("""
<div class="focus-wrap">
    <div class="focus-card">
        <div class="section-title">How to Use</div>
        <div class="section-text">
            Follow these simple steps to analyze the uploaded ultrasound image and review the generated output.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3, gap="medium")

with c1:
    st.markdown("""
    <div class="step-panel">
        <div class="step-number">1</div>
        <div class="step-title">Upload Scan</div>
        <div class="step-text">
            Upload one or more ultrasound images from your device for automated analysis.
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="step-panel">
        <div class="step-number">2</div>
        <div class="step-title">Run Detection</div>
        <div class="step-text">
            The model analyzes the scan and determines whether a renal stone is detected.
        </div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="step-panel">
        <div class="step-number">3</div>
        <div class="step-title">Review Output</div>
        <div class="step-text">
            View confidence, approximate size, Grad-CAM heatmap, overlay visualization, and downloadable report.
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# UPLOAD
# =========================================================
st.markdown("""
<div class="upload-card">
    <div class="upload-title">Upload Ultrasonography Images</div>
    <div class="upload-note">
        Supported file types: JPG, JPEG, PNG. You may upload one or multiple images for project demonstration and analysis.
    </div>
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Choose image(s)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    label_visibility="collapsed",
    key=f"uploaded_files_widget_{st.session_state.uploader_key}"
)

if uploaded_files and not st.session_state.results_ready:
    st.markdown("""
    <div class="upload-card">
        <div class="upload-title">Uploaded Image Preview</div>
        <div class="upload-note">
            A preview of the selected scan(s) is shown below. Once confirmed, click Start Detection to continue.
        </div>
    </div>
    """, unsafe_allow_html=True)

    preview_cols = st.columns(min(3, len(uploaded_files)))
    for i, file in enumerate(uploaded_files[:3]):
        image = Image.open(file)
        with preview_cols[i % len(preview_cols)]:
            st.image(image, use_container_width=True)

    btn_left, btn_mid, btn_right = st.columns([1, 1, 1])
    with btn_mid:
        st.button("Start Detection", use_container_width=True, on_click=trigger_detection)

# =========================================================
# RESULTS
# =========================================================
if st.session_state.results_ready and st.session_state.results_data:
    st.markdown("""
    <div class="upload-card">
        <div class="result-title">Results</div>
    </div>
    """, unsafe_allow_html=True)

    for result in st.session_state.results_data:
        scan_idx = result["scan_index"]

        st.markdown(f"<div class='small-scan-label'>{result['scan_name']}</div>", unsafe_allow_html=True)

        if not result["detected"]:
            st.warning("No stone detected in this image.")
            st.image(result["original"], caption="Uploaded Scan", use_container_width=True)
            continue

        left, middle, right = st.columns([1.05, 1.05, 0.9], gap="large")

        with left:
            st.image(
                result["detection_img"],
                caption="Detection Output with Bounding Box",
                use_container_width=True
            )

        with middle:
            st.image(
                result["gradcam_overlay"],
                caption="Focused Grad-CAM Overlay",
                use_container_width=True
            )

        with right:
            st.markdown(f"""
            <div class="summary-box-ui">
                <div class="summary-section-title">Prediction Summary</div>
                <b>Stone Detection Status:</b> Detected<br>
                <b>Approximate Size Category:</b> {result["size_label"]}<br>
                <b>Bounding Box Width:</b> {result["width_px"]:.1f} px<br>
                <b>Bounding Box Height:</b> {result["height_px"]:.1f} px<br>
                <b>Bounding Box Diagonal:</b> {result["diagonal_px"]:.1f} px
            </div>
            """, unsafe_allow_html=True)

        report_html = make_report_html(result)
        report_pdf = make_report_pdf(result)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="upload-card">
            <div class="result-title">Report</div>
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state.report_visibility.get(scan_idx, False):
            st.markdown("""
            <div class="report-shell">
                <div class="report-blur">
                    <div class="report-preview-card">
                        <div class="report-preview-title">Renal Stone Detection Report</div>
                        <p style="line-height:1.9; color:#5d7289;">
                            Case Information<br>
                            Examination Details<br>
                            AI Detection Summary<br>
                            Visual Findings<br>
                            Impression<br>
                            Size Estimation Note<br>
                            Disclaimer
                        </p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            left_sp, mid_sp, right_sp = st.columns([1, 1, 1])
            with mid_sp:
                st.button(
                    "Generate Report",
                    key=f"generate_report_{scan_idx}",
                    use_container_width=True,
                    on_click=show_report_for_scan,
                    args=(scan_idx,)
                )

        else:
            preview_col, icon_col = st.columns([12, 1])
            with preview_col:
                st.components.v1.html(report_html, height=980, scrolling=True)
            with icon_col:
                st.download_button(
                    "⬇️",
                    data=report_pdf,
                    file_name=f"renal_stone_report_scan_{scan_idx}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

        st.markdown("<br><hr><br>", unsafe_allow_html=True)

    cbl, cbm, cbr = st.columns([1, 1, 1])
    with cbm:
        st.markdown("<div class='clear-wrap'></div>", unsafe_allow_html=True)
        st.button("Clear All / Start New Case", use_container_width=True, on_click=clear_all)

# =========================================================
# NOTE
# =========================================================
st.markdown("""
<div class="note-card">
    <div class="badge">Research Use Note</div>
    <div class="note-text">
        Approximate size is derived from the detected bounding box in pixel space and should be interpreted as a visual estimate,
        not as an exact clinical measurement. This system is intended for project demonstration and research-oriented interpretation.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="footer-note">
    Project Demonstration Interface • Renal Stone Detection from Ultrasonography Images
</div>
""", unsafe_allow_html=True)
