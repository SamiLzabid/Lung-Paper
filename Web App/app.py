import io
import os
import math
import html
import base64
import cv2
import torch
import numpy as np
import streamlit as st
import torchvision.transforms as transforms
from PIL import Image
from model import FetleSwinHybrid

# 1. Page Configuration (Must be the first Streamlit command)
#    "centered" + the 794 px max-width in clinical_design.html = A4-portrait column (also fine on phones)
st.set_page_config(
    page_title="FeLASwin-Net Diagnostics",
    layout="centered",
    page_icon="🩺",
    initial_sidebar_state="collapsed",
)

# Grad-CAM target layer.
#   None  -> auto-pick the last feature layer (conv map or Swin token map) that influences the prediction
#   "name" -> pin a specific layer, using its name from `model.named_modules()`
GRADCAM_TARGET_LAYER = None

# 2. Helper function to load the external HTML/CSS file
def load_clinical_design(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            design_html = f.read()
        # Inject both the CSS styles and the HTML header into the app
        st.markdown(design_html, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("clinical_design.html not found. Proceeding with default Streamlit theme.")

# 3. Inject the UI design
load_clinical_design('clinical_design.html')

# 4. Sidebar Configuration (Keeping only Model Specs)
with st.sidebar:
    st.markdown("### 📊 Model Specifications")
    st.info(
        "**Architecture**: FeLASwin-Net\n\n"
        "**Feature Modules**: FETLE + AMSFA\n\n"
        "**Attention**: Swin Native\n\n"
        "**Status**: Deployed for Inference"
    )

# 5. Preprocessing & Transform Functions
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def apply_target_contrast_enhancement(pil_img: Image.Image) -> Image.Image:
    """Applies gamma correction and contrast stretching to Lung CT scans."""
    gray = np.array(pil_img.convert("L"), dtype=np.float32)
    non_bg_pixels = gray[gray > 15]
    
    if len(non_bg_pixels) > 0:
        p_low = np.percentile(non_bg_pixels, 20)
        p_high = np.percentile(non_bg_pixels, 98)
    else:
        p_low, p_high = 0, 255

    if p_high - p_low > 0:
        stretched = np.clip((gray - p_low) / (p_high - p_low) * 255.0, 0, 255)
    else:
        stretched = gray

    gamma = 1.4
    gamma_corrected = np.power(stretched / 255.0, gamma) * 255.0
    gamma_corrected = np.clip(gamma_corrected, 0, 255).astype(np.uint8)

    blurred = cv2.GaussianBlur(gamma_corrected, (3, 3), 1.0)
    sharpened = cv2.addWeighted(gamma_corrected, 1.4, blurred, -0.4, 0)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    rgb = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)

# 6. Model Loading
@st.cache_resource
def load_model(modality_selection):
    """Loads the model once and caches it in memory."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FetleSwinHybrid(num_classes=3, drop_path_rate=0.1, attn_drop_rate=0.1).to(device)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weight_filename = "felaswin_ct.pth" if modality_selection == "Lung CT Scan" else "felaswin_xray.pth"
    weight_file = os.path.join(base_dir, "models", weight_filename)
    
    try:
        model.load_state_dict(torch.load(weight_file, map_location=device, weights_only=True))
    except Exception as e:
        st.warning(f"Attempted to load weights from: {weight_file}\n\nFile not found. Using untrained weights for UI demonstration.")
        
    model.eval()
    return model, device

# 6a. Static disease overviews (shown under the "Predicted Class" bar)
#     Paraphrased from public patient-education pages: Mayo Clinic, Cleveland Clinic,
#     WebMD and National Jewish Health.
#     status -> colour of the predicted-class bar:  ok = green, watch = amber, alert = red
CLASS_INFO = {
    "Normal": {
        "status": "ok",
        "overview": (
            "No suspicious nodule, mass or consolidation pattern was detected. Healthy lungs appear "
            "clear and air-filled on chest imaging."
        ),
        "source": "General radiology reference",
    },
    "Benign": {
        "status": "watch",
        "overview": (
            "A non-cancerous growth in the lung, usually a small nodule under 3 cm. Most lung nodules are "
            "benign and often trace back to old infection, scarring or inflammation; hamartoma is the "
            "most common benign lung tumor."
        ),
        "source": "Cleveland Clinic · WebMD",
    },
    "Malignant": {
        "status": "alert",
        "overview": (
            "Lung cancer develops when abnormal lung cells grow uncontrollably and form a tumor that can "
            "invade nearby tissue and spread. Smoking causes most cases, and early detection widens treatment options."
        ),
        "source": "Mayo Clinic",
    },
    "COVID-19": {
        "status": "alert",
        "overview": (
            "A respiratory illness caused by the SARS-CoV-2 coronavirus. In severe cases, it progresses "
            "to bilateral pneumonia where fluid in the air sacs reduces oxygen exchange."
        ),
        "source": "WebMD · National Jewish Health",
    },
    "Pneumonia": {
        "status": "watch",
        "overview": (
            "An infection that inflames the air sacs in one or both lungs, which may fill with fluid or "
            "pus. Bacteria, viruses and fungi can all cause it."
        ),
        "source": "Mayo Clinic",
    },
}

# 6b. HTML builders (styles live in clinical_design.html -> classes starting with "fl-")
#     NOTE: every string is built on a single line with no indentation or blank lines,
#     otherwise Streamlit's markdown parser would treat it as a code block.
def pil_to_base64(img: Image.Image, max_side: int = 900) -> str:
    """Downscale (for the browser only) and encode a PIL image as base64 PNG."""
    img = img.copy()
    img.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def card_head(title: str, sub: str) -> str:
    return (
        '<div class="fl-card-head">'
        f'<div class="fl-card-title">{html.escape(title)}</div>'
        f'<div class="fl-card-sub">{html.escape(sub)}</div>'
        '</div>'
    )

def build_preview_card(img: Image.Image, label: str) -> str:
    b64 = pil_to_base64(img)
    return (
        '<div class="fl-preview">'
        f'<div class="fl-badge"><i></i>{html.escape(label)}</div>'
        '<div class="fl-preview-stage">'
        f'<img src="data:image/png;base64,{b64}" alt="Input image"/>'
        '</div></div>'
    )

def build_result_card(classes, probs, modality: str) -> str:
    order = np.argsort(probs)[::-1]          # highest probability first
    top_label = classes[order[0]]
    info = CLASS_INFO[top_label]
    status = info["status"]
    glyph = "✓" if status == "ok" else "!"

    rows = []
    for rank, idx in enumerate(order):
        pct = float(probs[idx]) * 100
        min_w = "4px" if pct >= 0.5 else "0"  # keep tiny values visible as a dot
        row_cls = "fl-row fl-top" if rank == 0 else "fl-row"
        rows.append(
            f'<div class="{row_cls}">'
            f'<div class="fl-bar"><span style="width:{pct:.2f}%;min-width:{min_w}"></span></div>'
            '<div class="fl-row-text">'
            f'<span>{html.escape(classes[idx])}</span><span class="fl-dots"></span><span>{pct:.1f}%</span>'
            '</div></div>'
        )

    return (
        f'<div class="fl-card fl-status-{status}">'
        + card_head("Diagnosis Report", f"{modality} · automated classification") +
        f'<div class="fl-result-title">{html.escape(top_label)}</div>'
        '<div class="fl-rows">' + "".join(rows) + '</div>'
        f'<div class="fl-predicted"><span class="fl-glyph">{glyph}</span>Predicted Class: {html.escape(top_label)}</div>'
        '<div class="fl-overview">'
        f'<div class="fl-overview-title">About {html.escape(top_label)}</div>'
        f'<div class="fl-overview-text">{info["overview"]}</div>'
        f'<div class="fl-overview-src">Source: {info["source"]}</div>'
        '</div></div>'
    )

def build_gradcam_card(original: Image.Image, overlay: Image.Image, label: str, layer_name: str) -> str:
    b64_in = pil_to_base64(original)
    b64_cam = pil_to_base64(overlay)
    return (
        '<div class="fl-card">'
        + card_head("Explainable AI · Grad-CAM", "Highlighted regions contributed most to the predicted class") +
        '<div class="fl-cam-grid">'
        '<div><div class="fl-cam-stage">'
        f'<img src="data:image/png;base64,{b64_in}" alt="Input image"/></div>'
        '<div class="fl-cam-cap">Input</div></div>'
        '<div><div class="fl-cam-stage">'
        f'<img src="data:image/png;base64,{b64_cam}" alt="Grad-CAM overlay"/></div>'
        '<div class="fl-cam-cap">Grad-CAM overlay</div></div>'
        '</div>'
        '<div class="fl-legend"><span>Low</span><div class="fl-legend-bar"></div><span>High</span></div>'
        f'<div class="fl-cam-note">Target class: {html.escape(label)} · Layer: {html.escape(layer_name)}</div>'
        '</div>'
    )

# 6c. Grad-CAM (works with conv feature maps and Swin-style token maps)
def _to_bchw(t: torch.Tensor):
    """Return a (B, C, H, W) view of a feature tensor, or None if it is not spatial."""
    if t.dim() == 4:
        _, d1, d2, d3 = t.shape
        if d1 == d2 and d3 != d1:                      # channels-last (B, H, W, C), Swin style
            return t.permute(0, 3, 1, 2)
        return t                                        # already (B, C, H, W)
    if t.dim() == 3:                                    # tokens (B, N, C)
        b, n, c = t.shape
        side = int(round(math.sqrt(n)))
        if side * side == n:
            return t.transpose(1, 2).reshape(b, c, side, side)
        side = int(round(math.sqrt(n - 1)))
        if side * side == n - 1:                        # drop a leading class token
            return t[:, 1:].transpose(1, 2).reshape(b, c, side, side)
    return None

def compute_gradcam(model, input_tensor, target_layer=None):
    """Grad-CAM for the predicted class.

    Returns (cam in [0, 1] as a 2-D float32 array, layer name, class index).
    """
    model.eval()
    named = dict(model.named_modules())
    if target_layer is not None:
        if target_layer not in named:
            raise ValueError(f"Layer '{target_layer}' not found in the model.")
        candidates = [(target_layer, named[target_layer])]
    else:
        candidates = [(n, m) for n, m in named.items() if n]   # skip the root module

    captured, handles = [], []

    def make_hook(name):
        def hook(_module, _inputs, output):
            out = output[0] if isinstance(output, (tuple, list)) and len(output) > 0 else output
            if torch.is_tensor(out) and out.requires_grad and out.dim() in (3, 4) and out.shape[0] == 1:
                captured.append((name, out))
        return hook

    for name, module in candidates:
        handles.append(module.register_forward_hook(make_hook(name)))

    try:
        with torch.enable_grad():
            x = input_tensor.detach().clone().requires_grad_(True)
            logits = model(x)
            class_idx = int(logits.argmax(dim=1).item())
            score = logits[0, class_idx]

            # keep spatial layers only (at least 2x2), then try them from the output side backwards
            spatial = []
            for name, act in captured:
                bchw = _to_bchw(act)
                if bchw is not None and bchw.shape[2] * bchw.shape[3] >= 4:
                    spatial.append((name, act))

            for name, act in reversed(spatial):
                try:
                    grad = torch.autograd.grad(score, act, retain_graph=True, allow_unused=True)[0]
                except RuntimeError:
                    continue
                if grad is None or not torch.isfinite(grad).all() or grad.abs().sum() == 0:
                    continue

                a = _to_bchw(act.detach())
                g = _to_bchw(grad.detach())
                weights = g.mean(dim=(2, 3), keepdim=True)
                cam = torch.relu((weights * a).sum(dim=1))[0]
                if float(cam.max()) <= 0:
                    cam = (weights * a).sum(dim=1)[0].abs()
                cam = cam - cam.min()
                cam = cam / (cam.max() + 1e-8)
                return cam.cpu().numpy().astype(np.float32), name, class_idx
    finally:
        for h in handles:
            h.remove()

    raise RuntimeError("No spatial feature layer with a usable gradient was found. "
                       "Set GRADCAM_TARGET_LAYER to a layer name from model.named_modules().")

def overlay_heatmap(pil_img: Image.Image, cam: np.ndarray) -> Image.Image:
    """Blend a JET heat-map onto the image; low-activation areas stay untouched."""
    base = np.asarray(pil_img.convert("RGB"), dtype=np.float32)
    h, w = base.shape[:2]
    cam_r = np.clip(cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC), 0.0, 1.0)
    heat = cv2.applyColorMap((cam_r * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32)
    alpha = (0.65 * cam_r)[..., None]
    out = base * (1.0 - alpha) + heat * alpha
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

# 6d. Button callbacks
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "gradcam_sig" not in st.session_state:
    st.session_state.gradcam_sig = None

def toggle_gradcam(sig):
    """Show / hide Grad-CAM for the current image + modality."""
    st.session_state.gradcam_sig = None if st.session_state.gradcam_sig == sig else sig

def reset_upload():
    """Re-upload: clear the current image and any Grad-CAM output."""
    st.session_state.uploader_key += 1
    st.session_state.gradcam_sig = None

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    # Step 1: imaging modality
    with st.container(key="modality_panel"):
        st.markdown(
            card_head("Imaging Modality", "Select the scan type to load the matching model"),
            unsafe_allow_html=True,
        )
        modality = st.radio(
            "Imaging modality",
            ["Lung CT Scan", "Chest X-Ray"],
            horizontal=True,
            key="modality",
            label_visibility="collapsed",
        )

    # Step 2: upload panel
    with st.container(key="upload_panel"):
        st.markdown(
            card_head("Upload an Image", "Supported formats: PNG, JPG, JPEG"),
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Upload DICOM, PNG, or JPG format",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
            key=f"uploader_{st.session_state.uploader_key}",
        )

        gradcam_on = False
        if uploaded_file is not None:
            original_image = Image.open(uploaded_file).convert('RGB')

            # Preview of the original upload
            st.markdown(build_preview_card(original_image, modality), unsafe_allow_html=True)

            sig = f"{uploaded_file.name}|{uploaded_file.size}|{modality}"
            gradcam_on = st.session_state.gradcam_sig == sig

            btn_cam, btn_reupload = st.columns(2)
            with btn_cam:
                st.button(
                    "Hide Grad-CAM" if gradcam_on else "Apply Grad-CAM",
                    key="gradcam",
                    on_click=toggle_gradcam,
                    args=(sig,),
                    icon=":material/visibility:",
                )
            with btn_reupload:
                st.button("Re-upload", key="reupload", on_click=reset_upload, icon=":material/upload:")

with right_col:
    if uploaded_file is not None:
        # Apply conditional preprocessing backend for the model tensor only
        if modality == "Lung CT Scan":
            processed_image = apply_target_contrast_enhancement(original_image)
        else:
            processed_image = original_image

        with st.spinner("Analyzing image..."):
            # Run inference
            model, device = load_model(modality)
            input_tensor = test_transform(processed_image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)[0].cpu().numpy()

            # Define class mappings
            if modality == "Lung CT Scan":
                classes = ["Benign", "Malignant", "Normal"]
            else:
                classes = ["COVID-19", "Pneumonia", "Normal"]

        # Step 3: diagnosis report (now on the right side)
        st.markdown(build_result_card(classes, probs, modality), unsafe_allow_html=True)

        # Step 4: Grad-CAM explanation, shown below the report when requested
        if gradcam_on:
            with st.spinner("Generating Grad-CAM..."):
                try:
                    cam, layer_name, cls_idx = compute_gradcam(model, input_tensor, GRADCAM_TARGET_LAYER)
                    display_img = original_image.copy()
                    display_img.thumbnail((900, 900))
                    overlay = overlay_heatmap(display_img, cam)
                    st.markdown(
                        build_gradcam_card(display_img, overlay, classes[cls_idx], layer_name),
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.error(f"Grad-CAM could not be generated: {e}")

        # Kept outside the cards so they stay clean for figures/screenshots
        st.caption("Clinical evaluation by a licensed radiologist is warranted to confirm this automated diagnosis.")