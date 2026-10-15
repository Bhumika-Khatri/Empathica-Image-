import json
from pathlib import Path

import cv2
import gdown
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms

# ============================================================
# EMPATHICA — Streamlit app using the SAME model/preprocessing
# as the user's Google Colab training/prediction code.
# ============================================================
st.set_page_config(page_title="EMPATHICA | Emotional Wellness", page_icon="🌿", layout="wide")

MODEL_PATH = Path(__file__).parent / "EMPATHICA_resnet18.pth"
DRIVE_FILE_ID = "1N28LwypLiqB_PwtsC1AwykfwVcAKmw48"
IMAGE_SIZE = 224
DEVICE = torch.device("cpu")
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# This is exactly the evaluation transform used in the training notebook.
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

RESPONSES = {
    "happy": ("☀️", "A little space for the good moments", "If you are feeling positive, take a moment to appreciate what is going well. It might be a small achievement, a kind conversation, a favourite activity, or a moment of calm. You can make room for positive moments by connecting with people you trust and recognising small wins.\n\n**A gentle reflection**\n- What is one thing that went well today?\n- What activity helps me feel refreshed?\n- Is there something good I would like to share?\n\nA smile does not always mean happiness. This is only a model estimate of a facial expression, not a confirmation of how you feel."),
    "sad": ("💙", "A gentle check-in with yourself", "If you are feeling low, disappointed, or overwhelmed, give yourself permission to pause. You do not have to solve everything at once. You might listen to music you enjoy, write down your thoughts, take a comfortable break, or speak with someone you trust.\n\n**A few questions to consider**\n- Is something bothering me today, or am I feeling something else?\n- Would I prefer quiet time or a conversation?\n- What is one small thing that might help me feel supported?\n\nYou deserve patience and understanding. Facial expressions can be misleading, so rely on your own experience rather than this prediction."),
    "angry": ("🌿", "Pause, breathe, and reflect", "The model cannot know whether you are angry or what caused an expression. If you are feeling frustrated, a short pause can give you time to understand the situation before responding. You could take a few comfortable breaths, name the issue, and decide what you want to communicate respectfully.\n\n**Consider asking yourself**\n- What, if anything, is bothering me?\n- Is there a misunderstanding I could clarify?\n- What response could help move the situation forward?\n\nFrustration is a normal human experience. If this prediction does not fit, it is okay to disregard it."),
    "fear": ("🤍", "Take a moment to feel grounded", "An image cannot confirm that you are afraid; lighting, concentration, and individual expressions can affect the result. If you are feeling worried or uncertain, identify what is concerning you, what is within your control, and one manageable next step. Talking with a trusted person may also help.\n\n**A gentle check-in**\n- Am I feeling worried, uncertain, or something different?\n- What would help me feel more comfortable?\n- Is there someone I trust whom I could talk to?\n\nYour own feelings and circumstances matter more than an automated prediction."),
    "disgust": ("🌱", "Notice your reaction without judging it", "This label describes a possible facial expression; the model cannot know what caused it or whether it reflects your actual feelings. If something is making you uncomfortable, give yourself a moment to notice your reaction. You may want a little space, a change of environment, or time to think.\n\n**You could reflect on**\n- Is there something I dislike or find uncomfortable?\n- Is there context the model cannot see?\n- What, if anything, would make this situation more comfortable?\n\nIf this result does not match your experience, trust your own interpretation."),
    "surprise": ("✨", "Make room for the unexpected", "Surprise can appear when something catches our attention, but a facial expression alone cannot tell whether an experience is pleasant, confusing, or uncomfortable. If something unexpected has happened, give yourself time to understand it. You can ask questions, gather information, or wait until you have a clearer picture before deciding how you feel.\n\n**Try reflecting**\n- What caught my attention?\n- Did the moment feel pleasant, confusing, or uncomfortable?\n- Do I need more information before reacting?\n\nYour thoughts and personal experience provide context that an image cannot. This prediction is only a starting point for reflection."),
    "neutral": ("🌼", "A calm, no-pressure check-in", "A neutral facial expression does not mean you have no emotions. You might be concentrating, thinking, tired, relaxed, or simply resting your face. Take a moment to check in with yourself without pressure to feel a particular way.\n\n**Ask yourself**\n- How am I actually feeling right now?\n- Is anything on my mind?\n- What would help me with the rest of my day?\n\nThere is no right or wrong expression to have. Your own description of your feelings is more meaningful than a facial-expression prediction."),
}

@st.cache_resource(show_spinner="Loading EMPATHICA model…")
def load_model():
    # Download model from the shared Google Drive file only if not already present.
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
        try:
            downloaded = gdown.download(id=DRIVE_FILE_ID, output=str(MODEL_PATH), quiet=True, fuzzy=True)
            if not downloaded or not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
                return None, None, "Could not download the model. In Google Drive, set General access to 'Anyone with the link'."
        except Exception as exc:
            return None, None, f"Model download failed: {exc}"

    try:
        checkpoint = torch.load(str(MODEL_PATH), map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
            return None, None, "This is not the expected EMPATHICA checkpoint. Please use EMPATHICA_resnet18.pth from your Colab training code."
        emotions = checkpoint.get("emotions")
        classes = checkpoint.get("classes")
        if not emotions or not classes or len(emotions) != len(classes):
            return None, None, "The checkpoint is missing its class/emotion labels. Please use the original saved checkpoint."

        model = models.resnet18(weights=None)
        model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, len(emotions)))
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.to(DEVICE)
        model.eval()
        return model, [str(x).lower().strip() for x in emotions], None
    except Exception as exc:
        return None, None, f"Could not load the trained model: {exc}"


def detect_and_crop_face(pil_image):
    """Match the Colab Haar-cascade crop. Never silently classify the whole photo."""
    rgb = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cascade_path = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError("OpenCV's face-detection file is missing. Check opencv-python-headless in requirements.txt.")
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return None, False
    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
    px, py = int(w * 0.10), int(h * 0.10)
    x1, y1 = max(0, x - px), max(0, y - py)
    x2, y2 = min(rgb.shape[1], x + w + px), min(rgb.shape[0], y + h + py)
    return Image.fromarray(rgb[y1:y2, x1:x2]), True


def predict_face(face_image, model, emotions):
    # Exact evaluation transform and softmax method from the Colab notebook.
    tensor = EVAL_TRANSFORM(face_image).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
    if len(probabilities) != len(emotions):
        raise RuntimeError("Model output count does not match the saved emotion labels.")
    order = np.argsort(probabilities)[::-1][:3]
    best = int(order[0])
    return {
        "emotion": emotions[best],
        "score": float(probabilities[best]),
        "top": [(emotions[int(i)], float(probabilities[int(i)])) for i in order],
    }

# ---------------- UI ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');
.stApp{background:linear-gradient(145deg,#faf8f2,#f0f4ed);color:#26352f;font-family:'DM Sans',sans-serif}
.block-container{max-width:1100px;padding-top:2rem;padding-bottom:3rem}
.hero{padding:2rem;border:1px solid #dce6d9;border-radius:26px;background:#ffffffc9;box-shadow:0 15px 40px #304a3610}
.eyebrow{color:#66806b;letter-spacing:.16em;font-size:.75rem;font-weight:700;text-transform:uppercase}
.hero h1{font-family:'Playfair Display',serif;font-size:clamp(2.4rem,6vw,4rem);color:#2d4437;margin:.4rem 0}
.muted{color:#68796e;line-height:1.7}
.card{padding:1.2rem 1.4rem;border:1px solid #dfe7dc;border-radius:20px;background:#ffffffc9}
.stButton>button{background:#476650;color:white;border-radius:999px;border:0;min-height:2.8rem;font-weight:700}
</style>
<div class="hero"><div class="eyebrow">🌿 Emotional wellness · A thoughtful check-in</div>
<h1>Meet yourself where you are.</h1>
<p class="muted">EMPATHICA estimates a facial expression from a clear, front-facing image and offers a supportive reflection. It cannot confirm how a person feels.</p></div>
""", unsafe_allow_html=True)

left, right = st.columns([1, 1], gap="large")
with left:
    st.subheader("Your check-in")
    st.write("Upload a clear, front-facing JPG or PNG containing one visible face.")
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    original = None
    if uploaded is not None:
        try:
            original = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
            st.image(original, caption="Uploaded image", use_container_width=True)
        except Exception:
            st.error("This image could not be opened. Please try a JPG or PNG file.")
            original = None
    run = st.button("✨ Detect expression", disabled=original is None, use_container_width=True)
with right:
    st.subheader("A mindful space")
    st.markdown("""<div class="card"><h3 style="font-family:Playfair Display,serif;color:#2d4437">No label defines you.</h3>
    <p class="muted">Lighting, pose, and individual differences affect predictions. This tool is a student project for reflection, not a diagnosis or a reliable way to know someone's feelings.</p></div>""", unsafe_allow_html=True)

if run and original is not None:
    model, emotions, error = load_model()
    if error:
        st.error(error)
    else:
        try:
            with st.spinner("Checking the face and running the model…"):
                face, found = detect_and_crop_face(original)
                if not found or face is None:
                    st.warning("No face was detected. To avoid an unreliable prediction, this app did not classify the full image. Upload a clear, front-facing photo with the face large and well lit.")
                    st.stop()
                result = predict_face(face, model, emotions)
            st.divider()
            st.subheader("Your EMPATHICA reflection")
            c1, c2 = st.columns([0.85, 1.15], gap="large")
            with c1:
                st.image(face, caption="Face crop used for prediction", use_container_width=True)
                st.markdown(f"### {result['emotion'].title()}")
                st.metric("Model score", f"{result['score'] * 100:.1f}%")
                st.caption("Model scores are not measures of emotional intensity.")
                st.markdown("**Top three model outputs**")
                for label, score in result["top"]:
                    st.write(f"{label.title()} — {score * 100:.1f}%")
                    st.progress(score)
            with c2:
                info = RESPONSES.get(result["emotion"])
                if info:
                    icon, title, message = info
                    st.markdown(f"### {icon} {title}")
                    st.markdown(message)
                else:
                    st.info("No supportive message is configured for this label: " + result["emotion"])
        except Exception as exc:
            st.error(f"Prediction could not be completed: {exc}")

st.divider()
st.caption("EMPATHICA · Student project · Facial-expression estimates can be incorrect; your own account of your feelings takes priority.")


