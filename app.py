import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps
import numpy as np
import cv2
import gdown
from pathlib import Path

# ------------------------------------------------------------
# EMPATHICA — facial-expression demo + supportive rule-based text
# Put EMPATHICA_resnet18.pth in the same folder as this file.
# ------------------------------------------------------------

st.set_page_config(
    page_title="EMPATHICA | Emotional Wellness",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = Path(__file__).parent / "EMPATHICA_resnet18.pth"
IMAGE_SIZE = 224
DEVICE = torch.device("cpu")  # Reliable for Streamlit Community Cloud

EMOTION_RESPONSES = {
    "happy": {
        "icon": "☀️",
        "title": "A little space for the good moments",
        "text": """Your uploaded image was classified as **happy** by EMPATHICA. If you are feeling positive, take a moment to appreciate what is going well in your day. It might be a small achievement, a kind conversation, a favourite activity, or simply a moment that feels good.

You can make room for positive moments by connecting with people you trust, spending time on activities you enjoy, or recognising something you have worked hard on. Small wins count, too.

**A gentle reflection**
- What is one thing that went well today?
- What activity helps me feel refreshed or positive?
- Is there something good I would like to share with someone?

A smile does not always mean happiness. This result is a model's estimate of a visible facial expression, not a confirmation of how you feel. Your own understanding of your feelings matters most."""
    },
    "sad": {
        "icon": "💙",
        "title": "A gentle check-in with yourself",
        "text": """Your uploaded image was classified as **sad** by EMPATHICA. If you are feeling low, disappointed, or overwhelmed, try giving yourself permission to pause. You do not have to solve everything at once, and it is okay to take things one step at a time.

You might listen to music you enjoy, write down what is on your mind, take a comfortable break, or speak with someone you trust. If a problem feels big, consider whether there is one small and manageable thing you can do next.

**A few questions to consider**
- Is something bothering me today, or am I feeling something else?
- Would I prefer quiet time or a conversation with someone?
- What is one small thing that might help me feel supported?

You deserve patience and understanding. Facial expressions can be misleading, so rely on your own experience rather than this prediction."""
    },
    "angry": {
        "icon": "🌿",
        "title": "Pause, breathe, and reflect",
        "text": """Your uploaded image was classified as **angry** by EMPATHICA. The model cannot know whether you are angry or what might have caused an expression. If you are feeling frustrated, a short pause can give you time to understand the situation before responding.

You could take a few comfortable breaths, name the issue that is bothering you, and decide what you want to communicate. If a conversation is difficult, it can help to focus on the specific concern and explain it respectfully.

**Consider asking yourself**
- What, if anything, is bothering me right now?
- Is there a misunderstanding I could clarify?
- What response could help move the situation forward constructively?

Frustration is a normal human experience. You can acknowledge a feeling without having to act on it immediately. If this prediction does not fit, it is okay to disregard it."""
    },
    "fear": {
        "icon": "🤍",
        "title": "Take a moment to feel grounded",
        "text": """Your uploaded image was classified as **fear** by EMPATHICA. An image cannot confirm that you are afraid; lighting, concentration, and individual facial expressions can affect the result.

If you are feeling worried or uncertain, you might pause and notice your surroundings. Try identifying what is concerning you, what is within your control, and one manageable next step. Talking with a trusted person may also help you get another perspective.

**A gentle check-in**
- Am I feeling worried, uncertain, or something different?
- What would help me feel more comfortable right now?
- Is there someone I trust whom I could talk to?

Your circumstances and your own feelings matter more than an automated prediction. You do not need to do anything simply because the model selected this category."""
    },
    "disgust": {
        "icon": "🌱",
        "title": "Notice your reaction without judging it",
        "text": """Your uploaded image was classified as **disgust** by EMPATHICA. This label may be associated with a particular facial expression, but the model cannot know what caused it or whether it reflects your actual feelings.

If something is making you uncomfortable, give yourself a moment to notice your reaction. You may want a little space, a change of environment, or time to think about what happened. There is no need to judge yourself for having a reaction.

**You could reflect on**
- Is there something I dislike or find uncomfortable?
- Is there context the model cannot see?
- What, if anything, would make this situation more comfortable?

Expressions vary from person to person. If this result does not match your experience, trust your own interpretation."""
    },
    "surprise": {
        "icon": "✨",
        "title": "Make room for the unexpected",
        "text": """Your uploaded image was classified as **surprise** by EMPATHICA. Surprise can appear when something catches our attention, but a facial expression alone cannot tell whether an experience is pleasant, confusing, or uncomfortable.

If something unexpected has happened, give yourself time to understand it. You can ask questions, gather more information, or wait until you have a clearer picture before deciding how you feel about it.

**Try reflecting on these questions**
- What caught my attention?
- Did the moment feel pleasant, confusing, or uncomfortable?
- Do I need more information before reacting?

Your thoughts, situation, and personal experience provide context that an image cannot. This prediction is only a starting point for reflection."""
    },
    "neutral": {
        "icon": "🌼",
        "title": "A calm, no-pressure check-in",
        "text": """Your uploaded image was classified as **neutral** by EMPATHICA. A neutral facial expression does not mean that you have no emotions. You might be concentrating, thinking, tired, relaxed, or simply resting your face.

Take a moment to check in with yourself without feeling pressure to experience a particular emotion. If you have been working for a while, you might consider whether a short break would feel helpful. Otherwise, it is perfectly fine to carry on with your day.

**Ask yourself**
- How am I actually feeling right now?
- Is anything on my mind?
- What would help me with the rest of my day?

There is no right or wrong expression to have. Your own description of your feelings is more meaningful than a facial-expression prediction."""
    },
}

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
PREPROCESS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

@st.cache_resource(show_spinner="Loading EMPATHICA model…")
def load_model():
    # Download the trained model from Google Drive when it is not in the repo.
    if not MODEL_PATH.exists():
        try:
            gdown.download(
                id="1N28LwypLiqB_PwtsC1AwykfwVcAKmw48",
                output=str(MODEL_PATH),
                quiet=False,
            )
        except Exception as exc:
            return None, None, f"Could not download the model from Google Drive: {exc}"

    if not MODEL_PATH.exists():
        return None, None, (
            "Model download failed. In Google Drive, set this file's General access "
            "to 'Anyone with the link' and try again."
        )

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        classes = checkpoint.get("classes")
        emotions = checkpoint.get("emotions")

        if not classes or not emotions:
            return None, None, (
                "The checkpoint does not contain the expected 'classes' and "
                "'emotions' lists. Use the checkpoint saved by your EMPATHICA training code."
            )

        model = models.resnet18(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, len(emotions))
        )
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        return model, list(emotions), None
    except Exception as exc:
        return None, None, (
            "Could not load the model checkpoint. Make sure it is the "
            "EMPATHICA_resnet18.pth file produced by your training notebook. "
            f"Technical detail: {exc}"
        )

def crop_largest_face(pil_image):
    """Try to crop the largest face; safely use the full image if detection is unavailable."""
    rgb = np.array(pil_image.convert("RGB"))
    try:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(cascade_path)
        faces = detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
    except (AttributeError, cv2.error, Exception):
        # Some hosted environments have a broken/missing OpenCV face detector.
        # Do not crash the app; classify the uploaded image instead.
        return pil_image.convert("RGB"), False

    if len(faces) == 0:
        return pil_image.convert("RGB"), False

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    px, py = int(w * 0.10), int(h * 0.10)
    x1, y1 = max(0, x - px), max(0, y - py)
    x2, y2 = min(rgb.shape[1], x + w + px), min(rgb.shape[0], y + h + py)
    return Image.fromarray(rgb[y1:y2, x1:x2]), True

def predict(pil_image, model, emotions):
    face_image, face_found = crop_largest_face(pil_image)
    tensor = PREPROCESS(face_image).unsqueeze(0)
    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

    best_index = int(np.argmax(probabilities))
    ranking = np.argsort(probabilities)[::-1][:3]
    return {
        "emotion": str(emotions[best_index]).lower().strip(),
        "score": float(probabilities[best_index]),
        "top": [(str(emotions[i]).lower().strip(), float(probabilities[i])) for i in ranking],
        "face_image": face_image,
        "face_found": face_found,
    }

# --------------------------- STYLE ---------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');
:root { --ink:#26352f; --muted:#697a70; --sage:#dfe9df; --cream:#faf8f2; --green:#4f6f5b; }
.stApp { background: radial-gradient(circle at 8% 0%, #edf3e9 0, transparent 32%), linear-gradient(145deg,#faf8f2,#f4f5ee 60%,#eef3ed); color:var(--ink); font-family:'DM Sans',sans-serif; }
.block-container { max-width:1120px; padding-top:2.2rem; padding-bottom:3rem; }
.hero { padding:2.1rem 2.2rem; border:1px solid rgba(94,122,99,.2); border-radius:28px; background:rgba(255,255,255,.68); box-shadow:0 18px 50px rgba(54,76,59,.07); }
.eyebrow { color:#66806b; letter-spacing:.19em; font-size:.72rem; font-weight:700; text-transform:uppercase; }
.hero h1 { font-family:'Playfair Display',serif; font-size:clamp(2.5rem,6vw,4.4rem); line-height:1.05; color:#2d4437; margin:.45rem 0 .7rem; }
.hero p { color:#64756a; font-size:1.03rem; max-width:690px; line-height:1.8; }
.pill { display:inline-block; background:#e6eee4; color:#45634d; padding:.42rem .8rem; border-radius:999px; font-size:.78rem; font-weight:600; margin-right:.35rem; }
.section-title { font-family:'Playfair Display',serif; color:#2d4437; font-size:1.7rem; }
.soft-card { background:rgba(255,255,255,.78); border:1px solid #e0e7dc; border-radius:22px; padding:1.25rem 1.4rem; box-shadow:0 10px 30px rgba(49,72,53,.045); }
.result-label { color:#708074; font-size:.78rem; letter-spacing:.12em; text-transform:uppercase; font-weight:700; }
.result-emotion { font-family:'Playfair Display',serif; font-size:2.45rem; color:#2d4437; text-transform:capitalize; margin:.25rem 0; }
.small-muted { color:#738177; font-size:.88rem; line-height:1.65; }
.stButton > button { border-radius:999px; border:0; background:#476650; color:white; padding:.72rem 1.45rem; font-weight:700; min-height:3rem; }
.stButton > button:hover { background:#35523e; color:white; border:0; }
div[data-testid="stFileUploader"] { background:rgba(255,255,255,.65); border:1px dashed #9db19e; border-radius:18px; padding:1rem; }
hr { border-color:#e1e8df; }
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="eyebrow">✦ Emotional wellness · A thoughtful check-in</div>
  <h1>Meet yourself<br>where you are.</h1>
  <p>EMPATHICA is an emotion-aware demo that estimates a facial expression from an uploaded image and offers a gentle, rule-based reflection. Take what feels useful, leave what does not.</p>
  <span class="pill">🌿 Thoughtful design</span>
  <span class="pill">🔒 Image processed in this app session</span>
  <span class="pill">✨ Seven expression labels</span>
</div>
""", unsafe_allow_html=True)

st.write("")
left, right = st.columns([1, 0.88], gap="large")

with left:
    st.markdown('<div class="section-title">Your check-in</div>', unsafe_allow_html=True)
    st.markdown('<p class="small-muted">Upload a clear JPG, JPEG, or PNG image with one visible face for the best chance of a useful prediction.</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if uploaded:
        try:
            original = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
            st.image(original, caption="Your uploaded image", use_container_width=True)
            run_prediction = st.button("✨ Explore expression", use_container_width=True)
        except Exception:
            st.error("This image could not be opened. Please try a JPG or PNG file.")
            run_prediction = False
    else:
        original = None
        run_prediction = False
        st.markdown("""
        <div class="soft-card">
          <div style="font-size:2rem">📷</div>
          <b>A little note before you begin</b>
          <p class="small-muted">Use an image you have permission to share. A front-facing, well-lit image generally works better than a distant or heavily angled photo.</p>
        </div>
        """, unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-title">A mindful space</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="soft-card">
      <div style="font-size:1.8rem">🫧</div>
      <h3 style="font-family:'Playfair Display',serif;color:#2d4437;margin:.5rem 0">No labels define you.</h3>
      <p class="small-muted">Facial expressions are complex. Lighting, pose, culture, accessibility, and personal differences can affect a model's output. EMPATHICA cannot read thoughts or confirm a person's emotional state.</p>
      <hr>
      <p class="small-muted"><b>Use this as reflection, not diagnosis.</b> The result is an estimate from an image and may be incorrect. Your own account of how you feel takes priority.</p>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    st.markdown("""
    <div class="soft-card">
      <div style="font-size:1.6rem">🌱</div>
      <b>What happens next?</b>
      <p class="small-muted">The app looks for a face, runs your trained ResNet18 model, shows its leading expression estimate, then displays a matching supportive message.</p>
    </div>
    """, unsafe_allow_html=True)

if run_prediction and original is not None:
    model, emotions, error = load_model()
    if error:
        st.error(error)
        st.info("Check the repository files and confirm that the checkpoint was uploaded next to app.py.")
    else:
        with st.spinner("Taking a moment to analyse the image…"):
            result = predict(original, model, emotions)

        st.write("")
        st.markdown('<div class="section-title">Your EMPATHICA reflection</div>', unsafe_allow_html=True)
        result_left, result_right = st.columns([0.8, 1.2], gap="large")

        with result_left:
            st.image(result["face_image"], caption="Image area used for prediction", use_container_width=True)
            st.markdown('<div class="soft-card">', unsafe_allow_html=True)
            st.markdown('<div class="result-label">Predicted facial expression</div>', unsafe_allow_html=True)
            response = EMOTION_RESPONSES.get(result["emotion"])
            icon = response["icon"] if response else "🌿"
            st.markdown(f'<div class="result-emotion">{icon} {result["emotion"]}</div>', unsafe_allow_html=True)
            st.metric("Model score", f'{result["score"] * 100:.1f}%')
            st.markdown('</div>', unsafe_allow_html=True)
            if result["face_found"]:
                st.caption("A face was detected and cropped before prediction.")
            else:
                st.warning("No face was detected, so the model used the full image. This can make the prediction less reliable.")

        with result_right:
            if response:
                st.markdown(f'<div class="soft-card"><div class="eyebrow">{response["icon"]} A moment for you</div><h2 style="font-family:Playfair Display,serif;color:#2d4437">{response["title"]}</h2></div>', unsafe_allow_html=True)
                st.write("")
                st.markdown(response["text"])
            else:
                st.info("No reflection message is configured for this model label yet.")

            st.markdown("#### Other model predictions")
            for emotion, score in result["top"]:
                st.write(f"**{emotion.title()}** · {score * 100:.1f}%")
                st.progress(float(score))

        st.caption("Important: these percentages are model scores, not a measure of emotional intensity or proof of how someone feels.")

st.write("")
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#829087;font-size:.82rem">EMPATHICA · A student project for emotion-aware interaction<br>Be kind to yourself. An expression is only one small part of a person.</p>',
    unsafe_allow_html=True
)

