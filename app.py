import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
import gdown
from pathlib import Path
from PIL import Image, ImageOps
from torchvision import models, transforms

# -------------------- APP SETTINGS --------------------
st.set_page_config(
    page_title="EMPATHICA | Emotion Support",
    page_icon="🌿",
    layout="wide"
)

MODEL_PATH = Path(__file__).parent / "EMPATHICA_resnet18.pth"
GOOGLE_DRIVE_FILE_ID = "1N28LwypLiqB_PwtsC1AwykfwVcAKmw48"
DEVICE = torch.device("cpu")

# Supportive, rule-based responses. These are prompts for reflection,
# not a diagnosis or a claim about what someone is truly feeling.
RESPONSES = {
    "happy": """
### ☀️ A Positive Moment
Take a moment to notice what is going well. Happiness can come from a small achievement, a good conversation, a favourite activity, or a little time to yourself.

**A gentle check-in:**
- What is one thing I appreciated today?
- Is there a positive moment I would like to remember or share?
- What helps me feel balanced and refreshed?

You do not have to feel happy all the time. Allow yourself to enjoy good moments without putting pressure on yourself to keep feeling the same way. Remember, this is only an estimate from an image; it may not match your actual feelings.
""",
    "sad": """
### 💙 Be Gentle With Yourself
If you are having a difficult day, you do not need to solve everything at once. Taking a small pause and reaching out to someone you trust may help.

**Things you could try:**
- Write down what is on your mind.
- Listen to music or do a familiar activity you enjoy.
- Take a comfortable break and return to one small task when ready.
- Talk with a trusted friend, family member, or counsellor if you want support.

**Ask yourself:** What would feel helpful right now: quiet time, practical help, or a conversation? Your own understanding of your feelings matters more than this model's prediction.
""",
    "angry": """
### 🌿 Pause and Reflect
Frustration can happen when something feels unfair, stressful, or different from what you expected. A short pause may give you space to decide how you want to respond.

**You could try:**
- Taking a few slow, comfortable breaths.
- Stepping away briefly from a stressful conversation if it is safe to do so.
- Writing down what is bothering you before discussing it.
- Explaining your concern calmly when you feel ready.

**Reflect:** What is within my control? Is there something I need to communicate or clarify? An image cannot explain the reason for an expression, and this prediction may be wrong.
""",
    "fear": """
### 🤍 A Grounding Check-In
Uncertainty can sometimes feel uncomfortable. If something is worrying you, it may help to separate what you know from what you are imagining might happen.

**You could try:**
- Naming the specific thing that concerns you.
- Focusing on one small next step you can control.
- Asking someone you trust for another perspective.
- Taking a break from information or activities that are making you feel overwhelmed.

**Ask yourself:** What information would help me understand the situation better? What might help me feel supported? This model cannot confirm that you are afraid or worried; only you can describe how you feel.
""",
    "disgust": """
### 🌱 Notice Your Reaction
Sometimes a situation, smell, memory, or experience can make us uncomfortable. You can notice a reaction without judging yourself for it.

**A few reflection questions:**
- Is something making me uncomfortable, or could my expression mean something else?
- Would a short break or a change of surroundings help?
- Is there more context I need before deciding how I feel about the situation?

Give yourself time to understand your reaction. A facial expression alone cannot tell the app what caused it or what you are experiencing.
""",
    "surprise": """
### ✨ Make Space for the Unexpected
Something unexpected can catch our attention. Surprise is not necessarily good or bad; sometimes we simply need a little time to understand what happened.

**You could reflect on:**
- What caught my attention?
- Do I feel pleased, confused, uncertain, or something else?
- Do I need more information before responding?
- What is one calm next step I can take?

There is no need to decide immediately what an unexpected moment means. This is only a model estimate based on an image, not a reliable reading of your inner feelings.
""",
    "neutral": """
### 🌼 A No-Pressure Check-In
A neutral facial expression does not mean that you have no emotions. You might be concentrating, resting, thinking, or simply expressing yourself naturally.

**Take a moment to check in:**
- How would I describe my mood in my own words?
- Is anything on my mind that needs attention?
- Would a break, some movement, or a conversation be helpful?
- What is one manageable thing I want to focus on next?

You do not need to display a particular expression to feel a certain way. Your own account of your experience is more meaningful than an automated prediction.
"""
}

# -------------------- MODEL LOADING --------------------
@st.cache_resource(show_spinner=False)
def load_model():
    # Download the checkpoint only if it is not already beside app.py.
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
        result = gdown.download(
            id=GOOGLE_DRIVE_FILE_ID,
            output=str(MODEL_PATH),
            quiet=False
        )
        if result is None or not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
            raise RuntimeError(
                "Model download failed. Open Google Drive sharing settings and set "
                "General access to 'Anyone with the link', then redeploy."
            )

    try:
        checkpoint = torch.load(
            str(MODEL_PATH),
            map_location=DEVICE,
            weights_only=False
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        checkpoint = torch.load(str(MODEL_PATH), map_location=DEVICE)

    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise RuntimeError(
            "This file is not the expected checkpoint. It must contain 'model_state'."
        )

    # Use the saved label order so prediction indexes match training.
    emotion_labels = checkpoint.get("emotions") or checkpoint.get("classes")
    if not emotion_labels:
        raise RuntimeError("No 'emotions' or 'classes' labels were found in the checkpoint.")

    emotion_labels = [str(label).strip().lower() for label in emotion_labels]

    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, len(emotion_labels))
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(DEVICE)
    model.eval()
    return model, emotion_labels

# -------------------- IMAGE PREPARATION --------------------
def prepare_face_or_image(pil_image):
    """Try to crop a face. If detection fails, return the original image instead of stopping."""
    rgb = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cascade_file = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_file))

    if not detector.empty():
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30)
        )
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
            pad_x = int(w * 0.10)
            pad_y = int(h * 0.10)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(rgb.shape[1], x + w + pad_x)
            y2 = min(rgb.shape[0], y + h + pad_y)
            crop = rgb[y1:y2, x1:x2]
            if crop.size > 0:
                return Image.fromarray(crop), True

    # Fallback: always let the classifier run on the uploaded image.
    return Image.fromarray(rgb), False

# -------------------- PREDICTION --------------------
def predict_emotion(image, model, emotion_labels):
    # Match the ImageNet normalization used in the training/evaluation code.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
    best_index = int(np.argmax(probabilities))
    return emotion_labels[best_index], float(probabilities[best_index]), probabilities

# -------------------- PAGE DESIGN --------------------
st.markdown("""
<style>
.stApp { background: #f5f7f1; color: #26352f; }
.block-container { max-width: 1050px; padding-top: 2rem; padding-bottom: 3rem; }
h1, h2, h3 { color: #365442; }
.stButton > button { background: #476650; color: white; border-radius: 10px; border: 0; }
</style>
""", unsafe_allow_html=True)

st.title("🌿 EMPATHICA")
st.subheader("Emotion-Aware Emotional Wellness")
st.write(
    "Upload an image to estimate a facial-expression category and receive a supportive, "
    "rule-based reflection. Facial expressions do not reliably reveal a person's actual emotions."
)
st.info("Tip: a clear, well-lit image with your face visible usually works best. If face detection fails, the app will still try the uploaded image.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        original_image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
        st.image(original_image, caption="Uploaded image", use_container_width=True)

        if st.button("✨ Detect Expression", type="primary"):
            with st.spinner("Loading your model and analysing the image..."):
                model, emotion_labels = load_model()
                image_for_prediction, face_found = prepare_face_or_image(original_image)
                emotion, confidence, probabilities = predict_emotion(
                    image_for_prediction, model, emotion_labels
                )

            if face_found:
                st.success("Face detected. Prediction completed.")
            else:
                st.warning(
                    "A face was not detected automatically, so the full uploaded image was used. "
                    "This may affect accuracy."
                )

            st.divider()
            st.header("Your EMPATHICA Result")
            left, right = st.columns(2)
            with left:
                st.image(
                    image_for_prediction,
                    caption="Image used for prediction",
                    use_container_width=True
                )
            with right:
                st.subheader(f"Estimated expression: {emotion.title()}")
                st.metric("Model confidence score", f"{confidence * 100:.2f}%")
                st.caption("Confidence is the model's score, not emotional intensity or certainty about how you feel.")

            response = RESPONSES.get(
                emotion,
                "### 🌼 Check In With Yourself\nTake a moment to notice how you feel in your own words. This model's estimate may be incorrect."
            )
            st.markdown(response)

            with st.expander("See the top 3 model predictions"):
                top_indices = np.argsort(probabilities)[::-1][:3]
                for idx in top_indices:
                    label = emotion_labels[int(idx)]
                    value = float(probabilities[int(idx)])
                    st.write(f"**{label.title()} — {value * 100:.2f}%**")
                    st.progress(value)

    except Exception as error:
        st.error(f"The app could not complete the prediction: {error}")
        st.caption("Check that the Google Drive model is shared with anyone who has the link and that the checkpoint matches the ResNet18 architecture used during training.")

st.divider()
st.caption("EMPATHICA is a student project for reflection and learning. It is not a mental-health diagnosis or a substitute for professional support.")
