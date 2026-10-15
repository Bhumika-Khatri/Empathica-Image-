import cv2
import gdown
import numpy as np
import streamlit as st
import torch
import torch.nn as nn

from pathlib import Path
from PIL import Image, ImageOps
from torchvision import models, transforms

# ---------------- CONFIGURATION ----------------

st.set_page_config(
    page_title="EMPATHICA",
    page_icon="🌿",
    layout="wide"
)

MODEL_PATH = Path(__file__).parent / "EMPATHICA_resnet18.pth"
DRIVE_FILE_ID = "1N28LwypLiqB_PwtsC1AwykfwVcAKmw48"

DEVICE = torch.device("cpu")

EMOTION_RESPONSES = {
    "happy": """
### ☀️ A Little Space for Positive Moments

Take a moment to appreciate the things going well in your life.
Happiness can come from small achievements, meaningful conversations,
personal progress, or simply having a peaceful moment.

**Questions to reflect on:**
- What is one good thing that happened today?
- What activity helps me feel refreshed?
- Is there someone I would like to share a positive moment with?

Allow yourself to enjoy positive experiences without feeling that
every day has to be perfect. Remember, a facial expression cannot
confirm how you actually feel.
""",

    "sad": """
### 💙 Be Gentle With Yourself

Everyone experiences difficult or disappointing moments.
You do not need to figure everything out at once. Give yourself
permission to pause and understand what you need.

**Things you could try:**
- Talk to someone you trust.
- Listen to music you enjoy.
- Write down your thoughts.
- Take a comfortable break.
- Focus on one manageable task at a time.

**Ask yourself:**
- Is something bothering me today?
- Would I prefer some quiet time or a conversation?
- What might help me feel supported?

Your feelings deserve patience and understanding. This prediction
may be incorrect, so trust your own experience.
""",

    "angry": """
### 🌿 Pause, Breathe, and Reflect

Sometimes situations can feel frustrating or overwhelming.
Taking a short pause can help you understand what is happening
before deciding how to respond.

**You could try:**
- Taking a few slow, comfortable breaths.
- Giving yourself time before responding to a difficult situation.
- Writing down what is bothering you.
- Explaining your concerns calmly when you feel ready.

**Reflect on these questions:**
- What, if anything, is frustrating me?
- Is there a misunderstanding I could clarify?
- What response might help improve the situation?

Frustration is a normal human experience. An image cannot tell
the app what caused an expression or how you really feel.
""",

    "fear": """
### 🤍 Take a Moment to Feel Grounded

Uncertainty can sometimes feel uncomfortable. If you are worried
about something, it may help to separate what you know from what
you are imagining might happen.

**You could try:**
- Identifying what is concerning you.
- Focusing on what is within your control.
- Choosing one small, manageable next step.
- Speaking with someone you trust.

**Ask yourself:**
- Am I feeling worried, uncertain, or something different?
- What information would help me understand the situation?
- What might help me feel more comfortable?

A facial expression cannot confirm fear or anxiety.
Your own feelings and circumstances matter most.
""",

    "disgust": """
### 🌱 Notice Your Reaction Without Judging It

Sometimes we react negatively to a situation, an experience,
or something that makes us uncomfortable. You do not need to
judge yourself for having a reaction.

**Take a moment to consider:**
- Is something making me uncomfortable?
- Is there context that this image cannot show?
- Would taking a break or changing my surroundings help?
- Is there another explanation for my expression?

Give yourself time to understand what you are experiencing.
This model can only estimate a facial expression; it cannot
know what caused it or how you actually feel.
""",

    "surprise": """
### ✨ Make Room for the Unexpected

Surprise can happen when something catches our attention.
It does not necessarily mean that the experience is positive
or negative. Sometimes we simply need more time to understand it.

**You could reflect on:**
- What caught my attention?
- Did the situation feel pleasant, confusing, or unexpected?
- Do I need more information before making a decision?
- What would help me understand what happened?

There is no need to react immediately to every unexpected event.
Give yourself time to process the situation.

Remember, the app cannot determine your actual feelings from
a facial expression alone.
""",

    "neutral": """
### 🌼 A Calm, No-Pressure Check-In

A neutral facial expression does not mean that you have no
emotions. You might be concentrating, thinking, resting, or
simply expressing yourself naturally.

**Take a moment to check in:**
- How am I actually feeling right now?
- Is anything on my mind?
- Do I need a break, some activity, or a conversation?
- What would help me with the rest of my day?

You do not need to display a particular expression to feel
a certain way. Your own understanding of your emotions is
more meaningful than an automated prediction.
"""
}


# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():

    # Download the trained model if it is not already present.
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:

        try:
            result = gdown.download(
                id=DRIVE_FILE_ID,
                output=str(MODEL_PATH),
                quiet=False
            )

            if (
                result is None
                or not MODEL_PATH.exists()
                or MODEL_PATH.stat().st_size == 0
            ):
                raise RuntimeError(
                    "Download failed. Set Google Drive access to "
                    "'Anyone with the link'."
                )

        except Exception as e:
            raise RuntimeError(
                f"Model download failed: {e}"
            )

    # Load your saved checkpoint.
    checkpoint = torch.load(
        str(MODEL_PATH),
        map_location=DEVICE,
        weights_only=False
    )

    if not isinstance(checkpoint, dict):
        raise RuntimeError("Invalid model checkpoint.")

    if "model_state" not in checkpoint:
        raise RuntimeError(
            "Checkpoint does not contain 'model_state'. "
            "Use your original EMPATHICA checkpoint."
        )

    emotions = checkpoint.get("emotions")

    if not emotions:
        raise RuntimeError(
            "Emotion labels are missing from the checkpoint."
        )

    emotions = [
        str(emotion).lower().strip()
        for emotion in emotions
    ]

    # Recreate the same ResNet18 architecture used in training.
    model = models.resnet18(weights=None)

    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(
            model.fc.in_features,
            len(emotions)
        )
    )

    model.load_state_dict(
        checkpoint["model_state"],
        strict=True
    )

    model.to(DEVICE)
    model.eval()

    return model, emotions


# ---------------- FACE DETECTION ----------------

def detect_face(image):

    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    cascade_path = (
        Path(cv2.data.haarcascades)
        / "haarcascade_frontalface_default.xml"
    )

    detector = cv2.CascadeClassifier(str(cascade_path))

    if detector.empty():
        raise RuntimeError(
            "Face detector could not be loaded. "
            "Check your requirements.txt."
        )

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    if len(faces) == 0:
        return None

    # Select the largest detected face.
    x, y, w, h = max(
        faces,
        key=lambda face: face[2] * face[3]
    )

    # Add 10% padding around the face.
    px = int(w * 0.10)
    py = int(h * 0.10)

    x1 = max(0, x - px)
    y1 = max(0, y - py)
    x2 = min(rgb.shape[1], x + w + px)
    y2 = min(rgb.shape[0], y + h + py)

    cropped_face = rgb[y1:y2, x1:x2]

    return Image.fromarray(cropped_face)


# ---------------- EMOTION PREDICTION ----------------

def predict_emotion(face, model, emotions):

    # Keep the same normalization used during evaluation.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    image_tensor = transform(face).unsqueeze(0)

    with torch.inference_mode():
        output = model(image_tensor)
        probabilities = torch.softmax(
            output.float(),
            dim=1
        )[0].cpu().numpy()

    best_index = int(np.argmax(probabilities))

    return (
        emotions[best_index],
        float(probabilities[best_index]),
        probabilities
    )


# ---------------- STREAMLIT INTERFACE ----------------

st.markdown("""
<style>
.stApp {
    background-color: #f5f7f1;
    color: #26352f;
}
.block-container {
    max-width: 1000px;
    padding-top: 2rem;
}
h1, h2, h3 {
    color: #365442;
}
.stButton > button {
    background-color: #476650;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.6rem 1rem;
}
</style>
""", unsafe_allow_html=True)

st.title("🌿 EMPATHICA")
st.subheader("Emotion-Aware Emotional Wellness")

st.write(
    "Upload a clear, front-facing image. EMPATHICA estimates "
    "a facial expression and provides a supportive, rule-based "
    "reflection. It cannot confirm a person's actual emotions."
)

uploaded_file = st.file_uploader(
    "Upload your image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    try:
        image = ImageOps.exif_transpose(
            Image.open(uploaded_file)
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded image",
            use_container_width=True
        )

        if st.button("✨ Detect Emotion"):

            with st.spinner("Loading model and analysing image..."):

                model, emotions = load_model()

                face = detect_face(image)

                if face is None:
                    st.warning(
                        "No face detected. Please upload a clear, "
                        "front-facing image with one visible face."
                    )

                else:
                    emotion, score, probabilities = predict_emotion(
                        face,
                        model,
                        emotions
                    )

                    st.divider()
                    st.header("Your EMPATHICA Result")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.image(
                            face,
                            caption="Face used for prediction",
                            use_container_width=True
                        )

                    with col2:
                        st.subheader(
                            f"Detected expression: {emotion.title()}"
                        )

                        st.metric(
                            "Model confidence score",
                            f"{score * 100:.2f}%"
                        )

                        st.caption(
                            "This score is the model's output, "
                            "not a measure of emotional intensity."
                        )

                    st.subheader("Supportive Reflection")

                    response = EMOTION_RESPONSES.get(
                        emotion,
                        "Take a moment to check in with yourself. "
                        "The model's prediction may not match "
                        "how you actually feel."
                    )

                    st.markdown(response)

                    with st.expander("View top model predictions"):

                        top_indices = np.argsort(
                            probabilities
                        )[::-1][:3]

                        for index in top_indices:
                            label = emotions[int(index)]
                            value = float(probabilities[int(index)])

                            st.write(
                                f"**{label.title()}** — "
                                f"{value * 100:.2f}%"
                            )

                            st.progress(value)

    except Exception as e:
        st.error(f"Something went wrong: {e}")

st.divider()

st.caption(
    "EMPATHICA is a student project. Facial-expression predictions "
    "can be incorrect and are not a diagnosis. Your own account "
    "of your feelings matters most."
)
