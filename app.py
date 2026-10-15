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
                    "'Anyone with the
