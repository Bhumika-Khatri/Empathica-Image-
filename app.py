import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps
import numpy as np
import cv2
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
