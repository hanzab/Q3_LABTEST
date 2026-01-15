# -*- coding: utf-8 -*-
# live_image_classifier.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# rest of your imports...


import streamlit as st
from PIL import Image
import requests
import pandas as pd

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(
    page_title="Real-Time Image Classifier",
    layout="centered"
)

st.title("🖼️ Real-Time Image Classification")
st.caption("Using Pretrained ResNet-18 (ImageNet)")

# -------------------------------
# Device configuration
# -------------------------------
device = torch.device("cpu")

# -------------------------------
# Load ImageNet Class Labels
# -------------------------------
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

@st.cache_data
def fetch_labels(url):
    response = requests.get(url)
    return [line.strip() for line in response.text.splitlines()]

labels = fetch_labels(LABELS_URL)

# -------------------------------
# Load Pretrained ResNet-18
# -------------------------------
@st.cache_resource
def get_model():
    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    net.eval()
    return net

model = get_model().to(device)

# -------------------------------
# Image Preprocessing
# -------------------------------
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------
# Image Capture or Upload
# -------------------------------
st.subheader("📸 Capture Image from Webcam")
img_file = st.camera_input("Click to take a picture")

if img_file:
    image = Image.open(img_file).convert("RGB")
    st.image(image, caption="Captured Image", use_column_width=True)

    # Preprocess image for model
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    # -------------------------------
    # Model Prediction
    # -------------------------------
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output[0], dim=0)

    top_probs, top_idxs = torch.topk(probabilities, 5)
    top_labels = [labels[i] for i in top_idxs]
    top_probs_percent = (top_probs * 100).cpu().numpy()

    results_df = pd.DataFrame({
        "Class": top_labels,
        "Confidence (%)": top_probs_percent
    })

    # -------------------------------
    # Display Results
    # -------------------------------
    st.subheader("🔍 Top 5 Predictions")
    st.dataframe(results_df, use_container_width=True)

    best_label = results_df.iloc[0]["Class"]
    best_conf = results_df.iloc[0]["Confidence (%)"]

    st.success(f"🎯 Best Prediction: **{best_label}** with **{best_conf:.2f}% confidence**")
    st.bar_chart(results_df.set_index("Class"), horizontal=True)

    st.info(
        "The confidence percentages are obtained via the Softmax function, "
        "showing the probability of each class."
    )
