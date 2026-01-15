# -*- coding: utf-8 -*-
# app.py
import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import requests
from io import BytesIO
import numpy as np

# ----------------------------
# 1) Load ImageNet Labels
# ----------------------------
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
response = requests.get(LABELS_URL)
imagenet_classes = [line.strip() for line in response.text.splitlines()]

# ----------------------------
# 2) Load Pretrained ResNet-18
# ----------------------------
model = models.resnet18(pretrained=True)
model.eval()  # set to evaluation mode

# ----------------------------
# 3) Image Preprocessing Pipeline
# ----------------------------
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    )
])

# ----------------------------
# 4) Streamlit UI
# ----------------------------
st.set_page_config(page_title="Real-Time Image Classifier", layout="wide")
st.title("📸 Real-Time Webcam Image Classification")
st.write("Capture an image of any object and let ResNet-18 predict what it is!")

# Upload image from webcam or file
uploaded_file = st.camera_input("Take a picture")  # streamlit-native webcam input

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Captured Image", use_column_width=True)

    # Preprocess image
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    # ----------------------------
    # 5) Model Inference
    # ----------------------------
    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get top 5 predictions
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    top5_labels = [imagenet_classes[i] for i in top5_catid]

    # Display results in a table
    st.subheader("Top 5 Predictions")
    results = { "Label": top5_labels, "Probability": [f"{p.item()*100:.2f}%" for p in top5_prob] }
    st.table(results)
