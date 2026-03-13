import streamlit as st
from PIL import Image
import torch
from transformers import pipeline

pipe = pipeline("image-classification", model="chriamue/bird-species-classifier")

st.title("Bird Species Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", width=300)

    result = pipe(image)

    top = result[0]

    label = top["label"]
    score_percent = top["score"] * 100

    st.write(f"{label} ==> {score_percent:.2f}%")