import os
import streamlit as st
from PIL import Image
import io
import time
from huggingface_hub import InferenceClient
from transformers import pipeline


client = InferenceClient(
    provider="auto",
    api_key=os.getenv("HF_TOKEN"),
)


def app():
    st.title("Bird Species Classifier")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")

        st.image(image, caption="Uploaded Image.", width=300)

        
        pipe = pipeline("image-classification", model="chriamue/bird-species-classifier")
        result = pipe(image)

        top = result[0]

        label = top["label"]
        score_percent = top["score"] * 100

        st.write("Predictions:", f"{label} ==> {score_percent:.2f}%")

if __name__ == "__main__":
    app()