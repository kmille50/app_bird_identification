import os
import streamlit as st
from PIL import Image
import io
from huggingface_hub import InferenceClient

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

        predictions = client.image_classification(
            buffer.getvalue(),
            model="chriamue/bird-species-classifier"
        )

        st.write("Predictions:", predictions)

    else:
        st.info("Please upload an image.")

if __name__ == "__main__":
    app()