import streamlit as st
from transformers import pipeline
from PIL import Image

st.title("Bird Species Identifier")

@st.cache_resource
def load_model():
    pipe = pipeline(
        "image-classification",
        model="chriamue/bird-species-classifier",
        device=-1
    )
    return pipe

pipe = load_model()

uploaded_file = st.file_uploader("Upload your pretty bird image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image)

    with st.spinner("Identifying bird..."):
        result = pipe(image)

    st.write(result[0]["label"])
    st.write(f"Confidence: {result[0]['score']:.2f}")