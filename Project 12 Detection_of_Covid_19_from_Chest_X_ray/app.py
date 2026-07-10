import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="COVID-19 Chest X-ray Detection",
    page_icon="🩻",
    layout="centered"
)

st.title("🩻 COVID-19 Detection from Chest X-ray")
st.write("Upload a Chest X-ray image to predict whether it is **COVID** or **NORMAL**.")

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_covid_model():
    model = load_model("my_model.keras")
    return model

model = load_covid_model()

# ----------------------------
# Image Preprocessing
# ----------------------------
def preprocess_image(img):
    img = img.resize((299, 299))
    img = img.convert("RGB")
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ----------------------------
# File Upload
# ----------------------------
uploaded_file = st.file_uploader(
    "Choose a Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    img = Image.open(uploaded_file)

    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_array = preprocess_image(img)

    prediction = model.predict(img_array)

    probability = prediction[0][0]

    # Since your model uses sigmoid
    if probability > 0.5:
        result = "COVID"
        confidence = probability * 100
        color = "red"
    else:
        result = "NORMAL"
        confidence = (1 - probability) * 100
        color = "green"

    st.markdown("---")

    st.subheader("Prediction")

    if result == "COVID":
        st.error(f"🦠 Prediction: **{result}**")
    else:
        st.success(f"✅ Prediction: **{result}**")

    st.write(f"**Confidence:** {confidence:.2f}%")

    st.progress(float(confidence / 100))

    st.markdown("### Raw Model Output")
    st.write(f"Sigmoid Score: **{probability:.4f}**")

st.markdown("---")
st.caption("Model: CNN trained on COVID-19 and Normal Chest X-ray dataset")
