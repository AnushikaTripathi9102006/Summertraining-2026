import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="🐶🐱 Cat vs Dog Classifier",
    page_icon="🐾",
    layout="centered"
)

# -------------------------
# Custom CSS
# -------------------------
st.markdown("""
<style>

.main{
    background-color:#F8F9FA;
}

h1{
    text-align:center;
    color:#1E3A8A;
}

.stButton>button{
    width:100%;
    background:#2563EB;
    color:white;
    border-radius:12px;
    height:3em;
    font-size:18px;
    font-weight:bold;
}

.result{
    padding:15px;
    border-radius:10px;
    font-size:24px;
    text-align:center;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("binary_image_classifier.keras")

model = load_model()

st.title("🐾 Cat vs Dog Image Classifier")

st.write(
"""
Upload an image of a **cat** or a **dog** and the model will predict its class.
"""
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image,
             caption="Uploaded Image",
             use_container_width=True)

    img = image.resize((150,150))

    x = np.array(img)/255.0
    x = np.expand_dims(x,axis=0)

    if st.button("Predict"):

        prediction = model.predict(x)[0][0]

        if prediction >= 0.5:
            label = "🐶 Dog"
            confidence = prediction*100
        else:
            label = "🐱 Cat"
            confidence = (1-prediction)*100

        st.success(f"Prediction : {label}")

        st.progress(float(confidence/100))

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

        st.write("Raw Prediction :", prediction)
