import streamlit as st
import numpy as np
from PIL import Image
import joblib

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="AI Gender Classifier",
    page_icon="🧠",
    layout="wide"
)

# -------------------------
# Custom CSS
# -------------------------
st.markdown("""
<style>
.main{
    background-color:#f5f7fa;
}
.title{
    text-align:center;
    font-size:40px;
    color:#1f77b4;
    font-weight:bold;
}
.subtitle{
    text-align:center;
    font-size:18px;
    color:gray;
}
.result{
    padding:20px;
    border-radius:15px;
    background:#ffffff;
    box-shadow:0px 0px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Load Model
# -------------------------
model = joblib.load("male_female_model.pkl")
IMG_SIZE = 64

# -------------------------
# Header
# -------------------------
st.markdown("<p class='title'>🧠 AI Gender Classification</p>", unsafe_allow_html=True)
st.markdown(
    "<p class='subtitle'>Upload an image and let the AI predict whether it is Male or Female.</p>",
    unsafe_allow_html=True
)

st.divider()

# -------------------------
# Layout
# -------------------------
left, right = st.columns([1, 1])

with left:
    uploaded_file = st.file_uploader(
        "📂 Upload an Image",
        type=["jpg", "jpeg", "png"]
    )

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    with left:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # -------------------------
    # Image Processing
    # -------------------------
    resized = image.resize((IMG_SIZE, IMG_SIZE))
    resized = np.array(resized).flatten()

    prediction = model.predict([resized])[0]
    probability = model.predict_proba([resized])[0]

    male_prob = probability[0]
    female_prob = probability[1]

    with right:

        st.markdown("## 📊 Prediction Result")

        if prediction == 0:
            st.success("### 👨 Male")
        else:
            st.success("### 👩 Female")

        st.write("### Confidence")

        st.write("👨 Male")
        st.progress(float(male_prob))

        st.write("👩 Female")
        st.progress(float(female_prob))

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Male Probability",
                f"{male_prob*100:.2f}%"
            )

        with c2:
            st.metric(
                "Female Probability",
                f"{female_prob*100:.2f}%"
            )

        st.divider()

        confidence = max(male_prob, female_prob) * 100

        st.info(f"🎯 Model Confidence: **{confidence:.2f}%**")

else:
    st.info("👈 Upload an image to begin prediction.")

st.divider()

st.caption("Built using Streamlit • Logistic Regression • Scikit-Learn")