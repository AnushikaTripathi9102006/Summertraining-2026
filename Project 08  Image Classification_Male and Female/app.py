
import streamlit as st
import numpy as np
from PIL import Image
import joblib
import time

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="🤖 AI Gender Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">

<style>

html,body,[class*="css"]{
font-family:'Poppins',sans-serif;
}

/* Background */

.stApp{

background:linear-gradient(-45deg,
#090979,
#020024,
#000428,
#004e92,
#6a11cb,
#2575fc);

background-size:400% 400%;

animation:bg 15s ease infinite;

}

@keyframes bg{

0%{background-position:0% 50%;}
50%{background-position:100% 50%;}
100%{background-position:0% 50%;}

}

/* Hide Streamlit */

#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}

/* Title */

.title{

font-size:60px;

font-weight:800;

text-align:center;

color:white;

margin-top:20px;

text-shadow:
0 0 10px cyan,
0 0 25px cyan,
0 0 45px cyan;

animation:glow 2s infinite alternate;

}

@keyframes glow{

from{

text-shadow:
0 0 10px cyan,
0 0 25px cyan;

}

to{

text-shadow:
0 0 20px white,
0 0 40px cyan,
0 0 70px cyan;

}

}

.subtitle{

text-align:center;

font-size:22px;

color:#f1f1f1;

margin-bottom:25px;

}

/* Glass Card */

.glass{

background:rgba(255,255,255,.12);

backdrop-filter:blur(18px);

padding:25px;

border-radius:25px;

border:1px solid rgba(255,255,255,.25);

box-shadow:0 8px 35px rgba(0,0,0,.4);

animation:float 4s ease-in-out infinite;

}

@keyframes float{

0%{transform:translateY(0);}
50%{transform:translateY(-8px);}
100%{transform:translateY(0);}

}

/* Image */

img{

border-radius:20px;

transition:.5s;

box-shadow:0 0 25px rgba(255,255,255,.4);

}

img:hover{

transform:scale(1.05);

}

/* Upload */

[data-testid="stFileUploader"]{

background:rgba(255,255,255,.15);

padding:20px;

border-radius:20px;

border:2px dashed cyan;

}

/* Buttons */

.stButton>button{

background:linear-gradient(45deg,#00c6ff,#0072ff);

color:white;

font-size:18px;

font-weight:600;

border-radius:12px;

border:none;

padding:12px;

transition:.3s;

}

.stButton>button:hover{

transform:scale(1.05);

box-shadow:0 0 20px cyan;

}

/* Progress */

.stProgress > div > div > div > div{

background:linear-gradient(90deg,#00ffcc,#00c6ff);

}

/* Prediction Card */

.pred{

background:linear-gradient(135deg,#00c6ff,#0072ff);

padding:25px;

border-radius:20px;

text-align:center;

color:white;

font-size:34px;

font-weight:bold;

box-shadow:0 0 30px rgba(0,0,0,.4);

}

/* Footer */

.footer{

text-align:center;

color:white;

opacity:.8;

margin-top:60px;

font-size:16px;

}

/* Metrics */

[data-testid="metric-container"]{

background:rgba(255,255,255,.15);

border-radius:20px;

padding:15px;

box-shadow:0 0 20px rgba(0,0,0,.3);

}

</style>

""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown("# 🤖 AI Dashboard")

    st.write("---")

    st.success("✔ Face Upload")

    st.success("✔ AI Analysis")

    st.success("✔ Prediction")

    st.write("---")

    st.metric("Model","Scikit-Learn")

    st.metric("Input","64 x 64")

    st.metric("Speed","<1 sec")

    st.write("---")

    st.info("""
This application predicts gender from
facial images using Machine Learning.

Developed with ❤️ using

• Streamlit

• NumPy

• Pillow

• Scikit-learn
""")

# =====================================================
# LOAD MODEL
# =====================================================

model=joblib.load("gender_model.pkl")

IMG_SIZE=64

# =====================================================
# HEADER
# =====================================================

st.markdown("""

<div class='glass'>

<div class='title'>

🤖 AI Gender Detection

</div>

<div class='subtitle'>

Deep Learning Inspired Face Classification System

</div>

</div>

""",unsafe_allow_html=True)

st.write("")

# =====================================================
# FEATURE CARDS
# =====================================================

c1,c2,c3,c4=st.columns(4)

with c1:
    st.info("📷 Image Upload")

with c2:
    st.info("⚡ Instant Prediction")

with c3:
    st.info("🎯 AI Accuracy")

with c4:
    st.info("🧠 Machine Learning")

st.divider()

# =====================================================
# FILE UPLOADER
# =====================================================

uploaded_file=st.file_uploader(
"📂 Upload Face Image",
type=["jpg","jpeg","png"]
)

# =====================================================
# PREDICTION
# =====================================================

if uploaded_file:

    image=Image.open(uploaded_file).convert("RGB")

    col1,col2=st.columns([1,1])

    with col1:

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    resized=image.resize((IMG_SIZE,IMG_SIZE))
    resized=np.array(resized).flatten()

    with st.spinner("🧠 AI is analyzing image..."):

        time.sleep(2)

        prediction=model.predict([resized])[0]

        probability=model.predict_proba([resized])[0]

    with col2:

        st.subheader("Prediction")

        if prediction==0:

            st.markdown("""

            <div class='pred'>

            👩 FEMALE

            </div>

            """,unsafe_allow_html=True)

            st.balloons()

        else:

            st.markdown("""

            <div class='pred'>

            👨 MALE

            </div>

            """,unsafe_allow_html=True)

            st.snow()

        st.write("")

        a,b=st.columns(2)

        with a:

            st.metric(
                "👩 Female",
                f"{probability[0]*100:.2f}%"
            )

        with b:

            st.metric(
                "👨 Male",
                f"{probability[1]*100:.2f}%"
            )

        st.write("### Confidence")

        st.progress(float(probability[0]))

        st.write(
            f"👩 Female : **{probability[0]*100:.2f}%**"
        )

        st.progress(float(probability[1]))

        st.write(
            f"👨 Male : **{probability[1]*100:.2f}%**"
        )

    st.divider()

    x,y,z=st.columns(3)

    with x:
        st.success("✔ Prediction Complete")

    with y:
        st.info("🧠 AI Processed")

    with z:
        st.warning("⚠ Not 100% Accurate")

    st.info("""
This prediction is generated by a Machine Learning model.

Results may vary depending upon

• Lighting

• Face Orientation

• Image Quality

• Dataset Bias
""")

else:

    st.markdown("""

    <div class='glass'>

    <h2 style='text-align:center;color:white;'>

    👆 Upload an image to begin AI prediction

    </h2>

    </div>

    """, unsafe_allow_html=True)


st.markdown("""

<div class='footer'>

<hr>

🚀 Developed using Streamlit | Scikit-Learn | NumPy | Pillow

<br><br>

© 2026 AI Gender Detection System

</div>

""", unsafe_allow_html=True)
