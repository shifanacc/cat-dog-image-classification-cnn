import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Cat vs Dog AI Classifier",
    page_icon="🐾",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');

* {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 45%, #111827 100%);
    color: white;
}

.block-container {
    padding-top: 2rem;
}

.hero {
    text-align: center;
    padding: 35px;
    border-radius: 30px;
    background: rgba(255, 255, 255, 0.08);
    box-shadow: 0 20px 60px rgba(0,0,0,0.35);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.15);
}

.hero h1 {
    font-size: 48px;
    font-weight: 800;
    background: linear-gradient(90deg, #FBBF24, #F472B6, #60A5FA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    font-size: 18px;
    color: #CBD5E1;
}

.card {
    padding: 28px;
    border-radius: 25px;
    background: rgba(255,255,255,0.10);
    box-shadow: 0 15px 45px rgba(0,0,0,0.30);
    border: 1px solid rgba(255,255,255,0.15);
}

.result-cat {
    padding: 25px;
    border-radius: 25px;
    background: linear-gradient(135deg, #F59E0B, #EF4444);
    text-align: center;
    font-size: 30px;
    font-weight: 800;
    color: white;
}

.result-dog {
    padding: 25px;
    border-radius: 25px;
    background: linear-gradient(135deg, #3B82F6, #8B5CF6);
    text-align: center;
    font-size: 30px;
    font-weight: 800;
    color: white;
}

.metric-box {
    padding: 20px;
    border-radius: 18px;
    background: rgba(255,255,255,0.12);
    text-align: center;
    border: 1px solid rgba(255,255,255,0.15);
}

.metric-box h3 {
    color: #F8FAFC;
}

.metric-box p {
    color: #CBD5E1;
    font-size: 22px;
    font-weight: 700;
}

.stButton > button {
    width: 100%;
    border-radius: 15px;
    height: 50px;
    font-size: 18px;
    font-weight: 700;
    background: linear-gradient(90deg, #FBBF24, #F472B6);
    color: black;
    border: none;
}

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.10);
    padding: 20px;
    border-radius: 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_cnn_model():
    return load_model("cat_dog_cnn.h5")

model = load_cnn_model()

# ---------------- HERO SECTION ----------------
st.markdown("""
<div class="hero">
    <h1>🐾 Cat vs Dog AI Classifier</h1>
    <p>Upload an image and let the CNN model identify whether it is a Cat or a Dog.</p>
</div>
""", unsafe_allow_html=True)

st.write("")

# ---------------- MAIN LAYOUT ----------------
left, right = st.columns([1, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📤 Upload Image")

    st.info(
        "📌 Please upload a clear image containing a single Cat or Dog for reliable prediction results."
    )

    uploaded_file = st.file_uploader(
        "Supported formats: JPG, JPEG, PNG",
        type=["jpg", "jpeg", "png"]
    )

    st.caption(
        "✔ Use a clear, high-quality image with good lighting and a visible Cat or Dog.\n"
        "Blurred images, multiple animals, partially visible subjects, or images containing multiple animals may reduce prediction accuracy."
    )

    st.markdown("</div>", unsafe_allow_html=True)
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("ℹ️ Project Details")
    st.write("""
    This application uses a **Convolutional Neural Network (CNN)** trained on cat and dog images.
    The model analyzes image features and predicts the correct class.
    """)
    st.markdown("""
    <div class="metric-box">
        <h3>Model Type</h3>
        <p>CNN</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if uploaded_file is not None:
    st.write("")
    col1, col2 = st.columns([1, 1])

    with col1:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

    with col2:
        img_resized = img.resize((128, 128))
        img_array = image.img_to_array(img_resized)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]

        if prediction > 0.5:
            confidence = prediction * 100
            st.markdown('<div class="result-dog">Prediction: Dog 🐶</div>', unsafe_allow_html=True)
        else:
            confidence = (1 - prediction) * 100
            st.markdown('<div class="result-cat">Prediction: Cat 🐱</div>', unsafe_allow_html=True)

        st.write("")
        st.markdown(f"""
        <div class="metric-box">
            <h3>Confidence Score</h3>
            <p>{confidence:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.write("")
st.markdown("""
<div style="text-align:center; color:#CBD5E1; padding:20px;">
    Developed using TensorFlow/Keras and Streamlit
</div>
""", unsafe_allow_html=True)