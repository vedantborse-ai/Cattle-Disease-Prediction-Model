import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CattleGuard AI",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. ADVANCED CSS (THE UI MAGIC) ---
st.markdown("""
    <style>
    /* IMPORT FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Manrope:wght@300;400;600&display=swap');

    /* GENERAL SETTINGS */
    .stApp {
        background-color: #0E1117;
        background-image: radial-gradient(circle at 50% 0%, #2b1c1c 0%, #0E1117 60%);
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Playfair Display', serif !important;
        color: #F0F2F6;
    }
    
    p, div, label, span {
        font-family: 'Manrope', sans-serif;
        color: #C0C3C9;
    }

    /* CUSTOM FILE UPLOADER */
    [data-testid='stFileUploader'] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px dashed rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 20px;
        transition: all 0.3s ease;
    }
    [data-testid='stFileUploader']:hover {
        background-color: rgba(255, 255, 255, 0.1);
        border-color: #FF4B4B;
    }

    /* GLASSMORPHISM CARDS */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }

    /* HIDE DEFAULT STUFF */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* PREDICTION TEXT STYLING */
    .prediction-title {
        font-size: 3rem;
        font-weight: 700;
        background: -webkit-linear-gradient(0deg, #FF9A9E, #FECFEF, #F0F2F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOGIC & HELPERS ---
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("best_cattle_disease_model")
    except OSError:
        return None

def clean_class_name(name):
    return name.replace("_aug", "").replace("_", " ").title()

class_names = [
    "Bovine Mastitis", "Dermatophilosis", "Healthy",
    "Pediculosis_aug", "Ringworm_aug", "lumpy skin", "pinkeye"
]
clean_names = [clean_class_name(n) for n in class_names]

model = load_model()

# --- 4. THE HERO SECTION ---
st.markdown("<div style='text-align: center; padding-top: 2rem; padding-bottom: 3rem;'>", unsafe_allow_html=True)
st.markdown("# 🐄 CattleGuard AI")
st.markdown("<p style='font-size: 1.2rem; opacity: 0.7;'>Advanced Dermatological Diagnostics for Livestock</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# --- 5. MAIN INTERFACE ---
if model is None:
    st.error("⚠️ Model file missing. Please check your directory.")
else:
    # Centered File Uploader
    col_spacer_l, col_upload, col_spacer_r = st.columns([1, 2, 1])
    with col_upload:
        uploaded_file = st.file_uploader("Upload Analysis Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Load and Process
        img = Image.open(uploaded_file)
        
        # Preprocessing
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Predict
        with st.spinner("Analyzing dermal patterns..."):
            prediction = model.predict(img_array)[0]
            idx = np.argmax(prediction)
            disease_name = clean_names[idx]
            confidence = prediction[idx]

        # --- RESULTS LAYOUT (Inspired by Oknoplast/Tack Shop) ---
        c1, c2 = st.columns([1, 1.2], gap="large")

        with c1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 📷 Input Sample")
            st.image(img, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 🔬 Diagnostic Report")
            
            # Dynamic Color based on result
            color = "#00FF7F" if disease_name == "Healthy" else "#FF4B4B"
            
            st.markdown(f"""
            <div style='margin-bottom: 20px;'>
                <span style='font-size: 1rem; text-transform: uppercase; letter-spacing: 2px; color: {color};'>Detected Condition</span>
                <h1 class="prediction-title">{disease_name}</h1>
                <p style='font-size: 1.1rem;'>Confidence Score: <b>{confidence*100:.2f}%</b></p>
            </div>
            """, unsafe_allow_html=True)

            # Elegant Plotly Chart (Dark Mode)
            df = pd.DataFrame({'Condition': clean_names, 'Probability': prediction})
            df = df.sort_values(by='Probability', ascending=True)

            fig = go.Figure(go.Bar(
                x=df['Probability'],
                y=df['Condition'],
                orientation='h',
                marker=dict(
                    color=df['Probability'],
                    colorscale='Reds' if disease_name != "Healthy" else 'Greens',
                    line=dict(color='rgba(255, 255, 255, 0.2)', width=1)
                )
            ))

            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=300,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False, color='#C0C3C9', tickfont=dict(family='Manrope')),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        # Empty State (Inspired by HackMol/IFI minimalist text)
        st.markdown("""
        <div style='text-align: center; padding: 50px; opacity: 0.5;'>
            <p>Awaiting Input Signal...</p>
        </div>
        """, unsafe_allow_html=True)