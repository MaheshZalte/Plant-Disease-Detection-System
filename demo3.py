import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import firebase_admin
from firebase_admin import auth, credentials, firestore
import base64
import os
import logging

# --------------------------
# 1. PAGE CONFIG & FIREBASE
# --------------------------
st.set_page_config(
    page_title="PlantDx: AI Plant Diagnosis",
    layout="centered",
    initial_sidebar_state="expanded",
    page_icon="🌿"
)

# Initialize Firebase
def initialize_firebase():
    cred_path = os.path.join(os.getcwd(), "plant_disease_detection.json")
    if not firebase_admin._apps:
        if not os.path.exists(cred_path):
            st.error(f"Firebase credentials not found at {cred_path}")
            return False
        try:
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
            return True
        except Exception as e:
            st.error(f"Firebase initialization failed: {e}")
            return False
    return True

if not initialize_firebase():
    st.stop()

db = firestore.client()

# -----------------------
# 2. AUTHENTICATION
# -----------------------
def handle_auth(action, email, password):
    try:
        if action == "login":
            user = auth.get_user_by_email(email)
            return {"idToken": "FAKE_TOKEN", "email": user.email}
        elif action == "signup":
            user = auth.create_user(email=email, password=password)
            return {"idToken": "FAKE_TOKEN", "email": user.email}
    except auth.UserNotFoundError:
        return {"error": "User not found. Please sign up."}
    except auth.EmailAlreadyExistsError:
        return {"error": "Email already exists. Please log in."}
    except Exception as e:
        return {"error": str(e)}

# -----------------------
# 3. SESSION MANAGEMENT
# -----------------------
def init_session_state():
    defaults = {
        "user": None,
        "app_mode": "landing",
        "prev_page": ""
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

# -------------------
# 4. STYLING & THEME
# -------------------
def apply_custom_style():
    st.markdown(f"""
    <style>
    :root {{
        --primary: #2E86C1;
        --secondary: #27AE60;
        --accent: #F1C40F;
        --text: #2C3E50;
        --background: rgba(255, 255, 255, 0.9);
    }}

    .stApp {{
        background: linear-gradient(rgba(255, 255, 255, 0.8), 
                    url("data:image/jpg;base64,{get_base64_image('background.jpg')}");
        background-size: cover;
        background-attachment: fixed;
    }}

    .card {{
        background: var(--background);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        margin: 1rem 0;
    }}

    .stButton>button {{
        background: var(--secondary);
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }}

    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 3px 6px rgba(0, 0, 0, 0.2);
    }}

    h1, h2, h3 {{
        color: var(--primary);
        font-family: 'Roboto', sans-serif;
        margin-bottom: 1rem;
    }}

    .sidebar .stButton>button {{
        width: 100%;
        margin: 0.5rem 0;
    }}
    </style>
    """, unsafe_allow_html=True)

apply_custom_style()

# -------------------------
# 5. UI COMPONENTS
# -------------------------
def navigation_bar():
    if st.session_state.user:
        with st.sidebar:
            st.title("🌱 PlantDx")
            if st.button("🏠 Home"):
                st.session_state.app_mode = "home"
            if st.button("🔍 Diagnosis"):
                st.session_state.app_mode = "diagnosis"
            if st.button("📚 Knowledge Base"):
                st.session_state.app_mode = "knowledge"
            if st.button("📤 Logout"):
                st.session_state.user = None
                st.session_state.app_mode = "landing"
                st.rerun()

def auth_form(type):
    with st.form(f"{type}_form"):
        email = st.text_input("Email", key=f"{type}_email")
        password = st.text_input("Password", type="password", key=f"{type}_pass")
        submitted = st.form_submit_button("Sign In" if type == "login" else "Create Account")
        
        if submitted:
            result = handle_auth(type, email, password)
            if "error" in result:
                st.error(result["error"])
            else:
                st.session_state.user = result
                st.session_state.app_mode = "home"
                st.rerun()

    if type == "login":
        st.markdown("Don't have an account? [Sign Up](#signup)")
    else:
        st.markdown("Already have an account? [Sign In](#login)")

# -------------------------
# 6. PAGE DEFINITIONS
# -------------------------
def landing_page():
    st.markdown("""
    <div class="card">
        <h1>🌿 Welcome to PlantDx</h1>
        <h3>AI-Powered Plant Health Diagnosis</h3>
        <p>Get instant plant disease detection and expert recommendations</p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(2)
    with cols[0]:
        if st.button("Sign In", key="landing_login"):
            st.session_state.app_mode = "login"
    with cols[1]:
        if st.button("Sign Up", key="landing_signup"):
            st.session_state.app_mode = "signup"

def home_page():
    navigation_bar()
    st.markdown("""
    <div class="card">
        <h1>Welcome Back, {}</h1>
        <p>Start by uploading a plant image for diagnosis or explore our knowledge base.</p>
    </div>
    """.format(st.session_state.user['email'].split('@')[0]), unsafe_allow_html=True)

    cols = st.columns(2)
    with cols[0]:
        st.markdown("""
        <div class="card">
            <h3>📸 New Diagnosis</h3>
            <p>Upload an image of a plant leaf for instant analysis</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Start Diagnosis"):
            st.session_state.app_mode = "diagnosis"

    with cols[1]:
        st.markdown("""
        <div class="card">
            <h3>🌱 Plant Health Guide</h3>
            <p>Learn about common plant diseases and prevention methods</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Explore Knowledge Base"):
            st.session_state.app_mode = "knowledge"

def diagnosis_page():
    navigation_bar()
    st.markdown("""
    <div class="card">
        <h2>Plant Health Diagnosis</h2>
        <p>Upload a clear photo of a plant leaf for analysis</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        cols = st.columns([1, 2])
        with cols[0]:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        with cols[1]:
            if st.button("Analyze Image"):
                with st.spinner("Analyzing..."):
                    # Add your prediction logic here
                    diagnosis_result = {
                        'disease': 'Tomato Early Blight',
                        'confidence': 0.92,
                        'treatment': 'Apply copper-based fungicides weekly'
                    }
                    
                st.markdown(f"""
                <div class="card">
                    <h3>Diagnosis Results</h3>
                    <p><strong>Detected Disease:</strong> {diagnosis_result['disease']}</p>
                    <p><strong>Confidence:</strong> {diagnosis_result['confidence']*100:.1f}%</p>
                    <div class="stProgress">
                        <div class="stProgressBar" style="width: {diagnosis_result['confidence']*100}%;"></div>
                    </div>
                    <h4>Recommended Treatment</h4>
                    <p>{diagnosis_result['treatment']}</p>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()

# -----------------------
# 7. MAIN APP CONTROLLER
# -----------------------
def main():
    pages = {
        "landing": landing_page,
        "login": lambda: auth_form("login"),
        "signup": lambda: auth_form("signup"),
        "home": home_page,
        "diagnosis": diagnosis_page,
        "knowledge": lambda: st.write("Knowledge Base Content")
    }
    
    if st.session_state.app_mode not in pages:
        st.session_state.app_mode = "landing"
    
    pages[st.session_state.app_mode]()

if __name__ == "__main__":
    main()