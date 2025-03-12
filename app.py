import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import firebase_admin
from firebase_admin import auth, credentials, firestore
import base64
import os
import logging
from datetime  import datetime



# --------------------------
# 1. PAGE CONFIG & FIREBASE
# --------------------------
st.set_page_config(
    page_title="Plant Disease Recognition",
    layout="centered",
    initial_sidebar_state="expanded"
)


# Initialize Firebase only once
cred_path = os.path.join(os.getcwd(), "plant_disease_detection.json")


if not firebase_admin._apps:
    if not os.path.exists(cred_path):
        st.error(f"Error: Firebase credentials file not found at {cred_path}.")
    else:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)

db = firestore.client()


# -----------------------
# 2. DEFINE LOGIN/SIGNUP
# -----------------------
# dimport firebase_admin
from firebase_admin import auth

def firebase_login(email: str, password: str):
    """
    Mock login function using Firebase Admin SDK.
    NOTE: This does NOT verify the actual password!
    1. Checks if a user record with the given email exists.
    2. Returns a mock 'idToken' if found, or an error message if not.
    """
    try:
        user_record = auth.get_user_by_email(email)
        # Since the Admin SDK can't verify the password, we return a mock token.
        return {
            "idToken": "FAKE_ID_TOKEN",
            "email": user_record.email,
            "displayName": user_record.display_name or "",
            "phoneNumber": user_record.phone_number or ""
        }
    except firebase_admin._auth_utils.UserNotFoundError:
        # More explicit error if the user doesn't exist
        return {"error": f"User with email '{email}' not found. Please sign up."}
    except Exception as e:
        return {"error": str(e)}

def firebase_signup(email: str, password: str):
    """
    Creates a user with the given email and password using Firebase Admin SDK.
    NOTE: This stores the password hash in Firebase Auth but doesn't verify it.
    1. Attempts to create a user with the specified email & password.
    2. Returns a mock 'idToken' on success, or an error message on failure.
    """
    try:
        user_record = auth.create_user(email=email, password=password)
        return {
            "idToken": "FAKE_ID_TOKEN",
            "email": user_record.email,
            "uid": user_record.uid
        }
    except firebase_admin._auth_utils.EmailAlreadyExistsError:
        # If the email is already in use, let the user know
        return {"error": f"Email '{email}' is already in use. Please log in instead."}
    except Exception as e:
        return {"error": str(e)}


# -----------------------
# 3. GLOBAL SESSION DATA
# -----------------------
def init_session_defaults():
    """
    Initialize default values for session state if they do not exist.
    This keeps your code organized and prevents re-initializing values on each run.
    """
    defaults = {
        "user": None,       # store authenticated user info
        "app_mode": "landing"  # start at landing page
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Call this function once, near the top of your script (after imports).
init_session_defaults()


# -------------------
# 4. LOAD MODEL ONCE
# -------------------
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("PlantDisease_Model.h5")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# -------------------------
# 5. HELPER FUNCTIONS (UI)
# -------------------------
import streamlit as st

def load_css():
    """
    Enhanced custom CSS injection:
    • Loads Sigmar (for headings) & Roboto (for body text) from Google Fonts
    • Applies Sigmar to all headings (h1–h6)
    • Applies Roboto to general text and form elements
    • Demonstrates styling for buttons, progress bars, etc.
    """
    custom_css = """
    <style>
    /* 1. Import Sigmar & Roboto from Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Sigmar&family=Roboto:wght@400;700&display=swap');

    /* 2. Sigmar class for special text usage (if needed) */
    .sigmar-regular {
        font-family: 'Sigmar', cursive;
        font-weight: 400;
        /* Example overrides:
        color: #2E86C1;
        font-size: 1.2rem;
        */
    }

    /* 3. Apply Sigmar to all headings (h1 - h6) */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Sigmar', cursive;
        color: #2E86C1; /* Example color for headings */
        margin-top: 0.5em;
        margin-bottom: 0.5em;
    }

    /* 4. Use Roboto for general body text and common elements */
    body, p, div, span, input, button, textarea, label, select {
        font-family: 'Roboto', sans-serif;
    }

    /* 5. Customize Streamlit default buttons */
    .stButton>button {
        background-color: #27AE60; /* Green background */
        color: #FFFFFF;           /* White text */
        border: none;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        font-size: 1rem;
        cursor: pointer;
        transition: background-color 0.3s ease, color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2ECC71; /* Lighter green on hover */
        color: #000000;           /* Black text on hover */
    }

    /* 6. Progress bar styling */
    div[data-testid="stProgressBar"] > div > div {
        background-color: #2ECC71 !important; /* Make progress bar green */
    }

    /* 7. Other optional global changes */
    /* Example: Changing link color
    a {color: #2980B9;}
    */
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
load_css()

def get_base64_image(image_path: str) -> str:
    """
    Reads an image file from the given path and returns its base64-encoded string.
    
    :param image_path: The file system path to the image.
    :return: A base64-encoded string of the image contents, or an empty string on error.
    """
    if not os.path.isfile(image_path):
        logging.warning(f"File not found or is a directory: {image_path}")
        return ""
    
    try:
        with open(image_path, "rb") as image_file:
            encoded_bytes = base64.b64encode(image_file.read())
            return encoded_bytes.decode("utf-8")
    except (IOError, OSError) as e:
        logging.error(f"Error reading file {image_path}: {e}")
        return ""
    except Exception as e:
        logging.exception(f"Unexpected error occurred while encoding {image_path}: {e}")
        return ""


# Set background image (if found)
background_image = get_base64_image("bag.jpg")

if background_image:
    page_bg_img = f"""
    <style>
    .stApp {{
        /* 1) Apply a subtle dark overlay via a linear-gradient. 
            The rgba(0,0,0,0.3) sets a 30% black overlay. 
              Adjust as needed (0.2 = 20%, 0.5 = 50%, etc.). */
        background: linear-gradient(
            rgba(0, 0, 0, 0.3),
            rgba(0, 0, 0, 0.3)
        ),
        url("data:image/jpg;base64,{background_image}") no-repeat center center fixed;

        /* 2) Scale the image to fill the viewport */
        background-size: cover;

        /* 3) (Optional) If you really want a slight blur or brightness tweak:
        filter: brightness(0.9) blur(2px);
        But we removed heavy blur for a cleaner look. */
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)


# -------------------------
# 6. DATA & PROCESSING FNS
# -------------------------
disease_info = {
    "Pepper_bell_Bacterial_spot": {
        "plant": "Pepper Bell",
        "solution": "Use copper-based bactericides and avoid overhead watering."
    },
    "Pepper_bell_healthy": {
        "plant": "Pepper Bell",
        "solution": "The plant is healthy. Maintain proper watering and nutrient levels."
    },
    "Potato_Early_blight": {
        "plant": "Potato",
        "solution": "Use fungicides with chlorothalonil and practice crop rotation."
    },
    "Potato_healthy": {
        "plant": "Potato",
        "solution": "The plant is healthy. Ensure good soil drainage."
    },
    "Potato_Late_blight": {
        "plant": "Potato",
        "solution": "Use copper-based fungicides and remove affected leaves."
    },
    "Tomato_Target_Spot": {
        "plant": "Tomato",
        "solution": "Apply fungicides like chlorothalonil and avoid excess moisture."
    },
    "Tomato_Tomato_mosaic_virus": {
        "plant": "Tomato",
        "solution": "Remove infected plants immediately and control aphids."
    },
    "Tomato_Tomato_YellowLeaf_Curl_Virus": {
        "plant": "Tomato",
        "solution": "Control whiteflies and use virus-resistant varieties."
    },
    "Tomato_Bacterial_spot": {
        "plant": "Tomato",
        "solution": "Use copper sprays and avoid overhead irrigation."
    },
    "Tomato_Early_blight": {
        "plant": "Tomato",
        "solution": "Use chlorothalonil-based fungicides regularly."
    },
    "Tomato_healthy": {
        "plant": "Tomato",
        "solution": "The plant is healthy. Maintain good cultivation practices."
    },
    "Tomato_Late_blight": {
        "plant": "Tomato",
        "solution": "Apply fungicides containing mancozeb and remove infected leaves."
    },
    "Tomato_Leaf_Mold": {
        "plant": "Tomato",
        "solution": "Ensure proper ventilation and use fungicides if needed."
    },
    "Tomato_Septoria_leaf_spot": {
        "plant": "Tomato",
        "solution": "Remove infected leaves and use copper-based fungicides."
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "plant": "Tomato",
        "solution": "Use neem oil or insecticidal soaps to control mites."
    }
}

def process_image(image_data):
    """Resize and normalize the uploaded image."""
    if not image_data:
        st.error("No image uploaded.")
        return None
    try:
        image = Image.open(image_data).convert("RGB")
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        arr = np.array(image) / 255.0
        return np.expand_dims(arr, axis=0)
    except Exception as e:
        st.error(f"🔥 Error Processing Image: {e}")
        return None

def model_prediction(image_data):
    """Run model prediction on processed image."""
    if model is None:
        st.error("⚠️ Model not loaded. Check the model file.")
        return None, None

    processed = process_image(image_data)
    if processed is None:
        return None, None

    try:
        preds = model.predict(processed)
        idx = int(np.argmax(preds))
        conf = float(np.max(preds))
        # Ensure index is within disease_info range
        if idx >= len(disease_info):
            st.error("⚠️ Prediction out of bounds.")
            return None, None
        return idx, conf
    except Exception as e:
        st.error(f"🔥 Prediction Error: {e}")
        return None, None

# --------------------
# 7. PAGE DEFINITIONS
# --------------------
def landing_page():
    """
    Enhanced landing page with:
    • Glassmorphism effect
    • Animated elements
    • Better typography and spacing
    • Feature cards
    """
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    .stApp > header {
        display: none !important;
    }
    .landing-hero {
        background: rgba(10, 27, 9, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 3rem;
        margin: 2% auto;
        max-width: 1000px;
        box-shadow: 0 10px 32px rgba(100, 100,100, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        animation: fadeIn 1s ease-in;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 2; transform: translateY(0); }
    }

    .landing-hero h1 {
        font-family: 'Poppins', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #2E86C1, #27AE60);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }

    .landing-hero h2 {
        font-family: 'Poppins', sans-serif;
        font-size: 1.8rem;
        color: #34495E;
        margin-bottom: 2rem;
        font-weight: 600;
    }

    .landing-hero p {
        font-family: 'Poppins', sans-serif;
        font-size: 1.2rem;
        line-height: 1.8;
        color: #2C3E50;
        max-width: 800px;
        margin: 0 auto 3rem auto;
    }

    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 2rem;
        margin-top: 3rem;
        text-align: left;
    }

    .feature-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 15px;
        transition: transform 0.3s ease;
    }

    .feature-card:hover {
        transform: translateY(-5px);
    }

    .feature-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
    }

    .top-right-buttons {
        position: absolute;
        top: 2rem;
        right: 2rem;
        display: flex;
        gap: 1rem;
    }

    .top-right-buttons .stButton>button {
        background: rgba(39, 174, 96, 0.9);
        color: white;
        border: none;
        padding: 0.8em 2em;
        font-size: 1.1rem;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
        backdrop-filter: blur(5px);
    }

    .top-right-buttons .stButton>button:hover {
        background: rgba(46, 204, 113, 0.9);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(46, 204, 113, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

    # Hero Section
    # st.markdown('<div class="landing-hero">', unsafe_allow_html=True)
    st.markdown("<h1>🌱 Plant Disease Recognition</h1>", unsafe_allow_html=True)
    st.markdown("<h2>AI-Powered Plant Health Analysis</h2>", unsafe_allow_html=True)
    st.markdown("""
        <p>
        Transform your agricultural practice with our advanced AI system. Upload leaf images 
        for instant disease detection and receive expert treatment recommendations. 
        Protect your crops and enhance yields with precision technology.
        </p>
        
        <div class="feature-grid">
            <div class="feature-card">
                <div class="feature-icon">🔍</div>
                <h3>Instant Analysis</h3>
                <p>Quick and accurate disease detection using state-of-the-art AI</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">💡</div>
                <h3>Expert Solutions</h3>
                <p>Tailored treatment recommendations from agricultural experts</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <h3>Detailed Reports</h3>
                <p>Comprehensive analysis with confidence scores and insights</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Login/Signup Buttons
    st.markdown('<div class="top-right-buttons">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        login_clicked = st.button("Login")
    with col2:
        signup_clicked = st.button("Sign Up")
    st.markdown('</div>', unsafe_allow_html=True)

    if login_clicked:
        st.session_state["app_mode"] = "login"
        st.rerun()
    if signup_clicked:
        st.session_state["app_mode"] = "signup"
        st.rerun()

def send_password_reset_email(email: str):
    """
    Sends a password reset email to the specified email address using Firebase Admin SDK.
    """
    try:
        link = auth.generate_password_reset_link(email)
        # You can use an email service to send the link to the user
        # For simplicity, we'll just return the link here
        return {"link": link}
    except firebase_admin._auth_utils.UserNotFoundError:
        return {"error": f"User with email '{email}' not found."}
    except Exception as e:
        return {"error": str(e)}

def login_page():
    """
    Enhanced Login page with:
    • Modern glassmorphism effect
    • Better color contrast
    • Improved typography
    • Polished input fields
    """
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
    
    /* Remove default Streamlit padding */
    .stApp {
        padding-top: 0rem;
    }
    
    .login-container {
        max-width: 450px;
        margin: 2% auto;
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        animation: fadeIn 0.5s ease-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .login-container h1 {
        font-family: 'Poppins', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }

    .login-container p {
        color: #E0E0E0;
        font-family: 'Poppins', sans-serif;
        text-align: center;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }

    /* Input field styling */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.15);
        border: 2px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        font-size: 1rem;
        color: #FFFFFF;
        font-family: 'Poppins', sans-serif;
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }

    .stTextInput > div > div > input:focus {
        background: rgba(255, 255, 255, 0.2);
        border-color: #4CAF50;
        box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.2);
    }

    .stTextInput > div > div > input::placeholder {
        color: rgba(255, 255, 255, 0.7);
    }

    /* Label styling */
    .stTextInput label {
        font-family: 'Poppins', sans-serif;
        font-weight: 500;
        color: #FFFFFF;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
    }

    /* Checkbox styling */
    .stCheckbox {
        color: #FFFFFF;
        font-family: 'Poppins', sans-serif;
    }

    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8em 1.2em;
        font-size: 1.1rem;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        cursor: pointer;
        margin-top: 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.2);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.3);
        background: linear-gradient(135deg, #45a049, #4CAF50);
    }

    /* Link styling */
    a {
        color: #4CAF50;
        text-decoration: none;
        font-weight: 500;
        transition: color 0.3s ease;
    }

    a:hover {
        color: #45a049;
    }

    /* Divider styling */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(to right, transparent, rgba(255,255,255,0.3), transparent);
        margin: 2rem 0;
    }

    /* Social login container */
    .social-login {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-top: 1.5rem;
    }

    .social-login button {
        padding: 0.8rem 1.5rem;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.2);
        background: rgba(255,255,255,0.1);
        color: #FFFFFF;
        font-family: 'Poppins', sans-serif;
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .social-login button:hover {
        background: rgba(255,255,255,0.2);
        transform: translateY(-2px);
    }
    </style>
    """, unsafe_allow_html=True)

    # st.markdown('<div class="login-container">', unsafe_allow_html=True)
    
    st.markdown("<h1>🌿 Welcome Back</h1>", unsafe_allow_html=True)
    st.markdown("<p>Sign in to continue to your dashboard</p>", unsafe_allow_html=True)

    email = st.text_input("📧 Email Address", placeholder="Enter your email")
    password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")

    col1, col2 = st.columns(2)
    with col1:
        st.checkbox("Remember me", key="remember")
    with col2:
        if st.button("Forgot Password?"):
            st.session_state["app_mode"] = "forgot_password"
            st.rerun()

    login_clicked = st.button("Sign In")
    
    if login_clicked:
        with st.spinner("Authenticating..."):
            result = firebase_login(email, password)
            
        if "idToken" in result:
            st.success("✅ Login successful!")
            st.session_state["user"] = result
            st.session_state["app_mode"] = "home"
            st.rerun()
        else:
            st.error("❌ Invalid credentials")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p style='color: #E0E0E0; text-align: center;'>Don't have an account?</p>", unsafe_allow_html=True)
    
    if st.button("Create Account"):
        st.session_state["app_mode"] = "signup"
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

def forgot_password_page():
    """
    Forgot Password page to handle password reset requests.
    """
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
    
    .forgot-password-container {
        max-width: 450px;
        margin: 2% auto;
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        animation: fadeIn 0.5s ease-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .forgot-password-container h1 {
        font-family: 'Poppins', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }

    .forgot-password-container p {
        color: #E0E0E0;
        font-family: 'Poppins', sans-serif;
        text-align: center;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }

    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.15);
        border: 2px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        font-size: 1rem;
        color: #FFFFFF;
        font-family: 'Poppins', sans-serif;
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }

    .stTextInput > div > div > input:focus {
        background: rgba(255, 255, 255, 0.2);
        border-color: #4CAF50;
        box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.2);
    }

    .stTextInput > div > div > input::placeholder {
        color: rgba(255, 255, 255, 0.7);
    }

    .stTextInput label {
        font-family: 'Poppins', sans-serif;
        font-weight: 500;
        color: #FFFFFF;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8em 1.2em;
        font-size: 1.1rem;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        cursor: pointer;
        margin-top: 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.2);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.3);
        background: linear-gradient(135deg, #45a049, #4CAF50);
    }
    </style>
    """, unsafe_allow_html=True)

    # st.markdown('<div class="forgot-password-container">', unsafe_allow_html=True)
    
    st.markdown("<h1>🔒 Forgot Password</h1>", unsafe_allow_html=True)
    st.markdown("<p>Enter your email address to receive a password reset link</p>", unsafe_allow_html=True)

    email = st.text_input("📧 Email Address", placeholder="Enter your email")

    if st.button("Send Reset Link"):
        with st.spinner("Sending reset link..."):
            result = send_password_reset_email(email)
            
        if "link" in result:
            st.success("✅ Password reset link sent! Please check your email.")
            st.info(f"Reset Link: {result['link']}")  # For testing purposes, display the link
        else:
            st.error(f"❌ Error: {result.get('error', 'Unknown error')}")

    if st.button("Back to Login"):
        st.session_state["app_mode"] = "login"
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

def signup_page():
    """
    Enhanced Sign Up page with matching login page styling
    """
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
    
    .signup-container {
        max-width: 450px;
        margin: 2% auto;
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        animation: fadeIn 0.5s ease-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .signup-container h1 {
        font-family: 'Poppins', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }

    .signup-container p {
        color: #E0E0E0;
        font-family: 'Poppins', sans-serif;
        text-align: center;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }

    /* Input field styling */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.15);
        border: 2px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        font-size: 1rem;
        color: #FFFFFF;
        font-family: 'Poppins', sans-serif;
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }

    .stTextInput > div > div > input:focus {
        background: rgba(255, 255, 255, 0.2);
        border-color: #4CAF50;
        box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.2);
    }

    .stTextInput > div > div > input::placeholder {
        color: rgba(255, 255, 255, 0.7);
    }

    .stTextInput label {
        font-family: 'Poppins', sans-serif;
        font-weight: 500;
        color: #FFFFFF;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
    }

    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8em 1.2em;
        font-size: 1.1rem;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        cursor: pointer;
        margin-top: 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.2);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.3);
        background: linear-gradient(135deg, #45a049, #4CAF50);
    }

    /* Social login styling */
    .social-login {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-top: 1.5rem;
    }

    .social-login button {
        padding: 0.8rem 1.5rem;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.2);
        background: rgba(255,255,255,0.1);
        color: #FFFFFF;
        font-family: 'Poppins', sans-serif;
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .social-login button:hover {
        background: rgba(255,255,255,0.2);
        transform: translateY(-2px);
    }

    hr {
        border: none;
        height: 1px;
        background: linear-gradient(to right, transparent, rgba(255,255,255,0.3), transparent);
        margin: 2rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # st.markdown('<div class="signup-container">', unsafe_allow_html=True)
    
    st.markdown("<h1>🌿 Create Account</h1>", unsafe_allow_html=True)
    st.markdown("<p>Join us to start detecting plant diseases</p>", unsafe_allow_html=True)

    email = st.text_input("📧 Email Address", placeholder="Enter your email")
    password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
    confirm_password = st.text_input("🔒 Confirm Password", type="password", placeholder="Confirm your password")

    signup_clicked = st.button("Sign Up")
    if signup_clicked:
        if password != confirm_password:
            st.error("❌ Passwords do not match!")
        else:
            with st.spinner("Creating account..."):
                result = firebase_signup(email, password)
            
            if "idToken" in result:
                st.success("🎉 Account created successfully!")
                st.session_state["app_mode"] = "login"
                st.rerun()
            else:
                st.error(f"⚠️ Error: {result.get('error', 'Unknown error')}")


    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p style='color: #E0E0E0; text-align: center;'>Already have an account?</p>", unsafe_allow_html=True)
    
    if st.button("Login"):
        st.session_state["app_mode"] = "login"
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

def home_page():
    """
    Enhanced home page with modern design, cards, and better visual hierarchy
    """
    # Security check
    if st.session_state["user"] is None:
        st.session_state["app_mode"] = "landing"
        st.rerun()

    # Enhanced CSS
    st.markdown("""
    <style>
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.15) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.2);
    }

    [data-testid="stSidebar"] .stButton>button {
        width: 100%;
        background: rgba(39, 174, 96, 0.9);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.8em 1.2em;
        font-size: 1rem;
        font-family: 'Poppins', sans-serif;
        font-weight: 500;
        margin-bottom: 0.8rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    [data-testid="stSidebar"] .stButton>button:hover {
        background: rgba(46, 204, 113, 0.9);
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.2);
    }

    /* Main Content Styling */
    .main-container {
        max-width: 1200px;
        margin: 2rem auto;
        padding: 0 2rem;
    }

    .welcome-section {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        animation: fadeIn 0.5s ease-out;
    }

    .welcome-section h1 {
        font-family: 'Poppins', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 1rem;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }

    .welcome-section p {
        color: #E0E0E0;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }

    /* Feature Cards Grid */
    .features-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1.5rem;
        margin-top: 2rem;
    }

    .feature-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(5px);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease;
    }

    .feature-card:hover {
        transform: translateY(-5px);
    }

    .feature-card h3 {
        color: #FFFFFF;
        font-size: 1.3rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .feature-card p {
        color: #E0E0E0;
        font-size: 1rem;
        line-height: 1.6;
    }

    /* Stats Section */
    .stats-section {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.5rem;
        margin-top: 2rem;
    }

    .stat-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(5px);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
    }

    .stat-card h4 {
        color: #FFFFFF;
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }

    .stat-card p {
        color: #E0E0E0;
        font-size: 0.9rem;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar Navigation
    st.sidebar.title("Plant Disease Detection System")
    
    nav_options = {
        "Disease Recognition": ("🔍", "disease"),
        "About": ("ℹ️", "about"),
        "Feedback": ("📝", "feedback"),
        "View Feedback": ("📋", "view_feedback"),
        "Logout": ("🚪", "landing")
    }

    # Create navigation buttons
    for label, (icon, page) in nav_options.items():
        if st.sidebar.button(f"{icon} {label}", key=f"nav_{page}"):
            if label == "Logout":
                st.session_state["user"] = None
            st.session_state["app_mode"] = page
            st.rerun()

    # Main Content
    st.markdown("""
        <div class="welcome-section">
            <h1>🌿 Welcome to Plant Disease Recognition</h1>
            <p>Your AI-powered assistant for detecting and treating plant diseases</p>
        </div>
    """, unsafe_allow_html=True)

    # Featured Image
    st.image("home_page.jpeg", use_container_width=True)

    # Feature Cards
    st.markdown("""
        <div class="features-grid">
            <div class="feature-card">
                <h3>🔍 Disease Detection</h3>
                <p>Upload leaf images for instant disease analysis using advanced AI technology.</p>
            </div>
            <div class="feature-card">
                <h3>💡 Smart Solutions</h3>
                <p>Get personalized treatment recommendations and prevention tips.</p>
            </div>
            <div class="feature-card">
                <h3>📊 Detailed Reports</h3>
                <p>Access comprehensive analysis with confidence scores and insights.</p>
            </div>
            <div class="feature-card">
                <h3>🌱 Plant Health</h3>
                <p>Monitor and maintain the health of your plants with expert guidance.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Stats Section
    st.markdown("""
        <div class="stats-section">
            <div class="stat-card">
                <h4>15+</h4>
                <p>Plant Diseases Detected</p>
            </div>
            <div class="stat-card">
                <h4>98%</h4>
                <p>Detection Accuracy</p>
            </div>
            <div class="stat-card">
                <h4>24/7</h4>
                <p>Available Support</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)




def disease_recognition_page():
    """
    Enhanced Disease Recognition page with modern UI and better user experience
    """
    if st.session_state["user"] is None:
        st.session_state["app_mode"] = "landing"
        st.rerun()

    st.markdown("""
    <style>
    /* Main container */
    .main-container {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem auto;
        max-width: 1200px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }

    /* Page title */
    .page-title {
        font-family: 'Poppins', sans-serif;
        color: #FFFFFF;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }

    /* Two-column layout */
    .content-wrapper {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 2rem;
        margin-top: 2rem;
    }

    /* Upload section */
    .upload-section {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        height: fit-content;
    }

    /* Results section */
    .results-section {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* File uploader */
    .uploadfile {
        border: 2px dashed rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    .uploadfile:hover {
        border-color: #4CAF50;
        background: rgba(255, 255, 255, 0.05);
    }

    /* Image preview */
    .image-preview {
        margin-top: 2rem;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    .image-preview img {
        max-width: 350px !important;
        max-height: 400px !important;
        object-fit: contain;
        display: block;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8em 1.2em;
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.2);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.3);
        background: linear-gradient(135deg, #45a049, #4CAF50);
    }

    /* Results cards */
    .result-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .result-card h3 {
        color: #FFFFFF;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }

    /* Progress bar */
    .stProgress > div > div > div {
        background-color: #4CAF50 !important;
    }

    /* Treatment expander */
    .treatment-expander {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        margin-top: 1rem;
    }

    /* Instructions list */
    .instructions {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }

    .instructions h3 {
        color: #FFFFFF;
        margin-bottom: 1rem;
    }

    .instructions ul {
        color: #E0E0E0;
        margin-left: 1.5rem;
    }

    .instructions li {
        margin-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Main Container
    # st.markdown('<div class="main-container"s>', unsafe_allow_html=True)
    
    # Page Title
    st.markdown('<h1 class="page-title">🔍 Disease Recognition</h1>', unsafe_allow_html=True)

    # Content Wrapper
    st.markdown('<div class="content-wrapper">', unsafe_allow_html=True)

    # Left Column - Upload Section
    # st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    test_image = st.file_uploader("📸 Upload a leaf image", type=["jpg", "png", "jpeg"])
    
    if test_image:
        st.markdown("""
        <style>
        .image-preview {
            margin-top: 2rem;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .image-preview img {
            max-width: 350px !important;
            max-height: 400px !important;
            object-fit: contain;
            display: block;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="image-preview">', unsafe_allow_html=True)
        st.image(test_image, caption="Preview", use_container_width=False)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="uploadfile">
                <h3>Drag and drop your image here</h3>
                <p>Supported formats: JPG, PNG, JPEG</p>
            </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Right Column - Results Section
    # st.markdown('<div class="results-section">', unsafe_allow_html=True)
    
    # Instructions
    st.markdown("""
        <div class="instructions">
            <h3>📝 How to Use</h3>
            <ul>
                <li>Upload a clear, well-lit image of the plant leaf</li>
                <li>Ensure the affected area is visible in the image</li>
                <li>Click "Analyze" to detect any diseases</li>
                <li>View detailed results and treatment recommendations</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    if test_image:
        if st.button("🔍 Analyze Image"):
            with st.spinner("🔄 Analyzing your image..."):
                result_index, confidence = model_prediction(test_image)

            if result_index is not None:
                class_names = list(disease_info.keys())
                disease_name = class_names[result_index]
                plant_name = disease_info[disease_name]["plant"]
                solution = disease_info[disease_name]["solution"]

                # Results Cards
                st.markdown("""
                    <div class="result-card">
                        <h3>🌿 Plant Identified</h3>
                        <p style="color: #E0E0E0;">{}</p>
                    </div>
                """.format(plant_name), unsafe_allow_html=True)

                st.markdown("""
                    <div class="result-card">
                        <h3>🔍 Diagnosis</h3>
                        <p style="color: #E0E0E0;">{}</p>
                    </div>
                """.format(disease_name.replace('_', ' ')), unsafe_allow_html=True)

                # Confidence Score
                st.markdown("<h3 style='color: #FFFFFF; margin-top: 1rem;'>Confidence Score</h3>", unsafe_allow_html=True)
                st.progress(confidence)
                st.markdown(f"<p style='color: #E0E0E0; text-align: center;'>{confidence:.1%}</p>", unsafe_allow_html=True)

                # Treatment Recommendations
                with st.expander("💡 Treatment Recommendations"):
                    st.markdown(f"""
                        <div style='color: #E0E0E0;'>
                            <p>{solution}</p>
                        </div>
                    """, unsafe_allow_html=True)

                st.balloons()
            else:
                st.error("⚠️ Could not analyze the image. Please try again with a clearer photo.")

    if st.button("← Back to Home"):
        st.session_state["app_mode"] = "home"
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # Close content-wrapper
    st.markdown('</div>', unsafe_allow_html=True)  # Close main-container


def about_page():
    """
    About page with more engaging layout and additional info.
    """
    # Redirect if user not logged in
    if st.session_state["user"] is None:
        st.session_state["app_mode"] = "landing"
        st.rerun()

    st.markdown("""
    <style>
    .about-container {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem auto;
        max-width: 800px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }

    .about-title {
        font-family: 'Poppins', sans-serif;
        color: #FFFFFF;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }

    .about-content {
        font-family: 'Poppins', sans-serif;
        color: #E0E0E0;
        font-size: 1.1rem;
        line-height: 1.8;
        text-align: justify;
    }

    .about-content h3 {
        color: #FFFFFF;
        font-size: 1.5rem;
        margin-top: 1.5rem;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
    }

    .about-content ul {
        margin-left: 1.5rem;
    }

    .about-content li {
        margin-bottom: 0.5rem;
    }

    .expander-content {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
    }

    .info-box {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
        text-align: center;
    }

    .go-back-button {
        display: block;
        margin: 2rem auto 0;
        background: #4CAF50;
        color: white;
        padding: 0.8rem 1.5rem;
        border: none;
        border-radius: 10px;
        font-size: 1.1rem;
        cursor: pointer;
        transition: background 0.3s ease;
    }

    .go-back-button:hover {
        background: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)

    # st.markdown('<div class="about-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="about-title">ℹ️ About This Project</h1>', unsafe_allow_html=True)

    # Check if the image file exists before displaying it
    image_path = "about_banner.jpeg"
    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)
    else:
        st.warning("Banner image not found. Please ensure 'about_banner.jpeg' is in the correct location.")

    st.markdown("""
    <div class="about-content">
        This project leverages **deep learning** to identify plant diseases from leaf images. By analyzing thousands of plant samples, our AI model delivers **accurate disease predictions** along with confidence scores and recommended treatments.
        
        ### 🔍 Key Features:
        - ✅ Trained on an extensive dataset of plant leaf images
        - ✅ Provides precise disease predictions with high accuracy
        - ✅ Suggests effective treatment methods for various plant diseases
        - ✅ Simple and user-friendly interface for easy access
        
        With the power of AI, we aim to help farmers and agriculturists **detect diseases early** and take **preventive actions** to ensure healthy crops.
    </div>
    """, unsafe_allow_html=True)

    with st.expander("References & Further Reading"):
        st.markdown(
            """
            - [FAO: Plant Health](http://www.fao.org/plant-health/en/)
            - [Research Paper: Deep Learning for Plant Disease Detection](https://arxiv.org/abs/1604.03169)
            - [GitHub Repository](https://github.com/your-username/plant-disease-recognition) *(if applicable)*
            """
        )

    st.markdown("""
    <div class="info-box">
        We welcome contributions and feedback! Feel free to reach out or submit pull requests.
    </div>
    """, unsafe_allow_html=True)

    if st.button("Go Back", key="go_back"):
        st.session_state["app_mode"] = "home"
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


def feedback_page():
    """
    A more detailed Feedback page.
    Allows users to rate their experience, enter comments, and optionally provide their name.
    """
    # Redirect if user not logged in
    if st.session_state["user"] is None:
        st.session_state["app_mode"] = "landing"
        st.rerun()

    st.markdown("""
    <style>
    .feedback-container {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem auto;
        max-width: 800px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }

    .feedback-title {
        font-family: 'Poppins', sans-serif;
        color: #FFFFFF;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }

    .feedback-content {
        font-family: 'Poppins', sans-serif;
        color: #E0E0E0;
        font-size: 1.1rem;
        line-height: 1.8;
        text-align: justify;
    }

    .feedback-content h3 {
        color: #FFFFFF;
        font-size: 1.5rem;
        margin-top: 1.5rem;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
    }

    .feedback-content ul {
        margin-left: 1.5rem;
    }

    .feedback-content li {
        margin-bottom: 0.5rem;
    }

    .info-box {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
        text-align: center;
    }

    .go-back-button {
        display: block;
        margin: 2rem auto 0;
        background: #4CAF50;
        color: white;
        padding: 0.8rem 1.5rem;
        border: none;
        border-radius: 10px;
        font-size: 1.1rem;
        cursor: pointer;
        transition: background 0.3s ease;
    }

    .go-back-button:hover {
        background: #45a049;
    }

    .submit-button {
        display: block;
        margin: 2rem auto 0;
        background: #4CAF50;
        color: white;
        padding: 0.8rem 1.5rem;
        border: none;
        border-radius: 10px;
        font-size: 1.1rem;
        cursor: pointer;
        transition: background 0.3s ease;
    }

    .submit-button:hover {
        background: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="feedback-title">Feedback & Suggestions</h1>', unsafe_allow_html=True)
    st.markdown('<p class="feedback-content">We value your feedback! Please share your thoughts on how we can improve.</p>', unsafe_allow_html=True)

    # Optional: Show user email if they're logged in
    user_email = st.session_state["user"].get("email", "Unknown") if st.session_state["user"] else "Guest"
    st.info(f"You are logged in as: **{user_email}**")

    # Rating Section
    st.markdown('<h3 class="feedback-content">Rate Your Experience</h3>', unsafe_allow_html=True)
    rating = st.slider("On a scale of 1-5, how would you rate your overall experience?", min_value=1, max_value=5, value=3)

    # Comments Section
    st.markdown('<h3 class="feedback-content">Your Comments</h3>', unsafe_allow_html=True)
    feedback_text = st.text_area("Please share any additional comments or suggestions:")

    # Optional Name Field
    name = st.text_input("Your Name (optional)")

    # Submit Button
    if st.button("Submit Feedback", key="submit_feedback"):
        if feedback_text.strip():
            # Store the feedback in Firestore
            db.collection("feedback").add({
                "email": user_email,
                "name": name,
                "rating": rating,
                "comments": feedback_text,
                "timestamp": datetime.utcnow()
            })

            st.success("Thank you for your feedback!")
            st.balloons()
        else:
            st.error("Please enter some feedback before submitting.")

    # Contact Us Section
    contact_name = st.text_input("Your Name", key="contact_name")
    contact_email = st.text_input("Your Email", value=user_email if user_email != "Guest" else "", key="contact_email")
    contact_message = st.text_area("Your Message", key="contact_message")

    if st.button("Send Message", key="send_message"):
        if not contact_name.strip() or not contact_email.strip() or not contact_message.strip():
            st.error("Please fill out all fields before sending.")
        else:
            # Store the contact message in Firestore
            db.collection("contacts").add({
                "name": contact_name,
                "email": contact_email,
                "message": contact_message,
                "timestamp": datetime.utcnow()
            })
            st.success("Thank you! Your message has been sent.")
            st.balloons()

    # Go Back Button
    if st.button("Go Back", key="go_back"):
        st.session_state["app_mode"] = "home"
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


def view_feedback_page():
    """
    Page to view all feedback and messages.
    """
    # Redirect if user not logged in
    if st.session_state["user"] is None:
        st.session_state["app_mode"] = "landing"
        st.rerun()

    st.title("📋 View Feedback & Messages")

    # Retrieve feedback from Firestore
    try:
        feedback_docs = db.collection("feedback").order_by("timestamp", direction=firestore.Query.DESCENDING).stream()
        contact_docs = db.collection("contacts").order_by("timestamp", direction=firestore.Query.DESCENDING).stream()

        st.markdown("""
        <style>
        .feedback-card, .contact-card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .feedback-card h4, .contact-card h4 {
            color: #FFFFFF;
            margin-bottom: 0.5rem;
        }
        .feedback-card p, .contact-card p {
            color: #E0E0E0;
            margin-bottom: 0.5rem;
        }
        .feedback-card .timestamp, .contact-card .timestamp {
            color: #B0B0B0;
            font-size: 0.9rem;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("## Feedback")
        feedback_count = 0
        for doc in feedback_docs:
            feedback = doc.to_dict()
            st.markdown(f"""
            <div class="feedback-card">
                <h4>{feedback.get('name', 'Anonymous')}</h4>
                <p><strong>Email:</strong> {feedback.get('email')}</p>
                <p><strong>Rating:</strong> {feedback.get('rating')}</p>
                <p><strong>Comments:</strong> {feedback.get('comments')}</p>
                <p class="timestamp"><strong>Timestamp:</strong> {feedback.get('timestamp').strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            """, unsafe_allow_html=True)
            feedback_count += 1

        if feedback_count == 0:
            st.info("No feedback available.")

        st.markdown("## Messages")
        contact_count = 0
        for doc in contact_docs:
            contact = doc.to_dict()
            st.markdown(f"""
            <div class="contact-card">
                <h4>{contact.get('name')}</h4>
                <p><strong>Email:</strong> {contact.get('email')}</p>
                <p><strong>Message:</strong> {contact.get('message')}</p>
                <p class="timestamp"><strong>Timestamp:</strong> {contact.get('timestamp').strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            """, unsafe_allow_html=True)
            contact_count += 1

        if contact_count == 0:
            st.info("No messages available.")

    except Exception as e:
        st.error(f"Error retrieving data: {e}")

    # Go Back Button
    if st.button("Go Back"):
        st.session_state["app_mode"] = "home"
        st.rerun()


# -----------------------
# 8. MAIN APP CONTROLLER
# -----------------------
def main():
    mode = st.session_state["app_mode"]

    if mode == "landing":
        landing_page()
    elif mode == "login":
        login_page()
    elif mode == "signup":
        signup_page()
    elif mode == "home":
        home_page()
    elif mode == "disease":
        disease_recognition_page()
    elif mode == "about":
        about_page()
    elif mode == "feedback":
        feedback_page()
    elif mode == "view_feedback":
        view_feedback_page()
    elif mode == "forgot_password":
        forgot_password_page()
    else:
        st.session_state["app_mode"] = "landing"
        st.rerun()

if __name__ == "__main__":
    main()
