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
    page_title="Plant Disease Recognition",
    layout="centered",
    initial_sidebar_state="expanded",
    page_icon="🌿"
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
        "app_mode": "landing",  # start at landing page
        "prediction_history": []  # store previous predictions
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
    • Loads Montserrat (for headings) & Roboto (for body text) from Google Fonts
    • Applies modern design elements with smooth transitions
    • Improved color scheme and visual hierarchy
    """
    custom_css = """
    <style>
    /* 1. Import Montserrat & Roboto from Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&family=Roboto:wght@300;400;500;700&display=swap');

    /* 2. Base styles */
    * {
        box-sizing: border-box;
    }

    /* 3. Apply Montserrat to all headings (h1 - h6) */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Montserrat', sans-serif;
        font-weight: 600;
        color: #2C7A7B; /* Teal shade */
        margin-top: 0.5em;
        margin-bottom: 0.5em;
        letter-spacing: -0.01em;
    }

    /* 4. Use Roboto for general body text and common elements */
    body, p, div, span, input, button, textarea, label, select {
        font-family: 'Roboto', sans-serif;
        color: #2D3748;
    }

    /* 5. Customize Streamlit buttons with modern styling */
    .stButton>button {
        background: linear-gradient(135deg, #38B2AC, #319795);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        font-size: 1rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(49, 151, 149, 0.1);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 14px rgba(49, 151, 149, 0.2);
        background: linear-gradient(135deg, #4FD1C5, #38B2AC);
    }
    .stButton>button:active {
        transform: translateY(1px);
    }

    /* 6. Card-like container styling */
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1);
    }

    /* 7. Input fields styling */
    input[type="text"], input[type="password"], input[type="email"], textarea {
        width: 100%;
        padding: 10px 15px;
        margin: 8px 0;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        box-sizing: border-box;
        transition: border-color 0.3s;
        background-color: #F7FAFC;
    }
    input[type="text"]:focus, input[type="password"]:focus, input[type="email"]:focus, textarea:focus {
        border-color: #38B2AC;
        outline: none;
        box-shadow: 0 0 0 3px rgba(56, 178, 172, 0.2);
    }

    /* 8. Progress bar styling */
    div[data-testid="stProgressBar"] > div > div {
        background-color: #38B2AC !important;
        background: linear-gradient(90deg, #4FD1C5, #38B2AC) !important;
        border-radius: 10px !important;
    }

    /* 9. Alert/message box styling */
    .stAlert {
        border-radius: 10px !important;
        border: none !important;
        padding: 16px !important;
    }
    
    /* 10. Sidebar styling */
    [data-testid="stSidebar"] {
        background-image: linear-gradient(180deg, rgba(38, 198, 218, 0.1) 0%, rgba(0, 150, 136, 0.1) 100%);
        border-right: 1px solid rgba(0, 150, 136, 0.1);
    }
    [data-testid="stSidebar"] .stButton>button {
        width: 100%;
        margin-bottom: 8px;
    }
    
    /* 11. Improved file uploader */
    [data-testid="stFileUploader"] {
        padding: 20px;
        border-radius: 10px;
        border: 2px dashed #38B2AC;
        background-color: rgba(56, 178, 172, 0.05);
        text-align: center;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploader"]:hover {
        background-color: rgba(56, 178, 172, 0.1);
    }
    
    /* 12. Table styling */
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.05);
    }
    thead tr {
        background-color: #38B2AC;
        color: white;
        text-align: left;
    }
    th, td {
        padding: 12px 15px;
    }
    tbody tr {
        border-bottom: 1px solid #E2E8F0;
    }
    tbody tr:nth-of-type(even) {
        background-color: #F7FAFC;
    }
    tbody tr:last-of-type {
        border-bottom: 2px solid #38B2AC;
    }
    
    /* 13. Footer styling */
    .footer {
        text-align: center;
        padding: 20px 0;
        font-size: 0.9rem;
        color: #718096;
        border-top: 1px solid #E2E8F0;
        margin-top: 30px;
    }
    
    /* 14. Custom badges */
    .badge {
        display: inline-block;
        padding: 0.35em 0.65em;
        font-size: 0.75em;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.25rem;
        margin-right: 5px;
    }
    .badge-primary {
        color: #fff;
        background-color: #38B2AC;
    }
    .badge-secondary {
        color: #fff;
        background-color: #718096;
    }
    
    /* 15. Responsive adjustments */
    @media (max-width: 768px) {
        .card {
            padding: 1rem;
        }
        h1 {
            font-size: 1.8rem;
        }
        h2 {
            font-size: 1.5rem;
        }
    }
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
background_image = get_base64_image("Background.jpg")

if background_image:
    page_bg_img = f"""
    <style>
    .stApp {{
        background: linear-gradient(
            rgba(0, 0, 0, 0.5),
            rgba(0, 0, 0, 0.5)
        ),
        url("data:image/jpg;base64,{background_image}") no-repeat center center fixed;
        background-size: cover;
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
        "solution": "Use copper-based bactericides and avoid overhead watering. Ensure good air circulation between plants and practice crop rotation.",
        "severity": "High",
        "organic_remedy": "Neem oil spray or copper-based organic fungicide"
    },
    "Pepper_bell_healthy": {
        "plant": "Pepper Bell",
        "solution": "The plant is healthy. Maintain proper watering and nutrient levels.",
        "severity": "None",
        "organic_remedy": "Regular compost application and balanced organic fertilizer"
    },
    "Potato_Early_blight": {
        "plant": "Potato",
        "solution": "Use fungicides with chlorothalonil and practice crop rotation. Remove and destroy infected plant material.",
        "severity": "Medium",
        "organic_remedy": "Copper sulfate spray and remove lower infected leaves"
    },
    "Potato_healthy": {
        "plant": "Potato",
        "solution": "The plant is healthy. Ensure good soil drainage and proper spacing between plants.",
        "severity": "None",
        "organic_remedy": "Add organic matter to soil and use mulch to retain moisture"
    },
    "Potato_Late_blight": {
        "plant": "Potato",
        "solution": "Use copper-based fungicides and remove affected leaves. Ensure good air circulation and avoid watering in the evening.",
        "severity": "High",
        "organic_remedy": "Milk spray (1:9 ratio with water) and copper fungicide"
    },
    "Tomato_Target_Spot": {
        "plant": "Tomato",
        "solution": "Apply fungicides like chlorothalonil and avoid excess moisture. Prune lower branches to improve air circulation.",
        "severity": "Medium",
        "organic_remedy": "Baking soda solution (1 tbsp in 1 gallon water with a few drops of dish soap)"
    },
    "Tomato_Tomato_mosaic_virus": {
        "plant": "Tomato",
        "solution": "Remove infected plants immediately and control aphids. There is no cure, so prevention is key.",
        "severity": "High",
        "organic_remedy": "No cure available. Use insecticidal soap for aphid control"
    },
    "Tomato_Tomato_YellowLeaf_Curl_Virus": {
        "plant": "Tomato",
        "solution": "Control whiteflies and use virus-resistant varieties. Remove infected plants to prevent spread.",
        "severity": "Very High",
        "organic_remedy": "Yellow sticky traps for whiteflies and reflective mulch"
    },
    "Tomato_Bacterial_spot": {
        "plant": "Tomato",
        "solution": "Use copper sprays and avoid overhead irrigation. Remove infected leaves and maintain good sanitation.",
        "severity": "High",
        "organic_remedy": "Copper-based organic fungicide and crop rotation"
    },
    "Tomato_Early_blight": {
        "plant": "Tomato",
        "solution": "Use chlorothalonil-based fungicides regularly and remove lower infected leaves promptly.",
        "severity": "Medium",
        "organic_remedy": "Compost tea spray and remove infected leaves"
    },
    "Tomato_healthy": {
        "plant": "Tomato",
        "solution": "The plant is healthy. Maintain good cultivation practices and regular monitoring.",
        "severity": "None",
        "organic_remedy": "Seaweed extract as foliar spray and regular organic fertilization"
    },
    "Tomato_Late_blight": {
        "plant": "Tomato",
        "solution": "Apply fungicides containing mancozeb and remove infected leaves. Ensure good spacing between plants.",
        "severity": "Very High",
        "organic_remedy": "Copper spray and removal of infected plant material"
    },
    "Tomato_Leaf_Mold": {
        "plant": "Tomato",
        "solution": "Ensure proper ventilation, reduce humidity and use fungicides if needed. Remove infected leaves.",
        "severity": "Medium",
        "organic_remedy": "Garlic spray (4 cloves blended with 1 liter water) and improved ventilation"
    },
    "Tomato_Septoria_leaf_spot": {
        "plant": "Tomato",
        "solution": "Remove infected leaves and use copper-based fungicides. Mulch around plants to prevent splash-up.",
        "severity": "Medium",
        "organic_remedy": "Organic copper fungicide and thick organic mulch"
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "plant": "Tomato",
        "solution": "Use neem oil or insecticidal soaps to control mites. Increase humidity as mites prefer dry conditions.",
        "severity": "Medium",
        "organic_remedy": "Neem oil spray or introducing predatory mites"
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
        
        # Get class names from the disease_info dictionary
        class_names = list(disease_info.keys())
        
        # Ensure index is within class_names range
        if idx >= len(class_names):
            st.error("⚠️ Prediction out of bounds.")
            return None, None
            
        return class_names[idx], conf
    except Exception as e:
        st.error(f"🔥 Prediction Error: {e}")
        return None, None

def get_severity_color(severity):
    """Return color based on severity level"""
    severity_colors = {
        "None": "#4CAF50",      # Green for healthy
        "Low": "#8BC34A",       # Light Green
        "Medium": "#FFC107",    # Amber/Yellow
        "High": "#FF9800",      # Orange
        "Very High": "#F44336"  # Red for severe
    }
    return severity_colors.get(severity, "#9E9E9E")  # Gray as default

# Function to save prediction to history
def save_prediction(plant, disease, confidence, solution):
    """Save prediction to session state history"""
    if len(st.session_state.prediction_history) >= 10:
        # Keep only the 9 most recent predictions
        st.session_state.prediction_history = st.session_state.prediction_history[1:]
    
    # Add the new prediction
    st.session_state.prediction_history.append({
        "plant": plant,
        "disease": disease,
        "confidence": confidence,
        "solution": solution,
        "timestamp": "Today"  # You could use datetime here for actual timestamps
    })

# --------------------
# 7. PAGE DEFINITIONS
# --------------------
def landing_page():
    """
    An enhanced landing page with:
    • Animated hero section
    • Feature highlights with icons
    • Call-to-action buttons
    """

    # -- Inject custom CSS for animations and layout --
    st.markdown("""
    <style>
    /* Animation keyframes */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Landing page container */
    .landing-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
        animation: fadeIn 0.8s ease-out;
    }
    
    /* Hero section styling */
    .hero {
        text-align: center;
        padding: 3rem 1rem;
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    }
    
    .hero h1 {
        font-size: 3rem;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #2C7A7B, #319795);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .hero h2 {
        font-size: 1.5rem;
        font-weight: 400;
        color: #4A5568;
        margin-bottom: 2rem;
    }
    
    /* Feature grid */
    .features {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 2rem;
        margin-bottom: 3rem;
    }
    
    .feature-card {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 12px 20px rgba(0, 0, 0, 0.1);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        color: #38B2AC;
    }
    
    .feature-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #2D3748;
    }
    
    /* CTA section */
    .cta {
        text-align: center;
        padding: 3rem;
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 16px;
        margin-bottom: 2rem;
    }
    
    .btn-container {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-top: 2rem;
    }
    
    .btn {
        display: inline-block;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        font-weight: 600;
        text-align: center;
        text-decoration: none;
        border-radius: 8px;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .btn-primary {
        background: linear-gradient(135deg, #38B2AC, #319795);
        color: white;
        border: none;
        box-shadow: 0 4px 6px rgba(49, 151, 149, 0.2);
    }
    
    .btn-primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 14px rgba(49, 151, 149, 0.3);
    }
    
    .btn-secondary {
        background: transparent;
        color: #38B2AC;
        border: 2px solid #38B2AC;
    }
    
    .btn-secondary:hover {
        background-color: rgba(56, 178, 172, 0.1);
        transform: translateY(-2px);
    }
    
    /* Animation classes for staggered appearance */
    .animate-1 { animation: fadeIn 0.5s ease-out 0.2s both; }
    .animate-2 { animation: fadeIn 0.5s ease-out 0.4s both; }
    .animate-3 { animation: fadeIn 0.5s ease-out 0.6s both; }
    </style>
    """, unsafe_allow_html=True)

    # -- Hero Section --
    st.markdown("""
    <div class="landing-container">
        <div class="hero">
            <h1>🌿 Plant Disease Recognition</h1>
            <h2>Protect your crops with AI-powered disease detection</h2>
            <p>Upload a photo of your plant leaf and get instant disease diagnosis with treatment recommendations</p>
        </div>
        
        <div class="features">
            <div class="feature-card animate-1">
                <div class="feature-icon">🔍</div>
                <div class="feature-title">Accurate Detection</div>
                <p>Our AI model analyzes leaf images to identify 15+ diseases with high accuracy across multiple crop types.</p>
            </div>
            
            <div class="feature-card animate-2">
                <div class="feature-icon">💊</div>
                <div class="feature-title">Treatment Guidance</div>
                <p>Get specific recommendations for both conventional and organic treatment options to address plant issues.</p>
            </div>
            
            <div class="feature-card animate-3">
                <div class="feature-icon">📱</div>
                <div class="feature-title">Easy to Use</div>
                <p>Simply upload a photo from your device and receive instant results that help you take action quickly.</p>
            </div>
        </div>
        
        <div class="cta">
            <h2>Ready to protect your plants?</h2>
            <p>Join thousands of farmers and gardeners using our platform to detect and treat plant diseases early.</p>
            <div class="btn-container">
                <div class="btn btn-primary" id="signup-btn">Get Started</div>
                <div class="btn btn-secondary" id="login-btn">Login</div>
            </div>
        </div>
    </div>
    
    <script>
        // JavaScript to handle button clicks
        document.getElementById('signup-btn').addEventListener('click', function() {
            // This will be handled by Streamlit
        });
        document.getElementById('login-btn').addEventListener('click', function() {
            // This will be handled by Streamlit
        });
    </script>
    """, unsafe_allow_html=True)

    # -- Streamlit Buttons (Hidden but functional) --
    col1, col2 = st.columns(2)
    with col1:
        login_clicked = st.button("Login", key="login_hidden")
    with col2:
        signup_clicked = st.button("Sign Up", key="signup_hidden")

    # -- Button Logic --
    if login_clicked:
        st.session_state["app_mode"] = "login"
        st.rerun()

    if signup_clicked:
        st.session_state["app_mode"] = "signup"
        st.rerun()


def login_page():
    """
    Enhanced Login page with a mock password check (Firebase Admin SDK).
    Includes a 'Sign Up' button for easy navigation if the user doesn't have an account.
    """

    # -- Inject custom CSS for a nicer layout --
    st.markdown("""
    <style>
    /* Container to center the login form and give it a card-like look */
    .login-container {
        max-width: 400px;        /* Limit width for readability */
        margin: 5% auto;         /* Center horizontally, 5% from the top */
        background-color: #f9f9f9;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 0 15px rgba(0,0,0,0.1);
        text-align: center;
    }

    /* Main heading style */
    .login-container h1 {
        margin-bottom: 1rem;
        color: #2E86C1; /* Example: deep blue */
    }

    /* Style the input labels for clarity */
    label {
        font-weight: 600;
        color: #333333;
    }

    /* Add spacing under the inputs */
    .stTextInput {
        margin-bottom: 1rem;
    }

    /* Button styling overrides */
    .login-container .stButton>button {
        width: 100%;             /* Make buttons full-width */
        background-color: #27AE60;
        color: #FFFFFF;
        border: none;
        border-radius: 6px;
        padding: 0.6em 1.2em;
        font-size: 1rem;
        cursor: pointer;
        transition: background-color 0.3s ease, color 0.3s ease;
        margin-top: 0.5rem;
    }
    .login-container .stButton>button:hover {
        background-color: #2ECC71;
        color: #000000;
    }

    /* Subtle divider line */
    hr {
        border: none;
        border-top: 1px solid #ddd;
        margin: 1.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # -- Container for the login form --
    st.markdown('<div class="login-container">', unsafe_allow_html=True)

    # -- Heading --
    st.markdown("<h1>🔑 Login</h1>", unsafe_allow_html=True)

    # -- Input Fields --
    email = st.text_input("📧 Email")
    password = st.text_input("🔒 Password", type="password")

    # -- Login Button --
    login_clicked = st.button("Login")
    if login_clicked:
        with st.spinner("Authenticating..."):
            result = firebase_login(email, password)

        if "idToken" in result:
            st.success("✅ Logged in successfully!")
            st.session_state["user"] = result
            st.session_state["app_mode"] = "home"
            st.rerun()
        else:
            st.error("❌ Login failed. Check your credentials or sign up.")

    # -- Divider & Sign Up Prompt --
    st.markdown("<hr>", unsafe_allow_html=True)
    st.write("Don't have an account?")

    signup_clicked = st.button("Sign Up")
    if signup_clicked:
        st.session_state["app_mode"] = "signup"
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


def signup_page():
    """
    Enhanced Sign Up page with a card-like container, matching the login page style.
    Uses Firebase Admin SDK (mock password check) to create a new user.
    """

    # -- Inject custom CSS for a nicer layout --
    st.markdown("""
    <style>
    /* Container to center the sign-up form and give it a card-like look */
    .signup-container {
        max-width: 400px;        /* Limit width for readability */
        margin: 5% auto;         /* Center horizontally, 5% from the top */
        background-color: #f9f9f9;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 0 15px rgba(0,0,0,0.1);
        text-align: center;
    }

    /* Main heading style */
    .signup-container h1 {
        margin-bottom: 1rem;
        color: #2E86C1; /* Example: deep blue */
    }

    /* Style the input labels for clarity */
    label {
        font-weight: 600;
        color: #333333;
    }

    /* Add spacing under the inputs */
    .stTextInput {
        margin-bottom: 1rem;
    }

    /* Button styling overrides */
    .signup-container .stButton>button {
        width: 100%;             /* Make buttons full-width */
        background-color: #27AE60;
        color: #FFFFFF;
        border: none;
        border-radius: 6px;
        padding: 0.6em 1.2em;
        font-size: 1rem;
        cursor: pointer;
        transition: background-color 0.3s ease, color 0.3s ease;
        margin-top: 0.5rem;
    }
    .signup-container .stButton>button:hover {
        background-color: #2ECC71;
        color: #000000;
    }

    /* Subtle divider line */
    hr {
        border: none;
        border-top: 1px solid #ddd;
        margin: 1.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # -- Container for the sign-up form --
    st.markdown('<div class="signup-container">', unsafe_allow_html=True)

    # -- Heading --
    st.markdown("<h1>📝 Create an Account</h1>", unsafe_allow_html=True)

    # -- Input Fields --
    email = st.text_input("📧 Email")
    password = st.text_input("🔒 Password", type="password")

    # -- Sign Up Button --
    signup_clicked = st.button("Sign Up")
    if signup_clicked:
        with st.spinner("Creating account..."):
            result = firebase_signup(email, password)

        if "idToken" in result:
            st.success("🎉 Account created successfully! Please login.")
            st.session_state["app_mode"] = "login"
            st.rerun()
        else:
            st.error(f"⚠️ Error: {result.get('error', 'Unknown error')}")

    # -- Divider & Login Prompt --
    st.markdown("<hr>", unsafe_allow_html=True)
    st.write("Already have an account?")

    login_clicked = st.button("Login")
    if login_clicked:
        st.session_state["app_mode"] = "login"
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

def home_page():
    """
    Home page with a transparent, attractive sidebar for navigation 
    and a centered card-like container for main content.
    """

    # 1. Redirect to landing if not logged in
    if st.session_state["user"] is None:
        st.session_state["app_mode"] = "landing"
        st.rerun()

    # 2. Inject custom CSS for the sidebar & main container
    st.markdown("""
    <style>
    /* -------------- SIDEBAR STYLING -------------- */
    /* Make the sidebar semi-transparent with a blur effect */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.1) !important; /* Light overlay */
        backdrop-filter: blur(8px); /* Blur behind the sidebar */
    }

    /* Style the navigation buttons in the sidebar */
    [data-testid="stSidebar"] .stButton>button {
        background-color: #27AE60;
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        font-size: 1rem;
        cursor: pointer;
        margin-bottom: 0.5rem; /* space between buttons */
        transition: background-color 0.3s ease, color 0.3s ease;
    }
    [data-testid="stSidebar"] .stButton>button:hover {
        background-color: #2ECC71;
        color: #000000;
    }

    /* -------------- MAIN CONTENT CONTAINER -------------- */
    .home-container {
        max-width: 800px;         /* Limit width for readability */
        margin: 2rem auto;        /* Center horizontally, some spacing from top */
        background-color: #ffffff; 
        padding: 2rem;
        border-radius: 8px;
        box-shadow: 0 0 15px rgba(0,0,0,0.1);
        text-align: center;
    }
    .home-container h1 {
        color: #2E86C1;           /* Example: deep blue heading */
        margin-bottom: 1rem;
    }
    .home-container p {
        line-height: 1.6;
    }
    .home-container img {
        margin-bottom: 1rem;
        border-radius: 8px;       /* Slightly rounded corners on the image */
    }
    </style>
    """, unsafe_allow_html=True)

    # 3. SIDEBAR NAVIGATION
    st.sidebar.title("Navigation")

    # Buttons for each page in the sidebar
    if st.sidebar.button("Disease Recognition"):
        st.session_state["app_mode"] = "disease"
        st.rerun()
    if st.sidebar.button("About"):
        st.session_state["app_mode"] = "about"
        st.rerun()
    if st.sidebar.button("Feedback"):
        st.session_state["app_mode"] = "feedback"
        st.rerun()
    if st.sidebar.button("Contact"):
        st.session_state["app_mode"] = "contact"
        st.rerun()

    # Logout button in the sidebar
    if st.sidebar.button("Logout"):
        st.session_state["user"] = None
        st.session_state["app_mode"] = "landing"
        st.rerun()

    # 4. MAIN CONTENT (CARD-LIKE CONTAINER)
    st.markdown('<div class="home-container">', unsafe_allow_html=True)
    st.markdown("<h1>Home Page</h1>", unsafe_allow_html=True)
    st.write("Welcome to the Plant Disease Recognition System!")

    # Display your home_page.jpeg here
    st.image("home_page.jpeg", use_container_width=True)

    # Short list of features or instructions
    st.markdown(
        """
        <ul style="text-align: left;">
        <li><strong>Disease Recognition:</strong> Upload a leaf image to detect potential diseases.</li>
        <li><strong>About:</strong> Learn more about this project.</li>
        <li><strong>Feedback:</strong> Submit your feedback.</li>
        <li><strong>Contact:</strong> Contact us for more information.</li>
        </ul>
        """,
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)




def disease_recognition_page():
    """
    Disease Recognition page with a two-column layout:
    • Left column: File uploader and image preview
    • Right column: Instructions, Predict button, and results
    """

    # 1. Ensure user is logged in
    if st.session_state["user"] is None:
        st.session_state["app_mode"] = "landing"
        st.rerun()

    # -- Inject custom CSS for the two-column layout and styling --
    st.markdown("""
    <style>
    /* Main container for disease recognition page */
    .disease-container {
        display: flex;
        flex-direction: row;
        gap: 2rem;                /* Space between left & right columns */
        max-width: 1200px;        /* Limit total width for readability */
        margin: 2rem auto;        /* Center horizontally, some spacing from top */
        background-color: #ffffff;
        border-radius: 8px;
        box-shadow: 0 0 15px rgba(0,0,0,0.1);
        padding: 2rem;
    }

    /* Left column for uploader & image preview */
    .left-col {
        flex: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
        border-right: 1px solid #eee; /* Subtle vertical divider */
        padding-right: 1rem;
    }

    /* Right column for instructions, Predict button, and results */
    .right-col {
        flex: 1;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        padding-left: 1rem;
    }

    /* Headings and text styling */
    .disease-container h1, .disease-container h2, .disease-container h3 {
        color: #2E86C1; /* Example: deep blue */
        margin-bottom: 0.5rem;
    }
    .disease-container p {
        line-height: 1.6;
        margin-bottom: 1rem;
    }

    /* Buttons within the right column */
    .right-col .stButton>button {
        width: 100%;             /* Make Predict & Go Back full-width */
        background-color: #27AE60;
        color: #FFFFFF;
        border: none;
        border-radius: 6px;
        padding: 0.6em 1.2em;
        font-size: 1rem;
        cursor: pointer;
        margin-bottom: 1rem;     /* Space between buttons/results */
        transition: background-color 0.3s ease, color 0.3s ease;
    }
    .right-col .stButton>button:hover {
        background-color: #2ECC71;
        color: #000000;
    }

    /* Info/Warning boxes margin adjustments */
    .stInfo, .stWarning, .stError, .stSuccess {
        margin-bottom: 1rem;
    }

    /* Progress bar color override */
    div[data-testid="stProgressBar"] > div > div {
        background-color: #2ECC71 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # -- Page Title / Heading --
    st.title("Disease Recognition")

    # -- Create the main container --
    st.markdown('<div class="disease-container">', unsafe_allow_html=True)

    # ----------------- LEFT COLUMN -----------------
    st.markdown('<div class="left-col">', unsafe_allow_html=True)

    # File Uploader
    test_image = st.file_uploader("Upload a leaf image (JPG, PNG, or JPEG):", type=["jpg", "png", "jpeg"])
    
    # Preview the uploaded image
    if test_image is not None:
    # Create three columns with the middle column wider to center the image.
        cols = st.columns([1, 2, 1])
        with cols[1]:
            st.image(test_image, caption="Uploaded Image", width=300)
    else:
        st.warning("No image uploaded yet.")

    st.markdown('</div>', unsafe_allow_html=True)  # Close left-col

    # ----------------- RIGHT COLUMN -----------------
    st.markdown('<div class="right-col">', unsafe_allow_html=True)

    # Instructions
    st.subheader("How to Use:")
    st.markdown("""
    1. Ensure the leaf is clearly visible and well-lit.<br>
    2. Upload your image on the left.<br>
    3. Click **Predict** to analyze the disease.<br>
    4. View results below, including confidence score and treatment tips.
    """, unsafe_allow_html=True)

    # If image is uploaded, show Predict button; else hide
    if test_image is not None:
        predict_clicked = st.button("Predict")
        if predict_clicked:
            with st.spinner("Analyzing... ⏳"):
                result_index, confidence = model_prediction(test_image)

            if result_index is not None:
                # Retrieve disease info
                class_names = list(disease_info.keys())
                disease_name = class_names[result_index]
                plant_name = disease_info[disease_name]["plant"]
                solution = disease_info[disease_name]["solution"]

                st.success(f"**Plant Detected:** {plant_name}")
                st.warning(f"**Disease:** {disease_name.replace('_', ' ')}")

                st.markdown("### Confidence Score:")
                st.progress(confidence)

                with st.expander("💡 Treatment & Prevention"):
                    st.markdown(f"**Recommended Solution:** {solution}")

                st.balloons()
            else:
                st.error("⚠️ Could not process the prediction. Please try again.")

        # Go Back button
        if st.button("Go Back"):
            st.session_state["app_mode"] = "home"
            st.rerun()
    else:
        # If no image, just show "Go Back" button
        if st.button("Go Back"):
            st.session_state["app_mode"] = "home"
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)  # Close right-col

    st.markdown('</div>', unsafe_allow_html=True)  # Close disease-container


def about_page():
    """
    About page with more engaging layout and additional info.
    """
    # Redirect if user not logged in
    if st.session_state["user"] is None:
        st.session_state["app_mode"] = "landing"
        st.rerun()

    st.title("ℹ️ About This Project")

    # Optional: Display a small banner or image (if you have one)
    # st.image("about_banner.jpeg", use_container_width=True)

    st.markdown(
        """
        Welcome to our **Plant Disease Recognition System**, an AI-driven approach to identifying 
        plant diseases from leaf images. Our mission is to help farmers and gardeners make 
        **data-driven decisions** to protect their crops.

        ### Key Features
        - **Extensive Dataset**: Trained on thousands of plant leaf images.
        - **High Accuracy**: Employs a deep learning model optimized for precision.
        - **Actionable Insights**: Provides recommended treatments and prevention tips.
        - **User-Friendly**: Simple interface with quick results.

        ### Why It Matters
        Early disease detection can **save crops**, reduce chemical use, and improve yields. 
        By leveraging cutting-edge AI, we empower you to address issues promptly and sustainably.
        """
    )

    # Optional: Additional info or references
    with st.expander("References & Further Reading"):
        st.markdown(
            """
            - [FAO: Plant Health](http://www.fao.org/plant-health/en/)
            - [Research Paper: Deep Learning for Plant Disease Detection](https://arxiv.org/abs/1604.03169)
            - [GitHub Repository](https://github.com/your-username/plant-disease-recognition) *(if applicable)*
            """
        )

    st.info(
        "We welcome contributions and feedback! "
        "Feel free to reach out or submit pull requests."
    )

    # "Go Back" Button
    if st.button("Go Back"):
        st.session_state["app_mode"] = "home"
        st.rerun()


def feedback_page():
    """
    A more detailed Feedback page.
    Allows users to rate their experience, enter comments, and optionally provide their name.
    """
    # Redirect if user not logged in
    if st.session_state["user"] is None:
        st.session_state["app_mode"] = "landing"
        st.rerun()

    # Page Title & Intro
    st.title("Feedback & Suggestions")
    st.write("We value your feedback! Please share your thoughts on how we can improve.")

    # Optional: Show user email if they're logged in
    user_email = st.session_state["user"].get("email", "Unknown") if st.session_state["user"] else "Guest"
    st.info(f"You are logged in as: **{user_email}**")

    # Rating Section
    st.markdown("### Rate Your Experience")
    rating = st.slider("On a scale of 1-5, how would you rate your overall experience?", min_value=1, max_value=5, value=3)

    # Comments Section
    st.markdown("### Your Comments")
    feedback_text = st.text_area("Please share any additional comments or suggestions:")

    # Optional Name Field
    name = st.text_input("Your Name (optional)")

    # Submit Button
    if st.button("Submit Feedback"):
        if feedback_text.strip():
            # You could store the feedback in Firestore or any DB here.
            # Example (uncomment if you have db and a 'feedback' collection):
            # db.collection("feedback").add({
            #     "email": user_email,
            #     "name": name,
            #     "rating": rating,
            #     "comments": feedback_text,
            #     "timestamp": datetime.utcnow()
            # })

            st.success("Thank you for your feedback!")
            st.balloons()
        else:
            st.error("Please enter some feedback before submitting.")

    # Go Back Button
    if st.button("Go Back"):
        st.session_state["app_mode"] = "home"
        st.rerun()


def contact_page():
    """
    A more engaging Contact Us page.
    Allows users to submit their name, email, and a message.
    """
    # Redirect if user not logged in
    if st.session_state["user"] is None:
        st.session_state["app_mode"] = "landing"
        st.rerun()

    st.title("Contact Us")
    st.markdown(
        """
        We'd love to hear from you! Fill out the form below, 
        and we'll get back to you as soon as possible.
        """
    )

    # Optionally show the logged-in user's email
    user_email = st.session_state["user"].get("email", "Unknown") if st.session_state["user"] else "Guest"
    st.info(f"You are logged in as: **{user_email}**")

    # Contact Form Fields
    name = st.text_input("Your Name")
    email = st.text_input("Your Email", value=user_email if user_email != "Guest" else "")
    message = st.text_area("Your Message")

    # Submit Button
    if st.button("Send Message"):
        if not name.strip() or not email.strip() or not message.strip():
            st.error("Please fill out all fields before sending.")
        else:
            # Example: store in Firestore or send via email
            # db.collection("contacts").add({
            #     "name": name,
            #     "email": email,
            #     "message": message,
            #     "timestamp": datetime.utcnow()
            # })
            st.success("Thank you! Your message has been sent.")
            st.balloons()

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
        home_page()  # The function we just modified
    elif mode == "disease":
        disease_recognition_page()
    elif mode == "about":
        about_page()
    elif mode == "feedback":
        feedback_page()
    elif mode == "contact":
        contact_page()
    else:
        st.session_state["app_mode"] = "landing"
        st.rerun()

if __name__ == "__main__":
    main()