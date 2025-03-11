import streamlit as st  # ✅ Import Streamlit first

# Other imports
import tensorflow as tf
import numpy as np
from PIL import Image
import firebase_admin
from firebase_admin import auth, credentials, firestore
import base64


st.set_page_config(page_title="Plant Disease Recognition", layout="wide")

import os
# ✅ Construct the absolute path to the credentials file
cred_path = os.path.join(os.getcwd(), "plant_disease_detection.json")

if not os.path.exists(cred_path):
    st.error(f"Error: Firebase credentials file not found at {cred_path}. Please check the file location.")
else:
    # ✅ Initialize Firebase only if it's not already initialized
    if not firebase_admin._apps:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)

    db = firestore.client()
    # st.success("Firebase initialized successfully!")

st.title("Plant Disease Detection System")
st.write("Upload an image to detect plant diseases.")
# Function to load custom CSS
def load_css():
    with open("styles.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

from firebase_config import firebase_login, firebase_signup


# User Session Handling
if "user" not in st.session_state:
    st.session_state["user"] = None  # Store authenticated user info

# Login Page
def login():
    st.header("🔑 Login to Your Account")
    email = st.text_input("📧 Email")
    password = st.text_input("🔒 Password", type="password")

    if st.button("Login", use_container_width=True):
        with st.spinner("Authenticating..."):
            result = firebase_login(email, password)

        if "idToken" in result:
            st.success("✅ Logged in successfully!")
            st.session_state["user"] = result  # Store user session
            st.rerun()
        else:
            st.error("❌ Login failed. Check your credentials.")

# Signup Page
def signup():
    st.header("📝 Create an Account")
    email = st.text_input("📧 Email")
    password = st.text_input("🔒 Password", type="password")

    if st.button("Sign Up", use_container_width=True):
        with st.spinner("Creating account..."):
            result = firebase_signup(email, password)

        if "idToken" in result:
            st.success("🎉 Account created successfully! Please login.")
        else:
            st.error(f"⚠️ Error: {result.get('error', {}).get('message', 'Unknown error')}")

# Sidebar Navigation
st.sidebar.title("🌿 Plant Disease Recognition")
if st.session_state["user"]:
    st.sidebar.write(f"👤 Logged in as: {st.session_state['user']['email']}")
    if st.sidebar.button("Logout"):
        st.session_state["user"] = None  # Clear session
        st.rerun()

app_mode = st.sidebar.radio("📌 Select Page", ["Home", "Login", "Sign Up"])

if app_mode == "Home":
    if st.session_state["user"]:
        st.header("🌱 Welcome to the Plant Disease Recognition System!")
        st.write("You are logged in. Upload an image to identify diseases.")
    else:
        st.warning("🔒 Please log in to use the application.")

elif app_mode == "Login":
    login()

elif app_mode == "Sign Up":
    signup()


# # Firebase Setup
# if not firebase_admin._apps:
#     try:
#         cred = credentials.Certificate("firebase_credentials.json")
#         firebase_admin.initialize_app(cred)
#     except Exception as e:
#         st.error(f"Error initializing Firebase: {e}")

# db = firestore.client()

# Function to Encode Image to Base64 (Optimized)
def get_base64_image(image_path):
    if not os.path.exists(image_path):
        return ""  # Avoid unnecessary warnings
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except Exception as e:
        st.warning(f"⚠️ Error loading background image: {e}")
        return ""


background_image = get_base64_image("Background.jpg")

# Apply Background Image with Transparency Reduction
if background_image:
    page_bg_img = f'''
    <style>
    .stApp {{
        background: url("data:image/jpg;base64,{background_image}") no-repeat center center fixed;
        background-size: cover;
        backdrop-filter: blur(8px) brightness(0.85);
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)


# Load Model Once (Cached to Avoid Reloading)
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("PlantDisease_Model.h5")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Image Processing with Dynamic Resizing
def process_image(image_data):
    try:
        image = Image.open(image_data).convert("RGB")
        img_size = (224, 224)
        image = image.resize(img_size, Image.Resampling.LANCZOS)
        input_arr = np.array(image) / 255.0
        return np.expand_dims(input_arr, axis=0)
    except Exception as e:
        st.error(f"🔥 Error Processing Image: {e}")
        return None


# Model Prediction with Validation
def model_prediction(image_data):
    if not model:
        st.error("⚠️ Model is not loaded. Please check the model file.")
        return None, None
    processed_image = process_image(image_data)
    if processed_image is None:
        return None, None
    try:
        predictions = model.predict(processed_image)
        predicted_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions))
        return predicted_index, confidence
    except Exception as e:
        st.error(f"🔥 Prediction Error: {e}")
        return None, None


# Disease Data
disease_info = {
    "Pepper_bell_Bacterial_spot": {"plant": "Pepper Bell", "solution": "Use copper-based bactericides and avoid overhead watering."},
    "Pepper_bell_healthy": {"plant": "Pepper Bell", "solution": "The plant is healthy. Maintain proper watering and nutrient levels."},
    "Potato_Early_blight": {"plant": "Potato", "solution": "Use fungicides with chlorothalonil and practice crop rotation."},
    "Potato_healthy": {"plant": "Potato", "solution": "The plant is healthy. Ensure good soil drainage."},
    "Potato_Late_blight": {"plant": "Potato", "solution": "Use copper-based fungicides and remove affected leaves."},
    "Tomato_Target_Spot": {"plant": "Tomato", "solution": "Apply fungicides like chlorothalonil and avoid excess moisture."},
    "Tomato_Tomato_mosaic_virus": {"plant": "Tomato", "solution": "Remove infected plants immediately and control aphids."},
    "Tomato_Tomato_YellowLeaf_Curl_Virus": {"plant": "Tomato", "solution": "Control whiteflies and use virus-resistant varieties."},
    "Tomato_Bacterial_spot": {"plant": "Tomato", "solution": "Use copper sprays and avoid overhead irrigation."},
    "Tomato_Early_blight": {"plant": "Tomato", "solution": "Use chlorothalonil-based fungicides regularly."},
    "Tomato_healthy": {"plant": "Tomato", "solution": "The plant is healthy. Maintain good cultivation practices."},
    "Tomato_Late_blight": {"plant": "Tomato", "solution": "Apply fungicides containing mancozeb and remove infected leaves."},
    "Tomato_Leaf_Mold": {"plant": "Tomato", "solution": "Ensure proper ventilation and use fungicides if needed."},
    "Tomato_Septoria_leaf_spot": {"plant": "Tomato", "solution": "Remove infected leaves and use copper-based fungicides."},
    "Tomato_Spider_mites_Two_spotted_spider_mite": {"plant": "Tomato", "solution": "Use neem oil or insecticidal soaps to control mites."}
}

# Display Login Button in Top Right Corner
st.markdown('<button id="login-button">Login</button>', unsafe_allow_html=True)


# Sidebar Navigation Handling
st.sidebar.title("🌿 Plant Disease Recognition")
app_mode = st.sidebar.radio("📌 Select Page", ["Home", "About", "Disease Recognition", "Feedback"], 
                            index=["Home", "About", "Disease Recognition", "Feedback"].index(st.session_state.get("app_mode", "Home")))
st.session_state["app_mode"] = app_mode


# User Authentication
def login():
    st.header("🔑 Login to Your Account")
    email = st.text_input("📧 Email")
    password = st.text_input("🔒 Password", type="password")
    if st.button("Login", use_container_width=True):
        try:
            user = auth.get_user_by_email(email)
            st.success("✅ Logged in successfully!")
            st.session_state["app_mode"] = "Home"
        except:
            st.error("❌ Invalid credentials.")

def signup():
    st.header("📝 Create an Account")
    email = st.text_input("📧 Email")
    password = st.text_input("🔒 Password", type="password")
    if st.button("Sign Up", use_container_width=True):
        try:
            user = auth.create_user(email=email, password=password)
            st.success("🎉 Account created successfully! Please login.")
        except:
            st.error("⚠️ Error creating account.")

if app_mode == "Landing Page":
    st.header("🌍 Welcome to the Plant Disease Recognition System!")
    st.markdown("### 🍃 AI-powered plant disease detection & expert recommendations!")
    st.markdown("This system helps farmers and gardeners detect plant diseases quickly and get expert recommendations for treatment and prevention.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login", use_container_width=True):
            st.session_state["app_mode"] = "Login"
    with col2:
        if st.button("Sign Up", use_container_width=True):
            st.session_state["app_mode"] = "Sign Up"

elif app_mode == "Login":
    login()

elif app_mode == "Sign Up":
    signup()

if app_mode == "Home":
    st.header("🌱 Welcome to the Plant Disease Recognition System!")
    st.image("home_page.jpeg", use_column_width=True)
    st.markdown("""
    Our AI-powered system helps farmers and plant enthusiasts detect plant diseases with ease. Simply upload an image of a plant leaf, and our deep learning model will analyze it to identify potential diseases and suggest treatments.
    
    🚀 **How It Works?**
    1. Upload a clear image of the affected plant leaf.
    2. Our AI model processes the image and predicts the disease.
    3. Get instant results with confidence scores and suggested treatments.
    
    Let’s take a step toward healthier crops with AI-driven disease detection!
    """)

elif app_mode == "About":
    st.header("ℹ️ About This Project")
    st.markdown("""
    This project leverages **deep learning** to identify plant diseases from leaf images. By analyzing thousands of plant samples, our AI model delivers **accurate disease predictions** along with confidence scores and recommended treatments.
    
    🔍 **Key Features:**
    ✅ Trained on an extensive dataset of plant leaf images
    ✅ Provides precise disease predictions with high accuracy
    ✅ Suggests effective treatment methods for various plant diseases
    ✅ Simple and user-friendly interface for easy access
    
    With the power of AI, we aim to help farmers and agriculturists **detect diseases early** and take **preventive actions** to ensure healthy crops.
    """)

elif app_mode == "Disease Recognition":
    st.header("🔍 Identify Plant Diseases in Seconds!")
    test_image = st.file_uploader("Upload an Image:", type=["jpg", "png", "jpeg"])

    if test_image is not None:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(test_image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            with st.spinner("Processing... ⏳"):
                result_index, confidence = model_prediction(test_image)

            # Class Names
            class_names = list(disease_info.keys())

            if result_index is not None and result_index < len(class_names):
                disease_name = class_names[result_index]
                plant_name = disease_info[disease_name]["plant"]
                solution = disease_info[disease_name]["solution"]

                with col2:
                    st.success(f"✅ **Plant:** {plant_name}")
                    st.warning(f"🌿 **Disease:** {disease_name.replace('_', ' ')}")

                    # Confidence Score Progress Bar
                    # Confidence Score Display with Percentage
                    st.markdown(f"### Confidence Score: **{confidence*100:.2f}%**")
                    st.progress(float(confidence))


                    with st.expander("💡 Treatment & Prevention"):
                        st.markdown(f"📌 **Solution:** {solution}")

                    st.balloons()
            else:
                st.error("⚠️ Error: Prediction out of bounds. Check class labels.")