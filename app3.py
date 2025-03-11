import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import logging

# Set Streamlit Page Config
st.set_page_config(page_title="Plant Disease Recognition", layout="wide")

# Configure logging
logging.basicConfig(filename='app.log', level=logging.ERROR)

# Load Model Once (Cached to Avoid Reloading)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("PlantDisease_Model.h5")

model = load_model()

# Function to Process Image (Auto-Resize)
def process_image(image_data):
    try:
        image = Image.open(image_data).convert("RGB")
        target_size = model.input_shape[1:3]  # (224,224)
        image = image.resize(target_size, Image.LANCZOS)
        input_arr = np.array(image) / 255.0
        input_arr = np.expand_dims(input_arr, axis=0)
        return input_arr
    except Exception as e:
        st.error("❌ Error processing the image. Please upload a valid image file.")
        logging.error(f"Error processing image: {str(e)}")
        return None

# Function for Model Prediction
def model_prediction(image_data):
    processed_image = process_image(image_data)
    if processed_image is None:
        return None, None  # Return None if processing failed
    try:
        predictions = model.predict(processed_image)
        predicted_index = np.argmax(predictions)
        confidence = np.max(predictions)
        return predicted_index, confidence
    except Exception as e:
        st.error("❌ Error during prediction. Please try again.")
        logging.error(f"Error during prediction: {str(e)}")
        return None, None

# Class Labels & Disease Solutions (Updated List)
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

# Sidebar Navigation
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

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
                    st.markdown("### Confidence Score:")
                    st.progress(float(confidence))

                    with st.expander("💡 Treatment & Prevention"):
                        st.markdown(f"📌 **Solution:** {solution}")

                    st.balloons()
            else:
                st.error("⚠️ Error: Prediction out of bounds. Check class labels.")