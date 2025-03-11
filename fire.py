import firebase_admin
from firebase_admin import credentials, firestore
import os

# Initialize Firebase only once
cred_path = os.path.join(os.getcwd(), "plant_disease_detection.json")

if not firebase_admin._apps:
    if not os.path.exists(cred_path):
        st.error(f"Error: Firebase credentials file not found at {cred_path}.")
    else:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        print("initialize")

db = firestore.client()