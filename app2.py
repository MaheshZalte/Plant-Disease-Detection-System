import streamlit as st

st.set_page_config(page_title="Plant Disease Detection", layout="wide")

st.sidebar.title("Navigation")
st.sidebar.markdown("""
    - [Home](Home)
    - [About](About)
    - [Disease Recognition](Disease Recognition)
""")

st.write("# Plant Disease Detection System")
st.markdown("""
Welcome to the AI-powered Plant Disease Detection System!  
Use the navigation on the left to explore the features:
- **Home**: Overview and introduction.
- **About**: Detailed project description.
- **Disease Recognition**: Upload a leaf image to detect disease.
""")
