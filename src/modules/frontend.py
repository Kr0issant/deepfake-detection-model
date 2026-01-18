import streamlit as st
import time
import random
from PIL import Image
import inference

inference = inference.Inference()

# --- Page Configuration ---
st.set_page_config(
    page_title="Veritas Shield | Deepfake Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for "Nice Looking" UI ---
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        height: 3em;
        border-radius: 10px;
    }
    .stProgress > div > div > div > div {
        background-color: #FF4B4B;
    }
    img {
        user-drag: none;
        -webkit-user-drag: none;
        draggable: false;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Mock Backend Function ---
def analyze_media(file_type, file_bytes):
    
    if file_type == "image":
        label, confidence = inference.analyze_image(file_bytes)
        result = "REAL" if label == 0 else "FAKE"
    elif file_type == "video":
        label, confidence = inference.analyze_video(file_bytes)
        result = "REAL" if len(label) == 0 else "FAKE"
    
    return result, label, confidence

# --- Sidebar ---
with st.sidebar:
    st.image("./veritas-shield-logo.png", width=100)
    st.title("Veritas Shield")
    st.caption("AI-Powered Media Forensics")
    st.markdown("---")
    
    st.header("Settings")
    sensitivity = st.slider("Detection Sensitivity", 0.0, 1.0, 0.5, 0.05)
    inertia = st.slider("Timestamp Inertia", 1, 30, 10, 1)
    st.markdown("---")
    
    st.link_button("GitHub", "https://github.com/Kr0issant/deepfake-detection-model")
    st.link_button("Kaggle", "https://kaggle.com/")
    st.markdown("---")
    
    st.markdown("Built with Streamlit")

# --- Main Layout ---
st.title("🕵️ Deepfake Detection Dashboard")
st.markdown("Upload an image or video to analyze it for manipulation artifacts.")

# File Uploader
uploaded_file = st.file_uploader("Choose a media file...", type=['jpg', 'jpeg', 'png', 'mp4', 'mov', 'avi'])

if uploaded_file is not None:
    # Determine file type
    file_type = uploaded_file.type.split('/')[0]
    
    # --- Layout: 2 Columns (Preview & Results) ---
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        st.subheader("Media Preview")
        if file_type == 'image':
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
        elif file_type == 'video':
            st.video(uploaded_file)
            
    with col2:
        st.subheader("Analysis")
        st.write("Click the button below to start the forensic analysis.")
        
        analyze_btn = st.button("🔍 Run Deepfake Detection")
        
        if analyze_btn:
            inference.update_variables(inertia, sensitivity)

            with st.spinner(f"Analyzing {file_type}..."):
                # Call the backend function
                result, label, confidence = analyze_media(file_type, uploaded_file)
            
            # --- Result Display ---
            st.markdown("### Results")
            
            # Logic for color coding
            if result == "FAKE":
                result_color = "red"
                msg = "⚠️ AI Manipulation Detected"
            else:
                result_color = "green"
                msg = "✅ Content appears Authentic"
                
            # Metric Cards
            m1, m2 = st.columns(2)
            m1.metric("Verdict", result, delta_color="inverse" if result == "REAL" else "normal")
            m2.metric("Confidence Score", f"{confidence:.2%}")
            
            # Progress Bar for visualization
            st.write("Confidence Level:")
            st.progress(float(confidence))
            
            # Alert Box
            if result == "FAKE":
                st.error(msg)
                with st.expander("See Technical Details"):
                    if file_type == "image":
                        st.write(f"- **Artifacts found:** High frequency noise irregularities.")
                        st.write(f"- **Confidence:** {confidence:.2%}")
                    else:
                        # label is now time_streaks: list of (start_t, end_t, conf)
                        st.write(f"- **Frame Analysis:** {len(label)} suspicious segments found.")
                        if len(label) > 0:
                            st.write("Suspicious Segments:")
                            for start_t, end_t, conf in label:
                                st.write(f"**{start_t} - {end_t}**")
                                st.progress(float(conf))
                                st.caption(f"Confidence: {conf:.2%}")
            else:
                st.success(msg)
                with st.expander("See Technical Details"):
                    st.write("- **Analysis:** Consistent noise patterns observed.")
                    st.write("- **Integrity:** Verified.")

else:
    # Empty State (What shows when nothing is uploaded)
    st.info("👆 Please upload a file to begin.")
    
    # Optional: Demo visuals
    with st.expander("How does it work?"):
        st.markdown("""
        1. **Upload**: We accept standard image and video formats.
        2. **Scan**: The system breaks down videos into frames and scans images for pixel-level inconsistencies.
        3. **Report**: You receive a probability score and a verdict (Real vs Fake).
        """)