import streamlit as st
import cv2
import tempfile
import os
import urllib.request
from tracker_logic import HazardDetector

# Page Configuration
st.set_page_config(
    page_title="Autonomous Driving Hazard Detection",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Autonomous Driving Scene Understanding & Real-Time Hazard Detection")
st.markdown("""
This interactive web application processes driving video clips to detect multi-class road actors, 
compute spatial proximity collision risks, and track wrong-way movement using **ByteTrack** and fine-tuned regional YOLO weights.
""")
st.sidebar.markdown("---")
st.sidebar.subheader("📐 Hazard Zone Geometry Tuning")
zone_width = st.sidebar.slider("Zone Width Spread", min_value=0.1, max_value=0.5, value=0.15, step=0.05)
zone_horizon = st.sidebar.slider("Zone Horizon Height", min_value=0.3, max_value=0.7, value=0.55, step=0.05)
# Load the Hazard Detector model using cached resource for speed
@st.cache_resource
def load_detector():
    return HazardDetector(model_path='best.pt')

with st.spinner("Loading fine-tuned regional model weights (`best.pt`)..."):
    detector = load_detector()

# Sidebar options for input source (Local upload or Pre-loaded sample)
st.sidebar.header("Input Video Source")
option = st.sidebar.radio("Choose input method:", ["Upload Local Video", "Use Pre-loaded Sample Traffic Video"])

vid_path = None

if option == "Use Pre-loaded Sample Traffic Video":
    if st.sidebar.button("Load Sample Video"):
        with st.spinner("Downloading sample traffic video..."):
            url = "https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4"
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            urllib.request.urlretrieve(url, tfile.name)
            st.session_state['vid_path'] = tfile.name
    if 'vid_path' in st.session_state:
        vid_path = st.session_state['vid_path']
else:
    uploaded_file = st.file_uploader("Upload a driving video clip (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        vid_path = tfile.name

if vid_path is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Video")
        st.video(vid_path)
        
    with col2:
        st.subheader("Real-Time Hazard Analysis Feed")
        run_button = st.button("Run Hazard Detection Pipeline")
        stframe = st.empty()
        
    if run_button:
        cap = cv2.VideoCapture(vid_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 30
            
        # Use WebM format with VP90 codec for seamless browser playback
        out_file = tempfile.NamedTemporaryFile(delete=False, suffix='.webm')
        out_path = out_file.name
        
        fourcc = cv2.VideoWriter_fourcc(*'VP90')
        out = cv2.VideoWriter(out_path, fourcc, fps, (640, 640))  # Updated to match 640x640 frame size
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0
        progress_bar = st.progress(0)
        
        with st.status("Processing frames and computing hazard heuristics...", expanded=True) as status:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame = detector.process_frame(frame, zone_width=zone_width, zone_horizon=zone_horizon)
                out.write(processed_frame)
                
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                stframe.image(rgb_frame, channels="RGB", use_container_width=True)
                
                current_frame += 1
                if frame_count > 0:
                    progress_bar.progress(min(current_frame / frame_count, 1.0))
            
            status.update(label="Hazard analysis complete!", state="complete", expanded=False)
            
        cap.release()
        out.release()
        
        st.success("Processing finished successfully!")
        st.subheader("Processed Output Video")
        st.video(out_path)
        
        # --- NEW: Automated Incident Summary Analytics Dashboard ---
        summary = detector.get_incident_summary()
        
        st.markdown("---")
        st.subheader("📊 Automated Hazard Analytics & Incident Report")
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Total Frames Analyzed", summary["total_frames"])
        col_m2.metric("Wrong-Way Violations", summary["wrong_way_count"])
        col_m3.metric("Zone Proximity Events", summary["proximity_count"])
        col_m4.metric("Critical TTC Alerts (<1.5s)", summary["ttc_critical_count"])
        
        st.write("**Detected Actor Distribution:**")
        st.json(summary["detected_classes"])
else:
    st.info("👈 Select a pre-loaded sample video or upload a file from the sidebar to begin analysis.")