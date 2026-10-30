# 🚗 Autonomous Driving Scene Understanding & Real-Time Hazard Detection Web App

An end-to-end, production-ready Advanced Driver Assistance System (ADAS) web application designed for unstructured, dense traffic environments (such as South Asian urban roads). Powered by **YOLOv8**, **ByteTrack**, and **Streamlit**, this system performs automated visual analysis, spatial proximity tracking, Time-to-Collision (TTC) estimation, and dynamic wrong-way hazard detection.

---

## 🌟 Key Features
* **Multi-Class Local Actor Detection**: Accurately recognizes localized traffic classes including cars, motorbikes, auto-rickshaws, pedestrians, and cyclists
* **Kinematic Safety Telemetry**: Computes bounding-box growth rates across sequential frames to estimate Time-to-Collision (TTC) and dynamic proximity risks
* **Restricted Wrong-Way Trajectory Analysis**: Leverages ByteTrack motion vectors combined with an active ego-lane polygon zone to flag wrong-way maneuvers .
* **Dynamic Hazard Zone Geometry**: Features adjustable interactive UI sliders allowing real-time adaptation to varying road widths and camera perspectives.
* **Automated KPI Dashboard**: Summarizes quantitative safety metrics, incident counts, and actor frequencies into an instant post-video report.

---

## 🏗️ System Architecture & Workflow
1. **Perception Engine**: Uses YOLOv8 object detection combined with ByteTrack multi-object tracking for persistent ID association across video frames[cite: 5].
2. **Spatial Risk Mapping**: Evaluates dynamic polygon zone containment to isolate the ego-lane from surrounding environment clutter[cite: 5].
3. **Deployment & MLOps**: Developed via modular Python scripts (`app.py`, `tracker_logic.py`), version-controlled through GitHub, and deployed live to the cloud[cite: 6].

---

## 📂 Project Repository Structure
```text
├── .github/
│   └── workflows/          # CI/CD automated pipeline configurations
├── app.py                  # Main Streamlit dashboard interface & UI 
├── tracker_logic.py        # Core YOLOv8, ByteTrack, TTC, and wrong-way logic
├── requirements.txt        # Project dependencies & Python package 
└── README.md               # Technical project documentation[cite: 6]