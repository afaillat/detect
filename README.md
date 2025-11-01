# 🎓 ExamSafe AI  
**Intelligent Exam Monitoring System**

---

## 📖 Overview
**ExamSafe AI** is an intelligent proctoring assistant that uses real-time object detection to enhance exam integrity. The system automatically recognizes forbidden objects  
like cellphones, books or laptops during examinations.  

It combines the power of YOLOv5, a state-of-the-art deep learning model for object detection, with Streamlit, a Python framework for web applications.

---

## Features
- **Real-time detection** directly from your webcam  
- **YOLOv5 model** for accurate object recognition  
- **Streamlit interface** for a simple and interactive user experience  
- **Local processing**, no data or images are stored online  
- Detects objects like cellphones, bottles, books, laptops, cups and scissors.

---

## 🛠️ Installation

1. **Clone this repository**
   ```bash
   git clone https://github.com/<username>/ExamSafe-AI.git
   cd ExamSafe-AI

2. **Create and activate a virtual environment**
python -m venv venv
source venv/bin/activate      # on macOS 
venv\Scripts\activate         # on Windows

3. **Install the dependencies**
pip install -r requirements.txt

##  How to run the application ? 

## Option 1 — Web App (Streamlit version)
This launches the full interactive interface inside your browser.
python -m streamlit run app.py

- Click ▶️ Start to activate the webcam
- Click ⏹ Stop to stop the detection
- The detected objects are displayed directly in the web interface

## Option 2 — Direct Python script (Terminal version)
This version runs YOLOv5 directly from the terminal without the Streamlit interface.
It opens a standard OpenCV camera window.

python detect.py --source 0

- source 0 → uses your computer’s webcam
You can also use:
- source path/to/video.mp4 → to test a saved video
- source path/to/image.jpg → to test a single image
