# 🧠 AI Object Detection Web App (Image + Video)

A full-stack AI-powered web application that performs object detection on images and videos using YOLOv8, built with Flask and analytics dashboard built as a part of my CodeAlpha internship.

---


## 🌐 Live Deployment

🚀 The project is successfully deployed on Render:

👉 Live URL: ((https://aiobjectdetection-1.onrender.com))



---

## ⚙️ Deployment Platform

- Hosted on: Render
- Backend: Flask (Python)
- Model: YOLOv8
- Database: SQLite

---

## 📌 How to Run Live Version

1. Open the link above
2. Sign up / login
3. Upload image or video
4. View detection results in real-time

---

## 🚀 Deployment Notes

- Uses `gunicorn` for production server
- Flask app hosted on Render Web Service
- Static uploads handled locally
- YOLOv8 model loaded at runtime

## 🚀 Features

- 📷 Image Object Detection  
- 🎥 Video Object Detection  
- 🧠 YOLOv8 integration  
- 📊 Analytics Dashboard (Chart.js)  
- 📁 Upload images & videos  
- 🗂 Detection history (SQLite)  
- 🔐 Login / Signup system  
- ⚡ Optimized video processing  
- 🎯 Confidence score tracking  

---

## 🛠️ Tech Stack

**Backend:** Flask, Python, SQLite  
**AI Model:** YOLOv8 (Ultralytics)  
**Frontend:** HTML, CSS, JS, Chart.js  
**Other:** OpenCV, Werkzeug  

---

## 📁 Project Structure

AIObjectDetection/
├── app.py
├── yolov8n.pt
├── database.py
├── check_db.py
├── requirements.txt
│
├── instance/
│   └── ai_detect.sqlite
│
├── uploads/
│   ├── images & videos
│
├── static/
│   └── dashboard.css
│
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── signup.html
│   ├── dashboard.html
│   ├── result.html
│
└── README.md

---

## ⚙️ Installation

### 1. Clone repo
```bash
git clone https://github.com/hazracodegit/AIObjectDetection.git

cd AIObjectDetection

### 2. Create virtual environment
python -m venv venv

venv\Scripts\activate

### 3. Install Dependencies

pip install -r requirements.txt

### 4. Run the project

python app.py

### 5. Open Browser

http://127.0.0.1:5000/

📷 How It Works
Image Detection
Upload image
YOLO detects objects
Results stored in database
Video Detection
Upload video
Frame-by-frame processing
YOLO runs on frames
Results aggregated

📊 Dashboard Features
Total uploads
Total detections
Most detected object
Confidence score
Charts (daily analytics)
Recent history


🧠 AI Model
YOLOv8 nano (yolov8n.pt)
80 COCO classes
Real-time detection support


⚡ Performance Optimizations
Frame skipping for video
Reduced inference size (640)
Limited frame processing
Efficient DB queries

🔐 Authentication
Signup / Login system
Session-based auth
