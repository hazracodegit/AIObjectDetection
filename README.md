# 🚀 AI Object Detection Web App (YOLOv8 + Flask + Auth System)

A full-stack AI-powered web application that performs **object detection using YOLOv8** with a complete **authentication system (Login/Signup)**, user dashboard, and results tracking system.

---

## 🧠 Project Overview

This project integrates:
- 🔐 User authentication system (Login / Signup)
- 📊 Dashboard for users
- 🔍 AI object detection using YOLOv8
- 📤 Image/video upload system
- 📄 Results page for detections

It is built using Flask and demonstrates a real-world AI + Web Development project.

---

## 📁 Project Structure
pycache/
instance/
static/
├── style.css
├── script.js

templates/
├── login.html
├── signup.html
├── dashboard.html
├── index.html
├── results.html

uploads/

app.py
check_db.py
database.py
database.db
database.db-journal
requirements.txt
yolov8n.pt


---

## ✨ Features

### 🔐 Authentication System
- User Signup
- User Login
- Session-based access control
- Secure database storage (SQLite)

### 📊 Dashboard
- User-specific dashboard
- Upload files for detection
- View detection history

### 🔍 AI Object Detection
- YOLOv8-powered detection
- Image & video processing
- Bounding box visualization

### 📄 Results Page
- Displays detection output
- Shows detected objects
- Stores processed results

---

## 🛠️ Tech Stack

**Frontend:**
- HTML
- CSS
- JavaScript

**Backend:**
- Python
- Flask

**AI/ML:**
- YOLOv8 (Ultralytics)
- OpenCV

**Database:**
- SQLite

---

## 📦 Installation

### 1. Clone repository
```bash
git clone https://github.com/your-username/ai-object-detection.git
2. Move into project
cd ai-object-detection
3. Create virtual environment
python -m venv venv

Activate it:

Windows

venv\Scripts\activate

Mac/Linux

source venv/bin/activate
4. Install dependencies
pip install -r requirements.txt
🚀 Run the Project
python app.py

Open browser:

http://127.0.0.1:5000
