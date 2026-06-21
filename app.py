import os
os.environ["YOLO_CONFIG_DIR"] = "/tmp"
import os
os.environ["YOLO_CONFIG_DIR"] = "/tmp"

import sqlite3
from collections import Counter
from datetime import datetime

from flask import Flask, flash, redirect, render_template, request, send_from_directory, session, url_for
from ultralytics import YOLO
from werkzeug.utils import secure_filename


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATABASE = os.path.join(BASE_DIR, "instance", "ai_detect.sqlite")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}
ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
MAX_VIDEO_FRAMES = 120

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SECRET_KEY"] = "ai-detect-dev-secret"

os.makedirs(os.path.dirname(DATABASE), exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
model = YOLO(os.path.join(BASE_DIR, "yolov8n.pt"))


def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=MEMORY")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_db():
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
            """
        )
        conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    image_name TEXT NOT NULL,
    objects TEXT NOT NULL,
    object_count INTEGER NOT NULL DEFAULT 0,
    confidence REAL,
    media_type TEXT NOT NULL DEFAULT 'image',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
            
        

        columns = {row["name"] for row in conn.execute("PRAGMA table_info(detections)")}
        if "image_name" not in columns:
            conn.execute("ALTER TABLE detections ADD COLUMN image_name TEXT")
        if "objects" not in columns:
            conn.execute("ALTER TABLE detections ADD COLUMN objects TEXT")
        if "object_count" not in columns:
            conn.execute("ALTER TABLE detections ADD COLUMN object_count INTEGER NOT NULL DEFAULT 0")
        if "created_at" not in columns:
            conn.execute("ALTER TABLE detections ADD COLUMN created_at DATETIME")
        if "confidence" not in columns:
            conn.execute("ALTER TABLE detections ADD COLUMN confidence REAL")
        if "media_type" not in columns:
            conn.execute("ALTER TABLE detections ADD COLUMN media_type TEXT NOT NULL DEFAULT 'image'")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def media_type_for(filename):
    extension = filename.rsplit(".", 1)[1].lower()
    return "video" if extension in VIDEO_EXTENSIONS else "image"


def detect_media(filepath, media_type):
    detected = []
    confidences = []

    if media_type == "video":
        results = model.predict(filepath, stream=True, vid_stride=1, verbose=False)

        for frame_number, result in enumerate(results):
            if frame_number > MAX_VIDEO_FRAMES:
                break

        for box in result.boxes:
            class_id = int(box.cls[0])
            detected.append(model.names[class_id])
            confidences.append(float(box.conf[0]) * 100)
    else:
        results = model(filepath, verbose=False)
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                detected.append(model.names[class_id])
                confidences.append(float(box.conf[0]) * 100)

    detected_objects = sorted(set(detected))
    confidence = round(sum(confidences) / len(confidences), 1) if confidences else 0
    return detected_objects, len(detected), confidence




import cv2

def process_video(filepath):
    cap = cv2.VideoCapture(filepath)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_path = filepath.rsplit(".", 1)[0] + "_output.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    detected = []
    confidences = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count > MAX_VIDEO_FRAMES:
            break

        results = model(frame)

        for box in results[0].boxes:
            class_id = int(box.cls[0])
            detected.append(model.names[class_id])
            confidences.append(float(box.conf[0]) * 100)

        annotated = results[0].plot()
        out.write(annotated)

        frame_count += 1

    cap.release()
    out.release()

    detected_objects = sorted(set(detected))
    confidence = round(sum(confidences) / len(confidences), 1) if confidences else 0

    return output_path, detected_objects, len(detected), confidence


def dashboard_data():
    with get_db() as conn:
       
        user_id = session.get("user_id")

        if not user_id:
            return {
        "total_uploads": 0,
        "total_detections": 0,
        "accuracy": 0,
        "top_object": "None",
        "top_percent": 0,
        "recent": [],
        "top_counts": [],
        "chart_labels": [],
        "chart_uploads": [],
        "chart_detections": [],
    }

        rows = conn.execute(
    """
    SELECT image_name, objects, object_count, confidence, media_type, created_at
    FROM detections
    WHERE user_id = ?
    ORDER BY created_at DESC
    """,
    (user_id,)
).fetchall()

    object_names = []
    for row in rows:
        object_names.extend([name.strip() for name in (row["objects"] or "").split(",") if name.strip()])

    counts = Counter(object_names)
    top_object, top_count = counts.most_common(1)[0] if counts else ("Person", 0)
    total_detections = sum(counts.values())
    avg_confidence = round(
        sum(row["confidence"] or 0 for row in rows) / len(rows),
        1,
    ) if rows else 95.4

    labels = []
    uploads = []
    detections = []
    daily = {}
    for row in rows:
        day = (row["created_at"] or "")[:10] or datetime.now().strftime("%Y-%m-%d")
        daily.setdefault(day, {"uploads": 0, "detections": 0})
        daily[day]["uploads"] += 1
        daily[day]["detections"] += row["object_count"] or 0

    for day in sorted(daily.keys())[-7:]:
        labels.append(datetime.strptime(day, "%Y-%m-%d").strftime("%b %d"))
        uploads.append(daily[day]["uploads"])
        detections.append(daily[day]["detections"])

    if not labels:
        labels = ["May 8", "May 9", "May 10", "May 11", "May 12", "May 13", "May 14"]
        uploads = [180, 310, 260, 430, 360, 490, 580]
        detections = [360, 500, 420, 720, 610, 780, 930]

    return {
        "total_uploads": len(rows),
        "total_detections": total_detections or 3782,
        "accuracy": avg_confidence,
        "top_object": top_object.title(),
        "top_percent": round((top_count / total_detections) * 100) if total_detections else 32,
        "recent": rows[:200],
        "top_counts": counts.most_common(4),
        "chart_labels": labels,
        "chart_uploads": uploads,
        "chart_detections": detections,
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        with get_db() as conn:
            user = conn.execute(
                "SELECT * FROM users WHERE email = ? AND password = ?",
                (email, password),
            ).fetchone()

        if user:
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("dashboard"))
        error = "Invalid email or password."

    return render_template("login.html", error=error)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if password != confirm_password:
            error = "Passwords do not match."
        else:
            try:
                with get_db() as conn:
                    conn.execute(
                        "INSERT INTO users(username, email, password) VALUES (?, ?, ?)",
                        (username, email, password),
                    )
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                error = "An account with this email already exists."

    return render_template("signup.html", error=error)



@app.route("/detect", methods=["POST"])
def detect():
    media = request.files.get("media") or request.files.get("image")
    if not media or media.filename == "":
        flash("Choose an image or video first.")
        return redirect(url_for("dashboard"))
    if not allowed_file(media.filename):
        flash("Please upload an image or video file.")
        return redirect(url_for("dashboard"))

    filename = secure_filename(media.filename)
    media_type = media_type_for(filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    media.save(filepath)

    if media_type == "video":
        filepath, detected_objects, object_count, confidence = process_video(filepath)
    else:
        detected_objects, object_count, confidence = detect_media(filepath, media_type)
    objects_text = ", ".join(detected_objects) if detected_objects else "No objects detected"

    with get_db() as conn:
        user_id = session.get("user_id")
       

        if not user_id:
            flash("Please login first")
            return redirect(url_for("login"))
        conn.execute(
            """
            INSERT INTO detections(user_id, image_name, objects, object_count, confidence, media_type)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, filename, objects_text, object_count, confidence, media_type),
        )

    return render_template(
        "result.html",
        media=filename,
        media_type=media_type,
        objects=detected_objects,
        confidence=confidence,
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        flash("Please login first")
        return redirect(url_for("login"))

    username = session.get("username")
    initials = "".join(part[0] for part in username.split()[:2]).upper() or "G"

    return render_template(
        "dashboard.html",
        username=username,
        initials=initials,
        **dashboard_data(),
    )


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


init_db()

if __name__ == "__main__":
    app.run(debug=True)

