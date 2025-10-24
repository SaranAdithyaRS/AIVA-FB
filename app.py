from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import os
import json
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore

app = Flask(__name__)
CORS(app)

# ---------------- FIREBASE CONFIG ----------------
firebase_json = os.getenv("FIREBASE_CREDENTIALS_JSON")
cred_dict = json.loads(firebase_json)
cred = credentials.Certificate(cred_dict)
firebase_admin.initialize_app(cred)
db = firestore.client()

# ---------------- FACE MODELS ----------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

TRAINER_FILE = "trainer.yml"
LABELS_FILE = "labels.npy"

# ---------------- UTILS ----------------
def upload_to_firestore(image, name):
    _, buffer = cv2.imencode('.png', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    doc_ref = db.collection('faces').document(name)
    doc_ref.set({'name': name}, merge=True)
    db.collection('faces').document(name).collection('images').add({'data': img_base64})
    return f"Stored {name}'s face image in Firestore."

def save_face(frame, name):
    if not name:
        return "No name provided"
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return "No face detected"
    for (x, y, w, h) in faces:
        face_img = cv2.resize(gray[y:y + h, x:x + w], (100, 100))
        upload_to_firestore(face_img, name)
    return f"Face data saved for {name}"

def train_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, ids, label_map, current_id = [], [], {}, 0

    docs = db.collection('faces').stream()
    for doc in docs:
        person = doc.id
        if person not in label_map:
            label_map[person] = current_id
            current_id += 1
        imgs = db.collection('faces').document(person).collection('images').stream()
        for img_doc in imgs:
            img_data = img_doc.to_dict().get('data')
            if img_data:
                img_bytes = base64.b64decode(img_data)
                img_arr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(img_arr, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    faces.append(img)
                    ids.append(label_map[person])

    if faces:
        recognizer.train(faces, np.array(ids))
        recognizer.save(TRAINER_FILE)
        np.save(LABELS_FILE, label_map)
        return f"Trained {len(faces)} faces."
    return "No face data found in Firestore."

def estimate_emotion(face_gray, coords):
    x, y, w, h = coords
    roi = face_gray[y:y + h, x:x + w]
    smiles = smile_cascade.detectMultiScale(roi, scaleFactor=1.7, minNeighbors=22)
    eyes = eye_cascade.detectMultiScale(roi)
    if len(smiles) > 0:
        return "Happy"
    elif len(eyes) > 0:
        return "Neutral"
    else:
        return "Sad"

def detect_faces(frame):
    if not os.path.exists(TRAINER_FILE):
        return [], "No trained data found"
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_FILE)
    labels = np.load(LABELS_FILE, allow_pickle=True).item()
    label_map = {v: k for k, v in labels.items()}

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    detected = []
    for (x, y, w, h) in faces:
        roi = cv2.resize(gray[y:y + h, x:x + w], (100, 100))
        id_, conf = recognizer.predict(roi)
        name = label_map.get(id_, "Unknown") if conf < 80 else "Unknown"
        emotion = estimate_emotion(gray, (x, y, w, h))
        if name != "Unknown":
            detected.append({"name": name, "emotion": emotion})
    return detected, None

# ---------------- ROUTES ----------------
@app.route('/process_face', methods=['POST'])
def process_face():
    data = request.json
    image_b64 = data.get('frame', '')
    cmd = data.get('cmd', '').lower()
    name = data.get('name', '')

    if ',' in image_b64:
        image_b64 = image_b64.split(',')[1]
    img_bytes = base64.b64decode(image_b64)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"response": "Invalid image"})

    if "add face" in cmd:
        res = save_face(frame, name)
    elif "train faces" in cmd:
        res = train_recognizer()
    elif "detect faces" in cmd:
        faces, err = detect_faces(frame)
        if err:
            res = err
        elif faces:
            res = ". ".join([f"I found {f['name']}, looks {f['emotion']}" for f in faces]) + "."
        else:
            res = "No familiar faces detected."
    else:
        res = "Command not recognized for core server."

    return jsonify({"response": res})

@app.route('/health')
def health():
    return jsonify({"status": "core face server running"})

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 Core Face + Firestore Server Running on Port 5000")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000)
