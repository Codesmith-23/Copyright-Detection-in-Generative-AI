import os
# --- OPTIMIZATION: Force CPU Mode & Reduce TF Logs ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import uuid
import secrets
import sqlite3
import cv2
import numpy as np
import dlib
import time
import shutil
import gc
import atexit
import jwt
import base64
import threading
import concurrent.futures
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.exceptions import RequestEntityTooLarge
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime, timedelta
from flask_cors import CORS
import requests
from moviepy.editor import VideoFileClip
import traceback
import tensorflow as tf
from huggingface_hub import hf_hub_download


app = Flask(__name__)

# Only the frontend may call this API from a browser. Set FRONTEND_ORIGIN to a
# comma-separated list to allow other addresses (e.g. a LAN IP or a domain).
FRONTEND_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "FRONTEND_ORIGIN", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if origin.strip()
]
CORS(app, resources={r"/*": {"origins": FRONTEND_ORIGINS}})

# JWT Configuration
# The signing secret comes from the JWT_SECRET environment variable. If it is
# not set, a random secret is generated on first start and kept in a local,
# git-ignored file so issued tokens stay valid across restarts.
JWT_SECRET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".jwt_secret")
MIN_JWT_SECRET_LENGTH = 32

def load_jwt_secret():
    secret = os.environ.get("JWT_SECRET", "").strip()
    if secret:
        if len(secret) < MIN_JWT_SECRET_LENGTH:
            raise RuntimeError(
                f"JWT_SECRET must be at least {MIN_JWT_SECRET_LENGTH} characters long"
            )
        return secret

    if os.path.exists(JWT_SECRET_FILE):
        with open(JWT_SECRET_FILE) as f:
            secret = f.read().strip()
        if len(secret) >= MIN_JWT_SECRET_LENGTH:
            return secret
        print(f"[AUTH] {JWT_SECRET_FILE} is too short, generating a new secret")

    secret = secrets.token_hex(32)
    # Owner read/write only.
    fd = os.open(JWT_SECRET_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(secret)
    print(f"[AUTH] Generated a new JWT secret in {JWT_SECRET_FILE}")
    return secret

app.config['SECRET_KEY'] = load_jwt_secret()
app.config['JWT_EXPIRATION_HOURS'] = 24

ALLOWED_AUDIO = {'.mp3', '.wav', '.flac', '.m4a'}
ALLOWED_VIDEO = {'.mp4', '.avi', '.mov', '.mkv'}

# Largest accepted upload, in MB. Matches the 500 MB limit in the frontend's upload box.
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "500"))
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_MB * 1024 * 1024


@app.errorhandler(RequestEntityTooLarge)
def upload_too_large(_error):
    return jsonify({"error": f"File is too large. The limit is {MAX_UPLOAD_MB} MB."}), 413

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config["UPLOAD_FOLDER"] = os.path.join(BASE_DIR, "uploads")
VIDEOS_DIR = os.path.join(app.config["UPLOAD_FOLDER"], "videos")
FRAMES_DIR = os.path.join(app.config["UPLOAD_FOLDER"], "flagged_frames")

os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)

# --- THREAD-LOCAL SQLite CONNECTIONS ---
# One persistent connection per thread instead of opening and closing
# a new connection on every single query.
DB_PATH = os.path.join(BASE_DIR, "database.db")
_thread_local = threading.local()

def get_db():
    if not hasattr(_thread_local, "conn") or _thread_local.conn is None:
        _thread_local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _thread_local.conn.execute("PRAGMA journal_mode=WAL")
        _thread_local.conn.execute("PRAGMA synchronous=NORMAL")
    return _thread_local.conn

# --- CLEANUP LOGIC ---
def robust_delete(file_path):
    """Retries deletion to handle Windows file locking."""
    if not file_path or not os.path.exists(file_path):
        return
    gc.collect()
    for i in range(5):
        try:
            os.remove(file_path)
            return
        except PermissionError:
            time.sleep(0.1)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

def cleanup_folder(folder_path):
    """Wipes all files in a folder."""
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Failed to clean {filename}: {e}")

cleanup_folder(VIDEOS_DIR)
cleanup_folder(FRAMES_DIR)
atexit.register(lambda: cleanup_folder(FRAMES_DIR))

# --- MODEL LOADING ---
print("Loading video model from Hugging Face...")
try:
    REPO_ID = "Codesmith-23/deepfake-detector-v1"
    FILENAME = "model_fine_final.h5"
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    model = load_model(model_path)
    print(f"Video Model loaded successfully from: {model_path}")
except Exception as e:
    print(f"CRITICAL ERROR: Could not download/load video model. {e}")
    model = None

detector = dlib.get_frontal_face_detector()

# --- DATABASE INIT ---
def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            analysisID TEXT,
            file_name TEXT,
            result TEXT,
            confidence TEXT,
            timestamp TEXT,
            file_size TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    # Which user registered which protected identity, with their consent record.
    c.execute("""
        CREATE TABLE IF NOT EXISTS creator_registrations (
            entity_id TEXT PRIMARY KEY,
            owner_user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            type TEXT NOT NULL,
            consent_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (owner_user_id) REFERENCES users (id)
        )
    """)
    conn.commit()

init_db()

def save_result(user_id, analysisID, file_name, result, confidence, file_size):
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        INSERT INTO results (user_id, analysisID, file_name, result, confidence, timestamp, file_size)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (user_id, analysisID, file_name, result, confidence, datetime.now().isoformat(), file_size))
    conn.commit()

# --- AUTHENTICATION UTILITIES ---
def generate_token(user_id, username):
    payload = {
        'user_id': user_id,
        'username': username,
        'exp': datetime.utcnow() + timedelta(hours=app.config['JWT_EXPIRATION_HOURS'])
    }
    return jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')

def verify_token(token):
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def token_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = None
        auth_header = request.headers.get('Authorization')
        if auth_header:
            try:
                token = auth_header.split(' ')[1]
            except IndexError:
                return jsonify({"error": "Invalid token format. Use: Bearer <token>"}), 401
        if not token:
            return jsonify({"error": "Authentication token missing"}), 401
        payload = verify_token(token)
        if not payload:
            return jsonify({"error": "Invalid or expired token"}), 401
        request.current_user = payload
        return f(*args, **kwargs)
    return decorated_function

# --- AUDIO EXTRACTION HELPER ---
# Runs in a background thread so the main frame loop and audio
# extraction happen concurrently instead of sequentially.
def extract_and_analyse_audio(filepath, videos_dir):
    """
    Extracts audio from the video file and posts it to the audio
    microservice. Designed to run in a ThreadPoolExecutor so it
    does not block the main request thread.
    Returns: (audio_data dict, temp_audio_path or None)
    """
    audio_data = {"label": "Not Detected", "confidence": 0}
    temp_audio_path = None
    try:
        clip = VideoFileClip(filepath)
        if not clip.audio:
            clip.close()
            return audio_data, None

        temp_audio_path = os.path.join(videos_dir, f"temp_{uuid.uuid4()}.wav")
        clip.audio.write_audiofile(temp_audio_path, logger=None, verbose=False)
        clip.close()
        del clip
        gc.collect()

        with open(temp_audio_path, "rb") as f:
            resp = requests.post(
                "http://127.0.0.1:5001/predict_audio",
                files={"file": f},
                timeout=120
            )
            if resp.status_code == 200:
                audio_data = resp.json()

    except Exception:
        if 'clip' in locals():
            try:
                clip.close()
            except Exception:
                pass

    return audio_data, temp_audio_path


# --- AUTHENTICATION ROUTES ---
@app.route("/auth/register", methods=["POST"])
def register():
    try:
        data = request.get_json()
        if not data or not all(k in data for k in ['username', 'email', 'password']):
            return jsonify({"error": "Missing required fields: username, email, password"}), 400

        username = data['username'].strip()
        email = data['email'].strip().lower()
        password = data['password']

        if len(username) < 3:
            return jsonify({"error": "Username must be at least 3 characters"}), 400
        if len(password) < 6:
            return jsonify({"error": "Password must be at least 6 characters"}), 400
        if '@' not in email or '.' not in email:
            return jsonify({"error": "Invalid email format"}), 400

        password_hash = generate_password_hash(password)

        conn = get_db()
        c = conn.cursor()
        try:
            c.execute("""
                INSERT INTO users (username, email, password_hash, created_at)
                VALUES (?, ?, ?, ?)
            """, (username, email, password_hash, datetime.now().isoformat()))
            conn.commit()
            user_id = c.lastrowid
            token = generate_token(user_id, username)
            return jsonify({
                "message": "Registration successful",
                "token": token,
                "user": {"id": user_id, "username": username, "email": email}
            }), 201
        except sqlite3.IntegrityError as e:
            if 'username' in str(e):
                return jsonify({"error": "Username already exists"}), 409
            elif 'email' in str(e):
                return jsonify({"error": "Email already registered"}), 409
            else:
                return jsonify({"error": "Registration failed"}), 500

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/auth/login", methods=["POST"])
def login():
    try:
        data = request.get_json()
        if not data or not all(k in data for k in ['username', 'password']):
            return jsonify({"error": "Missing username or password"}), 400

        username = data['username'].strip()
        password = data['password']

        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT id, username, email, password_hash FROM users WHERE username = ?", (username,))
        user = c.fetchone()

        if not user:
            return jsonify({"error": "Invalid username or password"}), 401

        user_id, username, email, password_hash = user

        if not check_password_hash(password_hash, password):
            return jsonify({"error": "Invalid username or password"}), 401

        token = generate_token(user_id, username)
        return jsonify({
            "message": "Login successful",
            "token": token,
            "user": {"id": user_id, "username": username, "email": email}
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/auth/verify", methods=["GET"])
@token_required
def verify():
    return jsonify({
        "valid": True,
        "user": {
            "id": request.current_user['user_id'],
            "username": request.current_user['username']
        }
    }), 200


# --- MAIN ROUTE ---
@app.route("/predict/media", methods=["POST"])
@token_required
def predict_media():
    first_fake_frame_b64 = None
    cleanup_folder(FRAMES_DIR)

    cv2.ocl.setUseOpenCL(False)
    filepath = None
    temp_extracted_audio = None
    start_time = time.time()

    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file"}), 400

        file = request.files["file"]
        filename = file.filename
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_AUDIO and ext not in ALLOWED_VIDEO:
            allowed = ", ".join(sorted(ALLOWED_VIDEO | ALLOWED_AUDIO))
            return jsonify({"error": f"Unsupported file type '{ext or 'none'}'. Allowed: {allowed}"}), 415
        user_id = request.current_user['user_id']
        analysisID = str(uuid.uuid4())

        save_name = f"{uuid.uuid4()}{ext}"
        filepath = os.path.join(VIDEOS_DIR, save_name)
        file.save(filepath)
        file_size = os.path.getsize(filepath)

        # --- A. AUDIO FILE ---
        if ext in ALLOWED_AUDIO:
            audio_data = {"label": "Error", "confidence": 0, "segments": []}
            try:
                with open(filepath, "rb") as f:
                    resp = requests.post(
                        "http://127.0.0.1:5001/predict_audio",
                        files={"file": f},
                        timeout=120
                    )
                    if resp.status_code == 200:
                        audio_data = resp.json()
            except Exception:
                pass

            result = "deepfake" if audio_data['label'] == 'fake' else "authentic"
            confidence = audio_data.get('confidence', 0)
            save_result(user_id, analysisID, filename, result, confidence, file_size)

            return jsonify({
                "type": "audio",
                "result": result,
                "confidence": confidence,
                "audio_analysis": audio_data,
                "flaggedFrames": [],
                "copyright_check": {
                    "violation_detected": False,
                    "reason": "audio_only_no_face_matching"
                },
                "processingTime": round(time.time() - start_time, 2)
            })

        # --- B. VIDEO FILE ---
        elif ext in ALLOWED_VIDEO:
            print(f"Processing Video: {filename}")

            # Submit audio extraction to background thread immediately
            # so it runs concurrently with the frame loop below.
            audio_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            audio_future = audio_executor.submit(
                extract_and_analyse_audio, filepath, VIDEOS_DIR
            )

            cap = cv2.VideoCapture(filepath)
            frameRate = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fake_probs = []
            flagged_frames = []
            first_fake_frame_b64 = None

            # Laptop-safe sampling: 1 frame every 3 seconds,
            # hard-capped at 50 frames total to prevent RAM spikes
            # on long videos.
            sample_interval = max(1, int(frameRate * 3))
            max_frames_to_process = 50
            frames_processed = 0

            while cap.isOpened():
                frameId = cap.get(1)
                ret, frame = cap.read()
                if not ret:
                    break
                if frames_processed >= max_frames_to_process:
                    break
                if frameRate > 0 and frameId % sample_interval != 0:
                    continue

                frames_processed += 1

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_rects = detector(gray, 0)

                for i, d in enumerate(face_rects):
                    x1 = max(0, d.left())
                    y1 = max(0, d.top())
                    x2 = min(frame.shape[1], d.right())
                    y2 = min(frame.shape[0], d.bottom())
                    if x2 - x1 < 10 or y2 - y1 < 10:
                        continue

                    try:
                        crop = cv2.resize(frame[y1:y2, x1:x2], (299, 299)) / 255.0
                        pred = model.predict(crop.reshape(1, 299, 299, 3), verbose=0)[0][0]
                        fake_probs.append(pred)

                        if pred <= 0.5:  # Fake Detected
                            t_name = f"{uuid.uuid4()}.jpg"

                            # --- VISUAL ANNOTATION ---
                            annotated_frame = frame.copy()

                            cx = (x1 + x2) // 2
                            cy = (y1 + y2) // 2
                            radius = max((x2 - x1), (y2 - y1)) // 2 + 10

                            cv2.circle(annotated_frame, (cx, cy), radius, (0, 0, 255), thickness=3)

                            frame_confidence = round((1 - pred) * 100, 1)
                            label = f"MANIPULATED - {frame_confidence}%"
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.9
                            thickness = 2
                            (text_w, text_h), baseline = cv2.getTextSize(
                                label, font, font_scale, thickness
                            )

                            text_x = max(0, cx - text_w // 2)
                            text_y = max(text_h + 5, cy - radius - 10)

                            cv2.rectangle(
                                annotated_frame,
                                (text_x - 4, text_y - text_h - 4),
                                (text_x + text_w + 4, text_y + baseline),
                                (0, 0, 255),
                                cv2.FILLED
                            )
                            cv2.putText(
                                annotated_frame, label,
                                (text_x, text_y),
                                font, font_scale,
                                (255, 255, 255),
                                thickness, cv2.LINE_AA
                            )
                            # --- END ANNOTATION ---

                            cv2.imwrite(os.path.join(FRAMES_DIR, t_name), annotated_frame)
                            flagged_frames.append(f"/uploads/flagged_frames/{t_name}")

                            if first_fake_frame_b64 is None:
                                # Encode the clean unannotated frame for FaceNet
                                # so the red circle does not corrupt the embedding.
                                _, buffer = cv2.imencode('.jpg', frame)
                                first_fake_frame_b64 = base64.b64encode(buffer).decode('utf-8')

                    except Exception:
                        continue

            cap.release()

            # Clear TensorFlow graph nodes accumulated during the loop.
            # Prevents slow memory creep across multiple requests on a
            # low-RAM machine.
            tf.keras.backend.clear_session()

            # Collect audio result. By now the background thread has
            # likely already finished since the frame loop took time.
            audio_data, temp_extracted_audio = audio_future.result()
            audio_executor.shutdown(wait=False)

            # Fusion Logic
            avg_prob = np.mean(fake_probs) if fake_probs else 1.0
            video_result = "deepfake" if avg_prob <= 0.5 else "authentic"
            video_conf = round(
                (sum(p <= 0.5 for p in fake_probs) / len(fake_probs) * 100), 2
            ) if fake_probs else 0

            final_result = video_result
            final_conf = video_conf
            if audio_data.get('label') == 'fake':
                final_result = "deepfake"
                final_conf = (
                    audio_data['confidence']
                    if video_result == "authentic"
                    else (video_conf + audio_data['confidence']) / 2
                )

            save_result(user_id, analysisID, filename, final_result, final_conf, file_size)

            # ==========================================
            # PHASE 3: COPYRIGHT VIOLATION DETECTION
            # ==========================================
            copyright_violation = {
                "violation_detected": False,
                "reason": "authentic_content" if final_result == "authentic" else "no_flagged_frames"
            }

            if final_result == "deepfake" and first_fake_frame_b64:
                print(f"[Phase 3] Synthetic content detected. Checking for copyright violations...")

                try:
                    print(f"[Phase 3] Calling identity service...")
                    identity_response = requests.post(
                        "http://127.0.0.1:5002/match_identity",
                        json={"image": first_fake_frame_b64, "threshold": 0.60},
                        timeout=30
                    )

                    if identity_response.status_code == 200:
                        identity_data = identity_response.json()

                        if identity_data.get("match_found"):
                            matched_entity = identity_data.get("matched_entity", {})
                            license_status = identity_data.get("license_status", "unknown")

                            print(f"[Phase 3] MATCH FOUND: {matched_entity.get('name')} "
                                  f"({matched_entity.get('entity_id')})")
                            print(f"[Phase 3] License Status: {license_status}")

                            try:
                                conn = get_db()
                                c = conn.cursor()
                                c.execute("""
                                    INSERT INTO violation_logs (
                                        analysis_id, entity_id, matched_confidence,
                                        violation_type, flagged_frame_path, timestamp
                                    )
                                    VALUES (?, ?, ?, ?, ?, ?)
                                """, (
                                    analysisID,
                                    matched_entity.get('entity_id'),
                                    matched_entity.get('confidence', 0),
                                    "unauthorized_likeness"
                                    if license_status == "unauthorized"
                                    else "unknown_rights",
                                    flagged_frames[0] if flagged_frames else "in_memory",
                                    datetime.now().isoformat()
                                ))
                                conn.commit()
                            except Exception as e:
                                print(f"[ERROR] Failed to log violation: {e}")

                            copyright_violation = {
                                "violation_detected": True,
                                "matched_entity": matched_entity,
                                "license_status": license_status,
                                "victim_name": matched_entity.get('name', 'Unknown Creator'),
                                "violation_type": (
                                    "unauthorized_likeness"
                                    if license_status == "unauthorized"
                                    else "unknown_rights"
                                )
                            }

                        else:
                            print(f"[Phase 3] No protected identity matched "
                                  f"(score: {identity_data.get('closest_match_score', 0):.2f})")
                            copyright_violation = {
                                "violation_detected": False,
                                "reason": "synthetic_but_unregistered"
                            }
                    else:
                        print(f"[Phase 3] Identity service error: {identity_response.status_code}")

                except Exception as e:
                    print(f"[Phase 3] Error during copyright check: {e}")
                    traceback.print_exc()

            response_data = {
                "type": "video",
                "result": final_result,
                "confidence": round(final_conf, 2),
                "video_analysis": {"label": video_result, "conf": video_conf},
                "audio_analysis": audio_data,
                "flaggedFrames": flagged_frames[:3] if final_result == "deepfake" else [],
                "copyright_check": copyright_violation,
                "processingTime": round(time.time() - start_time, 2)
            }

            return jsonify(response_data)

    except RequestEntityTooLarge:
        raise  # handled by the JSON 413 handler below
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        robust_delete(filepath)
        robust_delete(temp_extracted_audio)


@app.route("/uploads/flagged_frames/<path:filename>")
def serve_flagged_frame(filename):
    return send_from_directory(FRAMES_DIR, filename)


@app.route("/history", methods=["POST"])
@token_required
def get_results():
    user_id = request.current_user['user_id']
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute(
            "SELECT * FROM results WHERE user_id = ? ORDER BY timestamp DESC",
            (user_id,)
        )
        cols = [desc[0] for desc in c.description]
        return jsonify([dict(zip(cols, row)) for row in c.fetchall()])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/history/delete/<int:result_id>", methods=["DELETE"])
@token_required
def delete_result(result_id):
    user_id = request.current_user['user_id']
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT user_id FROM results WHERE id = ?", (result_id,))
        result = c.fetchone()

        if not result:
            return jsonify({"error": "Result not found"}), 404
        if result[0] != user_id:
            return jsonify({"error": "Unauthorized to delete this result"}), 403

        c.execute("DELETE FROM results WHERE id = ?", (result_id,))
        conn.commit()
        return jsonify({"message": "Result deleted successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================================
# CREATOR REGISTRATION ROUTES
# ==========================================

CREATOR_TYPES = {"creator", "celebrity", "brand_character"}
ALLOWED_IMAGE = {'.jpg', '.jpeg', '.png', '.webp'}
MAX_REFERENCE_IMAGES = 10


def get_owned_creator(entity_id, user_id):
    """Returns the registration if user_id owns entity_id, else None."""
    c = get_db().cursor()
    c.execute("""
        SELECT entity_id, name, email, type, consent_at
        FROM creator_registrations
        WHERE entity_id = ? AND owner_user_id = ?
    """, (entity_id, user_id))
    row = c.fetchone()
    if not row:
        return None
    return dict(zip(["entity_id", "name", "email", "type", "consent_at"], row))


def fetch_identities():
    """Returns the identity service's identities keyed by entity_id, or None on failure."""
    response = requests.get("http://127.0.0.1:5002/list_identities", timeout=10)
    if response.status_code != 200:
        return None
    return {c["entity_id"]: c for c in response.json().get("identities", [])}


@app.route("/creators/register", methods=["POST"])
@token_required
def register_creator():
    try:
        data = request.get_json(silent=True)
        required_fields = ["name", "email", "type", "consent"]
        if not data or not all(k in data for k in required_fields):
            return jsonify({"error": f"Missing required fields: {required_fields}"}), 400

        name = str(data["name"]).strip()
        email = str(data["email"]).strip().lower()
        creator_type = data["type"]
        consent = data.get("consent") is True

        if len(name) < 2:
            return jsonify({"error": "Name must be at least 2 characters"}), 400
        if '@' not in email or '.' not in email:
            return jsonify({"error": "Invalid email format"}), 400
        if creator_type not in CREATOR_TYPES:
            return jsonify({"error": "Invalid type. Must be: creator, celebrity, brand_character"}), 400
        if not consent:
            return jsonify({"error": "Consent to biometric data storage is required"}), 400

        entity_id = f"creator_{uuid.uuid4().hex[:8]}"
        now = datetime.now().isoformat()

        conn = get_db()
        conn.execute("""
            INSERT INTO creator_registrations
            (entity_id, owner_user_id, name, email, type, consent_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (entity_id, request.current_user['user_id'], name, email, creator_type, now, now))
        conn.commit()

        print(f"[CREATOR] Registering: {name} ({entity_id})")

        return jsonify({
            "success": True,
            "entity_id": entity_id,
            "message": "Registration successful. Please upload reference images.",
            "next_step": "upload-references"
        }), 201

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/creators/upload-reference", methods=["POST"])
@token_required
def upload_reference():
    try:
        entity_id = request.form.get("entity_id")
        if not entity_id:
            return jsonify({"error": "Missing entity_id"}), 400

        # Name, type and email come from the owner's registration, not the request.
        creator = get_owned_creator(entity_id, request.current_user['user_id'])
        if not creator:
            return jsonify({"error": "Creator not found"}), 404

        files = [f for f in request.files.getlist("files") if f.filename]
        if not files:
            return jsonify({"error": "At least one file is required"}), 400
        if len(files) > MAX_REFERENCE_IMAGES:
            return jsonify({"error": f"Upload at most {MAX_REFERENCE_IMAGES} images"}), 400

        embeddings_stored = 0
        errors = []

        for file in files:
            if os.path.splitext(file.filename)[1].lower() not in ALLOWED_IMAGE:
                errors.append(f"{file.filename}: unsupported image type (use jpg, png or webp)")
                continue
            try:
                file_data = file.read()
                image_b64 = base64.b64encode(file_data).decode('utf-8')

                identity_response = requests.post(
                    "http://127.0.0.1:5002/register_identity",
                    json={
                        "entity_id": entity_id,
                        "name": creator["name"],
                        "type": creator["type"],
                        "email": creator["email"],
                        "consent_date": creator["consent_at"],
                        "image": image_b64
                    },
                    timeout=30
                )

                if identity_response.status_code in [201, 200]:
                    embeddings_stored += 1
                    print(f"[UPLOAD] Embedding stored for {entity_id}: {file.filename}")
                else:
                    error_msg = identity_response.json().get("error", "Unknown error")
                    errors.append(f"{file.filename}: {error_msg}")
                    print(f"[ERROR] Failed to register {file.filename}: {error_msg}")

            except Exception as e:
                error_msg = f"Failed to process {file.filename}: {str(e)}"
                errors.append(error_msg)
                print(f"[ERROR] {error_msg}")

        if embeddings_stored == 0:
            return jsonify({
                "success": False,
                "error": "No embeddings could be stored",
                "details": errors
            }), 400

        return jsonify({
            "success": True,
            "entity_id": entity_id,
            "embeddings_stored": embeddings_stored,
            "status": "active",
            "errors": errors if errors else None
        }), 200

    except RequestEntityTooLarge:
        raise  # handled by the JSON 413 handler below
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/creators/profile/<entity_id>", methods=["GET"])
@token_required
def get_creator_profile(entity_id):
    try:
        creator = get_owned_creator(entity_id, request.current_user['user_id'])
        if not creator:
            return jsonify({"error": "Creator not found"}), 404

        identities = fetch_identities()
        if identities is None:
            return jsonify({"error": "Failed to fetch creator profile"}), 500

        # No identity row yet means no reference image has been stored.
        identity = identities.get(entity_id) or {}
        is_active = bool(identity.get("is_active"))
        references = identity.get("embedding_count", 0)
        return jsonify({
            "entity_id": entity_id,
            "name": creator["name"],
            "type": creator["type"],
            "status": "active" if is_active else "inactive",
            "registered_references": references,
            "created_at": identity.get("created_at"),
            "protection_status": "active" if is_active and references > 0 else "pending"
        }), 200
    except Exception as e:
        print(f"[ERROR] Failed to get profile for {entity_id}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/creators/<entity_id>", methods=["DELETE"])
@token_required
def delete_creator(entity_id):
    try:
        if not get_owned_creator(entity_id, request.current_user['user_id']):
            return jsonify({"error": "Creator not found"}), 404

        response = requests.delete(
            f"http://127.0.0.1:5002/delete_identity/{entity_id}",
            timeout=10
        )
        if response.status_code != 200:
            return jsonify({"error": "Failed to delete creator"}), 500

        conn = get_db()
        conn.execute("DELETE FROM creator_registrations WHERE entity_id = ?", (entity_id,))
        conn.commit()
        return jsonify({
            "success": True,
            "message": f"Creator {entity_id} deleted successfully"
        }), 200
    except Exception as e:
        print(f"[ERROR] Failed to delete {entity_id}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/creators/list", methods=["GET"])
@token_required
def list_creators():
    """Lists only the identities the current user registered."""
    try:
        c = get_db().cursor()
        c.execute("""
            SELECT entity_id, name, type, created_at FROM creator_registrations
            WHERE owner_user_id = ? ORDER BY created_at DESC
        """, (request.current_user['user_id'],))
        owned = c.fetchall()

        identities = fetch_identities()
        if identities is None:
            return jsonify({"error": "Failed to list creators"}), 500

        creators = []
        for entity_id, name, creator_type, created_at in owned:
            identity = identities.get(entity_id) or {}
            creators.append({
                "entity_id": entity_id,
                "name": name,
                "type": creator_type,
                "status": "active" if identity.get("is_active") else "inactive",
                "references": identity.get("embedding_count", 0),
                "created_at": created_at
            })
        return jsonify({"total": len(creators), "creators": creators}), 200
    except Exception as e:
        print(f"[ERROR] Failed to list creators: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False, port=5000)