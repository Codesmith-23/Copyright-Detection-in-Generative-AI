"""
Identity Service (Port 5002)
Handles face embedding extraction and identity matching against protected database.
Runs independently from main Flask backend.
"""

import os
import sqlite3
import cv2
import numpy as np
import pickle
import base64

from flask import Flask, request, jsonify

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.models import load_model

try:
    from keras_facenet import FaceNet
    FACENET_AVAILABLE = True
except ImportError:
    FACENET_AVAILABLE = False
    print("[WARNING] keras-facenet not installed. Install with: pip install keras-facenet")

# No CORS: only the video backend (app.py) calls this service, server to server.
app = Flask(__name__)

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database.db")
FACENET_MODEL_PATH = os.path.join(BASE_DIR, "models", "facenet_keras.h5")
FACE_DETECTOR = None

print("[Identity Service] Initializing on port 5002...")

# ==========================================
# MODEL LOADING
# ==========================================

def load_facenet_model():
    global facenet_model
    try:
        if FACENET_AVAILABLE:
            print("[FaceNet] Loading keras-facenet model...")
            facenet_model = FaceNet()
            print("[FaceNet] keras-facenet model loaded successfully (512-dim embeddings)")
            return facenet_model
        elif os.path.exists(FACENET_MODEL_PATH):
            print(f"[FaceNet] Loading from {FACENET_MODEL_PATH}")
            facenet_model = load_model(FACENET_MODEL_PATH, custom_objects={'tf': tf})
            print("[FaceNet] Model loaded successfully")
            return facenet_model
        else:
            print("[FaceNet] keras-facenet not available, using fallback embedding model...")
            facenet_model = _load_simple_embedding_model()
            print("[FaceNet] Fallback model loaded (MobileNetV2-based)")
            return facenet_model
    except Exception as e:
        print(f"[ERROR] Failed to load FaceNet: {e}")
        import traceback
        traceback.print_exc()
        return None

def _load_simple_embedding_model():
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import GlobalAveragePooling2D
    from tensorflow.keras.models import Model
    base_model = MobileNetV2(input_shape=(160, 160, 3), include_top=False, weights='imagenet')
    x = GlobalAveragePooling2D()(base_model.output)
    model = Model(inputs=base_model.input, outputs=x)
    return model

def load_face_detector():
    global FACE_DETECTOR
    try:
        prototxt = os.path.join(BASE_DIR, "models", "deploy.prototxt")
        model_file = os.path.join(BASE_DIR, "models", "res10_300x300_ssd_iter_140000.caffemodel")
        if os.path.exists(prototxt) and os.path.exists(model_file):
            FACE_DETECTOR = cv2.dnn.readNetFromCaffe(prototxt, model_file)
            print("[Face Detector] OpenCV DNN loaded")
        else:
            print("[Face Detector] Model files not found, using fallback (dlib)")
            import dlib
            FACE_DETECTOR = dlib.get_frontal_face_detector()
        return FACE_DETECTOR
    except Exception as e:
        print(f"[ERROR] Failed to load face detector: {e}")
        return None

facenet_model = None
try:
    facenet_model = load_facenet_model()
except Exception:
    print("[WARNING] FaceNet not available, using fallback embeddings")

face_detector = load_face_detector()

# ==========================================
# DATABASE INIT
# ==========================================

def init_identity_db():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS protected_entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                type TEXT CHECK(type IN ('creator', 'celebrity', 'brand_character', 'test')),
                email TEXT,
                consent_agreement_date TEXT,
                created_at TEXT NOT NULL,
                is_active BOOLEAN DEFAULT 1
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS identity_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id TEXT NOT NULL,
                embedding BLOB NOT NULL,
                source_file_name TEXT,
                registration_date TEXT NOT NULL,
                FOREIGN KEY (entity_id) REFERENCES protected_entities(entity_id)
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS license_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id TEXT NOT NULL,
                licensee_name TEXT,
                license_type TEXT CHECK(license_type IN ('exclusive', 'non_exclusive', 'unauthorized')),
                allowed_usage TEXT,
                start_date TEXT,
                end_date TEXT,
                FOREIGN KEY (entity_id) REFERENCES protected_entities(entity_id)
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS violation_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                matched_confidence REAL,
                violation_type TEXT,
                flagged_frame_path TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (entity_id) REFERENCES protected_entities(entity_id)
            )
        """)
        conn.commit()
        print("[DB] Identity tables initialized")

init_identity_db()

# ==========================================
# IN-MEMORY EMBEDDING CACHE
# ==========================================
# Loaded once at startup. Eliminates all repeated pickle.loads() calls
# and SQLite reads during every match request. On a laptop with
# 50 creators x 10 photos the RAM cost is roughly 30-50MB, loaded once.
# The cache is refreshed after every registration or deletion write.

_embedding_cache = {}

def reload_embedding_cache():
    """
    Reads all active embeddings from SQLite once and holds them in RAM.
    Structure: { entity_id: { "name": str, "type": str, "embeddings": [np_array, ...] } }
    Called at startup and after every write to identity_embeddings.
    """
    global _embedding_cache
    new_cache = {}
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("""
                SELECT ie.entity_id, ie.embedding, pe.name, pe.type
                FROM identity_embeddings ie
                JOIN protected_entities pe ON ie.entity_id = pe.entity_id
                WHERE pe.is_active = 1
            """)
            for entity_id, blob, name, entity_type in c.fetchall():
                try:
                    embedding = pickle.loads(blob)
                    if entity_id not in new_cache:
                        new_cache[entity_id] = {
                            "name": name,
                            "type": entity_type,
                            "embeddings": []
                        }
                    new_cache[entity_id]["embeddings"].append(embedding)
                except Exception as e:
                    print(f"[CACHE] Failed to load embedding for {entity_id}: {e}")

        _embedding_cache = new_cache
        total_embeddings = sum(len(v["embeddings"]) for v in new_cache.values())
        print(f"[CACHE] Loaded {len(new_cache)} identities, "
              f"{total_embeddings} embeddings into memory")
    except Exception as e:
        print(f"[CACHE] Failed to reload cache: {e}")

# Load once at startup
reload_embedding_cache()

# ==========================================
# FACE PROCESSING FUNCTIONS
# --- DO NOT MODIFY ---
# ==========================================

def extract_face_from_image(image_array, face_size=160):
    """
    Extract primary face from image.
    Returns: aligned face (160x160x3) or None
    """
    try:
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image_array

        h, w = rgb_image.shape[:2]
        blob = cv2.dnn.blobFromImage(rgb_image, 1.0, (300, 300), [104, 117, 123], False, False)

        if FACE_DETECTOR and hasattr(FACE_DETECTOR, 'setInput'):
            FACE_DETECTOR.setInput(blob)
            detections = FACE_DETECTOR.forward()
            if len(detections) > 0 and detections[0, 0, 0, 2] > 0.5:
                det = detections[0, 0, 0]
                x1 = int(det[3] * w)
                y1 = int(det[4] * h)
                x2 = int(det[5] * w)
                y2 = int(det[6] * h)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                face_crop = rgb_image[y1:y2, x1:x2]
                face_aligned = cv2.resize(face_crop, (face_size, face_size))
                return face_aligned
        else:
            import dlib
            detector = dlib.get_frontal_face_detector()
            rects = detector(rgb_image, 1)
            if len(rects) > 0:
                rect = rects[0]
                x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                face_crop = rgb_image[y1:y2, x1:x2]
                face_aligned = cv2.resize(face_crop, (face_size, face_size))
                return face_aligned

        return None
    except Exception as e:
        print(f"[ERROR] Face extraction failed: {e}")
        return None


def compute_embedding(face_image):
    """
    --- FROZEN: DO NOT MODIFY ---
    Compute face embedding from aligned face image.
    """
    try:
        if facenet_model is None:
            return None

        face_aligned = cv2.resize(face_image, (160, 160))
        samples = np.expand_dims(face_aligned, axis=0)

        if FACENET_AVAILABLE and hasattr(facenet_model, 'embeddings'):
            embedding = facenet_model.embeddings(samples)[0]
        else:
            samples = samples.astype('float32') / 255.0
            embedding = facenet_model.predict(samples, verbose=0)[0]

        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm

    except Exception as e:
        print(f"[ERROR] Embedding computation failed: {e}")
        return None


def similarity_score(embedding1, embedding2):
    """
    --- FROZEN: DO NOT MODIFY ---
    Compute cosine similarity between two embeddings.
    """
    try:
        e1 = embedding1 / np.linalg.norm(embedding1)
        e2 = embedding2 / np.linalg.norm(embedding2)
        return float(np.dot(e1, e2))
    except Exception:
        return 0.0

# ==========================================
# API ENDPOINTS
# ==========================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "identity_service"}), 200


@app.route("/match_identity", methods=["POST"])
def match_identity():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "Missing 'image' field"}), 400

        try:
            image_data = base64.b64decode(data["image"])
            image_array = cv2.imdecode(
                np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR
            )
        except Exception as e:
            return jsonify({"error": f"Invalid image format: {e}"}), 400

        face_crop = extract_face_from_image(image_array)
        if face_crop is None:
            return jsonify({"match_found": False, "error": "No face detected in image"}), 200

        embedding = compute_embedding(face_crop)
        if embedding is None:
            return jsonify({"match_found": False, "error": "Failed to compute embedding"}), 200

        threshold = data.get("threshold", 0.85)
        best_match = None
        best_score = -1.0

        # Search the in-memory cache — zero disk I/O, zero pickle calls.
        # All deserialization was done once at startup by reload_embedding_cache().
        for entity_id, entity_data in _embedding_cache.items():
            for stored_embedding in entity_data["embeddings"]:
                try:
                    score = similarity_score(embedding, stored_embedding)
                    print(f"[MATCH] Comparing against {entity_data['name']}: {score:.4f}")
                    if score > best_score:
                        best_score = score
                        best_match = {
                            "entity_id": entity_id,
                            "name": entity_data["name"],
                            "type": entity_data["type"],
                            "confidence": score
                        }
                except Exception as e:
                    print(f"[ERROR] Similarity failed for {entity_id}: {e}")
                    continue

        if best_match and best_score >= threshold:
            with sqlite3.connect(DB_PATH) as conn:
                c = conn.cursor()
                c.execute("""
                    SELECT license_type FROM license_records
                    WHERE entity_id = ?
                    ORDER BY start_date DESC LIMIT 1
                """, (best_match["entity_id"],))
                result = c.fetchone()
                license_status = result[0] if result else "unknown"

            return jsonify({
                "match_found": True,
                "matched_entity": best_match,
                "license_status": license_status
            }), 200
        else:
            return jsonify({
                "match_found": False,
                "closest_match_score": best_score
            }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/register_identity", methods=["POST"])
def register_identity():
    try:
        data = request.get_json(silent=True)
        required_fields = ["entity_id", "name", "type", "image"]
        if not data or not all(field in data for field in required_fields):
            return jsonify({"error": f"Missing required fields: {required_fields}"}), 400

        entity_id = data["entity_id"]
        name = data["name"]
        entity_type = data["type"]
        email = data.get("email", "")

        try:
            image_data = base64.b64decode(data["image"])
            image_array = cv2.imdecode(
                np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR
            )
        except Exception as e:
            return jsonify({"error": f"Invalid image format: {e}"}), 400

        face_crop = extract_face_from_image(image_array)
        if face_crop is None:
            return jsonify({"error": "No face detected in image"}), 400

        embedding = compute_embedding(face_crop)
        if embedding is None:
            return jsonify({"error": "Failed to compute embedding"}), 400

        from datetime import datetime
        timestamp = datetime.now().isoformat()
        # The consent time recorded when the creator registered, if provided.
        consent_date = data.get("consent_date") or timestamp

        try:
            with sqlite3.connect(DB_PATH) as conn:
                c = conn.cursor()
                c.execute("""
                    INSERT OR IGNORE INTO protected_entities
                    (entity_id, name, type, email, consent_agreement_date, created_at, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, 1)
                """, (entity_id, name, entity_type, email, consent_date, timestamp))

                embedding_blob = pickle.dumps(embedding)
                c.execute("""
                    INSERT INTO identity_embeddings
                    (entity_id, embedding, source_file_name, registration_date)
                    VALUES (?, ?, ?, ?)
                """, (entity_id, embedding_blob, f"ref_{timestamp}.jpg", timestamp))

                c.execute("""
                    INSERT INTO license_records (entity_id, license_type, start_date)
                    VALUES (?, ?, ?)
                """, (entity_id, "unauthorized", timestamp))

                conn.commit()

            # Refresh the in-memory cache so the new identity is
            # immediately available for matching without a server restart.
            reload_embedding_cache()

            return jsonify({
                "success": True,
                "entity_id": entity_id,
                "message": f"Identity registered: {name}",
                "embedding_dim": len(embedding)
            }), 201

        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                return jsonify({"error": f"Entity {entity_id} already exists"}), 409
            raise

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/list_identities", methods=["GET"])
def list_identities():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("""
                SELECT entity_id, name, type, email, created_at, is_active,
                       (SELECT COUNT(*) FROM identity_embeddings
                        WHERE entity_id = pe.entity_id) as embedding_count
                FROM protected_entities pe
                ORDER BY created_at DESC
            """)
            columns = [desc[0] for desc in c.description]
            identities = [dict(zip(columns, row)) for row in c.fetchall()]
            return jsonify({"total": len(identities), "identities": identities}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/delete_identity/<entity_id>", methods=["DELETE"])
def delete_identity(entity_id):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute(
                "UPDATE protected_entities SET is_active = 0 WHERE entity_id = ?",
                (entity_id,)
            )
            c.execute(
                "DELETE FROM identity_embeddings WHERE entity_id = ?",
                (entity_id,)
            )
            c.execute(
                "DELETE FROM license_records WHERE entity_id = ?",
                (entity_id,)
            )
            conn.commit()

        # Evict deleted entity from the in-memory cache immediately.
        reload_embedding_cache()

        return jsonify({"success": True, "message": f"Identity {entity_id} deleted"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Identity Service (Copyright Detection)")
    print("=" * 50)
    print("Listening on http://127.0.0.1:5002")
    print("=" * 50 + "\n")
    app.run(debug=False, port=5002, host="127.0.0.1")