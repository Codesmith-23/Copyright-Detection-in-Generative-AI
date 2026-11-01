import os
import torch
import torch.nn as nn
import librosa
import numpy as np 
import shutil
import time
import uuid
import gc
import atexit
import traceback
from flask import Flask, request, jsonify
from torchvision import models
from scipy.stats import entropy
from scipy.signal import find_peaks
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from huggingface_hub import hf_hub_download 

# No CORS: only the video backend (app.py) calls this service, server to server.
app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_PATH = "deepfake_model_resnet_lstm.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEGMENT_DURATION = 4.0
OVERLAP = 2.0
BATCH_SIZE = 32       # windows per model call
MAX_WINDOWS = 300     # about 10 minutes of audio; longer files are sampled evenly

# Dedicated folder for audio uploads
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_UPLOAD_FOLDER = os.path.join(BASE_DIR, "audio_uploads")
os.makedirs(AUDIO_UPLOAD_FOLDER, exist_ok=True)

ALLOWED_AUDIO = {'.mp3', '.wav', '.flac', '.m4a'}

# Largest accepted upload, in MB (same default as the video backend).
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "500"))
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_MB * 1024 * 1024


@app.errorhandler(RequestEntityTooLarge)
def upload_too_large(_error):
    return jsonify({"error": f"File is too large. The limit is {MAX_UPLOAD_MB} MB."}), 413

# --- CLEANUP LOGIC ---
def cleanup_folder():
    """Wipes the upload folder."""
    if os.path.exists(AUDIO_UPLOAD_FOLDER):
        for filename in os.listdir(AUDIO_UPLOAD_FOLDER):
            file_path = os.path.join(AUDIO_UPLOAD_FOLDER, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception: pass

def robust_delete(file_path):
    """Retries deletion to handle Windows file locking."""
    if not file_path or not os.path.exists(file_path):
        return
    
    # Force garbage collection to release file handles
    gc.collect()
    
    for i in range(5):
        try:
            os.remove(file_path)
            return
        except PermissionError:
            time.sleep(0.1)
        except Exception as e:
            print(f" Error deleting {file_path}: {e}")

# Startup & Shutdown Cleanup
cleanup_folder()
atexit.register(cleanup_folder)

# --- MODEL DEFINITION ---
class ResNetLSTM(nn.Module):
    def __init__(self):
        super(ResNetLSTM, self).__init__()
        resnet = models.resnet34(weights=None)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.mean(dim=2)
        x = x.permute(0, 2, 1)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.fc(x)

# --- LOAD MODEL ---
# print(f"Loading Audio Model on {DEVICE}...")
# model = ResNetLSTM()
# try:
#     state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
#     model.load_state_dict(state_dict)
#     model.to(DEVICE).eval()
#     print("Audio Model Loaded!")
# except Exception as e:
#     print(f"CRITICAL ERROR: Could not load audio model. {e}")
#     model = None


print(f"Loading Audio Model from Hugging Face...")
model = ResNetLSTM()
try:
    # Your Repo ID and Filename
    REPO_ID = "Codesmith-23/deepfake-detector-v1"
    FILENAME = "deepfake_model_resnet_lstm.pth"
    
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    print(f"Audio Model loaded successfully from: {model_path}")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load audio model. {e}")
    model = None



# --- PREPROCESSING ---
def preprocess_chunk_with_metadata(y_chunk, sr, chunk_idx, start_time):
    try:
        # Pad or truncate to fixed length
        target_len = int(SEGMENT_DURATION * 16000)
        if len(y_chunk) < target_len:
            y_chunk = np.pad(y_chunk, (0, target_len - len(y_chunk)), mode='constant')
        else:
            y_chunk = y_chunk[:target_len]

        # Compute metadata for quality filtering
        rms = librosa.feature.rms(y=y_chunk)[0]
        zcr = librosa.feature.zero_crossing_rate(y_chunk)[0]
        
        # Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y_chunk, sr=16000, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_db = (mel_spec_db + 80) / 80

        # Tensor conversion
        tensor = torch.tensor(mel_spec_db).float().unsqueeze(0)
        tensor = torch.nn.functional.interpolate(
            tensor.unsqueeze(0), size=(128, 128), mode='bilinear'
        ).squeeze(0)
        
        metadata = {
            'index': chunk_idx,
            'start_time': start_time,
            'is_silence': np.mean(rms) < 0.015,
            'is_noisy': np.mean(zcr) > 0.20
        }
        
        return tensor.unsqueeze(0).to(DEVICE), metadata
        
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None, None

# --- WINDOWING ---
def window_starts(n_samples, chunk_samples, stride_samples):
    """
    Start positions (in samples) of the analysis windows.
    - Empty audio: no windows.
    - Shorter than one window: one window at 0 (it gets zero-padded).
    - Otherwise every stride, plus one window aligned to the end, so the
      last seconds of the clip are always analysed.
    """
    if n_samples <= 0:
        return []
    if n_samples <= chunk_samples:
        return [0]
    starts = list(range(0, n_samples - chunk_samples + 1, stride_samples))
    tail = n_samples - chunk_samples
    if starts[-1] != tail:
        starts.append(tail)
    return starts


def cap_windows(starts, max_windows=MAX_WINDOWS):
    """Keeps at most max_windows starts, spread evenly over the whole clip."""
    if len(starts) <= max_windows:
        return starts
    picks = np.unique(np.linspace(0, len(starts) - 1, max_windows).round().astype(int))
    return [starts[i] for i in picks]


def predict_probs(tensors):
    """Runs the model in batches of BATCH_SIZE and returns the fake probability per window."""
    probs = []
    with torch.no_grad():
        for i in range(0, len(tensors), BATCH_SIZE):
            batch = torch.cat(tensors[i:i + BATCH_SIZE], dim=0)
            outputs = model(batch)
            probs.append(torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


# --- ANALYSIS HELPERS ---
def detect_anomaly_patterns(probs, window_size=3):
    if len(probs) < window_size:
        return {'has_spikes': False, 'has_plateau': False, 'spike_score': 0, 'plateau_score': 0, 'num_spikes': 0}
    
    peaks, _ = find_peaks(probs, prominence=0.15, width=1)
    has_spikes = len(peaks) > 0
    spike_score = np.max(probs[peaks]) if has_spikes else 0
    
    high_regions = probs > 0.55
    max_consecutive = 0
    current = 0
    for val in high_regions:
        if val: current += 1
        else:
            max_consecutive = max(max_consecutive, current)
            current = 0
    max_consecutive = max(max_consecutive, current)
    
    has_plateau = max_consecutive >= window_size
    plateau_score = np.mean(probs[probs > 0.55]) if has_plateau else 0
    
    return {
        'has_spikes': has_spikes, 'has_plateau': has_plateau, 
        'spike_score': spike_score, 'plateau_score': plateau_score,
        'num_spikes': len(peaks)
    }

def compute_statistics(probs):
    if len(probs) == 0: return {'median': 0, 'mean': 0, 'q75': 0, 'top3_mean': 0}
    sorted_probs = np.sort(probs)
    return {
        'mean': np.mean(probs),
        'median': np.median(probs),
        'q75': np.percentile(probs, 75),
        'top3_mean': np.mean(sorted_probs[-3:]) if len(probs) >= 3 else np.max(probs)
    }

# --- MAIN ENDPOINT ---
@app.route('/predict_audio', methods=['POST'])
def predict():
    if not model: return jsonify({"error": "Model not loaded"}), 500
    if 'file' not in request.files: return jsonify({"error": "No file"}), 400

    file = request.files['file']
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_AUDIO:
        return jsonify({"error": f"Unsupported audio type '{ext or 'none'}'. Allowed: {', '.join(sorted(ALLOWED_AUDIO))}"}), 415
    filename = secure_filename(file.filename)
    temp_path = os.path.join(AUDIO_UPLOAD_FOLDER, f"{uuid.uuid4()}_{filename}")
    
    try:
        file.save(temp_path)
        
        # Load Audio (Force 16kHz)
        y, sr = librosa.load(temp_path, sr=16000)
        
        chunk_samples = int(SEGMENT_DURATION * sr)
        stride_samples = int((SEGMENT_DURATION - OVERLAP) * sr)
        
        duration = len(y) / sr
        all_starts = window_starts(len(y), chunk_samples, stride_samples)
        if not all_starts:
            return jsonify({"error": "The audio file contains no audio."}), 422
        starts = cap_windows(all_starts)

        tensors = []
        meta_list = []
        
        # Processing Loop
        chunk_idx = 0
        for start in starts:
            chunk = y[start : start + chunk_samples]
            tensor, meta = preprocess_chunk_with_metadata(chunk, sr, chunk_idx, start/sr)
            if tensor is not None:
                tensors.append(tensor)
                meta_list.append(meta)
                chunk_idx += 1

        if not tensors:
            # No verdict: never report "real" when nothing could be analysed.
            return jsonify({"error": "None of the audio could be analysed."}), 422

        # Batch Inference (in mini-batches to keep memory bounded)
        probs = predict_probs(tensors)

        # Filter: Remove Silent/Noisy chunks
        active_indices = [i for i, m in enumerate(meta_list) if not m['is_silence'] and not m['is_noisy']]
        if len(active_indices) < 3: 
            active_indices = [i for i, m in enumerate(meta_list) if not m['is_silence']] # Fallback
        
        # If still too few, use all
        if len(active_indices) < 2: active_indices = list(range(len(probs)))
            
        active_probs = probs[active_indices]
        
        # Statistics & Pattern Detection
        stats = compute_statistics(active_probs)
        patterns = detect_anomaly_patterns(active_probs)
        
        # --- DECISION LOGIC (Rules 1-5) ---
        is_fake = False
        conf = 0.0
        
        if patterns['has_spikes'] and patterns['spike_score'] > 0.85:
            is_fake, conf = True, patterns['spike_score'] * 100
        elif patterns['has_plateau'] and patterns['plateau_score'] > 0.65:
            is_fake, conf = True, patterns['plateau_score'] * 100
        elif stats['q75'] > 0.60 and stats['top3_mean'] > 0.65:
            is_fake, conf = True, stats['top3_mean'] * 100
        elif patterns['num_spikes'] >= 2 and stats['top3_mean'] > 0.58:
            is_fake, conf = True, stats['top3_mean'] * 100
        elif stats['median'] > 0.55 and stats['mean'] > 0.50:
            is_fake, conf = True, stats['median'] * 100
        else:
            is_fake, conf = False, (1 - stats['top3_mean']) * 100

        # Build Response
        final_label = "fake" if is_fake else "real"
        segments = []
        if is_fake:
            for i in active_indices:
                if probs[i] > 0.5:
                    m = meta_list[i]
                    segments.append({
                        "start": round(float(m['start_time']), 2),
                        "end": round(float(min(m['start_time'] + SEGMENT_DURATION, duration)), 2),
                        "confidence": round(float(probs[i]) * 100, 2),
                        "label": "fake"
                    })

        return jsonify({
            "label": final_label,
            "confidence": round(float(conf), 2),
            "segments": segments,
            "windows_analysed": len(tensors),
            "windows_total": len(all_starts),
            "capped": len(starts) < len(all_starts),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    
    finally:
        robust_delete(temp_path)

if __name__ == '__main__':
    app.run(port=5001, debug=False)