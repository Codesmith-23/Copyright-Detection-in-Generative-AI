import subprocess
import time
import os
import signal
import sys
import threading
import shutil
import json

# ==============================================================================
#  CONFIGURATION
# ==============================================================================
# Paths are worked out from where this file lives, so nothing here needs
# editing after a clone or a pull. Each service uses the Python from its own
# virtual environment (.venv or venv inside its folder) when one exists,
# otherwise the Python running this script.
#
# To override anything on your machine only, create orchestrator_local.json
# next to this file (it is git-ignored), e.g.:
#   {"VIDEO_PYTHON": "C:\\path\\to\\python.exe", "FRONTEND_MODE": "dev"}
# Keys: AUDIO_PYTHON, VIDEO_PYTHON, AUDIO_DIR, VIDEO_DIR, FRONTEND_DIR, FRONTEND_MODE
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_CONFIG_FILE = os.path.join(ROOT_DIR, "orchestrator_local.json")


def load_local_config():
    if not os.path.exists(LOCAL_CONFIG_FILE):
        return {}
    with open(LOCAL_CONFIG_FILE, encoding="utf-8") as f:
        return json.load(f)


def find_venv_python(service_dir):
    """Returns the Python inside service_dir/.venv or service_dir/venv, else this Python."""
    for venv in (".venv", "venv"):
        for exe in (os.path.join("Scripts", "python.exe"), os.path.join("bin", "python")):
            candidate = os.path.join(service_dir, venv, exe)
            if os.path.exists(candidate):
                return candidate
    return sys.executable


local_config = load_local_config()

AUDIO_DIR = local_config.get("AUDIO_DIR") or os.path.join(ROOT_DIR, "Voice Deepfake")
VIDEO_DIR = local_config.get("VIDEO_DIR") or os.path.join(ROOT_DIR, "DeepFake", "backend")
FRONTEND_DIR = local_config.get("FRONTEND_DIR") or os.path.join(ROOT_DIR, "DeepFake", "frontend")

AUDIO_PYTHON = local_config.get("AUDIO_PYTHON") or find_venv_python(AUDIO_DIR)
VIDEO_PYTHON = local_config.get("VIDEO_PYTHON") or find_venv_python(VIDEO_DIR)

# "dev"  = hot reload, red overlays
# "prod" = clean UI, no error screens <- use for demo
FRONTEND_MODE = local_config.get("FRONTEND_MODE", "prod")
# ==============================================================================

# Global list to track processes for cleanup
processes = []

def install_and_import_psutil():
    """Ensures psutil is installed for accurate port killing."""
    try:
        import psutil
        return psutil
    except ImportError:
        print(" Installing required package 'psutil' for port management...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
        import psutil
        return psutil

# Load psutil
psutil = install_and_import_psutil()

def kill_process_on_port(port):
    """Finds and kills any process listening on a specific port."""
    killed = False
    try:
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                for con in proc.net_connections():
                    if con.laddr.port == port:
                        print(f" Port {port} is blocked by PID {proc.pid} ({proc.info['name']}) -> KILLING...")
                        proc.kill()
                        killed = True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception as e:
        print(f" Warning cleaning port {port}: {e}")
    
    if killed:
        time.sleep(1) # Give OS time to release port

import shutil

def cleanup():
    """Kills all started processes."""
    print("\n Shutting down services...")
    
    # 1. Kill child processes we spawned
    for p in processes:
        try:
            if sys.platform == 'win32':
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(p.pid)], 
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                p.terminate()
        except:
            pass

    # 2. Wipe flagged frames folder directly
    frames_dir = os.path.join(VIDEO_DIR, "uploads", "flagged_frames")
    if os.path.exists(frames_dir):
        for f in os.listdir(frames_dir):
            file_path = os.path.join(frames_dir, f)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f" Could not delete {f}: {e}")
        print(f" Flagged frames cleared: {frames_dir}")

    # 3. Final sweep of ports just in case
    kill_process_on_port(3000)
    kill_process_on_port(5000)
    kill_process_on_port(5001)
    kill_process_on_port(5002)


def signal_handler(sig, frame):
    cleanup()
    sys.exit(0)

def start_service(name, cmd, cwd):
    print(f" Starting {name}...")
    try:
        # shell=False for Python (cleaner), shell=True for NPM (required on Windows)
        use_shell = "npm" in cmd[0]
        
        # Windows group creation flag to help with kills
        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0
        
        p = subprocess.Popen(
            cmd, 
            cwd=cwd, 
            shell=use_shell,
            creationflags=creation_flags
        )
        processes.append(p)
        return p
    except Exception as e:
        print(f" Failed to start {name}: {e}")
        return None

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    print(" PRE-FLIGHT CHECK: Clearing ports...")
    kill_process_on_port(5002) # Identity Service
    kill_process_on_port(5001) # Audio
    kill_process_on_port(5000) # Video
    kill_process_on_port(3000) # Frontend

    print("\n Ports clear. Launching services...\n")

    # 1. Start Identity Service (Phase 1 - Copyright Detection Infrastructure)
    start_service("Identity Service (Copyright Detection)", [VIDEO_PYTHON, "identity_service.py"], VIDEO_DIR)
    time.sleep(2)

    # 2. Start Audio API
    start_service("Audio API", [AUDIO_PYTHON, "api_audio.py"], AUDIO_DIR)
    time.sleep(1)

    # 3. Start Video Backend (integrates with Identity Service for Phase 3)
    start_service("Video Backend API", [VIDEO_PYTHON, "app.py"], VIDEO_DIR)
    time.sleep(2)

    # 4. Start Frontend
    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"

    if FRONTEND_MODE == "prod":
        print(" Building frontend for production (this takes ~30s)...")
        result = subprocess.run([npm_cmd, "run", "build"], cwd=FRONTEND_DIR, shell=True)
        if result.returncode != 0:
            print(" Frontend build FAILED. Check errors above. Falling back to dev mode...")
            start_service("Frontend (Dev - fallback)", [npm_cmd, "run", "dev"], FRONTEND_DIR)
        else:
            print(" Build successful. Starting production server...")
            start_service("Frontend (Production)", [npm_cmd, "start"], FRONTEND_DIR)
    else:
        start_service("Frontend (Dev)", [npm_cmd, "run", "dev"], FRONTEND_DIR)
    
    # Keep alive loop
    try:
        while True:
            time.sleep(1)
            # Check if any process died unexpectedly
            for p in processes:
                if p.poll() is not None:
                    print(" A service died unexpectedly! Shutting down...")
                    cleanup()
                    sys.exit(1)
    except KeyboardInterrupt:
        cleanup()
