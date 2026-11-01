"""
Shared test setup.

Each test loads a service (app.py, identity_service.py or api_audio.py) from a
fresh copy in a temporary folder, so it gets its own empty database, uploads
folder and JWT secret file and never touches the real ones in the repo.

The heavy ML libraries (TensorFlow, PyTorch, dlib, OpenCV, librosa, moviepy,
Hugging Face) are replaced with small stand-ins, so the suite runs in seconds
with only requirements-dev.txt installed. Model downloads fail on purpose, so
each service starts with model = None, exactly as when a download fails.
"""

import importlib.util
import os
import shutil
import sys
import types

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_BACKEND = "DeepFake/backend/app.py"
IDENTITY_SERVICE = "DeepFake/backend/identity_service.py"
AUDIO_SERVICE = "Voice Deepfake/api_audio.py"

FRONTEND_ORIGIN = "http://localhost:3000"
TEST_JWT_SECRET = "test-secret-" + "x" * 32


def _offline(*_, **__):
    raise RuntimeError("offline test: model downloads are disabled")


def _stub_modules(monkeypatch):
    """Registers stand-ins for the ML libraries in sys.modules for one test."""

    def mod(name, **attrs):
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    # Video backend and identity service
    mod("cv2", ocl=types.SimpleNamespace(setUseOpenCL=lambda *_: None))
    mod("dlib", get_frontal_face_detector=lambda: None)
    keras = mod("tensorflow.keras")
    mod("tensorflow.keras.models", load_model=lambda *_, **__: None)
    tf = mod("tensorflow", keras=keras)
    keras.backend = types.SimpleNamespace(clear_session=lambda: None)
    tf.keras = keras
    mod("moviepy")
    mod("moviepy.editor", VideoFileClip=None)
    mod("huggingface_hub", hf_hub_download=_offline)
    # Makes "from keras_facenet import FaceNet" raise ImportError.
    monkeypatch.setitem(sys.modules, "keras_facenet", None)

    # Audio service
    class Module:
        def __init__(self, *_, **__):
            pass

    layer = lambda *_, **__: None
    nn = types.SimpleNamespace(Module=Module, Conv2d=layer, Sequential=layer, LSTM=layer, Linear=layer)
    mod("torch", cuda=types.SimpleNamespace(is_available=lambda: False), nn=nn, load=_offline)
    monkeypatch.setitem(sys.modules, "torch.nn", nn)
    resnet = types.SimpleNamespace(children=lambda: [None, None, None])
    mod("torchvision", models=types.SimpleNamespace(resnet34=lambda **_: resnet))
    mod("librosa")
    mod("scipy")
    mod("scipy.stats", entropy=None)
    mod("scipy.signal", find_peaks=None)


@pytest.fixture
def load_service(tmp_path, monkeypatch):
    """
    Returns a function that loads a service and returns its module.
    Usage: module = load_service(VIDEO_BACKEND, env={"FRONTEND_ORIGIN": "..."})
    By default JWT_SECRET is set; pass env={"JWT_SECRET": None} to unset it.
    """
    _stub_modules(monkeypatch)
    counter = {"n": 0}

    def load(rel_path, env=None):
        settings = {"JWT_SECRET": TEST_JWT_SECRET, "FRONTEND_ORIGIN": None}
        settings.update(env or {})
        for key, value in settings.items():
            if value is None:
                monkeypatch.delenv(key, raising=False)
            else:
                monkeypatch.setenv(key, value)

        service_dir = tmp_path / os.path.dirname(rel_path).replace(" ", "_")
        service_dir.mkdir(parents=True, exist_ok=True)
        dest = service_dir / os.path.basename(rel_path)
        shutil.copy(os.path.join(REPO_ROOT, rel_path), dest)

        counter["n"] += 1
        spec = importlib.util.spec_from_file_location(f"service_under_test_{counter['n']}", dest)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load


@pytest.fixture
def backend(load_service):
    """The video backend (app.py) module, loaded with the test JWT secret."""
    return load_service(VIDEO_BACKEND)


@pytest.fixture
def client(backend):
    return backend.app.test_client()


def register_user(client, name="alice", password="secret123"):
    """Registers a user and returns the Authorization header for them."""
    r = client.post("/auth/register", json={"username": name, "email": f"{name}@example.com", "password": password})
    assert r.status_code == 201, r.get_json()
    return {"Authorization": f"Bearer {r.get_json()['token']}"}
