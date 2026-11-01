"""
Failures must never turn into an "authentic" verdict (roadmap T2.2).

Each failure case checks the status code and that nothing was saved to history.
The video pipeline (OpenCV, dlib, the Keras model) is replaced with small fakes.
"""

import io
import types

import numpy as np
import pytest

from conftest import register_user


class FakeCapture:
    """Pretends to be a 3-frame video at 1 fps (every frame gets sampled)."""

    def __init__(self, frames=3):
        self.frames = frames
        self.pos = 0

    def isOpened(self):
        return True

    def get(self, prop):
        return {"fps": 1.0, "count": self.frames, 1: self.pos}[prop]

    def read(self):
        if self.pos >= self.frames:
            return False, None
        self.pos += 1
        return True, np.zeros((40, 40, 3), np.uint8)

    def release(self):
        pass


FACE = types.SimpleNamespace(left=lambda: 0, top=lambda: 0, right=lambda: 30, bottom=lambda: 30)


class FakeModel:
    def __init__(self, prob=0.9, fail=False):
        self.prob, self.fail = prob, fail

    def predict(self, *_, **__):
        if self.fail:
            raise RuntimeError("model exploded")
        return [[self.prob]]


@pytest.fixture
def video(backend, client, monkeypatch):
    """Wires fake OpenCV/dlib into the backend. Returns helpers to configure each test."""
    cv2 = backend.cv2
    monkeypatch.setattr(cv2, "CAP_PROP_FPS", "fps", raising=False)
    monkeypatch.setattr(cv2, "CAP_PROP_FRAME_COUNT", "count", raising=False)
    monkeypatch.setattr(cv2, "COLOR_BGR2GRAY", 0, raising=False)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: FakeCapture(), raising=False)
    monkeypatch.setattr(cv2, "cvtColor", lambda frame, _code: frame, raising=False)
    monkeypatch.setattr(cv2, "resize", lambda img, size: np.zeros((*size, 3)), raising=False)

    state = types.SimpleNamespace(headers=register_user(client))

    def configure(faces=True, model=None, audio=None):
        monkeypatch.setattr(backend, "detector", lambda _gray, _up: [FACE] if faces else [])
        monkeypatch.setattr(backend, "model", model)
        audio = audio or {"label": "unavailable", "confidence": 0, "segments": []}
        monkeypatch.setattr(backend, "extract_and_analyse_audio", lambda *_: (audio, None))

    def upload(name="clip.mp4"):
        return client.post("/predict/media", data={"file": (io.BytesIO(b"video"), name)},
                           headers=state.headers, content_type="multipart/form-data")

    def history():
        return client.post("/history", headers=state.headers).get_json()

    state.configure, state.upload, state.history = configure, upload, history
    return state


def audio_service_returns(backend, monkeypatch, status, body=None):
    """Makes the backend's call to the audio service get this response."""
    def fake_post(url, **_):
        assert url == backend.AUDIO_SERVICE_URL
        return types.SimpleNamespace(status_code=status, json=lambda: body or {}, text=str(body))
    monkeypatch.setattr(backend.requests, "post", fake_post)


# --- Video model not loaded ---

def test_video_without_model_returns_503(video):
    video.configure(model=None)
    r = video.upload()
    assert r.status_code == 503  # was: "authentic", saved to history
    assert "video model is not loaded" in r.get_json()["error"]
    assert video.history() == []


# --- Every face fails ---

def test_all_faces_failing_returns_500(video):
    video.configure(model=FakeModel(fail=True))
    r = video.upload()
    assert r.status_code == 500  # was: "authentic" with 0% confidence
    assert "failed for every detected face" in r.get_json()["error"]
    assert video.history() == []


# --- No faces ---

def test_no_faces_and_no_audio_returns_422(video):
    video.configure(faces=False, model=FakeModel())
    r = video.upload()
    assert r.status_code == 422  # was: "authentic"
    assert "No verdict possible" in r.get_json()["error"]
    assert video.history() == []


@pytest.mark.parametrize("label, expected", [("fake", "deepfake"), ("real", "authentic")])
def test_no_faces_with_audio_gives_audio_only_verdict(video, label, expected):
    video.configure(faces=False, model=FakeModel(), audio={"label": label, "confidence": 87.5, "segments": []})
    r = video.upload()
    assert r.status_code == 200
    body = r.get_json()
    assert body["result"] == expected
    assert body["confidence"] == 87.5
    assert body["video_analysis"]["label"] == "no_faces"
    assert [e["result"] for e in video.history()] == [expected]


# --- Normal video analysis still works ---

def test_working_video_still_gets_a_verdict(video):
    video.configure(model=FakeModel(prob=0.9))  # prob > 0.5 means real
    r = video.upload()
    assert r.status_code == 200
    assert r.get_json()["result"] == "authentic"
    assert r.get_json()["audio_analysis"]["label"] == "unavailable"
    assert len(video.history()) == 1


# --- Audio-only uploads ---

@pytest.mark.parametrize("status, body", [(500, {"error": "Model not loaded"}), (200, {"label": "weird"})])
def test_audio_upload_with_audio_service_failing_returns_503(backend, video, monkeypatch, status, body):
    audio_service_returns(backend, monkeypatch, status, body)
    r = video.upload("voice.wav")
    assert r.status_code == 503  # was: "authentic", saved to history
    assert video.history() == []


def test_audio_upload_with_audio_service_down_returns_503(backend, video, monkeypatch):
    def down(*_, **__):
        raise backend.requests.ConnectionError("connection refused")
    monkeypatch.setattr(backend.requests, "post", down)
    assert video.upload("voice.mp3").status_code == 503
    assert video.history() == []


def test_audio_upload_with_working_service_is_saved(backend, video, monkeypatch):
    audio_service_returns(backend, monkeypatch, 200, {"label": "fake", "confidence": 91.0, "segments": []})
    r = video.upload("voice.wav")
    assert r.status_code == 200 and r.get_json()["result"] == "deepfake"
    assert [e["result"] for e in video.history()] == ["deepfake"]


# --- The helper itself ---

def test_call_audio_service_never_raises(backend, monkeypatch, tmp_path):
    clip = tmp_path / "a.wav"
    clip.write_bytes(b"x")
    audio_service_returns(backend, monkeypatch, 503)
    assert backend.call_audio_service(str(clip))["label"] == "unavailable"
    assert backend.call_audio_service(str(tmp_path / "missing.wav"))["label"] == "unavailable"
