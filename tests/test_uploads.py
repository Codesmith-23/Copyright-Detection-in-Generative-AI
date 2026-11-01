"""Upload checks (roadmap T2.3): unsupported file types and the upload size limit."""

import io

from conftest import AUDIO_SERVICE, VIDEO_BACKEND, register_user


def post_media(client, headers, name, content=b"data"):
    return client.post("/predict/media", data={"file": (io.BytesIO(content), name)},
                       headers=headers, content_type="multipart/form-data")


def test_unsupported_type_is_rejected_before_saving(backend, client):
    headers = register_user(client)
    for name in ("notes.txt", "script.exe", "noextension"):
        r = post_media(client, headers, name)
        assert r.status_code == 415, name  # was a 500 crash
        assert "Unsupported file type" in r.get_json()["error"]
    import os
    assert os.listdir(backend.VIDEOS_DIR) == []  # nothing was written to uploads/


def test_extension_check_ignores_case(client):
    # An allowed extension in capitals gets past the type check (here it then
    # fails later because the model is not loaded, which is not a 415).
    headers = register_user(client)
    assert post_media(client, headers, "CLIP.MP4").status_code != 415


def test_video_backend_upload_over_limit_returns_413(load_service):
    client = load_service(VIDEO_BACKEND, env={"MAX_UPLOAD_MB": "1"}).app.test_client()
    headers = register_user(client)
    r = post_media(client, headers, "big.mp4", b"x" * (2 * 1024 * 1024))
    assert r.status_code == 413
    assert r.get_json()["error"] == "File is too large. The limit is 1 MB."


def test_reference_image_upload_over_limit_returns_413(load_service):
    client = load_service(VIDEO_BACKEND, env={"MAX_UPLOAD_MB": "1"}).app.test_client()
    headers = register_user(client)
    entity = client.post("/creators/register", headers=headers, json={
        "name": "Alice", "email": "a@example.com", "type": "celebrity", "consent": True}).get_json()["entity_id"]
    r = client.post("/creators/upload-reference", headers=headers, content_type="multipart/form-data",
                    data={"entity_id": entity, "files": [(io.BytesIO(b"x" * (2 * 1024 * 1024)), "a.jpg")]})
    assert r.status_code == 413


def test_default_limit_is_500_mb(backend):
    assert backend.app.config["MAX_CONTENT_LENGTH"] == 500 * 1024 * 1024


def test_audio_service_rejects_unsupported_type(load_service, monkeypatch):
    audio = load_service(AUDIO_SERVICE)
    monkeypatch.setattr(audio, "model", object())  # pretend the model loaded
    client = audio.app.test_client()
    r = client.post("/predict_audio", data={"file": (io.BytesIO(b"x"), "clip.mp4")},
                    content_type="multipart/form-data")
    assert r.status_code == 415
    import os
    assert os.listdir(audio.AUDIO_UPLOAD_FOLDER) == []


def test_audio_service_upload_over_limit_returns_413(load_service, monkeypatch):
    audio = load_service(AUDIO_SERVICE, env={"MAX_UPLOAD_MB": "1"})
    monkeypatch.setattr(audio, "model", object())
    r = audio.app.test_client().post("/predict_audio", content_type="multipart/form-data",
                                     data={"file": (io.BytesIO(b"x" * (2 * 1024 * 1024)), "clip.wav")})
    assert r.status_code == 413
    assert "limit is 1 MB" in r.get_json()["error"]
