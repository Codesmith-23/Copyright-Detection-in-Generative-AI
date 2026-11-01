"""T1.1: the JWT signing secret comes from the environment or a generated local file."""
import os
import subprocess
from datetime import datetime, timedelta

import jwt
import pytest

from conftest import REPO_ROOT, VIDEO_BACKEND, register_user

OLD_HARDCODED_KEY = "your-secret-key-change-in-production-2026"


def test_no_hardcoded_secret_in_source():
    with open(os.path.join(REPO_ROOT, VIDEO_BACKEND)) as f:
        assert OLD_HARDCODED_KEY not in f.read()


def test_token_signed_with_old_key_is_rejected(client):
    register_user(client)
    forged = jwt.encode(
        {"user_id": 1, "username": "alice", "exp": datetime.utcnow() + timedelta(hours=1)},
        OLD_HARDCODED_KEY, algorithm="HS256")
    assert client.post("/history", headers={"Authorization": f"Bearer {forged}"}).status_code == 401


def test_env_secret_is_used(backend):
    assert backend.app.config["SECRET_KEY"].startswith("test-secret-")
    assert not os.path.exists(backend.JWT_SECRET_FILE)


def test_short_env_secret_is_refused(load_service):
    with pytest.raises(RuntimeError, match="at least 32"):
        load_service(VIDEO_BACKEND, env={"JWT_SECRET": "short"})


def test_generated_secret_persists_across_restarts(load_service):
    first = load_service(VIDEO_BACKEND, env={"JWT_SECRET": None})
    assert os.path.exists(first.JWT_SECRET_FILE)
    with open(first.JWT_SECRET_FILE) as f:
        assert len(f.read().strip()) >= 32
    if os.name == "posix":
        assert os.stat(first.JWT_SECRET_FILE).st_mode & 0o777 == 0o600
    headers = register_user(first.app.test_client())

    second = load_service(VIDEO_BACKEND, env={"JWT_SECRET": None})  # simulated restart
    assert second.app.config["SECRET_KEY"] == first.app.config["SECRET_KEY"]
    assert second.app.test_client().get("/auth/verify", headers=headers).status_code == 200


def test_secret_file_is_git_ignored():
    result = subprocess.run(["git", "check-ignore", "-q", "DeepFake/backend/.jwt_secret"], cwd=REPO_ROOT)
    assert result.returncode == 0
