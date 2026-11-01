"""T1.3: creator identities require login, belong to the user who registered them, and record consent."""
import base64
import io
import sqlite3
import types

import numpy as np
import pytest

from conftest import IDENTITY_SERVICE, register_user


class FakeIdentityService:
    """In-memory stand-in for the identity service endpoints app.py calls."""

    def __init__(self):
        self.identities = {}
        self.registrations = []

    @staticmethod
    def _response(status, body):
        return types.SimpleNamespace(status_code=status, json=lambda: body)

    def post(self, url, json=None, **_):
        assert url.endswith("/register_identity")
        self.registrations.append(json)
        identity = self.identities.setdefault(json["entity_id"], {
            "entity_id": json["entity_id"], "name": json["name"], "type": json["type"],
            "email": json.get("email", ""), "created_at": "2026-11-01T00:00:00",
            "is_active": 1, "embedding_count": 0})
        identity["embedding_count"] += 1
        return self._response(201, {"success": True})

    def get(self, url, **_):
        assert url.endswith("/list_identities")
        return self._response(200, {"total": len(self.identities), "identities": list(self.identities.values())})

    def delete(self, url, **_):
        self.identities.pop(url.rsplit("/", 1)[1], None)
        return self._response(200, {"success": True})


@pytest.fixture
def env(backend, client, monkeypatch):
    fake = FakeIdentityService()
    for method in ("post", "get", "delete"):
        monkeypatch.setattr(backend.requests, method, getattr(fake, method))
    return types.SimpleNamespace(
        c=client, backend=backend, fake=fake,
        alice=register_user(client, "alice"), bob=register_user(client, "bob"))


def register(env, headers, **overrides):
    body = {"name": "Alice Star", "email": "alice@example.com", "type": "celebrity", "consent": True, **overrides}
    return env.c.post("/creators/register", json=body, headers=headers)


def upload(env, headers, entity_id, files=(("face.jpg", b"jpg-bytes"),)):
    data = {"files": [(io.BytesIO(content), name) for name, content in files]}
    if entity_id is not None:
        data["entity_id"] = entity_id
    return env.c.post("/creators/upload-reference", data=data, headers=headers, content_type="multipart/form-data")


def test_all_creator_routes_require_login(env):
    assert env.c.post("/creators/register", json={}).status_code == 401
    assert env.c.post("/creators/upload-reference").status_code == 401
    assert env.c.get("/creators/profile/creator_x").status_code == 401
    assert env.c.delete("/creators/creator_x").status_code == 401
    assert env.c.get("/creators/list").status_code == 401


def test_full_flow_records_owner_and_consent(env):
    r = register(env, env.alice)
    assert r.status_code == 201
    entity = r.get_json()["entity_id"]

    profile = env.c.get(f"/creators/profile/{entity}", headers=env.alice).get_json()
    assert profile["protection_status"] == "pending" and profile["registered_references"] == 0

    r = upload(env, env.alice, entity, [("a.jpg", b"1"), ("b.png", b"2")])
    assert r.status_code == 200 and r.get_json()["embeddings_stored"] == 2

    sent = env.fake.registrations[-1]
    assert (sent["name"], sent["type"], sent["email"]) == ("Alice Star", "celebrity", "alice@example.com")
    assert sent["consent_date"]

    profile = env.c.get(f"/creators/profile/{entity}", headers=env.alice).get_json()
    assert profile["protection_status"] == "active" and profile["registered_references"] == 2


def test_other_users_get_not_found(env):
    entity = register(env, env.alice).get_json()["entity_id"]
    upload(env, env.alice, entity)

    assert upload(env, env.bob, entity).status_code == 404
    assert env.c.get(f"/creators/profile/{entity}", headers=env.bob).status_code == 404
    assert env.c.delete(f"/creators/{entity}", headers=env.bob).status_code == 404
    assert entity in env.fake.identities

    assert env.c.delete(f"/creators/{entity}", headers=env.alice).status_code == 200
    assert entity not in env.fake.identities
    assert env.c.get(f"/creators/profile/{entity}", headers=env.alice).status_code == 404


def test_list_shows_only_own_identities(env):
    a = register(env, env.alice).get_json()["entity_id"]
    b = register(env, env.bob, name="Bob Brand", email="bob@example.com", type="brand_character").get_json()["entity_id"]
    upload(env, env.alice, a)
    upload(env, env.bob, b)

    alice_list = env.c.get("/creators/list", headers=env.alice).get_json()
    assert [x["entity_id"] for x in alice_list["creators"]] == [a] and alice_list["total"] == 1
    assert [x["entity_id"] for x in env.c.get("/creators/list", headers=env.bob).get_json()["creators"]] == [b]


def test_register_validation(env):
    assert register(env, env.alice, consent=False).status_code == 400
    assert register(env, env.alice, consent="yes").status_code == 400
    assert register(env, env.alice, type="admin").status_code == 400
    assert register(env, env.alice, email="nope").status_code == 400
    assert env.c.post("/creators/register", data="x", headers=env.alice).status_code == 400


def test_upload_validation(env):
    entity = register(env, env.alice).get_json()["entity_id"]
    assert upload(env, env.alice, None).status_code == 400
    r = upload(env, env.alice, entity, [("notes.txt", b"x")])
    assert r.status_code == 400 and "unsupported image type" in r.get_json()["details"][0]
    r = upload(env, env.alice, entity, [(f"{i}.jpg", b"x") for i in range(11)])
    assert r.status_code == 400 and "at most 10" in r.get_json()["error"]
    assert env.fake.identities == {}


def test_creator_registrations_table(env):
    columns = [row[1] for row in env.backend.get_db().execute("PRAGMA table_info(creator_registrations)")]
    assert columns == ["entity_id", "owner_user_id", "name", "email", "type", "consent_at", "created_at"]


def test_identity_service_register_identity(load_service, monkeypatch):
    service = load_service(IDENTITY_SERVICE)
    client = service.app.test_client()
    assert client.post("/register_identity", data="not json").status_code == 400
    assert client.post("/register_identity", json={"entity_id": "x"}).status_code == 400

    monkeypatch.setattr(service.cv2, "imdecode", lambda *_: np.zeros((10, 10, 3), np.uint8), raising=False)
    monkeypatch.setattr(service.cv2, "IMREAD_COLOR", 1, raising=False)
    monkeypatch.setattr(service, "extract_face_from_image", lambda image: image)
    monkeypatch.setattr(service, "compute_embedding", lambda face: np.ones(4) / 2)
    body = {"entity_id": "creator_1", "name": "Alice", "type": "celebrity", "email": "a@example.com",
            "consent_date": "2026-10-30T12:00:00", "image": base64.b64encode(b"img").decode()}
    assert client.post("/register_identity", json=body).status_code == 201

    row = sqlite3.connect(service.DB_PATH).execute(
        "SELECT consent_agreement_date, type, email FROM protected_entities WHERE entity_id = 'creator_1'").fetchone()
    assert row == ("2026-10-30T12:00:00", "celebrity", "a@example.com")
