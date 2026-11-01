"""Baseline behaviour of registration, login and analysis history (unchanged by Phase 1)."""
from conftest import register_user


def test_register_validation(client):
    def reg(**body):
        return client.post("/auth/register", json=body)
    assert reg(username="al", email="a@example.com", password="secret123").status_code == 400
    assert reg(username="alice", email="a@example.com", password="12345").status_code == 400
    assert reg(username="alice", email="not-an-email", password="secret123").status_code == 400
    assert reg(username="alice").status_code == 400


def test_duplicate_username_and_email(client):
    register_user(client, "alice")
    r = client.post("/auth/register", json={"username": "alice", "email": "new@example.com", "password": "secret123"})
    assert r.status_code == 409 and r.get_json()["error"] == "Username already exists"
    r = client.post("/auth/register", json={"username": "alice2", "email": "alice@example.com", "password": "secret123"})
    assert r.status_code == 409 and r.get_json()["error"] == "Email already registered"


def test_login(client):
    register_user(client, "alice")
    ok = client.post("/auth/login", json={"username": "alice", "password": "secret123"})
    assert ok.status_code == 200 and ok.get_json()["user"]["username"] == "alice"
    bad = client.post("/auth/login", json={"username": "alice", "password": "wrong-password"})
    assert bad.status_code == 401 and bad.get_json()["error"] == "Invalid username or password"
    unknown = client.post("/auth/login", json={"username": "nobody", "password": "secret123"})
    assert unknown.status_code == 401


def test_token_checks(client):
    headers = register_user(client)
    assert client.get("/auth/verify", headers=headers).get_json()["user"]["username"] == "alice"
    assert client.get("/auth/verify").status_code == 401
    assert client.get("/auth/verify", headers={"Authorization": "Bearer"}).status_code == 401
    assert client.get("/auth/verify", headers={"Authorization": "Bearer not.a.token"}).status_code == 401


def test_history_is_per_user(backend, client):
    alice = register_user(client, "alice")
    bob = register_user(client, "bob")
    backend.save_result(1, "a-1", "alice.mp4", "deepfake", 91.5, 1000)
    backend.save_result(2, "b-1", "bob.mp4", "authentic", 80.0, 2000)

    alice_rows = client.post("/history", headers=alice).get_json()
    assert [row["file_name"] for row in alice_rows] == ["alice.mp4"]
    bob_rows = client.post("/history", headers=bob).get_json()
    assert [row["file_name"] for row in bob_rows] == ["bob.mp4"]
    assert client.post("/history").status_code == 401


def test_history_delete_checks_owner(client, backend):
    alice = register_user(client, "alice")
    bob = register_user(client, "bob")
    backend.save_result(1, "a-1", "alice.mp4", "deepfake", 91.5, 1000)
    row_id = client.post("/history", headers=alice).get_json()[0]["id"]

    assert client.delete(f"/history/delete/{row_id}", headers=bob).status_code == 403
    assert client.delete("/history/delete/9999", headers=alice).status_code == 404
    assert client.delete(f"/history/delete/{row_id}", headers=alice).status_code == 200
    assert client.post("/history", headers=alice).get_json() == []
