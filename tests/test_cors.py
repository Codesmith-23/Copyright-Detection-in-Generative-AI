"""T1.2: only the frontend may call the video backend from a browser; internal services allow no browser calls."""
from conftest import AUDIO_SERVICE, FRONTEND_ORIGIN, IDENTITY_SERVICE, VIDEO_BACKEND, register_user

OTHER_SITE = "http://evil.example"


def preflight(client, path, origin, method="POST"):
    return client.open(path, method="OPTIONS", headers={
        "Origin": origin,
        "Access-Control-Request-Method": method,
        "Access-Control-Request-Headers": "authorization,content-type",
    })


def allowed_origin(response):
    return response.headers.get("Access-Control-Allow-Origin")


def test_backend_allows_frontend_origins(client):
    for origin in (FRONTEND_ORIGIN, "http://127.0.0.1:3000"):
        r = preflight(client, "/predict/media", origin)
        assert allowed_origin(r) == origin
        assert "authorization" in r.headers.get("Access-Control-Allow-Headers", "").lower()
    r = client.post("/auth/login", json={"username": "x", "password": "y"}, headers={"Origin": FRONTEND_ORIGIN})
    assert allowed_origin(r) == FRONTEND_ORIGIN


def test_backend_blocks_other_origins(client):
    assert allowed_origin(preflight(client, "/creators/list", OTHER_SITE, "GET")) is None
    assert allowed_origin(preflight(client, "/history", OTHER_SITE)) is None
    r = client.post("/auth/login", json={"username": "x", "password": "y"}, headers={"Origin": OTHER_SITE})
    assert allowed_origin(r) is None


def test_frontend_origin_setting_replaces_defaults(load_service):
    lan = "http://192.168.1.20:3000"
    client = load_service(VIDEO_BACKEND, env={"FRONTEND_ORIGIN": f"{lan}, https://app.example.com"}).app.test_client()
    assert allowed_origin(preflight(client, "/history", lan)) == lan
    assert allowed_origin(preflight(client, "/history", "https://app.example.com")) == "https://app.example.com"
    assert allowed_origin(preflight(client, "/history", FRONTEND_ORIGIN)) is None


def test_backend_requests_without_origin_still_work(client):
    headers = register_user(client)
    assert client.post("/history", headers=headers).status_code == 200


def test_identity_service_sends_no_cors_headers(load_service):
    client = load_service(IDENTITY_SERVICE).app.test_client()
    for origin in (OTHER_SITE, FRONTEND_ORIGIN):
        assert allowed_origin(preflight(client, "/delete_identity/x", origin, "DELETE")) is None
        assert allowed_origin(client.get("/list_identities", headers={"Origin": origin})) is None
    # Server-to-server calls from app.py carry no Origin and still work.
    assert client.get("/health").status_code == 200
    assert client.get("/list_identities").status_code == 200


def test_audio_service_sends_no_cors_headers(load_service):
    client = load_service(AUDIO_SERVICE).app.test_client()
    for origin in (OTHER_SITE, FRONTEND_ORIGIN):
        assert allowed_origin(preflight(client, "/predict_audio", origin)) is None
        assert allowed_origin(client.post("/predict_audio", headers={"Origin": origin})) is None
    # The model is not loaded in tests, but the route answers.
    r = client.post("/predict_audio")
    assert r.status_code == 500 and r.get_json()["error"] == "Model not loaded"
