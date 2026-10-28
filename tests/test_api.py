"""Web API: browsing stays inside the media folders, destructive actions need
confirmation, keys are never shown back, models are listed for install."""

import shutil

import pytest
from fastapi.testclient import TestClient

import synth
from aidub.api import create_app
from aidub.service import Service

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


@pytest.fixture
def client(cfg):
    svc = Service(cfg)  # not started: tests drive the queue directly
    return TestClient(create_app(cfg, svc)), svc


def test_browse_and_inspect(client, media_root, tmp_path):
    c, _ = client
    ep = synth.episode(media_root / "Show - 01.mkv")
    roots = c.get("/api/browse").json()
    assert [e["name"] for e in roots["entries"]] == ["Anime"]
    listing = c.get("/api/browse", params={"path": str(media_root)}).json()
    assert [e["name"] for e in listing["entries"]] == ["Show - 01.mkv"]
    assert c.get("/api/browse", params={"path": str(tmp_path)}).status_code == 403
    assert c.get("/api/browse", params={"path": str(media_root / "..")}).status_code == 403
    info = c.post("/api/inspect", json={"path": str(ep)}).json()
    assert info["english_audio"] is False and info["foreign_tracks"][0]["language"] == "ja"
    assert info["profile_guess"] == "Show"


def test_queue_add_and_remove(client, media_root, tmp_path):
    c, svc = client
    ep = synth.episode(media_root / "Show - 02.mkv")
    outside = tmp_path / "outside.mkv"
    shutil.copy(ep, outside)
    r = c.post("/api/jobs", json={"paths": [str(ep), str(outside)]}).json()
    assert len(r["added"]) == 1 and "outside" in r["errors"][0]["error"]
    again = c.post("/api/jobs", json={"paths": [str(ep)]}).json()
    assert again["errors"][0]["error"] == "already in the queue"
    q = c.get("/api/jobs", params={"view": "queue"}).json()
    assert len(q) == 1 and q[0]["status"] == "queued"
    assert c.delete(f"/api/jobs/{q[0]['id']}").status_code == 200
    assert c.get("/api/jobs", params={"view": "queue"}).json() == []


def test_confirmations_and_secrets(client, cfg):
    c, svc = client
    assert c.post("/api/originals/delete", json={"ids": [1]}).status_code == 400
    svc.profiles.get_or_create("Some Show")
    assert c.delete("/api/profiles/some-show").status_code == 400
    assert c.delete("/api/profiles/some-show", params={"confirm": "true"}).status_code == 200
    assert c.post("/api/secrets", json={"name": "BAZARR_API_KEY", "value": "abc123secret"}).status_code == 200
    s = c.get("/api/settings").json()
    assert s["secrets"]["BAZARR_API_KEY"] is True
    assert "abc123secret" not in str(s)
    assert c.post("/api/secrets", json={"name": "NOT_A_KEY", "value": "x"}).status_code == 400
    assert (cfg.appdata / "secrets.yaml").stat().st_mode & 0o077 == 0


def test_models_listing(client):
    c, _ = client
    models = c.get("/api/models").json()
    names = {m["name"] for m in models}
    assert {"asr", "llm", "tts-indextts", "separate"} <= names
    assert all("licence" in m and m["installed"] is False for m in models)


def test_listening_pages(client, cfg):
    c, _ = client
    assert c.get("/listen/").status_code == 404  # nothing to review
    page = cfg.work / "listen" / "voice"
    page.mkdir(parents=True)
    (page / "index.html").write_text("<p>test</p>")
    (page / "001-A.m4a").write_bytes(b"x" * 10)
    assert "voice/" in c.get("/listen/").text
    r = c.get("/listen/voice", follow_redirects=False)
    assert r.status_code in (307, 302) and r.headers["location"] == "/listen/voice/"
    assert c.get("/listen/voice/").text == "<p>test</p>"
    r = c.get("/listen/voice/001-A.m4a")
    assert r.content == b"x" * 10 and r.headers["content-type"] == "audio/mp4"
    assert c.get("/listen/../config.yaml").status_code == 404
    assert c.get("/listen/voice/%2e%2e/%2e%2e/secrets.yaml").status_code == 404


def test_listening_feedback(client, cfg):
    c, _ = client
    (cfg.work / "listen" / "check").mkdir(parents=True)
    assert c.get("/api/listen/feedback", params={"page": "check"}).json() == {}
    assert c.post("/api/listen/feedback", json={"page": "check", "item": "spy-001", "verdict": "poor",
                                                "comment": " man's voice for a girl "}).json()["ok"]
    c.post("/api/listen/feedback", json={"page": "check/", "item": "spy-002", "verdict": "acceptable"})
    got = c.get("/api/listen/feedback", params={"page": "check"}).json()
    assert got["spy-001"]["verdict"] == "poor" and got["spy-001"]["comment"] == "man's voice for a girl"
    assert got["spy-002"]["verdict"] == "acceptable"
    c.post("/api/listen/feedback", json={"page": "check", "item": "spy-002", "verdict": ""})  # cleared
    assert "spy-002" not in c.get("/api/listen/feedback", params={"page": "check"}).json()
    assert c.post("/api/listen/feedback", json={"page": "check", "item": "x", "verdict": "great"}).status_code == 400
    assert c.post("/api/listen/feedback", json={"page": "../..", "item": "x", "verdict": "poor"}).status_code == 404
    assert c.post("/api/listen/feedback", json={"page": "", "item": "x", "verdict": "poor"}).status_code == 404
    assert c.post("/api/listen/feedback", json={"page": "nope", "item": "x", "verdict": "poor"}).status_code == 404
    assert c.get("/listen-feedback.js").status_code == 200
