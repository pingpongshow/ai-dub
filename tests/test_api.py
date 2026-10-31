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


def test_paths_and_settings_stay_safe(client, cfg, media_root, tmp_path):
    c, svc = client
    ep = synth.episode(media_root / "Show - 03.mkv")
    # A reference-clip removal can't name a file outside the profile (it used to delete it).
    prof = svc.profiles.get_or_create("Show")
    prof.characters["c1"] = {"name": "A", "refs": [{"file": "refs/c1/a.flac"}]}
    (prof.root / "refs" / "c1").mkdir(parents=True)
    (prof.root / "refs" / "c1" / "a.flac").write_bytes(b"x")
    prof.save()
    for bad in (str(ep), "../../../media/Anime/Show - 03.mkv"):
        assert c.post("/api/profiles/show/characters/c1", json={"action": "remove_ref", "ref": bad}).status_code == 200
    assert ep.exists()
    c.post("/api/profiles/show/characters/c1", json={"action": "remove_ref", "ref": "refs/c1/a.flac"})
    assert not (prof.root / "refs" / "c1" / "a.flac").exists()
    assert svc.profiles.get("../appdata") is None
    # A broken link doesn't break the folder listing; set-aside files can't be queued.
    (media_root / "Gone - 01.mkv").symlink_to(tmp_path / "nowhere.mkv")
    names = [e["name"] for e in c.get("/api/browse", params={"path": str(media_root)}).json()["entries"]]
    assert names == ["Show - 03.mkv"]
    aside = media_root / "Show - 03.mkv.aidub-original"
    shutil.copy(ep, aside)
    assert c.post("/api/jobs", json={"paths": [str(aside)]}).json()["errors"][0]["error"] == "not a video file"
    # Settings: stored keys are never shown back and survive a save of the page's copy;
    # the media folders can't be changed from the page; a broken shape is refused.
    conf = cfg.user_overrides()
    conf["llm_endpoint"] = {"api_key": "sk-secret-123"}
    cfg.save_overrides(conf)
    s = c.get("/api/settings").json()
    assert "sk-secret-123" not in str(s)
    o = s["overrides"]
    o["media"] = [{"name": "All", "path": "/"}]
    o["fit"] = {"timing": "window"}
    assert c.post("/api/settings", json={"overrides": o}).status_code == 200
    saved = cfg.user_overrides()
    assert saved["llm_endpoint"]["api_key"] == "sk-secret-123" and saved["media"] == conf["media"]
    assert saved["fit"] == {"timing": "window"}
    assert c.post("/api/settings", json={"overrides": {"paths": None, "fit": 3}}).status_code == 400


def test_review_pauses_shows_and_applies_choices(client, cfg, media_root):
    from aidub import jobs as J
    c, svc = client
    ep = synth.episode(media_root / "Review Show - S01E01.mkv")
    jid = c.post("/api/jobs", json={"paths": [str(ep)], "review": "always"}).json()["added"][0]
    assert c.post("/api/jobs", json={"paths": [str(ep)], "review": "sometimes"}).status_code == 400
    svc.run_one(svc.store.next_queued())
    job = svc.store.get(jid)
    assert job["status"] == J.REVIEW and "waiting for review" in job["message"]
    assert job["id"] in [j["id"] for j in c.get("/api/jobs", params={"view": "queue"}).json()]
    assert svc.store.next_queued() is None                      # the queue moves on without it

    r = c.get(f"/api/jobs/{jid}/review").json()
    assert r["ready"] and r["editable"] and r["mode"] == "always" and r["duration"] > 10
    assert r["media"]["state"] == "ready" and r["speakers"] and r["lines"]
    spoken = [ln for ln in r["lines"] if ln["kind"] == "speech"]
    assert spoken
    # The preview plays and seeks (range requests), the subtitles and waveform load.
    v = c.get(f"/api/jobs/{jid}/review/media/preview.mp4", headers={"Range": "bytes=0-99"})
    assert v.status_code == 206 and len(v.content) == 100
    assert c.get(f"/api/jobs/{jid}/review/media/subs.ja.vtt").text.startswith("WEBVTT")
    assert c.get(f"/api/jobs/{jid}/review/media/peaks.json").json()["peaks"]
    assert c.get(f"/api/jobs/{jid}/review/media/state.json").status_code == 404

    # Keep the first spoken line in the original; give the second its own audio as the voice.
    first, second = spoken[0], spoken[1] if len(spoken) > 1 else spoken[0]
    choice = {"regions": [{"start": first["start"] - 0.05, "end": first["end"] + 0.05, "kind": "song",
                           "action": "keep", "source": "user"}],
              "cues": {}, "lines": {str(second["id"]): {"character": "own"}}}
    assert c.put(f"/api/jobs/{jid}/review", json=choice).status_code == 200
    assert c.get(f"/api/jobs/{jid}/review").json()["decisions"]["regions"][0]["action"] == "keep"
    assert c.post(f"/api/jobs/{jid}/review/start", json=choice).status_code == 200
    assert svc.store.get(jid)["status"] == J.QUEUED
    svc.run_one(svc.store.next_queued())
    job = svc.store.get(jid)
    assert job["status"] == J.DONE, job["message"]
    assert job["record"]["qc"]["review"]["changes"] >= 1
    # Once done, decisions can't be changed.
    assert c.put(f"/api/jobs/{jid}/review", json=choice).status_code in (404, 409)


def test_show_rules_settings(client):
    c, svc = client
    svc.profiles.get_or_create("Rules Show")
    r = c.get("/api/profiles/rules-show/rules").json()
    assert r["habits"]["romaji_lyrics"] is True and r["regions"] == []
    assert c.put("/api/profiles/rules-show/rules", json={"review": "always", "habits": {"tilde_signs": False,
                                                                                       "unknown": True}}).status_code == 200
    r = c.get("/api/profiles/rules-show/rules").json()
    assert r["review"] == "always" and r["habits"]["tilde_signs"] is False and "unknown" not in r["habits"]
    assert c.put("/api/profiles/rules-show/rules", json={"review": "sometimes"}).status_code == 400
