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
    # Once done, the review stays open for changes and a redub, until the dub is kept.
    assert c.get(f"/api/jobs/{jid}/review").json()["redub"]
    assert c.post(f"/api/jobs/{jid}/verify", json={"verdict": "good"}).status_code == 200
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


def test_check_a_finished_dub_then_keep_it_or_put_the_original_back(client, cfg, media_root):
    import hashlib
    from aidub import jobs as J, originals
    c, svc = client
    eps = [synth.episode(media_root / f"Check Show - S01E0{n}.mkv") for n in (1, 2)]
    digests = [hashlib.sha256(e.read_bytes()).hexdigest() for e in eps]
    ids = [c.post("/api/jobs", json={"paths": [str(e)], "review": "off"}).json()["added"][0] for e in eps]
    for _ in ids:
        svc.run_one(svc.store.next_queued())
    assert all(svc.store.get(i)["status"] == J.DONE for i in ids)
    pending = c.get("/api/verify").json()
    assert sorted(p["id"] for p in pending) == sorted(ids) and all(p["original_size"] > 0 for p in pending)
    v = c.get(f"/api/jobs/{ids[0]}/verify").json()
    assert v["ready"] and v["original"]["exists"] and v["duration"] > 10
    # The header counts them, and the folder browser marks both episodes dubbed.
    assert c.get("/api/status").json()["counts"] == {"queue": 0, "review": 0, "check": 2, "originals": 2}
    listing = c.get("/api/browse", params={"path": str(media_root)}).json()["entries"]
    assert all(e["dubbed"] for e in listing if e["name"].startswith("Check Show"))
    # The check files and what the job had at its review are left (for a redub); what
    # was made after it (voiced takes, the mix) is gone. The check files play.
    left = {p.name for p in (cfg.work / f"job-{ids[0]}").iterdir()}
    assert {"verify", "review", "lines.json", "state.json", "audio"} <= left and not {"mix", "deliver"} & left
    r = c.get(f"/api/jobs/{ids[0]}/verify/media/preview.mp4", headers={"Range": "bytes=0-9"})
    assert r.status_code == 206 and c.get(f"/api/jobs/{ids[0]}/verify/media/dub.m4a").status_code == 200
    # Good: the original and the check files go, the dub stays.
    assert c.post(f"/api/jobs/{ids[0]}/verify", json={"verdict": "good"}).status_code == 200
    assert not originals.aside_name(eps[0]).exists() and not (cfg.work / f"job-{ids[0]}").exists()
    assert any(s.title == "English (AI dub)" for s in __import__("aidub.media", fromlist=["probe"]).probe(eps[0]).audio)
    # Put the original back: the file is the untouched original again, and no dub copy is left.
    assert c.post(f"/api/jobs/{ids[1]}/verify", json={"verdict": "restore"}).status_code == 200
    assert hashlib.sha256(eps[1].read_bytes()).hexdigest() == digests[1]
    assert not originals.aside_name(eps[1]).exists() and not originals.aside_name(eps[1], "dubbed").exists()
    assert c.get("/api/verify").json() == []
    assert c.post(f"/api/jobs/{ids[1]}/verify", json={"verdict": "maybe"}).status_code == 409


def test_review_a_finished_dub_and_redub_it_with_changes(client, cfg, media_root):
    import hashlib
    from aidub import jobs as J, originals
    c, svc = client
    ep = synth.episode(media_root / "Redub Show - S01E01.mkv")
    digest = hashlib.sha256(ep.read_bytes()).hexdigest()
    jid = c.post("/api/jobs", json={"paths": [str(ep)], "review": "off"}).json()["added"][0]
    svc.run_one(svc.store.next_queued())
    assert svc.store.get(jid)["status"] == J.DONE
    assert c.get(f"/api/jobs/{jid}/verify").json()["can_redub"]
    # Nothing was flagged and it never stopped, yet the whole review page opens and can be changed.
    r = c.get(f"/api/jobs/{jid}/review").json()
    assert r["ready"] and r["editable"] and r["redub"] and r["lines"]
    assert c.post(f"/api/jobs/{jid}/review/media", json={}).status_code == 200
    first = next(ln for ln in r["lines"] if ln["kind"] == "speech")
    choice = {"regions": [{"start": first["start"] - 0.05, "end": first["end"] + 0.05, "kind": "song",
                           "action": "keep", "source": "user"}], "cues": {}, "lines": {}}
    assert c.put(f"/api/jobs/{jid}/review", json=choice).status_code == 200
    assert c.post(f"/api/jobs/{jid}/review/start", json=choice).status_code == 409   # a finished dub redubs
    assert c.post(f"/api/jobs/{jid}/review/redub", json=choice).status_code == 200
    # The original is back, the job is queued again from its review step.
    j = svc.store.get(jid)
    assert j["status"] == J.QUEUED and not originals.aside_name(ep).exists()
    assert hashlib.sha256(ep.read_bytes()).hexdigest() == digest
    state = __import__("json").loads((cfg.work / f"job-{jid}" / "state.json").read_text())
    assert "voicebank" in state["done"] and "review" not in state["done"] and "translate" not in state["done"]
    svc.run_one(svc.store.next_queued())
    j = svc.store.get(jid)
    assert j["status"] == J.DONE, j["message"]
    assert j["record"]["qc"]["review"]["changes"] >= 1 and j["record"]["verify"] == "pending"
    assert originals.aside_name(ep).exists()                       # the original set aside again
    assert any(s.title == "English (AI dub)" for s in __import__("aidub.media", fromlist=["probe"]).probe(ep).audio)
    assert c.post(f"/api/jobs/{jid}/verify", json={"verdict": "good"}).status_code == 200
    assert c.post(f"/api/jobs/{jid}/review/redub", json=choice).status_code == 409   # nothing left to redub


def test_stop_for_review_with_nothing_flagged(client, cfg, media_root):
    from aidub import jobs as J
    c, svc = client
    ep = synth.episode(media_root / "Hold Show - S01E01.mkv")
    jid = c.post("/api/jobs", json={"paths": [str(ep)], "review": "off"}).json()["added"][0]
    assert c.post(f"/api/jobs/{jid}/review/hold", json={}).status_code == 200
    assert svc.store.get(jid)["options"]["review"] == "always"
    svc.run_one(svc.store.next_queued())
    assert svc.store.get(jid)["status"] == J.REVIEW
    assert c.post(f"/api/jobs/{jid}/review/hold", json={}).status_code == 409    # already there


def test_redo_a_few_lines_of_a_finished_dub(client, cfg, media_root):
    import json
    from aidub import jobs as J, lines as L, originals
    c, svc = client
    ep = synth.episode(media_root / "Redo Show - S01E01.mkv")
    jid = c.post("/api/jobs", json={"paths": [str(ep)], "review": "off"}).json()["added"][0]
    svc.run_one(svc.store.next_queued())
    work = cfg.work / f"job-{jid}"
    r = c.get(f"/api/jobs/{jid}/review").json()
    assert r["redo_lines"] and (work / "lines").is_dir()
    spoken = [ln for ln in r["lines"] if ln["kind"] == "speech" and ln["en"]]
    assert spoken and all(ln["outcome"] in ("dubbed", "unclear", "japanese") for ln in spoken)
    one, other = spoken[0], spoken[-1]
    assert sorted((work / "lines" / f"{other['id']:05d}").glob("take_*.wav"))
    res = c.post(f"/api/jobs/{jid}/lines/redo", json={"lines": {str(one["id"]): {"en": "A brand new line."}}})
    assert res.status_code == 200, res.text
    j = svc.store.get(jid)
    assert j["status"] == J.QUEUED and not originals.aside_name(ep).exists()
    state = json.loads((work / "state.json").read_text())
    assert "translate" in state["done"] and "speak" not in state["done"]
    assert not (work / "lines" / f"{one['id']:05d}").exists()
    svc.run_one(svc.store.next_queued())
    j = svc.store.get(jid)
    assert j["status"] == J.DONE, j["message"]
    lines = {ln.id: ln for ln in L.load(work / "lines.json")}
    assert lines[one["id"]].en == "A brand new line." and lines[one["id"]].en_from == "user"
    # Only that line was voiced again (its two takes); every other take was reused.
    voiced = [x for x in (work / "job.log").read_text().splitlines() if "] takes: " in x]
    assert voiced[-1].endswith("takes: 2/2"), voiced[-1]
    assert originals.aside_name(ep).exists() and j["record"]["verify"] == "pending"
    # The English is remembered for a later redub of the whole episode.
    assert c.get(f"/api/jobs/{jid}/review").json()["decisions"]["lines"][str(one["id"])]["en"] == "A brand new line."


def test_add_a_whole_season_with_the_shows_setting(client, cfg, media_root):
    c, svc = client
    season = media_root / "Season Show" / "Season 1"
    season.mkdir(parents=True)
    eps = [synth.episode(season / f"Season Show - S01E0{n}.mkv") for n in (1, 2)]
    got = c.get("/api/expand", params={"path": str(media_root / "Season Show")}).json()
    assert [f["name"] for f in got["files"]] == [e.name for e in eps]
    res = c.post("/api/jobs", json={"paths": [f["path"] for f in got["files"]], "review": "always", "show_review": True}).json()
    assert len(res["added"]) == 2
    slug = svc.store.get(res["added"][0])["profile"]
    assert c.get(f"/api/profiles/{slug}/rules").json()["review"] == "always"
    # Queued episodes aren't offered again.
    assert c.get("/api/expand", params={"path": str(season)}).json() == {"files": [], "skipped": 2}
    assert c.get("/api/expand", params={"path": "/etc"}).status_code == 403


def test_pause_events_episodes_originals_and_bazarr_test(client, cfg, media_root):
    from aidub import jobs as J
    c, svc = client
    show = media_root / "Paused Show"
    show.mkdir()
    eps = [synth.episode(show / f"Paused Show - S01E0{n}.mkv") for n in (1, 2)]
    # Paused: nothing new starts until resumed.
    assert c.post("/api/queue", json={"paused": True}).json() == {"paused": True}
    assert c.get("/api/status").json()["paused"] is True
    jid = c.post("/api/jobs", json={"paths": [str(eps[0])], "review": "off"}).json()["added"][0]
    assert svc.paused and svc.store.next_queued()["id"] == jid      # queued, waiting for the resume
    c.post("/api/queue", json={"paused": False})
    svc.run_one(svc.store.next_queued())
    assert svc.store.get(jid)["status"] == J.DONE
    # The show's episodes and how far each is.
    slug = svc.store.get(jid)["profile"]
    got = {e["name"]: e["state"] for e in c.get(f"/api/profiles/{slug}/episodes").json()["episodes"]}
    assert got == {eps[0].name: "to check", eps[1].name: "not dubbed"}
    assert isinstance(c.get(f"/api/profiles/{slug}/similar").json()["suggestions"], list)
    # The header's events, newest first.
    ev = c.get("/api/events").json()
    assert ev[0]["id"] == jid and ev[0]["kind"] == "check"
    # Originals say which dubs aren't checked yet, and how much room the disks have.
    o = c.get("/api/originals").json()
    assert o["items"] and o["items"][0]["check_pending"] and o["disks"] and o["disks"][0]["free"] > 0
    # Bazarr: no address or key, a plain answer rather than an error.
    assert c.post("/api/bazarr/test", json={"url": ""}).json()["ok"] is False
