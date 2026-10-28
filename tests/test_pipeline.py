"""End to end on synthetic media with stand-in workers: file handling, tracks,
set-aside originals, skip rules, redo, restore and delete."""

import hashlib
import shutil

import pytest

import synth
from aidub import jobs as J
from aidub import media, originals
from aidub.service import Service

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run_service_job(svc, path, **opts):
    jid = svc.add(str(path), options=opts or None)
    svc.run_one(svc.store.next_queued())
    return svc.store.get(jid)


def test_dub_mkv_in_place(cfg, media_root):
    show = media_root / "Test Show" / "Season 01"
    show.mkdir(parents=True)
    ep = synth.episode(show / "Test Show - S01E01.mkv")
    before = media.probe(ep)
    digest = sha(ep)

    svc = Service(cfg)
    job = run_service_job(svc, ep)
    assert job["status"] == J.DONE, job["message"]

    aside = originals.aside_name(ep)
    assert aside.exists() and sha(aside) == digest, "original must be set aside untouched"
    after = media.probe(ep)
    dubs = [s for s in after.audio if s.title == "English (AI dub)"]
    assert len(dubs) == 1 and dubs[0].language == "en" and not dubs[0].default
    assert any(k.upper() == "AIDUB_VERSION" for k in dubs[0].tags)
    assert len(after.audio) == len(before.audio) + 1
    assert len(after.attachments) == len(before.attachments) == 1
    assert len(after.chapters) == len(before.chapters) == 2
    titles = {s.title for s in after.subtitles}
    assert "Japanese (AI transcript)" in titles and "English (AI translation)" in titles
    rec = job["record"]
    assert rec["qc"]["dubbed_lines"] > 0
    assert rec["qc"]["fallback_rate"] < 0.5
    # Working files are gone, the set-aside original is registered, a profile exists.
    assert not (cfg.work / f"job-{job['id']}").exists()
    assert [o["path"] for o in svc.store.originals()] == [str(aside)]
    prof = svc.profiles.get("test-show")
    assert prof and len(prof.characters) == 2
    genders = sorted(c["gender"] for c in prof.characters.values())
    assert genders == ["female", "male"]
    assert all(c["refs"] for c in prof.characters.values())


def test_existing_subtitles_are_not_replaced(cfg, media_root):
    ep = synth.episode(media_root / "Film (2020).mkv")
    synth.srt(media_root / "Film (2020).en.srt", offset=1.5)
    svc = Service(cfg)
    job = run_service_job(svc, ep)
    assert job["status"] == J.DONE, job["message"]
    titles = {s.title for s in media.probe(ep).subtitles}
    assert "English (AI translation)" not in titles      # a sidecar English subtitle exists
    assert "Japanese (AI transcript)" in titles
    subs = job["record"]["text_source"]
    assert "sidecar" in subs                              # the English subtitle won scoring
    assert (media_root / "Film (2020).en.srt").exists()   # sidecar untouched


def test_skip_english_audio(cfg, media_root):
    ep = synth.episode(media_root / "Dual.mkv", english_audio=True)
    digest = sha(ep)
    svc = Service(cfg)
    job = run_service_job(svc, ep)
    assert job["status"] == J.SKIPPED and "English audio" in job["message"]
    assert sha(ep) == digest and not originals.aside_name(ep).exists()


def test_mp4_and_redo_restore_delete(cfg, media_root):
    ep = synth.episode(media_root / "Clip.mp4", container="mp4")
    digest = sha(ep)
    svc = Service(cfg)
    assert run_service_job(svc, ep)["status"] == J.DONE
    assert any(s.title == "English (AI dub)" for s in media.probe(ep).audio)

    # Queueing again without redo is skipped; with redo it rebuilds from the set-aside original.
    assert run_service_job(svc, ep)["status"] == J.SKIPPED
    job = run_service_job(svc, ep, redo=True)
    assert job["status"] == J.DONE, job["message"]
    assert sum(1 for s in media.probe(ep).audio if s.title == "English (AI dub)") == 1
    aside = originals.aside_name(ep)
    dubbed_aside = originals.aside_name(ep, "dubbed")
    assert sha(aside) == digest and dubbed_aside.exists()

    # Restore the original: the dubbed copy is set aside in its place.
    entry = next(o for o in svc.store.originals() if o["kind"] == "original")
    dubbed_aside.unlink()
    svc.store.drop_original(next(o["id"] for o in svc.store.originals() if o["kind"] == "dubbed"))
    originals.restore(svc.store, entry["id"])
    assert sha(ep) == digest and dubbed_aside.exists()

    # Delete only removes set-aside files.
    ids = [o["id"] for o in svc.store.originals()]
    n, freed = originals.delete(svc.store, ids)
    assert n == 1 and freed > 0 and not dubbed_aside.exists() and ep.exists()


def test_preview_never_touches_source(cfg, media_root, tmp_path):
    from aidub import pipeline
    from aidub.job import Job
    from aidub.profiles import Profiles
    ep = synth.episode(media_root / "Preview.mkv")
    digest = sha(ep)
    out = tmp_path / "preview.mkv"
    prof = Profiles(cfg.appdata).get_or_create("Preview")
    job = Job(cfg, "preview", ep, prof, {"preview_s": 15, "output": str(out)})
    pipeline.run(job)
    assert sha(ep) == digest and not originals.aside_name(ep).exists()
    pr = media.probe(out)
    assert any(s.title == "English (AI dub)" for s in pr.audio)
    assert pr.duration == pytest.approx(synth.DURATION, abs=1.0)


def test_failed_delivery_keeps_original(cfg, media_root, monkeypatch):
    from aidub import deliver
    ep = synth.episode(media_root / "Broken.mkv")
    digest = sha(ep)
    monkeypatch.setattr(deliver, "verify", lambda *a, **k: ["simulated mismatch"])
    svc = Service(cfg)
    job = run_service_job(svc, ep)
    assert job["status"] == J.FAILED and "verification failed" in job["message"]
    assert sha(ep) == digest and not originals.aside_name(ep).exists()
    assert not list(media_root.glob(".*aidub-partial"))
    # The failed job keeps its working folder for a retry from the failed step.
    assert (cfg.work / f"job-{job['id']}" / "state.json").exists()
    monkeypatch.undo()
    svc.retry(job["id"])
    svc.run_one(svc.store.next_queued())
    assert svc.store.get(job["id"])["status"] == J.DONE


def test_overrunning_lines_get_shorter_english(cfg, media_root):
    """With speed factors and stretching off and a translator that writes too
    much, lines only fit through the condense step's shorter wording."""
    cfg.data["translate"]["syllables_per_second"] = 7.0
    cfg.data["fit"].update({"factor_min": 0.999, "factor_max": 1.001, "stretch_max": 1.0})
    ep = synth.episode(media_root / "Long Lines.mkv")
    svc = Service(cfg)
    job = run_service_job(svc, ep)
    assert job["status"] == J.DONE, job["message"]
    qc = job["record"]["qc"]
    assert qc["fit"]["condensed"] > 0
    assert qc["fallback_rate"] <= 0.15, qc["fallback_lines"]


def test_songs_are_not_translated(cfg, media_root):
    """Lines a subtitle marks as sung keep the original audio and get no English."""
    ep = synth.episode(media_root / "Song.mkv")
    srt = synth.srt(media_root / "Song.en.srt", text="Hello there")
    text = srt.read_text(encoding="utf-8").replace("Hello there 1", "♪ la la la ♪").replace("Hello there 2", "♪ la la ♪")
    srt.write_text(text, encoding="utf-8")
    svc = Service(cfg)
    job = run_service_job(svc, ep)
    assert job["status"] == J.DONE, job["message"]
    assert job["record"]["qc"]["lines"].get("song", 0) >= 2
