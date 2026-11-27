"""End to end on synthetic media with stand-in workers: file handling, tracks,
set-aside originals, skip rules, redo, restore and delete."""

import subprocess
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
    # Working files made after the review are gone (the check set and the analysis stay
    # for a redub until the dub is checked), the set-aside original is registered, a
    # profile exists.
    left = {p.name for p in (cfg.work / f"job-{job['id']}").iterdir()}
    assert {"verify", "review"} <= left and not {"mix", "deliver"} & left
    assert job["record"]["verify"] == "pending"
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


def test_decode_check_finds_what_cant_be_played(media_root):
    from aidub import deliver
    ep = synth.episode(media_root / "Decode.mkv")
    assert deliver.decode_check(ep, "a", 0.0) == "" and deliver.decode_check(ep, "v", 0.0) == ""
    silent = media_root / "Silent.mkv"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(ep), "-map", "0:v", "-c", "copy", str(silent)], check=True)
    assert "nothing could be decoded" in deliver.decode_check(silent, "a", 0.0)
    bad = media_root / "Garbage.mkv"
    bad.write_bytes(b"not a video " * 5000)
    assert deliver.decode_check(bad, "v", 0.0)


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
    cfg.data["fit"].update({"factor_min": 0.999, "factor_max": 1.001, "stretch_max": 1.0, "timing": "window"})
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


def test_until_stops_when_step_already_done(cfg, media_root, tmp_path):
    from aidub import pipeline
    from aidub.job import Job
    from aidub.profiles import Profiles
    ep = synth.episode(media_root / "Until.mkv")
    prof = Profiles(cfg.appdata).get_or_create("Until")
    job = Job(cfg, "until", ep, prof, {})
    pipeline.run(job, until="align")
    again = Job(cfg, "until", ep, prof, {})
    pipeline.run(again, until="align")
    assert not again.is_done("voicebank") and not originals.aside_name(ep).exists()


def test_subtitle_over_no_speech_keeps_original(cfg, media_root):
    """A subtitle line where nothing is spoken (a sigh, breathing) isn't dubbed."""
    ep = synth.episode(media_root / "Sigh.mkv")
    srt = synth.srt(media_root / "Sigh.en.srt", text="Hello there")
    srt.write_text(srt.read_text(encoding="utf-8") + "\n99\n00:00:17,000 --> 00:00:19,000\nWhat's wrong?\n",
                   encoding="utf-8")
    svc = Service(cfg)
    job = run_service_job(svc, ep)
    assert job["status"] == J.DONE, job["message"]
    lines = job["record"]["qc"]["lines"]
    assert lines.get("wordless", 0) >= 1 and lines.get("speech") == len(synth.LINES)


def test_subtitles_as_script(cfg, media_root):
    cfg.data.setdefault("translate", {})["lead"] = "subtitle_only"
    ep = synth.episode(media_root / "Script (2020).mkv")
    synth.srt(media_root / "Script (2020).en.srt", offset=1.5)
    svc = Service(cfg)
    job = run_service_job(svc, ep)
    assert job["status"] == J.DONE, job["message"]
    assert "subtitle_track" in job["record"]["qc"]


def test_free_timing_starts_with_the_original(cfg, media_root):
    cfg.data["mix"]["polish"] = "gentle"  # (the voice filter's tail must fit the mix too)
    cfg.data.setdefault("fit", {})["timing"] = "start"
    ep = synth.episode(media_root / "Free (2020).mkv")
    svc = Service(cfg)
    job = run_service_job(svc, ep)
    assert job["status"] == J.DONE, job["message"]
    assert "free_timing" in job["record"]["qc"]


def test_sync_timing_lays_lines_out_with_the_speaker(cfg, media_root):
    cfg.data.setdefault("fit", {})["timing"] = "sync"
    ep = synth.episode(media_root / "Sync (2020).mkv")
    svc = Service(cfg)
    job = run_service_job(svc, ep)
    assert job["status"] == J.DONE, job["message"]
    assert job["record"]["qc"]["sync"]["lines"] > 0 and "free_timing" in job["record"]["qc"]


def test_a_5_1_original_gets_a_5_1_dub(cfg, media_root):
    src = media_root / "Show" / "Season 1" / "Show - S01E03.mkv"
    src.parent.mkdir(parents=True)
    plain = src.with_name("plain.mkv")
    synth.episode(plain)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(plain), "-map", "0", "-c", "copy", "-c:a", "aac",
                    "-af", "pan=5.1|FL=c0|FR=c0|FC=c0|LFE=c0|BL=c0|BR=c0", str(src)], check=True)
    plain.unlink()
    job = run_service_job(Service(cfg), src)
    assert job["status"] == J.DONE
    assert any(t.is_aidub and t.channels == 6 for t in media.probe(src).audio)


def _ready(cfg, name):
    from aidub import workers
    d = workers.env_dir(cfg, name)
    d.mkdir(parents=True, exist_ok=True)
    (d / ".aidub-ready").write_text(workers.requirements_hash(name))


def test_house_voices_are_spoken_by_drama3_when_installed(cfg, media_root):
    _ready(cfg, "voicedesign")
    _ready(cfg, "tts-drama3")
    cfg.data["voice"]["house"]["design_min_s"] = 0
    ep = synth.episode(media_root / "Drama Show" / "Drama Show - S01E01.mkv")
    svc = Service(cfg)
    job = run_service_job(svc, ep, voice="house")
    assert job["status"] == J.DONE, job["message"]
    assert job["record"]["models"]["voice"] == cfg.get("stages.house_drama3.model")


def test_japanese_words_come_from_their_own_model_timed_by_the_aligner(cfg, media_root):
    _ready(cfg, "asr-haas")
    ep = synth.episode(media_root / "Haas (2021).mkv")
    svc = Service(cfg)
    job = run_service_job(svc, ep)
    assert job["status"] == J.DONE, job["message"]
    own = job["record"]["qc"]["recognition_own_words"]
    assert own["changed"] > 0
    assert "(words)" in job["record"]["models"]["recognition"] and "(timing)" in job["record"]["models"]["recognition"]


def test_house_voices_replace_the_actors(cfg, media_root):
    from aidub import lines as L
    from aidub import workers
    stamp = workers.env_dir(cfg, "voicedesign")
    stamp.mkdir(parents=True, exist_ok=True)
    (stamp / ".aidub-ready").write_text(workers.requirements_hash("voicedesign"))
    cfg.data["voice"]["house"]["design_min_s"] = 0
    show = media_root / "House Show"
    show.mkdir(parents=True)
    ep = synth.episode(show / "House Show - S01E01.mkv")
    svc = Service(cfg)
    job = run_service_job(svc, ep, voice="house")
    assert job["status"] == J.DONE, job["message"]
    prof = svc.profiles.get("house-show")
    houses = [c["house"] for c in prof.characters.values() if c.get("house")]
    assert len(houses) == 2 and all((prof.root / h["file"]).exists() and h["description"] for h in houses)
    assert houses[0]["file"] != houses[1]["file"]
    assert all(c.get("traits", {}).get("seconds") for c in prof.characters.values())
    assert job["record"]["qc"]["house_voices"]["designed_now"] == 2
    # Spoken by the model that designed them, not the cloning model.
    assert job["record"]["models"]["voice"] == cfg.get("stages.house_speak.model")
    # A second episode keeps the same voices.
    ep2 = synth.episode(show / "House Show - S01E02.mkv")
    job2 = run_service_job(svc, ep2, voice="house")
    assert job2["status"] == J.DONE, job2["message"]
    assert job2["record"]["qc"]["house_voices"]["designed_now"] == 0
    assert sorted(c["house"]["file"] for c in svc.profiles.get("house-show").characters.values() if c.get("house")) \
        == sorted(h["file"] for h in houses)


def test_lines_whose_one_take_fails_are_voiced_again_by_the_check(cfg, media_root, monkeypatch):
    monkeypatch.setenv("FAKE_TTS_GARBLE", "1")  # every first take comes out as mumbling
    cfg.data["speak"].update({"takes": 1, "retakes": 2, "retake_rounds": 2})
    ep = synth.episode(media_root / "Retake (2020).mkv")
    svc = Service(cfg)
    job = run_service_job(svc, ep)
    assert job["status"] == J.DONE, job["message"]
    qc = job["record"]["qc"]
    assert qc["retakes"]["lines"] == qc["retakes"]["first_take_failed"] > 0
    assert qc["retakes"]["saved"] == qc["retakes"]["lines"] and qc["failed_takes"] == 0
    log = (cfg.work / f"job-{job['id']}" / "job.log").read_text()
    assert "retakes, round 1:" in log and "retakes, round 2:" not in log  # all saved in one round
