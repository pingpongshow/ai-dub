"""Unit tests for the pure parts: subtitle cleaning and scoring, fit rules,
speaker linking, profile names, translation helpers."""

import numpy as np
import pytest

from aidub import fit, lines as L, profiles, speakers, translate
from aidub.subtitles import clean, score
from aidub.subtitles.model import DIALOGUE, SIGN, SONG, Cue


def test_clean_text():
    assert clean.clean_text("[door slams] Get out!") == ("Get out!", "", DIALOGUE)
    t, name, kind = clean.clean_text("JOHN: We have to go.")
    assert (t, name, kind) == ("We have to go.", "John", DIALOGUE)
    assert clean.clean_text("♪ la la la ♪")[2] == SONG
    assert clean.clean_text("(sighs)")[2] != DIALOGUE
    assert clean.clean_text("（太郎）行くぞ")[:2] == ("行くぞ", "太郎")
    assert clean.clean_text("{\\an8}Hello{\\i1}there")[0] == "Hellothere"


def test_ass_styles(tmp_path):
    p = tmp_path / "a.ass"
    p.write_text("""[Script Info]
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1
Style: OP,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1
Style: Sign,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.00,Default,Taro,0,0,0,,行くぞ
Dialogue: 0,0:00:04.00,0:00:06.00,OP,,0,0,0,,{\\k20}ra{\\k20}la
Dialogue: 0,0:00:07.00,0:00:08.00,Sign,,0,0,0,,{\\pos(10,10)}Tokyo Station
""", encoding="utf-8")
    cues = clean.load(p)
    assert [c.kind for c in cues] == [DIALOGUE, SONG, SIGN]
    assert cues[0].name == "Taro" and cues[0].text == "行くぞ"


def test_script_language():
    assert clean.script_language(["こんにちは、元気ですか"]) == "ja"
    assert clean.script_language(["안녕하세요"]) == "ko"
    assert clean.script_language(["你好，你在哪里"]) == "zh"
    assert clean.script_language(["Where are you going?"]) == "en"


def test_best_offset_finds_shift():
    rng = np.random.default_rng(0)
    spans, t = [], 1.0
    while t < 280:
        d = float(rng.uniform(1, 3))
        spans.append((t, t + d))
        t += d + float(rng.uniform(0.5, 3))
    speech = score.mask(spans, 300)
    cues = [Cue(a - 2.5, b - 2.5, "x") for a, b in spans]
    off, scale, frac, sharp = score.best_offset(cues, speech, 300)
    assert off == pytest.approx(2.5, abs=0.1) and scale == 1.0 and frac > 0.9
    # A frame-rate (25 -> 23.976) stretch is found too.
    k = 23.976 / 25
    cues = [Cue(a * k, b * k, "x") for a, b in spans]
    off, scale, frac, _ = score.best_offset(cues, speech, 300)
    assert scale == pytest.approx(25 / 23.976, rel=1e-3) and frac > 0.85


def test_discrimination():
    assert score.discrimination([0.6] * 10, [0.3] * 10) == 1.0
    assert score.discrimination([0.4, 0.3] * 5, [0.3, 0.4] * 5) == 0.0
    assert score.discrimination([], []) == 0.0


def test_cer_and_content():
    assert score.cer("今日はいい天気", "今日はいい天気") == 0
    assert score.cer("今日はいい天気", "明日は雨") > 0.5


def test_fit_rules():
    rules = fit.FitRules()
    assert fit.factor_for(2.0, 1.8, rules) == pytest.approx(0.9)
    assert fit.factor_for(2.0, 1.5, rules) is None          # beyond natural speed
    assert fit.stretch_ratio(2.2, 2.0, rules) == pytest.approx(1.1)
    assert fit.stretch_ratio(2.4, 2.0, rules) is None
    assert fit.needs_fit(2.3, 2.0, 2.0, rules) == "over"
    assert fit.needs_fit(1.2, 2.25, 2.0, rules) == "under"
    assert fit.needs_fit(1.9, 2.25, 2.0, rules) == ""
    a = L.Line(1, 0.0, 2.0, character="c1")
    b = L.Line(2, 2.1, 3.0, character="c1")
    c = L.Line(3, 5.0, 6.0, character="c2")
    w = fit.windows([a, b, c], cuts=[6.1], rules=rules, duration=10)
    assert w[1] == pytest.approx(2.1)      # stops at the same character's next line
    assert w[3] == pytest.approx(1.1)      # borrows silence only up to the shot change


def test_link_respects_cannot_link():
    v = [[1, 0, 0], [0.98, 0.1, 0], [0, 1, 0], [0.97, 0.05, 0.1]]
    labels = speakers.link(v, 0.8)
    assert labels[0] == labels[1] == labels[3] != labels[2]
    labels = speakers.link(v, 0.8, cannot_link={(0, 1), (1, 0)})
    assert labels[0] != labels[1]


def test_guess_profile_name(tmp_path):
    roots = [tmp_path]
    assert profiles.guess_name(tmp_path / "Show" / "Season 01" / "Show - S01E07 - Title.mkv", roots) == "Show"
    assert profiles.guess_name(tmp_path / "Movie (2001)" / "Movie (2001).mkv", roots) == "Movie (2001)"
    assert profiles.guess_name(tmp_path / "Loose File - S01E02.mkv", roots) == "Loose File"
    assert profiles.slugify("Frieren: Beyond Journey's End") == "frieren-beyond-journeys-end"


def test_translate_helpers():
    assert translate.syllables("Hello there, how are you?") == 6
    assert translate.parse_json('noise ```json\n{"lines": [{"id": 1}]}\n``` tail') == {"lines": [{"id": 1}]}
    assert translate.parse_json('{"lines": [{"id": 1}]} trailing }') == {"lines": [{"id": 1}]}


def test_profile_merge_split(tmp_path):
    ps = profiles.Profiles(tmp_path)
    p = ps.get_or_create("Show")
    a = p.new_character("A")
    b = p.new_character("B")
    for cid in (a, b):
        f = p.root / "refs" / cid / "x.flac"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"x")
        p.characters[cid]["refs"] = [{"file": f"refs/{cid}/x.flac", "duration": 5, "score": 1}]
        p.characters[cid]["embedding"] = [1.0, 0.0]
    p.merge(a, b)
    assert b not in p.characters and len(p.characters[a]["refs"]) == 2
    assert all((p.root / r["file"]).exists() for r in p.characters[a]["refs"])
    new = p.split(a, [p.characters[a]["refs"][1]["file"]], "B again")
    assert len(p.characters[a]["refs"]) == 1 and len(p.characters[new]["refs"]) == 1
    p.save()
    ps.delete("show")
    assert not (tmp_path / "profiles" / "show").exists()


def test_pgs_reader_handles_empty(tmp_path):
    from aidub.subtitles import pgs
    p = tmp_path / "empty.sup"
    p.write_bytes(b"")
    assert list(pgs.read(p)) == []


def test_stray_words_join_nearby_line():
    from aidub import align
    ln = L.Line(1, 1.0, 2.0, source_text="行く", text_origin="asr",
                words=[L.Word("行", 1.0, 1.4), L.Word("く", 1.4, 2.0)])
    assert align._attach([ln], {"text": "ぞ", "start": 2.2, "end": 2.4}, "ja")
    assert ln.end == 2.4 and ln.source_text == "行くぞ"
    assert not align._attach([ln], {"text": "は", "start": 5.0, "end": 5.2}, "ja")


def test_backcheck_wer():
    from aidub import speak
    assert speak.wer("It's really starting to hurt now.", "It is really starting to hurt now.") == 0
    assert speak.wer("Kikoru.", "Kikaru.") < 0.2
    assert speak.wer("Then I must work harder.", "Then I must work harder") == 0
    assert speak.wer("Run, everyone!", "Sit down, nobody") > 0.5


def test_take_passes_relative_similarity():
    from aidub import speak
    cfg = {"speak.max_wer": 0.35, "speak.min_similarity": 0.1, "speak.max_similarity_drop": 0.25}
    get = type("C", (), {"get": lambda self, k, d=None: cfg.get(k, d)})()
    assert speak._passes({"wer": 0.0, "sim": 0.22, "sim_median": 0.3}, get)
    assert not speak._passes({"wer": 0.0, "sim": 0.02, "sim_median": 0.3}, get)
    assert speak._passes({"wer": 0.1, "sim": None, "sim_median": None}, get)
    assert not speak._passes({"wer": 0.6, "sim": 0.4, "sim_median": 0.3}, get)
    assert speak._passes({"wer": 0.0, "sim": 0.01, "sim_median": 0.3, "seconds": 0.6}, get)


def test_trim_cuts_noise_lead_and_long_tail():
    sr = 22050
    rng = np.random.default_rng(0)
    lead = rng.normal(0, 10 ** (-58 / 20), int(0.25 * sr))
    t = np.arange(int(0.4 * sr)) / sr
    speech = 0.05 * np.sin(2 * np.pi * 220 * t)
    tail = 0.012 * np.exp(-np.arange(int(0.5 * sr)) / (0.15 * sr)) * np.sin(2 * np.pi * 220 * np.arange(int(0.5 * sr)) / sr)
    x = np.concatenate([lead, speech, tail]).astype(np.float32)
    a, b = fit.speech_bounds(x, sr)
    assert abs(a / sr - 0.25) < 0.04
    assert 0.4 < (b - a) / sr < 0.62


def test_short_lines_borrow_more_silence():
    rules = fit.FitRules()
    short = L.Line(1, 0.0, 0.5, character="c1")
    long_ = L.Line(2, 5.0, 7.0, character="c1")
    w = fit.windows([short, long_], cuts=[], rules=rules, duration=20)
    assert w[1] == pytest.approx(1.0) and w[2] == pytest.approx(2.25)


def test_short_text_rules_are_looser():
    rules = fit.FitRules()
    loose = rules.for_short_text()
    assert loose.factor_min == 0.75 and loose.stretch_max == 1.25
    assert fit.factor_for(1.0, 0.8, rules) is None and fit.factor_for(1.0, 0.8, loose) == pytest.approx(0.8)


def test_split_line_keeps_subtitle_only_as_context():
    from aidub import align
    ln = L.Line(1, 0.0, 2.0, source_text="わかったよ行くぞ", english_ref="I know. Let's go.", text_origin="asr",
                words=[L.Word("わかった", 0.0, 0.8), L.Word("よ", 0.8, 0.9), L.Word("行く", 1.2, 1.7), L.Word("ぞ", 1.7, 2.0)])
    turns = [{"start": 0.0, "end": 1.0, "speaker": "S1"}, {"start": 1.1, "end": 2.0, "speaker": "S2"}]
    parts = align.split_by_speaker(ln, turns, "ja")
    assert len(parts) == 2
    assert all(p.english_ref == "" and p.english_context == "I know. Let's go." for p in parts)


def test_wrong_subtitle_cannot_pass():
    # Perfect timing and coverage but unrelated content stays under the 0.65 threshold.
    assert score.combine(1.0, 1.0, 0.0, 0.0, 1.0) < 0.65
    # Good meaning with timing hurt by a song still passes.
    assert score.combine(0.71, 0.67, 0.89, 0.0, 0.1) >= 0.65


def test_no_drift_when_halves_fit_at_zero():
    rng = np.random.default_rng(1)
    spans, t = [], 30.0
    while t < 280:
        d = float(rng.uniform(1, 3))
        spans.append((t, t + d))
        t += d + float(rng.uniform(0.5, 3))
    speech = score.mask([(0, 25)] + spans, 300)   # a 25 s song at the start counts as "speech"
    cues = [Cue(a, b, "x") for a, b in spans]
    assert score.drift(cues, speech, 300) == 0.0


def test_high_band_share():
    from aidub import bwe
    sr = 48000
    t = np.arange(sr) / sr
    low = np.sin(2 * np.pi * 300 * t)
    bright = low + 0.5 * np.sin(2 * np.pi * 12000 * t)
    assert bwe.high_band_share(low, sr) < 0.01 < bwe.high_band_share(bright, sr)


class _StubJob:
    def __init__(self, cfg_data, data):
        self._cfg, self._data = cfg_data, data
        self.cfg = type("C", (), {"get": lambda s, k, d=None: cfg_data.get(k, d)})()

    def __getitem__(self, k):
        return self._data[k]


def test_named_characters_merged_by_diarization_keep_own_voices(tmp_path):
    from aidub import voicebank
    prof = profiles.Profiles(tmp_path).get_or_create("Spy")
    man, girl = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    job = _StubJob({}, {"diarization": {"speakers": {"S1": {"embedding": man, "seconds": 30, "gender": "male",
                                                             "age": "adult"}}}})
    lines = [L.Line(i, i, i + 1.5, speaker="S1", name_hint=name) for i, name in
             enumerate(["Loid", "Loid", "Loid", "Anya", "Anya"])]
    voices = {ln.id: {"emb": man if ln.name_hint == "Loid" else girl,
                      "gender": "male" if ln.name_hint == "Loid" else "female"} for ln in lines}
    voicebank.link_characters(job, prof, lines, voices)
    loid, anya = lines[0].character, lines[3].character
    assert loid != anya
    assert prof.characters[anya].get("voice_of") is None      # not sharing Loid's voice
    assert prof.characters[anya]["gender"] == "female"
    # Same voice really shared (one actor, two roles) stays shared.
    prof2 = profiles.Profiles(tmp_path / "b").get_or_create("Two Roles")
    lines2 = [L.Line(i, i, i + 1.5, speaker="S1", name_hint=n) for i, n in enumerate(["A", "A", "A", "B", "B"])]
    voices2 = {ln.id: {"emb": man} for ln in lines2}
    voicebank.link_characters(job, prof2, lines2, voices2)
    assert prof2.characters[lines2[3].character].get("voice_of") == lines2[0].character


def test_two_speaker_subtitles_are_split(tmp_path):
    p = tmp_path / "two.srt"
    p.write_text("1\n00:00:01,000 --> 00:00:04,000\n- Let's play dodgeball.\n- Dodgeball?\n\n"
                 "2\n00:00:05,000 --> 00:00:06,000\n- Run. - Where?\n\n"
                 "3\n00:00:07,000 --> 00:00:08,000\nA well-known thing - really.\n", encoding="utf-8")
    cues = clean.load(p)
    assert [c.text for c in cues] == ["Let's play dodgeball.", "Dodgeball?", "Run.", "Where?",
                                      "A well-known thing - really."]
    assert cues[0].end == pytest.approx(cues[1].start) and cues[0].end - cues[0].start > cues[1].end - cues[1].start


def test_invisible_marks_are_ignored():
    assert clean.clean_text("\u200e(sighs)")[2] != "dialogue"
    assert clean.speaker_turns("\u200e- 아이…\n\u200e- 잠 깨, 가자") == []  # raw marks: callers strip first
    assert clean.speaker_turns(clean.INVISIBLE.sub("", "\u200e- 아이…\n\u200e- 잠 깨, 가자")) == ["아이…", "잠 깨, 가자"]
