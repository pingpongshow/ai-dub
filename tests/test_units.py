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
    # A subtitle nothing was heard under ("Six.", a child's quiet 六つ) doesn't take the next sound.
    six = L.Line(2, 57.1, 57.79, source_text="", text_origin="asr", english_ref="Six.")
    assert not align._attach([six], {"text": "うん", "start": 58.04, "end": 58.28}, "ja")
    assert six.end == 57.79 and six.source_text == ""


def test_backcheck_wer():
    from aidub import speak
    assert speak.wer("It's really starting to hurt now.", "It is really starting to hurt now.") == 0
    assert speak.wer("Kikoru.", "Kikaru.") < 0.2
    assert speak.wer("Then I must work harder.", "Then I must work harder") == 0
    assert speak.wer("Run, everyone!", "Sit down, nobody") > 0.5
    assert speak.wer("Who's this kid?", "Who is this kid?") == 0
    assert speak.wer("They're here, we've won.", "They are here, we have won.") == 0
    assert speak._words("Karen's dad") == ["karen's", "dad"]  # a possessive stays


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

    def get(self, k, d=None):
        return self._data.get(k, d)

    def set(self, k, v):
        self._data[k] = v


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


def test_two_people_alike_on_average_but_not_line_by_line_stay_two_characters(tmp_path):
    from aidub import voicebank
    rnd = np.random.default_rng(4)
    base = rnd.normal(0, 1, 32)

    def voice():  # a person: a shared "male" direction plus their own
        return base + rnd.normal(0, 1.0, 32)

    def lines_of(v, n):  # each line wobbles around its person
        return [list(v + rnd.normal(0, 0.6, 32)) for _ in range(n)]
    a, b = voice(), voice()
    pa, pb = lines_of(a, 8), lines_of(b, 8)
    assert voicebank.same_person(pa[:4], pa[4:], 0.86) is True          # one person split in two
    assert voicebank.same_person(pa, pb, 0.86) is False                 # two people
    assert voicebank.same_person(pa[:2], pb, 0.86) is None              # too few to judge
    # Diarization told them apart, and their average voiceprints are alike enough to match one
    # character: line by line they aren't one person, so each gets a character of its own.
    prof = profiles.Profiles(tmp_path).get_or_create("Paper")
    job = _StubJob({}, {"diarization": {"speakers": {
        "S1": {"embedding": list(base), "seconds": 60, "gender": "male"},
        "S2": {"embedding": list(base * 0.98 + 0.02), "seconds": 50, "gender": "male"}}}})
    lines = [L.Line(i, i * 3.0, i * 3.0 + 2, speaker="S1" if i < 8 else "S2") for i in range(16)]
    voices = {ln.id: {"emb": (pa + pb)[ln.id]} for ln in lines}
    job.log = lambda m: None
    voicebank.link_characters(job, prof, lines, voices)
    assert lines[0].character and lines[8].character and lines[0].character != lines[8].character
    # The same person in two diarized clusters still becomes one character.
    prof2 = profiles.Profiles(tmp_path / "b").get_or_create("One")
    lines2 = [L.Line(i, i * 3.0, i * 3.0 + 2, speaker="S1" if i < 8 else "S2") for i in range(16)]
    voices2 = {ln.id: {"emb": lines_of(a, 16)[ln.id]} for ln in lines2}
    voicebank.link_characters(job, prof2, lines2, voices2)
    assert lines2[0].character == lines2[8].character


def test_a_line_uses_the_reference_clip_closest_to_its_voice(tmp_path):
    from aidub import voicebank
    prof = profiles.Profiles(tmp_path).get_or_create("Mixed")
    cid = prof.new_character("Character 1")
    prof.characters[cid]["refs"] = [{"file": "refs/x/one.flac", "score": 0.9, "embedding": [1.0, 0.0]},
                                    {"file": "refs/x/two.flac", "score": 0.5, "embedding": [0.0, 1.0]}]
    assert voicebank.closest_ref(prof, cid, [0.1, 0.9])["file"] == "refs/x/two.flac"
    assert voicebank.closest_ref(prof, cid, None)["file"] == "refs/x/one.flac"   # nothing to go by: the best clip
    lines = [L.Line(1, 0, 2, speaker="S1", character=cid), L.Line(2, 3, 5, speaker="S1", character=cid),
             L.Line(3, 6, 8, speaker="S2", character=cid), L.Line(4, 9, 11, speaker="S2", character=cid),
             L.Line(5, 12, 14, speaker="S2", character=cid), L.Line(6, 15, 17, speaker="S2", character=cid),
             L.Line(7, 18, 20, speaker="S1", character=cid), L.Line(8, 21, 23, speaker="S1", character=cid)]
    rnd = np.random.default_rng(1)
    voices = {ln.id: {"emb": list((np.array([1.0, 0.0]) if ln.speaker == "S1" else np.array([0.0, 1.0])) * 5
                                  + rnd.normal(0, 0.3, 2))} for ln in lines}
    mixed = voicebank.mixed_characters(prof, lines, voices, 0.86)
    assert len(mixed) == 1 and sorted(mixed[0]["lines"]) == [3, 4, 5, 6] and mixed[0]["main_lines"] == 4


def test_unmarked_voice_takes_the_named_character(tmp_path):
    from aidub import voicebank
    prof = profiles.Profiles(tmp_path).get_or_create("Spy")
    man = [1.0, 0.0, 0.0]
    job = _StubJob({}, {"diarization": {"speakers": {"S1": {"embedding": man, "seconds": 30}}}})
    lines = [L.Line(1, 1, 2.5, speaker="S1", name_hint="Edgar"), L.Line(2, 3, 3.8, name_hint="Edgar"),
             L.Line(3, 4, 4.8, name_hint="Nobody Yet"), L.Line(4, 5, 5.8)]
    voicebank.link_characters(job, prof, lines)
    assert lines[1].character == lines[0].character != ""
    assert lines[2].character == "" and lines[3].character == ""   # no new characters from one unmarked line


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


def test_each_word_belongs_to_one_cue():
    from aidub import align
    words = [{"text": "よし", "start": 141.3, "end": 141.6}, {"text": "まず", "start": 172.7, "end": 173.0}]
    cues = [Cue(141.2, 141.7, "Daddy!"), Cue(141.25, 141.65, "Close enough."), Cue(172.6, 173.4, "First...")]
    owned = align.assign_words(words, cues)
    assert sum(len(o) for o in owned) == 2 and owned[2] == [words[1]]


def test_duplicate_cues_become_one_line():
    from aidub import align
    a = L.Line(id=0, start=168.6, end=169.1, source_text="サイレン")
    b = L.Line(id=0, start=168.6, end=170.8, source_text="サイレンサー付きの拳銃だと")
    c = L.Line(id=0, start=175.3, end=176.0, source_text="それとこいつ")
    d = L.Line(id=0, start=175.3, end=176.0, source_text="それとこいつ")
    e = L.Line(id=0, start=176.5, end=177.0, source_text="別の話")
    out = align.merge_duplicates([a, b, c, d, e], "ja")
    assert [(x.start, x.end, x.source_text) for x in out] == [
        (168.6, 170.8, "サイレンサー付きの拳銃だと"), (175.3, 176.0, "それとこいつ"), (176.5, 177.0, "別の話")]


def test_song_spreads_to_neighbouring_lines_over_music():
    from aidub import align

    def mk(a, b, kind, text, ev, db):
        return L.Line(id=0, start=a, end=b, kind=kind, source_text=text, events=ev, music_db=db)
    lines = [mk(0.0, 0.3, L.WORDLESS, "Take my", ["BGM"], -3), mk(0.3, 5.2, L.SONG, "heartbeat", ["BGM"], 3),
             mk(34.7, 35.3, L.SPEECH, "뛰지 마", [], -7), mk(41.7, 43.0, L.SPEECH, "No effect", ["BGM"], -1),
             mk(72.8, 75.3, L.SPEECH, "교감 쌤", ["BGM"], -26)]
    align.extend_songs(lines, "ko")
    assert [x.kind for x in lines] == [L.SONG, L.SONG, L.SPEECH, L.SONG, L.SPEECH]


def test_unvoiced_lines_inside_a_song_are_sung():
    from aidub import align

    def mk(a, b, kind, spk, db=-10):
        return L.Line(id=0, start=a, end=b, kind=kind, speaker=spk, source_text="이번에", music_db=db, events=["BGM"])
    lines = [mk(25.7, 29.7, L.SONG, "S1"), mk(33.5, 34.9, L.SPEECH, "S1", -3), mk(35.8, 37.7, L.SPEECH, ""),
             mk(38.7, 39.8, L.SPEECH, ""), mk(41.7, 43.0, L.SONG, "S1"), mk(71.1, 72.0, L.SPEECH, "")]
    align.extend_songs(lines, "ko")
    # The singer's own slow line over the music between two sung lines is the song too.
    assert [x.kind for x in lines] == [L.SONG, L.SONG, L.SONG, L.SONG, L.SONG, L.SPEECH]


def _ln(a, b, text, spk="S1", kind=L.SPEECH, words=(), heard=None, i=0):
    ws = [L.Word(start=t, end=t + 0.15, text=w) for w, t in words]
    return L.Line(id=i, start=a, end=b, kind=kind, speaker=spk, source_text=text, words=ws, text_origin="asr",
                  heard_chars=len(text) if heard is None else heard)


def test_relistened_words_of_neighbours_are_not_this_lines():
    from aidub import align
    before = _ln(172.7, 174.6, "まずは必要なものを揃えないとな")
    after = _ln(175.3, 175.9, "それとこいつ")
    rest, owner = align.strip_neighbours("必要なものを揃えないとなそれと", [before, after])
    assert rest == "" and owner is before
    rest, _ = align.strip_neighbours("わくわく", [before, after])
    assert rest == "わくわく"
    # Only the neighbours' words heard inside this line's window: "무슨" ends the line before, out of reach.
    prev = _ln(103.57, 107.49, "그래 봤자 무슨", words=[("그래", 103.57), ("무슨", 107.41)])
    nxt = _ln(109.01, 111.73, "소리 야 어 요즘", words=[("소리", 109.01), ("야", 109.17), ("요즘", 109.17)])
    rest, owner = align.strip_neighbours("무슨소리야", [prev, nxt], (107.56, 109.11))
    assert rest == "무슨야" and owner is nxt


def test_pieces_of_neighbouring_lines_go():
    from aidub import align
    cool = _ln(171.07, 172.06, "わくわく。", "S3", i=73)
    between = _ln(171.39, 173.03, "", "S6", L.WORDLESS, heard=0, i=74)          # end of one, start of the next
    supplies = _ln(172.71, 174.55, "まずは必要なものを揃えないとな", i=75, words=[("まず", 172.71)])
    girls = _ln(33.47, 34.88, "여자들 하지 마.", heard=6, i=12)
    run = _ln(34.66, 35.22, "뛰지 마", "S2", words=[("뛰지", 34.66), ("마", 34.74)], i=13)  # same audio, cut twice
    hm = _ln(57.1, 58.28, "うん", kind=L.WORDLESS, words=[("うん", 58.04)], i=23)  # the director's
    six = _ln(58.59, 59.5, "六つ。", "", i=24)                                   # re-listen heard うん六つ
    breath = _ln(80.2, 80.6, "", "", L.WORDLESS, heard=0, words=[("あ", 80.3)], i=30)
    go = _ln(80.8, 82.0, "行くぞ", "S2", words=[("行く", 80.8)], i=31)           # re-listen heard あ行くぞ
    out = align.drop_pieces([cool, between, supplies, girls, run, hm, six, breath, go], {23: six, 30: go}, "ja")
    assert out == [cool, supplies, girls, hm, six, go]
    assert six.start == 58.59 and go.start == pytest.approx(80.2)


def test_subtitle_line_nobody_carries_gets_a_line(monkeypatch):
    from aidub import align
    from aidub.subtitles.model import DIALOGUE, Cue
    title = _ln(163.0, 164.72, "冒険アニメスパイウォーズ", "S2", words=[("ウォーズ", 164.52)])
    title.english_ref = "The adventure cartoon, Spy Wars!"
    nxt = _ln(167.43, 169.19, "よりによって", "S1")
    nxt.english_ref = "Of all shows..."
    cues = [Cue(start=162.6, end=165.37, text="The adventure cartoon, Spy Wars!", kind=DIALOGUE),
            Cue(start=164.21, end=165.74, text="This is my favourite!", kind=DIALOGUE),
            Cue(start=167.4, end=168.69, text="Of all shows...", kind=DIALOGUE)]
    monkeypatch.setattr(align, "load_chosen", lambda job, which: cues if which == "english" else [])
    job = {"diarization": {"segments": [{"start": 162.75, "end": 164.8, "speaker": "S2", "overlap": True},
                                        {"start": 164.25, "end": 165.32, "speaker": "S3", "overlap": True},
                                        {"start": 167.4, "end": 169.2, "speaker": "S1"}]},
           "scenes": {"cuts": []}}
    new = align.missed_cues(job, [title, nxt], "ja")
    assert len(new) == 1
    fav = new[0]
    assert (fav.start, fav.end, fav.speaker, fav.english_ref) == (164.72, 165.42, "S3", "This is my favourite!")
    assert fav.source_text == "" and fav.heard_chars == 0


def test_cut_sentence_is_one_line():
    from aidub import align
    a = _ln(175.27, 175.91, "それとこいつ", words=[("それと", 175.27), ("こいつ", 175.7)])
    b = _ln(175.91, 177.03, "の身分証の偽造", words=[("の", 175.91)])
    c = _ln(177.75, 178.79, "ちょっと出かけてくる", words=[("ちょっと", 177.75)])
    d = _ln(179.27, 179.91, "おとなしく", "S2")
    title = _ln(163.0, 164.2, "冒険アニメスパイ", "S2")
    wars = _ln(164.52, 164.72, "ウォーズ", "S2")
    done = _ln(160.0, 160.5, "よし。", "S2")
    nxt = _ln(160.6, 161.0, "行こう", "S2")
    looks = _ln(54.81, 56.81, "この子はどう見ても四五歳過疎から", words=[("この", 54.81), ("から", 56.57)])
    hm = _ln(57.1, 58.28, "うん", words=[("うん", 58.04)])     # a subtitle stretched its start
    assert len(align.merge_fragments([looks, hm], "ja")) == 2
    sub1, sub2 = _ln(55.0, 56.6, "너 작년 생일 생각해 봐"), _ln(56.9, 58.1, "올해도 금방 생일 됐지?")
    sub1.text_origin = sub2.text_origin = "subtitle:embedded:5"
    assert len(align.merge_fragments([sub1, sub2], "ko")) == 2
    out = align.merge_fragments([done, nxt, title, wars, a, b, c, d], "ja")
    assert [x.source_text for x in out] == ["よし。", "行こう", "冒険アニメスパイウォーズ", "それとこいつの身分証の偽造",
                                            "ちょっと出かけてくる", "おとなしく"]
    assert out[3].end == 177.03 and len(out[3].words) == 3


def test_overlap_share():
    from aidub import align
    turns = [{"start": 0.0, "end": 2.0}, {"start": 1.0, "end": 3.0}]
    assert align.overlap_share(turns, 0.0, 2.0) == pytest.approx(0.5)
    assert align.overlap_share(turns, 1.0, 2.0) == pytest.approx(1.0)


def test_backcheck_misheard_key_word_fails():
    from aidub import speak
    assert speak.missed_words("But shady places often have incomplete records.",
                              "But shabby places often have incomplete records.") == ["shady"]
    assert speak.missed_words("I'll take her, Anya.", "I will take her Ania") == []
    assert speak.missed_words("Hey, Ga-yeong, get down.", "Hey, Ga Young, get down!") == []
    assert speak.missed_words("Look at Ok-hui.", "Look at old Kwee.") == []
    s = {"wer": 0.14, "missed": ["shady"], "words": 7, "sim": None}
    assert not speak._passes(s, _Cfg())
    assert speak._passes({**s, "missed": []}, _Cfg())


class _Cfg:
    def get(self, key, default=None):
        return default


def test_interjections_keep_the_original():
    from aidub import align
    assert align.is_interjection("아이…", "ko") and align.is_interjection("うわぁ!", "ja")
    assert not align.is_interjection("잠 깨, 가자", "ko")


def test_thin_characters_speak_with_their_own_audio(tmp_path, monkeypatch):
    from aidub import voicebank
    prof = profiles.Profiles(tmp_path).get_or_create("Spy")
    loid = prof.new_character("Loid", gender="male", age="adult")
    anya = prof.new_character("Anya", gender="female", age="child")
    prof.characters[loid]["embedding"] = [1.0, 0.0, 0.0]
    prof.characters[loid]["refs"] = [{"file": "refs/loid/a.flac", "duration": 9.0, "score": 0.9}]
    monkeypatch.setattr(voicebank, "generic_voices", lambda job: [
        {"name": "female young 1", "gender": "female", "age": "young", "path": "/v/f.wav"}])
    from aidub import media
    job = _StubJob({"speakers.prefer_own_clip": False}, {})
    job.qc = {}
    job.path = lambda *parts: (tmp_path / "job").joinpath(*parts)
    (tmp_path / "job" / "audio").mkdir(parents=True)
    (tmp_path / "job" / "voices").mkdir(parents=True)
    media.write_wav(job.path("audio", "dialogue.wav"), np.zeros((48000 * 12, 1), np.float32), 48000)
    lines = [L.Line(1, 0, 2.0, character=anya), L.Line(2, 3, 3.4, character=anya),
             L.Line(3, 5, 7.0, character=loid), L.Line(4, 8, 10.0, character=loid)]
    voices = {4: {"emb": [1.0, 0.0, 0.0], "gender": "male", "age": "child"}}
    voicebank.assign_voices(job, prof, lines, voices)
    assert lines[0].voice == "" and lines[0].voice_note == "own line as reference"   # long enough: own audio
    assert lines[1].voice_note == "own voice from another line"                     # too short: her longest line
    assert lines[1].voice.endswith(f"{anya}.wav")
    assert lines[2].voice.endswith("a.flac")                                          # main character: voice bank
    assert lines[3].voice == "" and "age" in lines[3].voice_note                     # a child's line, adult voice


def test_words_follow_their_speaker_past_a_tv_cue():
    from aidub import align
    turns = [{"start": 0.0, "end": 10.0, "speaker": "S1"}, {"start": 20.0, "end": 24.0, "speaker": "S1"},
             {"start": 20.0, "end": 21.0, "speaker": "S6"}]
    words = [{"text": t, "start": a, "end": a + 0.3} for t, a in
             [("いい", 1.0), ("か", 1.4), ("アニャ", 1.8), ("今日", 2.5), ("から", 3.0),
              ("まず", 20.1), ("必要な", 20.5), ("もの", 21.5)]]
    cues = [Cue(0.8, 3.5, "Listen, Anya.", name="Loid"), Cue(20.0, 20.9, "You must be a pro!", name="Bondman"),
            Cue(20.05, 22.0, "First, I'll have to secure...", name="")]
    owned = align.assign_words(words, cues, turns=turns)
    assert [w["text"] for w in owned[2]] == ["まず", "必要な", "もの"] and owned[1] == []


def test_backcheck_hears_the_same_sound_spelled_differently(tmp_path):
    from aidub import pronounce, speak
    lex = tmp_path / "cmudict.dict"
    lex.write_text("brakes B R EY1 K S\nbreaks B R EY1 K S\nshady SH EY1 D IY0\nshoddy SH AA1 D IY0\n"
                   "toupee T UW2 P EY1\ntwo T UW1\np P IY1\nlloyd L OY1 D\nten T EH1 N\npent P EH1 N T\n"
                   "comes K AH1 M Z\nbomb B AA1 M\nbum B AH1 M\nnguyen N UW0 Y EH1 N\nno N OW1\nyen Y EH1 N\n"
                   "new N UW1\nnew(2) N Y UW1\nyin Y IH1 N\ngwen G W EH1 N\nproud P R AW1 D\n")
    try:
        assert pronounce.load(lex)
        assert speak.missed_words("Brakes!", "Breaks.") == []
        assert speak.missed_words("Kidnap the Minister's toupee.", "Kidnap the minister's two P.") == []
        assert speak.missed_words("Daddy's toupee was stolen.", "Daddy's topee was stolen.") == []
        assert speak.missed_words("Loid Forger. Occupation: psychiatrist.", "Lloyd Forger, occupation, psychiatrist.") == []
        assert speak.missed_words("That costs a dalc. A ten-pent coin won't do.",
                                  "That costs a dalk. A tenpence coin won't do.") == []
        assert speak.missed_words("Secret comms.", "Secret comes.") == []
        assert speak.missed_words("That's a dalc. Ten pents won't buy it.", "That is a dalk. Ten pence won't buy it.") == []
        assert speak.missed_words("How's Karen?", "How is Karen?") == []
        assert speak.missed_words("West's cipher?", "West Cipher.") == []
        assert speak.missed_words("Anya Levski.", "Anielewski.") == []
        # Dictionary words must sound the same: a wrong vowel is still a miss.
        assert speak.missed_words("But shady places.", "But shoddy places.") == ["shady"]
        assert speak.wer("Bomb! Bomb!", "Bum bum.") > 0.35
        assert speak.wer("Brakes, now!", "Breaks now.") == 0
        # A name the voice says from its spelled-out sound, heard as two words.
        assert pronounce.respell("Nguyen") == "noo yen" and pronounce.respell("proud") is None
        assert speak.missed_words("Nguyen!", "No yen.") == [] and speak.missed_words("Nguyen!", "New Yin.") == []
        assert speak.missed_words("Nguyen!", "Gwen.") == ["nguyen"]
        assert pronounce.skeleton("gh") == "" and not pronounce.near("ugh", "gh")
        # A made-up word heard one sound off counts; a dictionary word doesn't get that slack.
        assert speak.missed_words("Eeny, meeny...", "Ifny meany.") == []
        assert speak.missed_words("Eeny, meeny...", "Disney Mini.") == ["eeny"]
        assert speak.missed_words("But shady places.", "But shoddy places.") == ["shady"]
        assert speak.wer("Nguyen! Get the car.", "No yen, get the car.") == 0
        ln = L.Line(1, 0, 1, en="Nguyen! Get the car.", take_scores={"t": {"missed": ["nguyen", "car"]}})
        assert speak.respelled(ln) == "noo yen! Get the car."
        # Japanese names are respelled on a failed line even though names aren't "missed".
        ln = L.Line(2, 0, 1, en="Useless, Kikoru-chan.", take_scores={"t": {"missed": ["useless"]}})
        assert speak.respelled(ln) == "Useless, kee koh roo-chahn."
        # A made-up word heard in two longer pieces with its consonants.
        assert speak.wer("Daikaiju in Sumida!", "Dai Kajou in Sumida.") == 0
        assert speak.wer("Brakes, now!", "Bray kess now.") > 0      # dictionary words: no such slack
        assert speak.wer("It's useless, Kikoru-chan.", "It is useless, Kikorichan.") == 0   # hyphenated name, one word
        assert speak.wer("Down! Kat-chan!", "Down. Kachan.") == 0   # said as Japanese "Kacchan"
        assert speak.wer("Down! Kat-chan!", "Down. Mochan.") > 0.35
        assert pronounce.only_sounds(speak._words("Mm-hmm.")) and pronounce.only_sounds(speak._words("Shh."))
        assert not pronounce.only_sounds(speak._words("Oh, no."))
    finally:
        pronounce.load(tmp_path / "none.dict")  # back to no dictionary for the other tests
    assert speak.missed_words("Brakes!", "Breaks.") == ["brakes"]  # without it, only exact spellings


def test_backcheck_takes_from_e07_that_were_right():
    from aidub import speak
    from aidub.config import Config
    # "Dammit" heard as "Damn it", a quick "you'd let" heard as "you let", "Nah" heard as "No".
    assert speak.missed_words("Dammit. As if any of you losers are any", "Damn it! As if any of you losers are any.") == []
    assert speak.wer("Dammit. As if any of you losers are any", "Damn it! As if any of you losers are any.") == 0
    assert speak.missed_words("It's obvious you'd let another man steal her under your eyes!",
                              "It is obvious you let another man steal her under your eyes.") == []
    assert speak.missed_words("Nah.", "No.") == []
    assert speak.missed_words("You'd better hurry.", "Better hurry.") == ["you'd"]   # gone entirely: still missed
    # No take passed, but one is nearly right: it is used rather than keeping the original.
    cfg = Config({}, None, {})
    near = {"wer": 0.2, "missed": ["sandwich"], "sim": 0.4, "sim_median": 0.35, "seconds": 2.4, "words": 9}
    assert not speak._passes(near, cfg) and speak._rescue(near, cfg)
    assert not speak._rescue({**near, "wer": 0.4}, cfg) and not speak._rescue({**near, "missed": ["a1", "b2"]}, cfg)
    assert speak._rescue({**near, "sim": 0.02}, cfg)                           # a doubtful voice still beats the original
    assert speak._passes({**near, "rescued": True}, cfg)


def test_voice_model_reads_hyphenated_words_as_two():
    from aidub import speak
    assert speak.spoken("Build self-esteem, Ga-yeong.") == "Build self esteem, Ga yeong."
    assert speak.spoken("Ten-20 at 3-4.") == "Ten-20 at 3-4."  # only between letters
    # A count is said in words, a long run of dots is one pause.
    assert speak.spoken("3..... 4....") == "Three... Four..." and speak.spoken("Wait..... what?") == "Wait... what?"
    assert speak.spoken("Room 12.") == "Room 12."


def test_misheard_lines_follow_their_subtitle():
    from aidub import translate
    heard = L.Line(1, 0, 1, source_text="おじい。", relisten="おじい。", english_ref="Daddy!")
    plain = L.Line(2, 1, 2, source_text="えらいえらい", english_ref="You bet I am!")
    assert translate.source_unsure(heard) and not translate.source_unsure(plain)
    msgs = translate.translate_prompt([heard, plain], [], "", 4.0, {})
    assert msgs[1]["content"].count('"source_unsure": true') == 1
    assert "source_unsure" in translate.reflect_prompt([heard], "", 4.0, {}, {})[1]["content"]


def test_backcheck_spoken_forms_and_numbers():
    from aidub import speak
    assert speak.missed_words("I'd be thrilled to get her outta my hair.", "I'd be thrilled to get her out of my hair.") == []
    assert speak.wer("Alright, hurry up!", "All right, hurry up.") == 0
    assert speak.wer("K-212... K-212...", "K two hundred and twelve. K two hundred and twelve.") == 0
    assert speak.wer("K-212... K-212...", "K two one two. K two one two.") == 0
    assert speak.missed_words("K-212.", "K two hundred.") == ["212"]
    assert speak.missed_words("K-212.", "K two one two.") == [] and speak.missed_words("Room 4.", "Room 4.") == []
    assert speak.number_words(1999) == ["one", "thousand", "nine", "hundred", "ninety", "nine"]


class _SimJob(_StubJob):
    """A job whose speaker worker scores text pairs from a table (unknown pairs: unrelated)."""
    def __init__(self, data, sims):
        super().__init__({"stages.speaker.worker": "speaker"}, data)
        self.qc, self.sims = {}, sims

    def log(self, msg):
        pass

    def worker(self, *a, **k):
        job = self

        class W:
            def __enter__(self):
                return self

            def __exit__(self, *e):
                pass

            def call(self, op, args):
                return {"similarity": [job.sims.get(tuple(p), 0.1) for p in args.get("pairs", [])]}
        return W()


def test_subtitle_kept_where_recognition_misheard_and_english_agrees():
    from aidub import align
    misheard = L.Line(1, 0, 2, source_text="日々日々の重さ", text_origin="asr", subtitle_text="日比野カフカ",
                      english_ref="Kafka Hibino...")
    other_cut = L.Line(2, 3, 5, source_text="行くぞ みんな", text_origin="asr", subtitle_text="待って",
                       english_ref="Let's go, everyone!")
    fine = [L.Line(i, i * 3, i * 3 + 2, source_text=f"せりふ{i}です", english_ref=f"Line {i}.") for i in range(3, 7)]
    sims = {("Kafka Hibino...", "日比野カフカ"): 0.8, ("Kafka Hibino...", "日々日々の重さ"): 0.2,
            ("Let's go, everyone!", "行くぞ みんな"): 0.7, ("Let's go, everyone!", "待って"): 0.3}
    sims.update({(f"Line {i}.", f"せりふ{i}です"): 0.7 for i in range(3, 7)})
    job = _SimJob({"subtitles": {"source": "embedded:3"}}, sims)
    align.check_english_refs(job, [misheard, other_cut] + fine)
    assert misheard.source_text == "日比野カフカ" and misheard.text_origin == "subtitle:embedded:3"
    assert misheard.english_ref == "Kafka Hibino..."            # now matches: kept as the meaning
    assert other_cut.source_text == "行くぞ みんな" and other_cut.text_origin == "asr"   # the speech still wins
    assert job.qc["subtitle_lines_kept_over_recognition"] == 1


def test_loud_score_under_subtitled_dialogue_is_not_background(monkeypatch):
    from aidub import align
    from aidub.subtitles.model import Cue
    job = _StubJob({}, {"diarization": {"segments": []}, "events": [], "ingest": {},
                        "subtitles": {"source_language": "ja"}})
    def lines():
        return [L.Line(1, 0, 2, source_text="総員退避!", english_ref="Everyone, retreat!", music_db=8.0,
                       loudness_db=-31.0),
                L.Line(2, 3, 5, source_text="出入り口ドアが閉まります", music_db=27.0, loudness_db=-52.0),
                L.Line(3, 6, 8, source_text="夢の中こぼれ落ちた", music_db=9.0, loudness_db=-30.0),
                L.Line(4, 9, 11, source_text="ここは危ない", english_ref="It's dangerous here.", music_db=1.0,
                       loudness_db=-30.0)]
    monkeypatch.setattr(align, "load_chosen", lambda j, w: [Cue(start=0, end=2, text="x")] if w == "english" else [])
    ls = lines()
    align.classify(job, ls)
    assert [ln.kind for ln in ls] == [L.SPEECH, L.BACKGROUND, L.BACKGROUND, L.SPEECH]
    # Without English subtitles there's nothing to tell dialogue from lyrics: loud music keeps the original.
    monkeypatch.setattr(align, "load_chosen", lambda j, w: [])
    ls = lines()
    align.classify(job, ls)
    assert ls[0].kind == L.BACKGROUND


def test_subtitle_led_prompt_offers_checked_subtitles_only():
    from aidub import translate as T
    plain = L.Line(1, 0, 2, source_text="総員退避!", english_ref="Everyone, retreat!")
    unsure = L.Line(2, 3, 4, source_text="ちち", english_ref="Daddy!", relisten="ちち")
    none = L.Line(3, 5, 6, source_text="はい")
    led = T.translate_prompt([plain, unsure, none], [], "", 4.3, {}, "subtitle")[1]["content"]
    assert '"subtitle": "Everyone, retreat!"' in led and "subtitle_ok" in led
    assert '"subtitle_hint": "Daddy!"' in led                     # misheard lines keep their own rule
    audio = T.translate_prompt([plain, unsure, none], [], "", 4.3, {})[1]["content"]
    assert '"subtitle": ' not in audio and "subtitle_ok" not in audio and "subtitle-led" not in audio
    plain.en_from = "subtitle"
    assert T.flags(plain) == {"subtitle_led": True, "subtitle_hint": "Everyone, retreat!"}
    assert T.flags(none) == {}


def test_japanese_names_spelled_as_they_sound(tmp_path):
    from aidub import pronounce as P
    # Without the dictionary nothing is respelled: "home" and "made" spell like romaji too.
    P.load(tmp_path / "none.dict")
    assert P.respell("Kikoru") is None and P.respell("home") is None
    lex = tmp_path / "cmudict.dict"
    lex.write_text("home HH OW1 M\nmade M EY1 D\nname N EY1 M\n")
    try:
        assert P.load(lex)
        assert P.respell("Kikoru") == "kee koh roo" and P.respell("chan") == "chahn"
        assert P.respell("Kaiju") == "kye joo" and P.respell("Shinomiya") == "shee noh mee yah"
        assert P.respell("Kafka") is None and P.respell("klicks") is None      # not romanised Japanese
        assert P.respell("home") is None and P.respell("made") is None         # English words
    finally:
        P.load(tmp_path / "none.dict")


def test_subtitles_as_script_share_cues_and_leave_short_unsubtitled_lines():
    from aidub import translate as T
    from aidub.subtitles.model import Cue
    a1, a2 = L.Line(1, 10.0, 11.5, source_text="ちょっと通ります"), L.Line(2, 11.6, 12.4, source_text="すいません")
    b1 = L.Line(3, 30.0, 32.0, source_text="大泉エリア戦況悪化")
    cues = [Cue(start=10.0, end=12.4, text="Hey! Sorry, can I pass through?"),
            Cue(start=30.1, end=32.0, text="The Oizumi area is worsening!"), Cue(start=50, end=51, text="Nobody")]
    shares = T.script_cues(cues, [[a1, a2], [b1]])
    assert [c["text"] for c in shares[0]] == ["Hey! Sorry, can I pass through?"]   # one cue, two lines
    assert [c["text"] for c in shares[1]] == ["The Oizumi area is worsening!"]     # no cue twice, none stray
    # A subtitle shown just after its line still belongs to it; an on-screen sign doesn't.
    mina, later = L.Line(7, 5.375, 5.615), L.Line(8, 6.46, 8.3)
    near = [Cue(start=5.66, end=6.09, text="Mina!"), Cue(start=6.2, end=6.4, text="KAIJU NO. 15")]
    assert [c["text"] for c in T.script_cues(near, [[mina, later]])[0]] == ["Mina!"]
    roger = L.Line(4, 20.0, 20.6, source_text="了解!")
    speech = L.Line(5, 40.0, 43.0, source_text="こんな事になるとは思わなかった")
    assert T.covered_ids([a2, roger], cues) == {2}
    # A short line just before its own cue is covered, so its neighbour never voices it.
    six, seven = L.Line(6, 3.0, 5.4, source_text="任せて"), L.Line(7, 5.45, 5.65, source_text="ミナ")
    assert T.covered_ids([six, seven], [Cue(start=5.66, end=6.0, text="Mina!")]) == {7}
    # Only a sound the subtitles left out keeps the original; a short line with words is dubbed.
    assert T.short_reaction(L.Line(9, 1.0, 1.4, source_text="ええ。"), "ja")
    assert not T.short_reaction(roger, "ja") and not T.short_reaction(speech, "ja")
    assert not T.short_reaction(L.Line(9, 1.0, 2.5, source_text="すごいね"), "ja")
    # The rest of a speaker's sentence, said just before or after a subtitle too long for
    # their subtitled line, shares that subtitle and gets some of its words.
    s1 = L.Line(224, 1019.365, 1020.165, source_text="今度の試合", character="c1")
    s2 = L.Line(225, 1021.125, 1021.925, source_text="応援に行くんだろう", character="c1")
    s3 = L.Line(226, 1024.155, 1024.955, source_text="どうしようかな", character="c2")
    qs = [{"start": 1018.64, "end": 1020.7, "text": "You'll go support him, right?"},
          {"start": 1024.0, "end": 1025.0, "text": "Don't know."}]
    assert T.line_subtitles([s1, s2, s3], qs) == {224: [0], 225: [0], 226: [1]}
    got = {224: {"en": "You'll go support him, right?"}, 225: {"en": ""}}
    T.verbatim_lines([s1, s2, s3], qs, got)
    assert got[225]["en"] and f"{got[224]['en']} {got[225]['en']}" == "You'll go support him, right?"
    e1 = L.Line(254, 1292.98, 1293.94, source_text="皆さんこれまで", character="c1")
    e2 = L.Line(255, 1295.92, 1298.38, source_text="応援ありがとうございました", character="c1")
    assert T.line_subtitles([e1, e2], [{"start": 1294.6, "end": 1298.5,
                                        "text": "Everyone, thank you for supporting me all this time."}]) == {254: [0], 255: [0]}
    # ... but not another character's line, a sound, or a subtitle that fits its own line.
    other = L.Line(225, 1021.125, 1021.925, source_text="応援に行くんだろう", character="c2")
    assert T.line_subtitles([s1, other], qs[:1]) == {224: [0], 225: []}
    fits = [{"start": 1018.64, "end": 1020.7, "text": "Next match?"}]
    assert T.line_subtitles([s1, s2], fits) == {224: [0], 225: []}
    oh = L.Line(225, 1021.125, 1021.925, source_text="ああ", character="c1")
    assert T.line_subtitles([s1, oh], qs[:1]) == {224: [0], 225: []}
    user = T.script_prompt([a1, a2], shares[0], [], "", 4.3, {})[1]["content"]
    assert "Hey! Sorry, can I pass through?" in user and '"from_subtitle"' in user and '"matches"' in user
    # Each line gets the subtitles shown while it is spoken, by time; a shared one is marked.
    c1, c2, c3 = (L.Line(i, a, b) for i, a, b in ((11, 853.9, 855.9), (12, 858.8, 859.5), (13, 860.4, 862.1)))
    cs = [{"start": 853.5, "end": 856.4, "text": "Right, Kafka?"},
          {"start": 858.8, "end": 862.5, "text": "Please watch your step. It's okay. Please don't rush."}]
    assert T.line_subtitles([c1, c2, c3], cs) == {11: [0], 12: [1], 13: [1]}
    user = T.script_prompt([c1, c2, c3], cs, [], "", 4.3, {})[1]["content"]
    assert '"subtitle": "Right, Kafka?"' in user and '"shared_with": [\n13\n]' in user
    a1.en_from = "script"
    assert T.flags(a1) == {"subtitle_script": True} and T.led(a1)
    assert "subtitle_script" in T.source_notes([T.flags(a1)])


def test_subtitle_track_must_be_trusted_to_be_the_script():
    from aidub import translate as T
    def job(content, disagree):
        j = _StubJob({}, {"subtitles": {"english": "e", "candidates": [
            {"id": "e", "score": {"content": content, "total": 0.85}}]}})
        j.qc = {"subtitle_lines_checked": 200, "subtitle_lines_disagreeing_with_speech": disagree}
        return j
    assert T.subtitles_trusted(job(1.0, 8))[0]
    ok, why = T.subtitles_trusted(job(0.5, 8))       # meaning doesn't follow the speech: another cut, machine-made
    assert not ok and "meaning" in why
    assert not T.subtitles_trusted(job(1.0, 90))[0]  # too many lines disagree one by one


def test_subtitles_word_for_word():
    from aidub import translate as T
    assert T.spoken_form("No. 12.") == "Number 12." and T.spoken_form("No. I won't.") == "No. I won't."
    assert T.only_deletes("Twenty kilometers!", "It's twenty kilometers away!")
    assert not T.only_deletes("Twenty klicks!", "It's twenty kilometers away!")
    assert not T.only_deletes("Kilometers twenty", "It's twenty kilometers away!")    # reordered
    assert T.split_words("Hey! Sorry, can I pass through?", [1.3, 0.8]) == ["Hey! Sorry,", "can I pass through?"]
    own = L.Line(1, 10.0, 12.0)
    a, b = L.Line(2, 20.0, 21.3), L.Line(3, 21.4, 22.2)
    cues = [{"start": 10.0, "end": 12.0, "text": "The disaster caused by No. 9 will not end at this level."},
            {"start": 20.0, "end": 22.2, "text": "Hey! Sorry, can I pass through?"}]
    assert [([l.id for l in g], ks) for g, ks in T.subtitle_groups([own, a, b], cues)] == [([1], [0]), ([2, 3], [1])]
    # The translator's rewording is replaced by the subtitle; its division of a shared one is kept
    # when it used exactly the subtitle's words, and its shorter version only if it just leaves words out.
    drafts = {1: {"en": "No. 9's disaster won't end this easily.", "short": "The disaster won't end at this level."},
              2: {"en": "Hey! Sorry,", "short": ""}, 3: {"en": "can I pass through?", "short": "Can I pass?"}}
    assert T.verbatim_lines([own, a, b], cues, drafts) == 0
    assert drafts[1]["en"] == "The disaster caused by Number 9 will not end at this level."
    assert drafts[1]["short"] == "The disaster won't end at this level." or drafts[1]["short"] == ""
    assert drafts[2]["en"] == "Hey! Sorry," and drafts[3]["en"] == "can I pass through?"
    assert drafts[3]["short"] == "Can I pass?"
    drafts = {2: {"en": "Excuse me!"}, 3: {"en": "Can I get by?"}}      # reworded: divided by time instead
    assert T.verbatim_lines([a, b], cues[1:], drafts) == 1
    assert [drafts[2]["en"], drafts[3]["en"]] == ["Hey! Sorry,", "can I pass through?"]


def test_word_for_word_numbers_names_and_splits():
    from aidub import speak, translate as T
    assert speak.spoken("84%!") == "84 percent!" and speak.spoken("Fortitude is 9.0!") == "Fortitude is 9 point 0!"
    assert speak.wer("84%!", "Eighty-four percent.") == 0
    assert speak.wer("Estimated Fortitude is 9.0!", "Estimated fortitude is nine point oh.") == 0
    assert speak.missed_words("Fortitude is 9.0!", "Fortitude is nine point.") == ["0"]   # the voice dropped it
    # The translator's division of a shared subtitle, mapped back onto its own words.
    assert T.align_split("I will take care of it. Mina!", ["I'll take care of it.", "Mina!"]) == \
        ["I will take care of it.", "Mina!"]
    assert T.align_split("Hey! Sorry, can I pass through? You guys...", ["Sorry, can I pass?", "You guys..."]) == \
        ["Hey! Sorry, can I pass through?", "You guys..."]


def test_free_timing_prompts_have_no_length_target():
    from aidub import translate as T
    ln = L.Line(1, 0, 2, source_text="総員退避!", english_ref="Everyone, retreat!")
    timed = T.translate_prompt([ln], [], "", 4.3, {}, "subtitle")
    free = T.translate_prompt([ln], [], "", None, {}, "subtitle")
    assert "target_syllables" in timed[1]["content"] and "target syllable" in timed[0]["content"]
    assert "target_syllables" not in free[1]["content"] and "Length is free" in free[1]["content"]
    assert "target syllable" not in free[0]["content"]
    assert "target_syllables" not in T.reflect_prompt([ln], "", None, {}, {1: {"en": "Retreat!"}})[1]["content"]
    assert "target_syllables" not in T.rewrite_prompt([ln], "", None, {}, {1: {"en": "Retreat!"}}, {})[1]["content"]


def test_auto_timing_frees_only_the_word_for_word_dub():
    from aidub import fit
    def job(timing, lead_used):
        cfg = {"fit.timing": timing} if timing else {}
        return _StubJob(cfg, {"translate_lead": lead_used})
    assert fit.free_timing(job(None, "subtitle_verbatim"))         # default: auto
    assert fit.free_timing(job(None, "audio"))                     # speech only: natural length keeps more words
    assert not fit.free_timing(job("auto", "subtitle"))
    assert fit.free_timing(job("start", "audio")) and not fit.free_timing(job("window", "subtitle_verbatim"))


def test_queue_claim_is_atomic(tmp_path):
    from aidub import jobs as J
    st = J.Store(tmp_path / "db.sqlite")
    a, b = st.add("/m/a.mkv", "p"), st.add("/m/b.mkv", "p")
    st.update(a, status=J.CANCELLED)     # cancelled after the runner picked it
    st.remove(b)                         # removed after the runner picked it
    assert not st.claim(a) and not st.claim(b)
    c = st.add("/m/c.mkv", "p")
    assert st.claim(c) and st.get(c)["status"] == J.RUNNING and not st.claim(c)


def test_model_replies_read_safely():
    T = translate
    got = T._collect('{"lines": ["junk", {"id": 1, "en": "Run!", "short": null}, {"id": 2, "en": null},'
                     ' {"id": 9, "en": "Not asked for"}]}', {1, 2})
    assert got[1]["en"] == "Run!" and got[1]["short"] == "" and got[2]["en"] == "" and 9 not in got
    assert T.parse_json('Notes [1]: {"lines": [{"id": 1}]}') == {"lines": [{"id": 1}]}
    assert T.parse_json('```json\n[{"id": 2}]\n```') == {"lines": [{"id": 2}]}
    assert "target_syllables" not in T.VERBATIM_TASK_FREE and "target syllable" not in T.SCRIPT_SYSTEM_FREE


def test_subtitle_track_trust_without_line_checks():
    from aidub.subtitles import step as S
    job = _StubJob({}, {"subtitles": {"english": "e"}})
    job.qc = {}
    orig = S.candidate
    S.candidate = lambda j, cid: {"score": {"content": 0.9, "total": 0.8}}
    try:
        ok, why = translate.subtitles_trusted(job)          # too few lines to check one by one
        assert ok and "no lines checked" in why
        job.qc = {"subtitle_lines_checked": 8, "subtitle_lines_disagreeing_with_speech": 4}
        assert not translate.subtitles_trusted(job)[0]
    finally:
        S.candidate = orig


def test_numbers_read_and_checked_as_said():
    from aidub import speak as S
    assert S.said_as("It costs 1,000 yen.") == "It costs 1000 yen."
    assert S.said_as("12.75 seconds, $4.99, $5") == "12 point 7 5 seconds, 4 dollars 99, 5 dollars"
    assert S.said_as("1,2,3") == "1,2,3"
    assert S.missed_words("It costs 1,000 yen.", "it costs one thousand yen") == []
    assert S.missed_words("The 1st squad.", "the first squad") == []
    assert S.wer("The 21st squad moves.", "the twenty first squad moves") == 0
    # The first word of every sentence is checked, not taken for a name.
    assert S.missed_words("Wait. Brakes failed on the truck.", "wait cakes failed on the truck") == ["brakes"]
    assert S.missed_words("Wait, Kafka is here.", "wait, Kafuka is here") == []


def test_dub_run_on_keeps_original_lines_audible():
    from aidub.mix import outside, outside_all
    # A dub running on from 1.5 to 3.0 s over a kept line at 2.0-2.5 s mutes only around it.
    assert outside((1.5, 3.0), [(2.0, 2.5)]) == [(1.5, 2.0), (2.5, 3.0)]
    assert outside((1.5, 3.0), [(0.0, 9.0)]) == []
    # ... but never un-mutes the dubbed lines' own original speech.
    assert outside_all([(2.0, 2.5)], [(2.2, 2.3)]) == [(2.0, 2.2), (2.3, 2.5)]


def test_subtitle_languages_and_files(tmp_path):
    from aidub import media
    from aidub.subtitles import pgs, sources
    assert [sources._lang_from_name(t) for t in (["it"], ["pt", "br"], ["en", "hi"], ["spanish"], ["ita"])] == \
        ["it", "pt", "en", "es", "it"]
    assert media.lang_code("ita") == "it" and media.iso3("it") == "ita" and media.iso3("yue") == "chi"
    assert clean.script_language(["Non lo so. Andiamo a casa, per favore.", "Che cosa fai?"]) == ""
    assert clean.script_language(["I don't know. Let's go home.", "What are you doing?"]) == "en"
    # Hearing-impaired tags are not speaker names; a Japanese speaker label is.
    assert clean.clean_text("(SIGHS) I don't know.") == ("I don't know.", "", DIALOGUE)
    assert clean.clean_text("(健太)行くぞ")[:2] == ("行くぞ", "健太")
    for style, song in (("Lyrics", True), ("Song_EN", True), ("OP", True), ("Editor", False), ("Default", False)):
        assert bool(clean.SONG_STYLES.search(style)) is song, style
    # Subtitle files in legacy encodings are read, not dropped.
    srt = "1\n00:00:01,000 --> 00:00:02,000\n{}\n"
    for text, enc in (("こんにちは、元気ですか", "shift_jis"), ("你好，你好吗", "gbk"), ("Café déjà vu", "cp1252")):
        p = tmp_path / f"{enc}.srt"
        p.write_bytes(srt.format(text).encode(enc))
        assert [q.text for q in clean.load(p)] == [text], enc
    # Sidecar copies are named by file, so a rescan never overwrites an earlier copy.
    video = tmp_path / "Ep.mkv"
    video.touch()
    (tmp_path / "Ep.en.srt").write_text(srt.format("English"))
    (tmp_path / "Ep.ja.srt").write_text(srt.format("日本語"))
    work = tmp_path / "work"
    work.mkdir()
    first = {c.id: c.path for c in sources.from_sidecars(work, video)}
    (tmp_path / "Ep.en.hi.srt").write_text(srt.format("HI"))
    second = {c.id: c.path for c in sources.from_sidecars(work, video)}
    assert all(second[k] == v for k, v in first.items())
    assert open(first["sidecar:Ep.ja.srt"], encoding="utf-8").read().count("日本語") == 1
    # Truncated bitmap data keeps what was decoded.
    assert pgs._rle_decode(bytes([5, 5, 0, 0x80]), 4, 2)[0].tolist() == [5, 5, 0, 0]


def test_last_window_long_enough_to_diarize():
    from aidub import scenes
    assert scenes.windows(120.9, [], [], 120) == [[0.0, 100.9], [100.9, 120.9]]


def test_delivery_swap_resumes_after_a_crash(tmp_path):
    from aidub import deliver

    class J:
        def __init__(self):
            self.data = {}

        def get(self, k, d=None):
            return self.data.get(k, d)

        def set(self, k, v):
            self.data[k] = v

        def log(self, m):
            pass

    live, aside, tmp = tmp_path / "X.mkv", tmp_path / "X.mkv.aidub-original", tmp_path / ".X.aidub-partial"
    job = J()
    assert not deliver._finish_swap(job)                     # nothing pending
    # Stopped between the two renames: the original is aside, the dub still has its temporary name.
    aside.write_text("original")
    tmp.write_text("dub")
    job.set("deliver_swap", {"tmp": str(tmp), "final": str(live), "set_aside": str(aside),
                             "delivered": {"file": str(live)}})
    assert deliver._finish_swap(job)
    assert live.read_text() == "dub" and aside.read_text() == "original" and not tmp.exists()
    assert job.get("delivered") == {"file": str(live)} and job.get("deliver_swap") is None
    # Stopped before the swap began: deliver again from the start.
    job.set("deliver_swap", {"tmp": str(tmp_path / "none"), "final": str(tmp_path / "Y.mkv"),
                             "set_aside": str(tmp_path / "Y.mkv.aidub-original"), "delivered": {}})
    assert not deliver._finish_swap(job) and job.get("deliver_swap") is None


def test_plain_subtitle_songs_credits_and_signs(tmp_path):
    # An SRT has no styles: an opening song's lyrics in romaji (with the translation under
    # them), its title, the credits shown over it, sign text and title cards are not dialogue.
    p = tmp_path / "op.en.srt"
    p.write_text("1\n00:00:02,000 --> 00:00:05,000\n<i>Opening Song\nOmohide (Memories) by Suzuki</i>\n\n"
                 "2\n00:00:06,000 --> 00:00:10,000\n<i>STARRING\nKobayashi Kaoru</i>\n\n"
                 "3\n00:00:12,000 --> 00:00:16,000\n<i>Ayata Toshiki</i>\n\n"
                 "4\n00:00:26,000 --> 00:00:28,000\n<i>kimi ga hai ta shiroi iki ga\nThe white breath you exhale</i>\n\n"
                 "5\n00:00:31,000 --> 00:00:34,000\n<i>ima, yukkuri kaze notte\nIs slowly riding in the wind</i>\n\n"
                 "6\n00:00:40,000 --> 00:00:42,000\n<i>Directed by Matsuoka Joji</i>\n\n"
                 "7\n00:01:46,000 --> 00:01:48,000\n<i>As the day comes to an end,</i>\n\n"
                 "8\n00:01:54,000 --> 00:01:56,000\n~ Beer ~ 600 yen\n\n"
                 "9\n00:02:27,000 --> 00:02:32,000\nEpisode 1\nRed Wiener & Tamagoyaki\n\n"
                 "10\n00:02:44,000 --> 00:02:47,000\n<i>(*Pork soup with veggies and miso)</i>\n"
                 "Can you make something\nelse other than traditional Tonjiru*?\n\n"
                 "11\n00:03:00,000 --> 00:03:02,000\n(lit. kettle, pejorative slang for gay men, originated in the "
                 "Edo period) I don't like okama myself.\n\n"
                 "12\n00:03:10,000 --> 00:03:12,000\nKenji Tanaka\n", encoding="utf-8")
    cues = clean.load(p)
    assert [c.kind for c in cues] == [SONG, SIGN, SIGN, SONG, SONG, SIGN, DIALOGUE, SIGN, SIGN, DIALOGUE, DIALOGUE,
                                      DIALOGUE]
    assert cues[9].text == "Can you make something else other than traditional Tonjiru?"
    assert cues[10].text == "I don't like okama myself."
    assert cues[11].text == "Kenji Tanaka"  # a bare name far from any credit is someone calling out
    for said in ["Hi... You're back home?", "I do so", "Oh no, I know.", "Mama, I'm home", "Go ahead.",
                 "Sake, please, Master", "Make them scramble over the frying pan,"]:
        assert clean.text_kind(said) == DIALOGUE, said
    assert clean.is_romaji("zutto mukashi no koto no you da ne") and clean.is_romaji("pokkari to ukan de iru")


def test_free_timing_late_lines_do_not_pile_up():
    # Ten back-to-back lines, each English take 0.4 s longer than the gap to the next
    # original line: unpaced, the tenth would start ~4.7 s late.
    lines = [(i * 2.0, 2.4) for i in range(10)]
    plan = fit.pace(lines, gap=0.12, max_late=0.5, max_tempo=1.3)
    late = [s - o for (o, _), (s, _) in zip(lines, plan)]
    assert max(late) <= 0.5 + 1e-6
    assert all(t <= 1.3 for _, t in plan) and plan[-1][1] == 1.0 and max(t for _, t in plan) > 1.0
    # A line far too long for the speed-up limit makes the next one later, but the line after recovers.
    plan = fit.pace([(0.0, 6.0), (2.0, 1.0), (10.0, 1.0)], gap=0.12, max_late=0.5, max_tempo=1.3)
    assert plan[0][1] == 1.3 and plan[1][0] == pytest.approx(6.0 / 1.3 + 0.12, abs=1e-3) and plan[2] == (10.0, 1.0)
    # Lines with room keep their natural speed and start with the original.
    assert fit.pace([(0.0, 1.0), (3.0, 1.0)], 0.12, 0.5, 1.3) == [(0.0, 1.0), (3.0, 1.0)]


def test_free_timing_starts_early_only_when_it_helps():
    lines = [(5.0, 3.0), (7.0, 1.0)]  # the first runs 1.12 s into the second
    plan = fit.pace(lines, 0.12, 0.35, 1.3, early=0.4, gentle_tempo=1.12)
    assert plan[0][0] == pytest.approx(4.6) and plan[1][0] - 7.0 <= 0.35 + 1e-6
    # Nothing crowding it: on time. And no earlier than a shot change at 4.9.
    assert fit.pace([(5.0, 1.0)], 0.12, 0.35, 1.3, early=0.4) == [(5.0, 1.0)]
    assert fit.pace(lines, 0.12, 0.35, 1.3, early=0.4, floors=[4.9, 0.0])[0][0] == pytest.approx(4.9)
    # Gently first: several lines share a little before one goes fast.
    run = [(i * 2.0, 2.2) for i in range(5)]
    assert max(t for _, t in fit.pace(run, 0.12, 0.35, 1.3, gentle_tempo=1.12)) <= 1.12 + 1e-6


def test_intro_and_end_credits_keep_the_original(tmp_path, monkeypatch):
    from aidub import align
    # Opening song and credits (0:02-1:27), dialogue 1:46-23:06, then an unsubtitled closing song to 24:37.
    p = tmp_path / "ep.en.srt"
    p.write_text("1\n00:00:02,300 --> 00:00:05,400\nOpening Song\nOmohide by Suzuki\n\n"
                 "2\n00:00:06,600 --> 00:00:10,200\nSTARRING\nKobayashi Kaoru\n\n"
                 "3\n00:00:12,600 --> 00:00:16,200\nAyata Toshiki\n\n"
                 "4\n00:00:18,700 --> 00:00:22,100\nAndo Tamae\n\n"
                 "5\n00:00:26,500 --> 00:00:28,200\nkimi ga hai ta shiroi iki ga\nThe white breath you exhale\n\n"
                 "5\n00:00:36,500 --> 00:00:41,500\nsora ni ukabu kumo no naka ni\nAnd little by little vanishing\n\n"
                 "6\n00:01:46,200 --> 00:01:48,100\nAs the day comes to an end,\n\n"
                 "7\n00:12:00,000 --> 00:12:02,000\nTwo red wieners, please.\n\n"
                 "8\n00:23:04,300 --> 00:23:06,000\nGood night!\n", encoding="utf-8")
    cues = clean.load(p)
    monkeypatch.setattr(align, "load_chosen", lambda job, which: cues if which == "english" else [])
    job = {"ingest": {"duration": 1477.0, "song_ranges": [[600.0, 690.0]]}}
    ranges = align.intro_credits(job)
    assert [600.0, 690.0] in ranges                                   # a chapter named Opening/Ending/Credits
    assert [2.3, 41.5] in [[round(a, 1), round(b, 1)] for a, b in ranges]  # the opening song and its credits
    assert any(a == 0.0 and 100 < b < 106.2 for a, b in ranges)       # nothing before the first dialogue
    assert any(1386 < a < 1392 and b == 1477.0 for a, b in ranges)    # the unsubtitled closing song
    assert not align._in_ranges(106.2, 108.1, ranges)                  # the narration after the song is dubbed
    assert not align._in_ranges(1384.3, 1386.0, ranges)                # ... and the last line before the credits
    # Without English subtitles only chapters and song/credit runs count.
    monkeypatch.setattr(align, "load_chosen", lambda job, which: [])
    assert align.intro_credits({"ingest": {"duration": 1477.0}}) == []


def test_subtitle_flashes_and_offscreen_rows(tmp_path):
    p = tmp_path / "e02.en.srt"
    p.write_text("1\n00:11:16,400 --> 00:11:17,830\nthey really don't\nneed that much salt.\n\n"
                 "2\n00:11:17,830 --> 00:11:18,180\nBut I love salty foods...\nthey really don't\nneed that much salt.\n\n"
                 "3\n00:11:18,180 --> 00:11:19,760\nBut I love salty foods...\n\n"
                 "4\n00:14:22,760 --> 00:14:29,280\nI can't really explain it,\nbut I'm just giving my best for now.\n"
                 "<i>Roaming the streets of Shinjuku</i>\n\n"
                 "5\n00:14:29,620 --> 00:14:34,820\n<i>Is the stray cat...</i>\n\n"
                 "6\n00:14:39,750 --> 00:14:44,930\n- The Red & White Contest!\n- She's something.\n<i>To a woman...</i>\n",
                 encoding="utf-8")
    texts = [c.text for c in clean.load(p)]
    # The 0.35 s flash repeating its neighbours is gone; a lyric row under speech isn't spoken.
    assert texts == ["they really don't need that much salt.", "But I love salty foods...",
                     "I can't really explain it, but I'm just giving my best for now.", "Is the stray cat...",
                     "The Red & White Contest!", "She's something."]


def test_a_singers_performance_stays_a_song():
    from aidub import align

    def ln(i, a, b, text, spk="S5", ev=("BGM",), music=-7.0):
        return L.Line(id=i, start=a, end=b, source_text=text, speaker=spk, events=list(ev), music_db=music)
    lines = [ln(1, 584.0, 584.7, "行き当たり渡り", ev=("Sing",), music=-0.2),
             ln(2, 589.8, 598.4, "行き当たり渡りを"),               # 5 s later, music a little quieter
             ln(3, 602.0, 605.6, "になっても"),
             ln(4, 617.9, 620.8, "路地裏見つけて"),                  # 12 s gap: reached through line 5
             ln(5, 610.0, 612.0, "行き当たりバタリ"),
             ln(6, 625.0, 630.3, "よくわからないですけどとにかく一生懸命歌わせていただいてます"),  # her interview: spoken
             ln(7, 621.0, 623.0, "次の方どうぞ", spk="S9"),           # someone else
             ln(8, 640.0, 642.0, "とてもなのよ")]                      # too long after
    lines[0].kind = L.SONG
    align.extend_songs(lines, "ja")
    assert [l.id for l in lines if l.kind == L.SONG] == [1, 2, 3, 4, 5]


def _f0(*parts):
    """Pitch track (Hz every 10 ms) from (seconds, kind, semitones) parts: 'sing' holds notes
    with vibrato, 'speak' glides, 'rest' is unvoiced."""
    out = []
    for seconds, kind, st in parts:
        t = np.arange(int(seconds * 100)) * 0.01
        if kind == "rest":
            out.append(np.zeros(t.size))
            continue
        if kind == "sing":   # a new note every 0.6 s, 0.4 st vibrato at 6 Hz
            curve = st + 2 * (np.floor(t / 0.6) % 3) + 0.4 * np.sin(2 * np.pi * 6 * t)
        else:                # syllables gliding up and down by 4 st, a gap every 0.3 s
            curve = st + 4 * np.sin(2 * np.pi * 2.5 * t)
        f = 55 * 2 ** (curve / 12)
        if kind == "speak":
            f[(t % 0.3) > 0.24] = 0
        out.append(f)
    return np.concatenate(out).astype(np.float32)


def test_pitch_tells_singing_from_speech():
    from aidub.pitch import Pitch
    p = Pitch(_f0((20, "sing", 20), (5, "rest", 0), (20, "speak", 20)))
    assert p.held_share(5, 15) > 0.6 and p.held_share(30, 40) < 0.1
    assert p.held_share(21, 22) is None                     # nothing voiced to judge
    assert abs(p.median(30, 40) - 20) < 1.5


def test_singing_marks_unsubtitled_lines_and_leaves_subtitled_ones_to_the_lyrics_check(tmp_path, monkeypatch):
    from aidub import align
    np.save(tmp_path / "pitch.npy", _f0((40, "sing", 20), (40, "speak", 10)))
    p = tmp_path / "e.en.srt"
    p.write_text("1\n00:00:10,000 --> 00:00:14,000\nWherever fate leads me,\n\n"
                 "2\n00:00:50,000 --> 00:00:53,000\nI'm buying!\n", encoding="utf-8")
    cues = clean.load(p)
    monkeypatch.setattr(align, "load_chosen", lambda job, which: cues if which == "english" else [])
    job = _StubJob({}, {"pitch": {"path": str(tmp_path / "pitch.npy"), "hop": 0.01}})
    job.qc = {}

    def ln(i, a, b, music):
        return L.Line(id=i, start=a, end=b, source_text="x", music_db=music)
    lines = [ln(1, 10, 14, -3), ln(2, 20, 24, -3), ln(3, 30, 33, -30), ln(4, 50, 53, -3), ln(5, 60, 63, -3)]
    align.mark_singing(job, lines)
    assert [l.kind for l in lines] == [L.SPEECH, L.SONG, L.SPEECH, L.SPEECH, L.SPEECH]
    assert align.sung_candidate(lines[0])                    # subtitled: the lyrics check decides
    assert not align.sung_candidate(lines[2])               # no music under it
    assert not align.sung_candidate(lines[3]) and not align.sung_candidate(lines[4])   # spoken
    assert job.qc["songs_by_pitch"] == {"sung": 1, "lyrics_to_check": 1}


def test_lyrics_check_keeps_lyrics_and_dubs_lines_spoken_over_a_song():
    from aidub import translate

    class FakeLLM:
        def __init__(self, reply):
            self.reply, self.asked = reply, []

        def chat_many(self, prompts, max_tokens=4096):
            self.asked += prompts
            return [self.reply] * len(prompts)

    def lines():
        out = [L.Line(id=i, start=a, end=a + 3, source_text=t, english_ref=e, music_db=-3, sung=0.6)
               for i, (a, t, e) in enumerate([(10, "行き当たりばったり", "Wherever fate leads me,"),
                                               (14, "もしもし", "Ahh, are you there? Just a moment."),
                                               (100, "迷い猫", "was the stray cat.")], 1)]
        return out + [L.Line(id=9, start=5, end=7, source_text="ね", english_ref="Isn't it?")]
    job = _StubJob({}, {"subtitles": {"source_language": "ja"}})
    job.qc, job.log = {}, lambda m: None
    ls = lines()
    llm = FakeLLM('{"lines": [{"id": 1, "sung": true}, {"id": 2, "sung": false}, {"id": 3, "sung": true}]}')
    assert translate.lyrics_check(job, llm, ls, []) == 3 - 1
    assert [l.kind for l in ls] == [L.SONG, L.SPEECH, L.SONG, L.SPEECH]
    assert len(llm.asked) == 2 and "Isn't it?" in llm.asked[0][1]["content"]   # two runs; neighbours as context
    ls = lines()
    translate.lyrics_check(job, FakeLLM("no idea"), ls, [])  # unreadable: songs are never dubbed
    assert [l.kind for l in ls] == [L.SONG, L.SONG, L.SONG, L.SPEECH]


def test_recurring_opening_found_in_a_sibling_episode():
    from aidub import recurring
    rng = np.random.default_rng(7)
    sr = recurring.SR

    def sound(seconds):  # music-like: noise shaped by a changing envelope
        n = int(seconds * sr)
        env = np.repeat(rng.uniform(0.2, 1.0, n // 512 + 1), 512)[:n]
        return (rng.standard_normal(n) * env).astype(np.float32)
    theme = sound(60)
    a = np.concatenate([sound(30), theme, sound(120)])
    b = np.concatenate([sound(75), theme + 0.05 * rng.standard_normal(theme.size).astype(np.float32), sound(90)])
    found = recurring.common(recurring.fingerprint(a), recurring.fingerprint(b))
    assert len(found) == 1 and abs(found[0][0] - 30) < 3 and abs(found[0][1] - 90) < 3
    assert recurring.common(recurring.fingerprint(sound(120)), recurring.fingerprint(sound(120))) == []


def test_recurring_siblings_nearest_episode_first(tmp_path):
    from aidub import recurring
    for n in (1, 2, 3, 5, 12):
        (tmp_path / f"Show - S01E{n:02d} - Title.mkv").write_bytes(b"")
    (tmp_path / "Show - S01E04 - Title.mkv.aidub-original").write_bytes(b"")
    (tmp_path / "Show - S01E04 - Title.en.srt").write_bytes(b"")
    me = tmp_path / "Show - S01E04 - Title.mkv"
    assert [p.name[7:13] for p in recurring.siblings(me, 3)] == ["S01E03", "S01E05", "S01E02"]


def test_short_reply_in_the_other_persons_voice_moves_by_pitch(tmp_path):
    from aidub import voicebank
    prof = profiles.Profiles(tmp_path).get_or_create("Diner")
    woman, master = prof.new_character("Miyuki", gender="female"), prof.new_character("Master", gender="male")
    # Her lines sit at 25 st, his at 10; his 0.4 s reply "Uh? For what?" was filed under her.
    np.save(tmp_path / "pitch.npy", _f0((10, "speak", 25), (1, "speak", 10), (9, "speak", 25), (10, "speak", 10)))
    job = _StubJob({}, {"pitch": {"path": str(tmp_path / "pitch.npy"), "hop": 0.01}})
    job.qc, job.log = {}, lambda m: None
    lines = [L.Line(i, a, a + 1.5, character=woman) for i, a in enumerate((1, 4, 7, 12, 15), 1)]
    lines += [L.Line(10, 10.3, 10.7, character=woman, source_text="え？何が"),
              L.Line(11, 13.2, 13.6, character=woman, source_text="ふふ")]           # her own short laugh
    lines += [L.Line(20 + i, a, a + 1.5, character=master) for i, a in enumerate((21, 24, 27))]
    assert voicebank.move_by_pitch(job, prof, lines) == 1
    assert lines[5].character == master and lines[5].voice_note == "moved by pitch"
    assert lines[6].character == woman
    # A man's line measured far higher (a raised voice, a woman talking over him) stays his.
    np.save(tmp_path / "pitch.npy", _f0((10, "speak", 10), (1, "speak", 25), (9, "speak", 10), (10, "speak", 25)))
    lines = [L.Line(i, a, a + 1.5, character=master) for i, a in enumerate((1, 4, 7, 12, 15), 1)]
    lines += [L.Line(10, 10.3, 10.7, character=master, source_text="いらっしゃい")]
    lines += [L.Line(20 + i, a, a + 1.5, character=woman) for i, a in enumerate((21, 24, 27))]
    assert voicebank.move_by_pitch(job, prof, lines) == 0


def test_lyric_subtitles_are_not_given_to_the_next_spoken_line():
    from aidub import translate
    from aidub.subtitles.model import Cue
    cues = [Cue(start=805.6, end=810.9, text="was the stray cat.", raw="", kind=DIALOGUE),
            Cue(start=810.9, end=815.0, text="Today we feature the hit singer.", raw="", kind=DIALOGUE)]
    lines = [L.Line(id=1, start=805.6, end=810.5, kind=L.SONG), L.Line(id=2, start=810.7, end=815.4)]
    assert [q.text for q in translate.outside_songs(cues, lines)] == ["Today we feature the hit singer."]


def test_long_slow_held_lines_are_songs_without_asking(tmp_path):
    from aidub import translate

    class Quiet:
        def chat_many(self, prompts, max_tokens=4096):
            return ['{"lines": []}'] * len(prompts)
    # 10 s of held notes over "膝に抱かれて聴いている" (1 character a second): sung, even if the
    # model would call it narration; a spoken line over the same song is still asked about.
    np.save(tmp_path / "pitch.npy", _f0((30, "sing", 20)))
    job = _StubJob({}, {"pitch": {"path": str(tmp_path / "pitch.npy"), "hop": 0.01},
                        "subtitles": {"source_language": "ja"}})
    job.qc, job.log = {}, lambda m: None
    verse = L.Line(id=1, start=2, end=12.6, source_text="膝に抱かれて聴いている", english_ref="it listened, in her lap",
                   music_db=-2, sung=0.75)
    talk = L.Line(id=2, start=14, end=18.7, source_text="猫どんな曲ですか迷い猫がヒット中の千鳥みゆきさんです",
                  english_ref="Today we feature the hit singer", music_db=-2, sung=0.59)
    translate.lyrics_check(job, Quiet(), [verse, talk], [])
    assert verse.kind == L.SONG and job.qc["lyrics_check"]["sure"] == 1
    assert talk.kind == L.SONG  # the model gave no answer for it: songs are never dubbed


def test_opening_run_stops_at_the_narration_and_quoted_italics_are_lyrics(tmp_path, monkeypatch):
    from aidub import align, translate
    p = tmp_path / "e03.en.srt"
    p.write_text("1\n00:00:02,000 --> 00:00:05,000\nOpening Song\nOmohide by Suzuki\n\n"
                 "2\n00:00:26,000 --> 00:00:28,000\nkimi ga hai ta shiroi iki ga\nThe white breath\n\n"
                 "3\n00:00:48,000 --> 00:00:57,160\nkimi ga hai ta iki o sutte\nAnd drink the breath\n\n"
                 "4\n00:00:57,260 --> 00:00:59,020\n<i>As the day comes to an end,</i>\n\n"
                 "5\n00:01:02,790 --> 00:01:04,980\n<i>my own day begins.</i>\n\n"
                 "6\n00:01:05,290 --> 00:01:06,480\n~ Beer ~ 600 yen\n\n"
                 "7\n00:14:43,000 --> 00:14:47,000\n<i>\"This waltz is no laughing matter, either.\"</i>\n\n"
                 "8\n00:20:00,000 --> 00:20:02,000\nGood night!\n", encoding="utf-8")
    cues = clean.load(p)
    monkeypatch.setattr(align, "load_chosen", lambda job, which: cues if which == "english" else [])
    ranges = align.intro_credits({"ingest": {"duration": 1300.0, "song_ranges": []}})
    assert not align._in_ranges(57.3, 59.0, ranges) and align._in_ranges(30, 40, ranges)
    assert translate.quoted_lyric(L.Line(id=1, start=884.0, end=888.6), cues)
    assert not translate.quoted_lyric(L.Line(id=2, start=57.3, end=59.0), cues)   # narration: italic, no quotes


def test_a_spoken_line_right_after_a_song_is_not_asked_about(tmp_path):
    from aidub import translate

    class Never:
        def chat_many(self, prompts, max_tokens=4096):
            raise AssertionError("asked")
    # The theme ends as the narration starts: the window around the line is mostly song,
    # the line itself is speech.
    np.save(tmp_path / "pitch.npy", _f0((20, "sing", 20), (10, "speak", 12)))
    job = _StubJob({}, {"pitch": {"path": str(tmp_path / "pitch.npy"), "hop": 0.01},
                        "subtitles": {"source_language": "ja"}})
    job.qc, job.log = {}, lambda m: None
    narration = L.Line(id=1, start=20.5, end=22.5, source_text="一日が終わり", english_ref="As the day comes to an end,",
                       music_db=-8, sung=0.44)
    assert translate.lyrics_check(job, Never(), [narration], []) == 0 and narration.kind == L.SPEECH


def test_a_good_track_is_trusted_whatever_its_coverage_and_shows_remember(tmp_path):
    from aidub import translate as T
    prof = profiles.Profiles(tmp_path).get_or_create("Diner")

    def job(name, content, total, disagree):
        j = _StubJob({}, {"subtitles": {"english": "e", "candidates": [
            {"id": "e", "score": {"content": content, "total": total}}]}})
        j.qc = {"subtitle_lines_checked": 200, "subtitle_lines_disagreeing_with_speech": disagree}
        j.profile, j.src = prof, tmp_path / name
        return j
    # S01E04: every word right, long silences: overall 0.74. Meaning and lines decide.
    ok, why = T.subtitles_trusted(job("E04.mkv", 0.86, 0.74, 14))
    assert ok, why
    assert not T.subtitles_trusted(job("E04.mkv", 0.86, 0.60, 14))[0]   # under what chose the track at all
    # A borderline track (meaning 0.75) on a show whose last episodes' tracks were trusted.
    assert not T.subtitles_trusted(job("E05.mkv", 0.75, 0.80, 30))[0]
    for n in ("E01.mkv", "E02.mkv", "E03.mkv"):
        T.remember_trust(job(n, 1.0, 0.83, 10), True, "ok")
    ok, why = T.subtitles_trusted(job("E05.mkv", 0.75, 0.80, 30))
    assert ok and "earlier episodes" in why
    assert not T.subtitles_trusted(job("E05.mkv", 0.5, 0.80, 30))[0]   # a wrong track still fails
    T.remember_trust(job("E04.mkv", 0.5, 0.7, 10), False, "meaning 0.50 below 0.80")
    assert not T.subtitles_trusted(job("E05.mkv", 0.75, 0.80, 30))[0]  # the last episode's track failed


def test_a_subtitled_line_over_a_song_recording_is_not_carried_into_the_song():
    from aidub import align

    def ln(i, a, b, text, ev=("BGM",), music=-7.0):
        return L.Line(id=i, start=a, end=b, source_text=text, speaker="S4", events=list(ev), music_db=music)
    # The theme plays under the conversation; the diarizer filed the singer under the son.
    lines = [ln(1, 1250.6, 1251.2, "息を", ev=("Sing", "BGM")), ln(2, 1263.7, 1270.0, "ことのようだねあたしちょくちょく帰ってやれよ"),
             ln(3, 1272.8, 1274.0, "雲が"), ln(4, 1253.0, 1254.0, "ポカリと")]
    lines[0].kind = L.SONG
    align.extend_songs(lines, "ja", subtitled={2, 3})
    # The subtitled lines are left to the lyrics check; the unsubtitled one carries the song on.
    assert [l.kind for l in lines] == [L.SONG, L.SPEECH, L.SPEECH, L.SONG]
    # Inside an on-stage performance (singing all around), a subtitled verse still joins the song.
    verse = ln(5, 1256.0, 1258.0, "路地裏見つけて")
    verse.sung = 0.76
    lines = [lines[0], verse]
    align.extend_songs(lines, "ja", subtitled={5})
    assert verse.kind == L.SONG


def test_a_short_line_as_the_credits_start_is_spoken_without_asking(tmp_path):
    from aidub import translate

    class Never:
        def chat_many(self, prompts, max_tokens=4096):
            raise AssertionError("asked")
    np.save(tmp_path / "pitch.npy", _f0((20, "sing", 20), (0.6, "speak", 22), (10, "sing", 20)))
    job = _StubJob({}, {"pitch": {"path": str(tmp_path / "pitch.npy"), "hop": 0.01},
                        "subtitles": {"source_language": "ja"}})
    job.qc, job.log = {}, lambda m: None
    bye = L.Line(id=1, start=20.0, end=20.6, source_text="おやすみなさい", english_ref="Good night everyone.",
                 music_db=-4, sung=0.66)
    assert translate.lyrics_check(job, Never(), [bye], []) == 0 and bye.kind == L.SPEECH


def test_back_check_reads_clipped_words_sighs_own_voices_and_run_together_names(monkeypatch):
    from aidub import pronounce, speak
    # Clipped words are said and checked in full.
    assert speak.said_as("Here... How 'bout it?") == "Here... How about it?"
    assert speak.said_as("'Bout time. Leave 'em.") == "About time. Leave them."
    assert speak.missed_words("Here... How 'bout it...", "Here, how about it?") == []
    # A sigh in the script isn't a word to hear back.
    assert speak.missed_words("Ahh... Well, I let the ecstasy of the moment capture me,",
                              "Well, I let the ecstasy of the moment capture me.") == []
    # A line cloned from its own audio isn't held to the voice-likeness floor.
    cfg = type("C", (), {"get": lambda s, k, d=None: d})()
    take = {"wer": 0.0, "missed": [], "words": 4, "seconds": 1.3, "sim": 0.05, "sim_median": 0.047}
    assert not speak._passes(take, cfg) and speak._passes({**take, "own_ref": True}, cfg)
    assert not speak._passes({**take, "own_ref": True, "sim": -0.3}, cfg)   # still far below the character's own
    # ... unless it was given its own voice because it isn't that character at all.
    assert speak._passes({**take, "own_ref": True, "not_character": True, "sim": 0.217, "sim_median": 0.467}, cfg)
    # A contraction heard for the word it sounds like.
    assert speak.wer("were all teachers.", "We're all teachers.") == 0.0
    assert speak.wer("We're all teachers.", "We're all teachers.") == 0.0
    assert speak.wer("Is that your bag?", "Is that you're bag?") == 0.0
    # A stage name run into one word by recognition.
    monkeypatch.setattr(pronounce, "LEXICON", {w: [w.upper()] for w in ("erect", "juice", "gushing", "bonanza")})
    assert speak.wer('"Erect Ohki\'s Juice Gushing Bonanza"', "Erectokey's juice gushing bonanza.") <= 0.2


def test_review_decisions_go_into_the_lines(cfg, tmp_path):
    from aidub import review
    from aidub.job import Job
    prof = profiles.Profiles(cfg.appdata).get_or_create("Diner")
    master, miyuki, extra = prof.new_character("Master", gender="male"), prof.new_character("Miyuki"), prof.new_character("X")
    job = Job(cfg, "rv1", tmp_path / "ep.mkv", prof)
    job.set("ingest", {"duration": 100.0})
    lines = [L.Line(1, 10, 12, kind=L.SONG, character=miyuki, kept_why="held notes over music (0.55), no subtitle"),
             L.Line(2, 20, 22, character=miyuki, source_text="え？何が"),
             L.Line(3, 30, 32, character=extra),
             L.Line(4, 40, 42, character=master),
             L.Line(5, 50, 52, character=master)]
    prof.merge(master, extra)          # on the Voices tab: X is Master
    d = {"regions": [{"start": 9.5, "end": 12.5, "action": "dub", "kind": "song", "source": "ai"},
                     {"start": 49.9, "end": 52.1, "action": "keep", "kind": "song", "source": "user"}],
         "cues": {}, "lines": {"2": {"character": master}, "4": {"character": "new:Waiter", "gender": "male"}}}
    n = review.apply(job, lines, d)
    assert n == 4
    assert lines[0].kind == L.SPEECH and lines[0].decided == "dub"         # "Dub it" on a kept region
    assert lines[1].character == master and lines[1].voice_note.startswith("own line")  # Master has no clips yet
    assert lines[2].character == master                                   # merged speaker
    waiter = next(c for c, ch in prof.characters.items() if ch["name"] == "Waiter")
    assert lines[3].character == waiter and prof.characters[waiter]["gender"] == "male"
    assert lines[4].kind == L.SONG and lines[4].decided == "keep" and lines[4].kept_why == "kept by you"
    from aidub import align
    lines[0].sung, lines[0].music_db = 0.6, -3
    assert not align.sung_candidate(lines[0])                              # the lyrics check can't overrule it
    regs = review.regions(lines, 100.0)
    assert [(r["start"], r["kind"]) for r in regs] == [(50, "song")]
    # Reasons are grouped without each line's own score.
    many = [L.Line(10 + i, 60 + 2 * i, 61.5 + 2 * i, kind=L.SONG, kept_why=f"held notes over music (0.5{i}), no subtitle")
            for i in range(4)]
    assert review.regions(many, 100.0)[0]["reasons"] == ["held notes over music, no subtitle (4 lines)"]


def test_tuned_from_the_s01e05_review():
    from aidub import translate as T
    from aidub.subtitles.model import Cue

    class J(dict):
        qc = {}
        def log(self, m): pass
    job = J(subtitles={"source_language": "ja"})

    def cue(a, b, text):
        return Cue(start=a, end=b, text=text, raw=text, kind=DIALOGUE)
    # The guitarist: quoted lyrics, sung slowly over music.
    song = [L.Line(id=i, start=a, end=a + 6, source_text=t, music_db=-7)
            for i, (a, t) in enumerate([(465, "ハルバル北ぜ箱だて"), (474, "さあかまく波乗り越えて"), (482, "思い出すたびに")], 1)]
    cues = [cue(465, 471, '"It was a long and winding road to Hakodate."'), cue(474, 479, '"Past the storming waves I went."'),
            cue(482, 488, '"As you sent me away, you hid the tears."')]
    assert T.quoted_songs(job, song, cues) == 3 and all(l.kind == L.SONG for l in song)
    talk = [L.Line(id=i, start=a, end=a + 2, source_text="これは手紙に書いてあったことなんですけどね", music_db=-30)
            for i, a in enumerate((10, 14, 18), 1)]           # someone reading quotes aloud: spoken, no music
    assert T.quoted_songs(job, talk, [cue(10, 12, '"Dear John,"'), cue(14, 16, '"I miss you."'), cue(18, 20, '"Love, Mary."')]) == 0
    # Short subtitled replies kept as sounds are dubbed; pure exclamations and captions are not.
    yes = L.Line(id=1, start=100, end=101.3, source_text="はい", kind=L.WORDLESS, music_db=-12)
    ohh = L.Line(id=2, start=110, end=110.3, source_text="ほう", kind=L.WORDLESS, music_db=-28)
    tv = L.Line(id=3, start=120, end=121.2, source_text="はい", kind=L.BACKGROUND, music_db=110)
    edge = L.Line(id=4, start=130, end=130.2, source_text="恋", kind=L.WORDLESS, music_db=8)
    sung = L.Line(id=5, start=131, end=140, kind=L.SONG)
    ls = [yes, ohh, tv, edge, sung]
    n = T.subtitled_reactions(ls, [cue(100, 101.5, "Yes."), cue(110, 110.5, "Ohh!"), cue(120, 121.5, "Is gang warfare back?"),
                                   cue(130, 135, "From the lips of a hostess")])
    assert n == 1 and yes.kind == L.SPEECH and ohh.kind == L.WORDLESS and tv.kind == L.BACKGROUND and edge.kind == L.WORDLESS


def test_untagged_english_audio_is_skipped():
    from aidub import recognize as R
    assert R.already_english("en", None, {}, True)
    assert not R.already_english("ja", None, {}, True)
    assert not R.already_english("en", "ja", {}, True)                     # the track said Japanese
    assert not R.already_english("en", None, {"language": "ja"}, True)     # the user chose the language
    assert not R.already_english("en", None, {"redo": True}, True)
    assert not R.already_english("en", None, {}, False)                    # skipping English audio is off


def test_backcheck_knows_the_shows_names():
    from types import SimpleNamespace
    from aidub import speak
    prof = SimpleNamespace(glossary={"ネロ": {"en": "Nello", "type": "name"}, "アロア": {"en": "Alois", "type": "name"},
                                     "ミシェル": {"en": "Michel", "type": "name"}, "犬": {"en": "dog", "type": "term"}},
                           characters={"c1": {"name": "Character 1"}, "c2": {"name": "Grandpa Jehan"}})
    names = speak.show_names(prof)
    assert {"nello", "alois", "michel", "grandpa", "jehan"} <= names and "dog" not in names and "character" not in names
    # Dog of Flanders E02: names heard with other spellings, at the start of a line or alone.
    assert speak.missed_words("Alois, are you all right", "Alwa, are you all right?", names) == []
    assert speak.wer("Alois, are you all right", "Alwa, are you all right?", names) == 0
    assert speak.wer("Nello, goodbye", "Hello, goodbye.", names) == 0
    assert speak.wer("Alois", "Alua.", names) == 0 and speak.wer("Nello", "Hello.", names) == 0
    assert speak.wer("Nello", "Ahhhh!", names) == 1.0                  # a scream is not the name
    assert speak.wer("Alois", "Oh yeah, all of us.", names) == 1.0     # nor a whole other line
    assert speak.missed_words("Alois, are you all right", "Alwa, are you all right?") == ["alois"]  # without the names
    # Calls and sounds.
    assert speak.missed_words("Ouch", "Uh.") == [] and speak.missed_words("Yo ho~", "Yeah.") == []


def test_backcheck_two_words_heard_as_one():
    from aidub import speak
    # Dog of Flanders E02 7:41 stayed Japanese: "some time" heard as "sometime".
    assert speak.missed_words("The flowers turn into fruits some time", "The flowers turn into fruits sometimes.") == []
    assert speak.wer("The flowers turn into fruits some time", "The flowers turn into fruit sometime.") <= 0.15
    assert speak.missed_words("Any way you like", "Anyway you like.") == []


def test_sounds_and_fillers_said_over_and_over():
    from aidub.align import is_interjection
    for t in ("ね。", "うんうんうんうん", "はいはいはい", "パンパンパンパンパンパンパンパンパンパン", "ええ。"):
        assert is_interjection(t, "ja"), t
    for t in ("カットカット", "わくわく", "ようやく見つけたぜ", "やはり", "皆さん大丈夫"):   # "Cut, cut!" and real words are speech
        assert not is_interjection(t, "ja"), t


def test_track_filter_matches():
    from aidub.tracks import matches
    ja = {"audio": [{"lang": "ja", "aidub": False, "commentary": False}]}
    dubbed = {"audio": ja["audio"] + [{"lang": "en", "aidub": True, "commentary": False}]}
    with_en = {"audio": ja["audio"] + [{"lang": "en", "aidub": False, "commentary": False}]}
    en_comment = {"audio": ja["audio"] + [{"lang": "en", "aidub": False, "commentary": True}]}
    untagged = {"audio": [{"lang": "", "aidub": False, "commentary": False}]}
    assert matches(ja, {"ja"}, {"en"}) and not matches(ja, {"ko"}, set())
    assert not matches(dubbed, {"ja"}, {"en"})     # an Ai Dub track counts as having English audio
    assert not matches(dubbed, {"en"}, set())      # but not as a title in English to begin with
    assert not matches(with_en, {"ja"}, {"en"})
    assert matches(en_comment, set(), {"en"})      # nor does an English commentary
    assert matches(untagged, {"und"}, set()) and not matches(untagged, {"ja"}, set())
    assert matches({"error": "x"}, set(), {"en"}) and not matches({"error": "x"}, {"ja"}, set())


def test_tracks_read_in_background_and_cached(tmp_path):
    import synth
    from aidub.tracks import Tracks, videos_in
    show = tmp_path / "Show" / "Season 1"
    show.mkdir(parents=True)
    ep = synth.episode(show / "Show - S01E01.mkv")
    assert videos_in(tmp_path / "Show") == [ep]  # a show's seasons
    t = Tracks(tmp_path)
    first, left = t.lookup([ep], wait=0)          # returns at once; read in the background
    assert left in (0, 1) and (left == 1 or first[str(ep)]["audio"][0]["lang"] == "ja")
    info, left = t.lookup([ep])                   # waits until read
    assert left == 0 and info[str(ep)]["audio"][0]["lang"] == "ja"
    again, left = Tracks(tmp_path).lookup([ep], wait=0)   # from tracks.json, no reading
    assert left == 0 and again[str(ep)]["audio"][0]["lang"] == "ja"


def test_long_audio_keeps_its_end(tmp_path, monkeypatch):
    import soundfile as sf
    from aidub import media
    data = np.zeros((1000, 2), np.float32)
    assert sf.info(str(media.write_wav(tmp_path / "plain.wav", data, 48000))).format == "WAV"
    monkeypatch.setattr(media, "WAV_LIMIT", 1000)
    big = media.write_wav(tmp_path / "big.wav", data, 48000)
    assert sf.info(str(big)).format == "RF64"
    assert media.read_wav(big)[0].shape == (1000, 2)
    seen = []
    monkeypatch.setattr(media, "run", seen.append)
    media.extract_audio(tmp_path / "film.mkv", 1, tmp_path / "out.wav")
    assert seen[0][seen[0].index("-rf64") + 1] == "auto"


def test_read_wav_refuses_a_file_cut_short(tmp_path):
    from aidub import media
    path = media.write_wav(tmp_path / "cut.wav", np.zeros((1000, 2), np.float32), 48000)
    assert len(media.read_wav(path)[0]) == 1000
    with open(path, "ab") as f:  # what a WAV past 4 GiB looks like: more audio than its header counts
        f.write(b"\0" * (2 << 20))
    with pytest.raises(ValueError, match="cut short"):
        media.read_wav(path)


def test_relisten_skips_lines_past_the_audio(tmp_path):
    import soundfile as sf
    from types import SimpleNamespace
    from aidub import align, media
    media.write_wav(tmp_path / "dialogue16k.wav", np.zeros((16000 * 10, 1), np.float32), 16000)
    sent = []

    class Worker:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def call(self, op, args):
            if op != "transcribe":
                return {}
            sent.extend(sf.info(it["path"]).frames for it in args["items"])
            return {"results": [{"text": ""} for _ in args["items"]]}

    job = SimpleNamespace(cfg=SimpleNamespace(get=lambda k, d=None: {"stages.recognize.worker": "asr"}.get(k, {})),
                          path=lambda *p: tmp_path.joinpath(*p[1:]), worker=lambda name: Worker(), qc={},
                          log=lambda m: None)
    inside, past = L.Line(1, 2.0, 3.0), L.Line(2, 20.0, 21.0)
    align.relisten(job, [inside, past], "ja")
    assert len(sent) == 1 and sent[0] > 0
    assert align.relisten(job, [past], "ja") == {}
    assert len(sent) == 1


def test_voice_text_gets_past_its_normaliser():
    from aidub import speak
    assert speak.spoken("No.") == "No ." and speak.spoken("No... No.") == "No ... No ."
    assert speak.spoken("It's been fun.") == "It's been fun." and speak.spoken("Oh no.") == "Oh no."
    assert speak.spoken("Room No. 12.") == "Room Number 12."
    assert speak.spoken("Sat down. Unlike you, Gen. You must see TV.") == "sat down. Unlike you, Gen . You must see T V."
    assert speak.spoken("Gen. Yamada.") == "Gen. Yamada."
    assert speak.spoken("73,000.") == "Seventy three thousand." and speak.spoken("1980.") == "Nineteen eighty."
    assert speak.spoken("7:00.") == "7 o'clock." and speak.spoken("7:05") == "Seven oh five"
    assert speak.wer("7:00.", "Seven o'clock.") == 0
    assert speak.spoken("¥1,224,560 Yen.") == "1224560 yen." and speak.spoken("Those are 25cm guns.") == "Those are 25 centimeters guns."


def test_backcheck_hears_units_and_large_numbers():
    from aidub import speak
    assert speak.wer("Those are 25cm guns.", "Those are twenty five centimeters guns.") == 0
    assert speak.wer("¥1,224,560 Yen.", "One million two hundred twenty four thousand five hundred and sixty yen.") == 0
    assert speak.missed_words("You need 40cm.", "You need forty centimeters.") == []


def test_backcheck_counts_negations():
    from aidub import speak
    assert speak.missed_words("Men aren't dogs.", "Men are dogs.") == ["not"]
    assert speak.missed_words("You can go.", "You can't go.") == ["not"]
    assert speak.missed_words("I did not kill him.", "I did kill him.") == ["not"]
    assert speak.missed_words("No. Just touch it.", "Number, just touch it.")
    assert speak.missed_words("No.", "Oh.") == ["not"] or speak.missed_words("No.", "Oh.")
    assert speak.missed_words("I know.", "I no.") == [] and speak.missed_words("Don't go.", "Do not go.") == []
    assert speak.missed_words("Nah, it's fine.", "No, it's fine.") == []


def test_take_with_every_word_heard_is_used_despite_a_doubtful_voice():
    from aidub import speak
    from types import SimpleNamespace
    cfg = SimpleNamespace(get=lambda k, d=None: d)
    clear = {"wer": 0.0, "missed": [], "words": 3, "sim": -0.01, "sim_median": 0.3, "seconds": 2.0, "own_ref": False}
    assert not speak._passes(clear, cfg) and speak._rescue(clear, cfg)
    assert speak._rescue({**clear, "wer": 0.3}, cfg) and not speak._rescue({**clear, "wer": 0.4}, cfg)


def test_a_reply_dash_is_a_second_speaker():
    from aidub.subtitles import clean
    assert clean.speaker_turns("Who? -Dad. Repeat.") == ["Who?", "Dad. Repeat."]
    assert clean.speaker_turns("Who are you?\n-I'm Mio.") == ["Who are you?", "I'm Mio."]
    assert clean.speaker_turns("What's that for? - The sand seeps in.") == ["What's that for?", "The sand seeps in."]
    assert clean.speaker_turns("OK.\u3000-OK.") == ["OK.", "OK."]
    assert clean.speaker_turns("Special - like hell.") == [] and clean.speaker_turns("A well-made bowl.") == []


def test_ocr_capital_i():
    from aidub.subtitles import clean
    assert clean.clean_text("ln the end, l'm here. lt is. All well.")[0] == "In the end, I'm here. It is. All well."


def test_listen_back_finds_a_silent_stretch():
    from aidub import readback as R
    original = np.full(4000, -30.0)
    assert R.silent_runs(np.full(4000, -30.0), original) == []
    # A film whose last hour was lost:
    dub = np.concatenate([np.full(3728, -30.0), np.full(272, -120.0)])
    assert R.silent_runs(dub, original) == [(3728, 4000)]
    assert R.silent_runs(np.full(3728, -30.0), original) == [(3728, 4000)]  # a track that simply stops
    # The original goes quiet too: a pause, not a fault.
    quiet = original.copy(); quiet[100:400] = -90.0
    pause = np.full(4000, -30.0); pause[100:400] = -120.0
    assert R.silent_runs(pause, quiet) == []
    assert R.clock(3728) == "1:02:08" and R.clock(75) == "1:15"


def test_listen_back_hard_to_hear():
    from aidub import readback as R
    assert R.hard_to_hear("Get out of here.", "", [])
    assert R.hard_to_hear("Watch out.", "Watch.", ["out"])
    assert not R.hard_to_hear("Leave his hat on the table.", "Leave his hut on the table.", ["hat"])
    assert R.hard_to_hear("I want fatty pork in mine.", "I want ninety.", ["fatty", "pork", "mine"])
    ln = L.Line(1, 10.0, 12.0, fit={"placed": 10.5, "final": 2.6, "paced_tempo": 1.3})
    assert R.placed(ln) == (10.5, 12.5)


def _said(id, start, end, en, **kw):
    return L.Line(id=id, start=start, end=end, en=en, kind=L.SPEECH, **kw)


def test_one_sentence_across_two_lines_is_one_take():
    from types import SimpleNamespace
    from aidub import translate as T
    job = SimpleNamespace(qc={}, log=lambda m: None)
    lines = [_said(1, 0.0, 1.0, "They're the same", character="c1", source_text="A"),
             _said(2, 1.3, 2.0, "as always.", character="c2", source_text="B"),  # lower case: the rest of it, whoever
             _said(3, 2.2, 3.0, "Ask him.", character="c1"), _said(4, 3.1, 4.0, "He knows.", character="c1"),  # ended
             _said(5, 9.0, 10.0, "Thank", character="c1"), _said(6, 10.2, 11.0, "you.", character="c1"),
             _said(7, 20.0, 21.0, "About hunting boars in...", character="c1"), _said(8, 21.1, 22.0, "winter.", character="c1"),
             _said(9, 30.0, 31.0, "Then I", character="c1"), _said(10, 35.0, 36.0, "left.", character="c1"),  # too far apart
             _said(11, 40.0, 41.0, "somehow laugh-", character="c1"), _said(12, 41.2, 42.0, "ing.", character="c1")]
    got = T.join_sentences(job, lines)
    said = [l.en for l in got]
    assert "They're the same as always." in said and "Thank you." in said and "Ask him." in said and "He knows." in said
    assert "winter." in said and "left." in said and "somehow laughing." in said
    whole = got[0]
    assert (whole.start, whole.end, whole.source_text) == (0.0, 2.0, "A B") and job.qc["sentences_joined"] == 3


def test_two_sentences_two_speakers_get_two_voices():
    from aidub import translate as T
    class Job(dict):
        qc = {}
        def log(self, m): pass
    turns = [{"start": 0.0, "end": 2.0, "speaker": "S1"}, {"start": 2.0, "end": 3.0, "speaker": "S2"}]
    job = Job(diarization={"segments": turns}, characters={"S1": "c1", "S2": "c2"})
    lines = [_said(1, 0.0, 3.0, "You hid it from me. Sorry.", character="c1", voice="a.wav"),
             _said(2, 5.0, 6.0, "Hello.", character="c2", voice="b.wav")]
    got = T.split_two_voices(job, lines)
    assert [(l.en, l.character, l.voice) for l in got] == [("You hid it from me.", "c1", "a.wav"), ("Sorry.", "c2", "b.wav"), ("Hello.", "c2", "b.wav")]
    assert got[1].id == 3 and got[0].end == 2.0 and got[1].start == 2.0
    # One sentence, or words that don't fit the change: left whole.
    assert len(T.split_two_voices(job, [_said(1, 0.0, 3.0, "You hid it from me, sorry.", character="c1")])) == 1
    assert len(T.split_two_voices(job, [_said(1, 0.0, 3.0, "No. You hid it from me and then you lied to me.", character="c1")])) == 1


def test_unsubtitled_scraps_keep_the_original():
    from types import SimpleNamespace
    from aidub import align as A
    q = SimpleNamespace(start=0.0, end=2.0, text="Hello there.", kind=DIALOGUE)
    lines = [L.Line(1, 0.0, 2.0, source_text="こんにちは", music_db=0.0),      # subtitled
             L.Line(2, 5.0, 6.0, source_text="どうも", music_db=-3.0),         # scrap under music
             L.Line(3, 8.0, 9.0, source_text="いらっしゃいませ", music_db=-30.0),   # clear, no music: translated and flagged
             L.Line(4, 10.0, 13.0, source_text="はははははは", music_db=-30.0)]  # a laugh said over and over
    assert A.unsubtitled_scraps(lines, [q], "ja") == 2
    assert [l.kind for l in lines] == [L.SPEECH, L.BACKGROUND, L.SPEECH, L.BACKGROUND]
    assert A.unsubtitled_scraps(lines, [], "ja") == 0


def test_corrections_compare_by_what_gets_dubbed(tmp_path):
    from aidub import corrections as C
    row = lambda i, a, b, k: {"id": i, "start": a, "end": b, "kind": k, "source_text": "x", "english_ref": ""}
    saved = {"name": "a", "automatic": [row(1, 0, 1, "song"), row(2, 2, 3, "speech"), row(3, 4, 5, "speech")],
             "final": [row(1, 0, 1, "speech"), row(2, 2, 3, "song"), row(3, 4, 5, "speech")]}
    c = C.compare(saved, saved["automatic"])
    assert (c["wrong_before"], c["wrong_now"], len(c["should_dub"]), len(c["should_keep"])) == (2, 2, 1, 1)
    assert C.compare(saved, saved["final"])["wrong_now"] == 0
    # A file is kept per job, with what the review changed.
    work = tmp_path / "job-7"
    (work / "review").mkdir(parents=True)
    (work / "review" / "base-lines.json").write_text(__import__("json").dumps(saved["automatic"]))
    (work / "lines.json").write_text(__import__("json").dumps(saved["final"]))
    C.save(tmp_path, "Film", 7, work, {"review_changes": [{"was": "song"}]})
    assert C.summary(tmp_path) == [{"job": 7, "name": "Film", "lines": 3, "changed": 2, "kept_to_dubbed": 1, "dubbed_to_kept": 1}]


def test_check_list_leaves_nothing_out_and_puts_problems_first():
    from aidub import verify as V
    say = lambda i, a, en, **kw: L.Line(id=i, start=a, end=a + 2, en=en, kind=L.SPEECH, take="t.wav", **kw)
    lines = [L.Line(id=1, start=0, end=30, kind=L.SONG, kept_why="lyric"),
             say(2, 40, "Hello.", check_note="the subtitle says something different from the speech"),
             say(3, 50, "Lost.", fit={"mix_missed": ["lost"], "mix_heard": ""}),
             L.Line(id=4, start=60, end=61, en="No.", kind=L.SPEECH, fallback="every take failed"),
             say(5, 70, "Fast.", fit={"paced_tempo": 1.3, "placed": 70.8})]
    got = V.moments(lines, 100.0)
    by = {}
    for m in got:
        by.setdefault(m["group"], []).append(m["kind"])
    assert by["songs"] == ["song"] and got[0]["end"] == 30
    assert by["check"] == ["masked", "kept", "late"] and by["notes"] == ["note", "fast"]


def test_faces_name_characters_only_when_their_lines_agree():
    from aidub import faces as F
    a, b = [1.0] + [0.0] * 9, [0.0, 1.0] + [0.0] * 8
    talking = lambda p: {"faces": [{"score": 2.0, "print": p}]}
    lines = [L.Line(i, i, i + 1, character="c1") for i in range(1, 6)] + [L.Line(6, 6, 7, character="c2")]
    rows = {i: talking(a) for i in range(1, 7)}
    got = F.characters_by_face(lines, rows)
    assert got[6] == "c1" and got[1] == "c1"  # five lines of one face name it; the sixth goes with them
    # Two faces talking, or none: the picture says nothing.
    both = {"faces": [{"score": 2.0, "print": a}, {"score": 1.8, "print": b}]}
    assert F.speaking_face(both) is None and F.speaking_face({"faces": [{"score": -1.0, "print": a}]}) is None
    # Too few lines, or lines that disagree: no name.
    assert F.characters_by_face(lines[:2], {1: talking(a), 2: talking(a)}) == {}
    mixed = [L.Line(i, i, i + 1, character="c1" if i % 2 else "c2") for i in range(1, 7)]
    assert F.characters_by_face(mixed, {i: talking(a) for i in range(1, 7)}) == {}
    assert F.cluster([a, a, b]) == [0, 0, 1]


def test_audit_finds_what_was_never_heard():
    from types import SimpleNamespace
    from aidub import audit
    cue = lambda t, x: SimpleNamespace(start=t, end=t + 1, text=x)
    cues = [cue(0, "Good morning everyone."), cue(10, "The river is very high and the bridge is out today."), cue(20, "Go home now.")]
    heard = audit._words("good morning everyone go home now")
    share, missing = audit.missing_stretches(cues, heard)
    assert share < 0.5 and len(missing) == 1 and missing[0]["t"] == "0:10"
    assert audit.missing_stretches(cues[:1], audit._words("Good morning, everyone."))[0] == 1.0
    data = np.zeros(16000 * 40, np.float32)
    data[:16000 * 5] = 0.2
    assert audit.silent_stretches(data, 16000) == [[5, 40]]


def test_one_face_two_characters_is_said_not_acted_on():
    from aidub import faces as F
    a = [1.0] + [0.0] * 9
    lines = [L.Line(i, i, i + 1, character="c1" if i <= 6 else "c2") for i in range(1, 13)]
    rows = {i: {"faces": [{"score": 2.0, "print": a}]} for i in range(1, 13)}
    assert F.split_people(lines, rows) == [["c1", "c2"]]
    assert F.split_people(lines[:6], rows) == []


def test_backcheck_keeps_accented_words_whole():
    from aidub import speak
    assert speak._words("Consommé. Sole meunière.") == ["consomme", "sole", "meuniere"]
    assert speak.wer("Consommé, please.", "Consomme please.") == 0
    assert speak.missed_words("Sole meunière, please.", "Sole meuniere please.") == []


def test_other_voice_swaps_between_own_clip_and_usual_voice():
    from pathlib import Path
    from types import SimpleNamespace
    from aidub import review
    prof = SimpleNamespace(root=Path("/p"), refs=lambda c: [{"file": "a.flac"}] if c == "c1" else [])
    ln = L.Line(1, 0, 1, character="c1", voice="", voice_note="own line as reference")
    review.other_voice(prof, ln)
    assert ln.voice == "/p/a.flac" and ln.voice_note.endswith("(your choice)")
    review.other_voice(prof, ln)
    assert ln.voice == "" and ln.voice_note.endswith("(your choice)")
    nobody = L.Line(2, 0, 1, character="c9", voice="")
    review.other_voice(prof, nobody)
    assert nobody.voice == ""


def test_lines_are_voiced_from_their_own_clip_by_default(tmp_path, monkeypatch):
    from aidub import media, voicebank
    prof = profiles.Profiles(tmp_path).get_or_create("Spy")
    loid = prof.new_character("Loid", gender="male", age="adult")
    prof.characters[loid]["embedding"] = [1.0, 0.0, 0.0]
    prof.characters[loid]["refs"] = [{"file": "refs/loid/a.flac", "duration": 9.0, "score": 0.9}]
    prof.characters[loid]["clean_seconds"] = 60.0
    monkeypatch.setattr(voicebank, "generic_voices", lambda job: [])
    monkeypatch.setattr(prof, "clean_seconds", lambda c: 60.0)
    job = _StubJob({}, {})
    job.qc = {}
    job.path = lambda *parts: (tmp_path / "job").joinpath(*parts)
    (tmp_path / "job" / "audio").mkdir(parents=True)
    (tmp_path / "job" / "voices").mkdir(parents=True)
    media.write_wav(job.path("audio", "dialogue.wav"), np.zeros((48000 * 20, 1), np.float32), 48000)
    lines = [L.Line(1, 0.0, 2.0, character=loid), L.Line(2, 3.0, 3.3, character=loid),
             L.Line(3, 5.0, 7.0, character=loid, overlap=True)]
    voicebank.assign_voices(job, prof, lines, {})
    assert lines[0].voice == "" and lines[0].voice_note == "own line as reference"
    assert lines[1].voice.endswith("a.flac") and lines[2].voice.endswith("a.flac")  # too short; a second voice in it


def test_a_subtitle_in_another_language_becomes_an_english_track(monkeypatch):
    from aidub import translate
    from aidub.subtitles import translated as X
    from aidub.subtitles.model import Candidate, Cue
    asked = []

    class Fake:
        def __init__(self, job): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def chat_many(self, convs, max_tokens=0):
            asked.extend(convs)
            out = []
            for c in convs:
                ids = [int(i) for i in __import__("re").findall(r'"i": (\d+)', c[1]["content"])]
                out.append(__import__("json").dumps({"lines": [{"i": i, "en": f"line {i}"} for i in ids]}))
            return out
    monkeypatch.setattr(translate, "LLM", Fake)
    cues = [Cue(start=float(i), end=i + 0.9, text=f"せりふ{i}", raw="") for i in range(5)] + [Cue(start=9.0, end=10.0, text="", raw="♪", kind=SONG)]
    music = Cue(start=11.0, end=30.0, text="・～", raw="")
    job = type("J", (), {"profile": None, "check": lambda s: None, "status": lambda s, m: None, "log": lambda s, m: None})()
    guide = [Cue(start=0.0, end=0.9, text="A person's line.", raw="")]
    got = X.translate_cues(job, Candidate(id="embedded:2", label="", origin="embedded", language="ja", format="srt", path="", cues=cues + [music]), "Japanese", guide)
    assert [q.text for q in got[:5]] == [f"line {i}" for i in range(5)] and got[5].kind == SONG and got[1].start == 1.0
    assert len(got) == 6 and "・～" not in asked[0][1]["content"]   # music marks aren't lines
    assert "A person's line." in asked[0][1]["content"] and "guide" in asked[0][1]["content"]
    assert X.guide_for(cues[3], guide) == ""


def test_transcript_repair_stays_close_to_what_was_heard():
    from aidub import translate as T
    assert T._sounds_alike("電圧は瞬間二千万ボルト", "電圧は瞬間二千万ボルト。")
    assert not T._sounds_alike("はい", "東京の空は今日も青く晴れ渡っている")
    lines = [L.Line(1, 0, 1, source_text="お前がタイの肉までもらんなうき", text_origin="asr"),
             L.Line(2, 2, 3, source_text="はい", text_origin="asr"),
             L.Line(3, 4, 5, source_text="字幕", text_origin="subtitle:x")]

    class Llm:
        def chat_many(self, convs, max_tokens=0):
            assert "字幕" not in convs[0][1]["content"]
            return ['{"lines": [{"id": 1, "text": "お前が隊の肉までもらうなんて"}, {"id": 2, "text": "東京の空は今日も青く晴れ渡っている"}]}']
    job = type("J", (), {"qc": {}, "status": lambda s, m: None, "log": lambda s, m: None})()
    assert T.repair_transcript(job, Llm(), lines, "ja") == 1
    assert lines[0].source_text.endswith("なんて") and lines[0].subtitle_text.endswith("うき") and lines[1].source_text == "はい"


def test_translated_subtitles_say_names_as_heard_and_leave_no_characters(monkeypatch):
    import json as J
    import re as R
    from aidub import translate
    from aidub.subtitles import translated as X
    from aidub.subtitles.model import Candidate, Cue
    asked, rounds = [], []

    class Fake:
        def __init__(self, job): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def chat_many(self, convs, max_tokens=0):
            out = []
            for c in convs:
                asked.append(c[1]["content"])
                if "List every person" in c[1]["content"]:
                    out.append(J.dumps({"names": [{"subtitle": "角田", "english": "Tsunoda"}]}))
                    continue
                ids = [int(i) for i in R.findall(r'"i": (\d+)', c[1]["content"])]
                rounds.append(ids)
                out.append(J.dumps({"lines": [{"i": i, "en": "the 慰劳 trip" if len(rounds) == 1 else "Tsunoda left."} for i in ids]}))
            return out
    monkeypatch.setattr(translate, "LLM", Fake)
    cues = [Cue(start=0.0, end=1.0, text="角田走了", raw="")]
    job = type("J", (), {"profile": None, "check": lambda s: None, "status": lambda s, m: None, "log": lambda s, m: None})()
    cand = Candidate(id="embedded:2", label="", origin="embedded", language="zh", format="srt", path="", cues=cues)
    got = X.translate_cues(job, cand, "Chinese", None, [{"start": 0.0, "end": 1.0, "text": "ツノダさん"}])
    assert got[0].text == "Tsunoda left." and len(rounds) == 2  # the line with 慰劳 went round again
    assert "ツノダさん" in asked[0] and "Tsunoda" in asked[1]
    assert X.heard_between([{"start": 5, "end": 6, "text": "x"}], 0, 1) == ""


def test_name_list_covers_the_film_in_pieces_short_enough_for_the_model():
    import json as J
    from aidub.subtitles import translated as X
    from aidub.subtitles.model import Cue
    sizes = []

    class Llm:
        def chat_many(self, convs, max_tokens=0):
            sizes.extend(len(c[1]["content"]) for c in convs)
            return [J.dumps({"names": [{"subtitle": "角田", "english": "Tsunoda"}]}) for _ in convs[:-1]] + \
                [J.dumps({"names": [{"subtitle": "角田", "english": "Kakuta"}]})]
    talk = [(i, Cue(start=float(i), end=i + 1.0, text="如果角田也能挽回名声，事情就很难说了", raw="")) for i in range(1100)]
    got = X.name_list(Llm(), talk, [{"start": float(i), "end": i + 1.0, "text": "もし彼の名誉も取り戻せるなら"} for i in range(1100)], "Chinese")
    assert got == {"角田": "Tsunoda"} and len(sizes) == 10 and max(sizes) < 14000  # most common spelling wins


def test_names_come_from_the_english_subtitles_when_there_are_any():
    import json as J
    from aidub.subtitles import translated as X
    from aidub.subtitles.model import Cue
    guide = [Cue(start=0.0, end=1.0, text="Hey, Shoji is the new manager.", raw=""), Cue(start=2.0, end=3.0, text="hey you, Shoji.", raw="")]
    assert X.english_names(guide) == ["Shoji"]
    asked = []

    class Llm:
        def chat_many(self, convs, max_tokens=0):
            asked.extend(convs)
            return [J.dumps({"names": [{"subtitle": "東海林", "english": "Tokairin"}, {"subtitle": "東海林", "english": "Tokairin"},
                                       {"subtitle": "東海林", "english": "Shoji"}, {"subtitle": "西園寺", "english": "Saionji"},
                                       {"subtitle": "Shoji", "english": "Shoji"}]})]
    talk = [(0, Cue(start=0.0, end=1.0, text="東海林部長と西園寺", raw=""))]
    got = X.name_list(Llm(), talk, [], "Japanese", guide=guide)
    assert got == {"東海林": "Shoji", "西園寺": "Saionji"}  # backed by the subtitles beats most votes; nothing to back it: as voted
    talk = [(0, Cue(start=0.0, end=1.0, text="東海林", raw="")), (1, Cue(start=2.0, end=3.0, text="雅人さん", raw=""))]

    class Guess:
        def chat_many(self, convs, max_tokens=0):
            return [J.dumps({"names": [{"subtitle": "東海林", "english": "Shoji"}, {"subtitle": "雅人", "english": "Shoji"}]})]
    assert X.name_list(Guess(), talk, [], "Japanese", guide=[Cue(start=0.0, end=1.0, text="hey you, Shoji.", raw=""),
                                                            Cue(start=2.0, end=3.0, text="hey there.", raw="")]) == {"東海林": "Shoji"}
    assert '"english": "Hey, Shoji is the new manager."' in asked[0][1]["content"] and '["Shoji"]' in asked[0][1]["content"]


def test_lines_given_one_guide_do_not_say_it_twice():
    from aidub.subtitles import translated as X
    from aidub.subtitles.model import Cue
    guide = [Cue(start=0.0, end=4.0, text="Give your full attention to the ingredients.", raw=""),
             Cue(start=5.0, end=9.0, text="Good morning.", raw="")]
    talk = [(0, Cue(start=0.0, end=2.0, text="その材料を親しく見つめ", raw="")),
            (1, Cue(start=2.0, end=4.0, text="あつかわねばならぬ", raw="")),
            (2, Cue(start=5.0, end=6.0, text="おはようございます", raw="")),
            (3, Cue(start=6.0, end=7.0, text="おはようございます", raw="")),
            (4, Cue(start=7.0, end=8.0, text="今日は", raw="")),
            (5, Cue(start=8.0, end=9.0, text="明日は", raw=""))]
    english = {0: "Give your full attention to the ingredients.", 1: "to the ingredients.", 2: "Good morning to you all.",
               3: "Good morning to you all.", 4: "It is very cold today.", 5: "It will be sunny tomorrow."}
    span = X.merge_repeats(talk, english, guide)
    assert 1 not in english and span[0] == (0.0, 4.0)        # the fuller line stays and takes both lines' time
    assert 2 in english and 3 in english                     # the subtitle itself says it twice
    assert 4 in english and 5 in english and span[5] == (8.0, 9.0)
    talk = [(0, Cue(start=0.0, end=2.0, text="こちら", raw="")), (1, Cue(start=6.0, end=8.0, text="そちら", raw=""))]
    english = {0: "Good morning to you all.", 1: "Good morning to you all."}
    X.merge_repeats(talk, english, [Cue(start=0.0, end=2.0, text="a", raw=""), Cue(start=6.0, end=8.0, text="b", raw="")])
    assert len(english) == 2                                 # different guides: not this mistake
    talk = [(0, Cue(start=0.0, end=1.0, text="お前は間違ってる", raw="")), (1, Cue(start=1.2, end=2.0, text="何がだ", raw=""))]
    english = {0: "You're wrong.", 1: "You're wrong."}
    X.merge_repeats(talk, english, [Cue(start=0.0, end=2.0, text="You're wrong. How?", raw="")])
    assert list(english) == [0]                              # short, but word for word and the Japanese differs


def test_the_trim_keeps_a_last_consonant_and_cuts_the_noise_after_it():
    from aidub import fit as F
    sr, rnd = 24000, np.random.default_rng(1)
    t = np.arange(int(sr * 0.5)) / sr
    vowel = 0.3 * np.sin(2 * np.pi * 220 * t) + rnd.normal(0, 0.004, len(t))
    hiss = np.diff(rnd.normal(0, 0.012, int(sr * 0.12) + 1)).astype(np.float64)   # an "s", far below the vowel
    room = rnd.normal(0, 0.00005, int(sr * 0.5))
    a, b = F.speech_bounds(np.concatenate([vowel, hiss, room]).astype(np.float32), sr)
    assert b / sr >= 0.60     # vowel + the whole "s"
    assert b / sr <= 0.72     # and not the half second of room tone


def test_the_background_holds_steady_under_a_dubbed_line():
    from aidub import mix
    rnd = np.random.default_rng(3)
    n = mix.SR * 20
    noise = rnd.normal(0, 0.02, (n, 2)).astype(np.float32)
    dlg = noise.copy()
    dlg[mix.SR * 8: mix.SR * 12] += 0.3     # someone speaking
    lines = [L.Line(1, 8.0, 12.0)]
    g = mix.keep_gain(n, [(8.0, 12.0)], 0.03)
    room = mix.room_tone(dlg, g, mix.between_lines(n, lines))
    under = room[int(mix.SR * 8.5): int(mix.SR * 11.5)]
    assert abs(float(under.std()) / 0.02 - 1) < 0.1            # as loud as the room
    assert float(np.abs(under.mean(axis=0)).max()) < 0.01      # with none of the speech in it
    assert float(np.abs(room[: mix.SR * 7]).max()) == 0.0      # and nothing elsewhere
    silent = mix.room_tone(dlg, g, np.zeros(n, dtype=bool))    # nowhere quiet to take it from
    assert float(np.abs(silent).max()) == 0.0


def test_room_tone_takes_only_the_quietest_moments_not_voices_between_lines():
    from aidub import mix
    rnd = np.random.default_rng(7)
    n = mix.SR * 30
    hiss = rnd.normal(0, 0.002, (n, 1)).astype(np.float32)
    dlg = hiss.copy()
    for s in range(1, 29, 3):                       # someone talking in the "gaps", no line covers it
        dlg[mix.SR * s: mix.SR * s + mix.SR // 2] += rnd.normal(0, 0.05, (mix.SR // 2, 1)).astype(np.float32)
    lines = [L.Line(1, 14.6, 15.4)]
    g = mix.keep_gain(n, [(14.6, 15.4)], 0.03)
    room = mix.room_tone(dlg, g, mix.between_lines(n, lines))
    under = room[int(mix.SR * 14.7): int(mix.SR * 15.3)]
    assert float(under.std()) < 0.004               # the hiss, not the talk (0.05)


def test_surrounds_are_separated_only_when_the_speech_reaches_them():
    from aidub import separate
    rnd = np.random.default_rng(5)
    sr, n = 8000, 8000 * 120
    gate = (np.sin(np.arange(n) / sr * 2 * np.pi * 0.2) > 0).astype(np.float32)
    voice = rnd.normal(0, 0.1, n).astype(np.float32) * (gate + 0.001)
    music = rnd.normal(0, 0.02, (n, 2)).astype(np.float32)
    silent = np.zeros((n, 1), dtype=np.float32)
    dry = np.concatenate([music, voice[:, None], silent, music[:, ::-1]], axis=1)
    wet = dry.copy()
    wet[:, 4:] = voice[:, None] * 0.1
    assert not separate.speech_in_surrounds(dry, sr) and separate.speech_in_surrounds(wet, sr)
    assert not separate.speech_in_surrounds(dry[:, :2], sr)


def test_pieces_of_a_spoken_sentence_are_not_voiced_as_the_whole_of_it_twice():
    from aidub import translate as T
    job = type("J", (), {"qc": {}, "log": lambda s, m: None})()
    a = L.Line(1, 0.0, 1.0, source_text="ケンはそのことを", text_origin="asr", en="Ken wanted to explain that properly.", character="A")
    b = L.Line(2, 2.0, 3.0, source_text="きちんと説明したかった", text_origin="asr", en="He wanted to explain it properly.", character="A")
    c = L.Line(3, 3.5, 4.0, source_text="おはようございます", text_origin="asr", en="Good morning to you all.", character="A")
    d = L.Line(4, 4.5, 5.0, source_text="おはようございます", text_origin="asr", en="Good morning to you all.", character="A")
    e = L.Line(5, 5.5, 6.0, source_text="字幕です", text_origin="subtitle:x", en="Good morning to you all.", character="A")
    out = T.said_once(job, [a, b, c, d, e])
    assert [l.id for l in out] == [1, 3, 4, 5] and out[0].end == 3.0 and out[0].en.startswith("Ken wanted")
    assert "きちんと" in out[0].source_text and job.qc["repeats_merged"] == 1


def test_the_voice_filter_tames_harsh_moments_and_leaves_the_rest():
    from aidub import polish as P
    sr, rnd = 48000, np.random.default_rng(2)
    t = np.arange(sr * 2) / sr
    voice = 0.2 * np.sin(2 * np.pi * 200 * t) + 0.05 * np.sin(2 * np.pi * 600 * t)
    hiss = np.zeros(len(t))
    hiss[sr: sr + sr // 5] = np.diff(np.diff(rnd.normal(0, 0.1, sr // 5 + 2)))    # a burst of hiss in the second half
    x = (voice + hiss).astype(np.float32)
    out = P.polish(x, sr, P.LEVELS["strong"])
    assert len(out) == len(x) + int(sr * 0.25) - 1                     # the room's tail follows the line
    body = out[: len(x)]
    assert np.isfinite(out).all() and float(np.abs(out).max()) < 1.0

    def top(y):
        return float(np.sqrt((np.diff(np.diff(y)) ** 2).mean()))
    burst = slice(sr, sr + sr // 5)
    assert top(body[burst]) < top(x[burst]) * 0.8                      # the hiss is turned down
    calm = slice(sr // 4, sr // 2)
    assert abs(float(np.sqrt((body[calm] ** 2).mean())) / float(np.sqrt((x[calm] ** 2).mean())) - 1) < 0.25   # the voice isn't
    tail = out[len(x) + int(sr * 0.05):]
    assert 0 < float(np.abs(tail).max()) < float(np.abs(x).max()) * 0.3
    assert P.level_for("off") is None and P.level_for(False) is None and P.level_for("gentle") is P.LEVELS["gentle"]
    short = np.zeros(100, dtype=np.float32)
    assert np.isfinite(P.polish(short, sr, P.LEVELS["gentle"])).all()  # too short to do much with: no crash


def test_a_bad_bracket_in_a_reply_does_not_lose_a_batch_of_lines():
    from aidub.subtitles import translated as X
    from aidub.translate import parse_json
    forgot = '{\n"lines": [\n{"i": 8, "en": "Fine."},\n{"i": 9, "en": "Oh no, we upset her."]\n}'
    assert [x["i"] for x in parse_json(forgot)["lines"]] == [8, 9]
    assert parse_json('{"ok": ["a"]}') == {"ok": ["a"]}             # a good reply ending that way is left alone
    assert X._replies(forgot) == [(8, "Fine."), (9, "Oh no, we upset her.")]
    assert X._replies('{"i": 3, "en": "She said \\"no\\"."') == [(3, 'She said "no".')]


def test_lines_a_reply_missed_go_round_again_in_smaller_pieces(monkeypatch):
    import json as J
    from aidub import translate
    from aidub.subtitles import translated as X
    from aidub.subtitles.model import Candidate, Cue
    sizes = []

    class Fake:
        def __init__(self, job): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def chat_many(self, convs, max_tokens=0):
            out = []
            for c in convs:
                ids = [int(i) for i in __import__("re").findall(r'"i": (\d+)', c[1]["content"])]
                sizes.append(len(ids))
                out.append("not a reply" if len(ids) == X.BATCH else J.dumps({"lines": [{"i": i, "en": f"line {i}"} for i in ids]}))
            return out
    monkeypatch.setattr(translate, "LLM", Fake)
    logged = []
    job = type("J", (), {"profile": None, "check": lambda s: None, "status": lambda s, m: None, "log": lambda s, m: logged.append(m)})()
    cues = [Cue(start=float(i), end=i + 0.9, text=f"せりふ{i}", raw="") for i in range(X.BATCH)]
    got = X.translate_cues(job, Candidate(id="e", label="", origin="embedded", language="ja", format="srt", path="", cues=cues), "Japanese")
    assert len(got) == X.BATCH and sizes[0] == X.BATCH and max(sizes[1:]) == X.BATCH // 4
    assert not any("no translation" in m for m in logged)


def test_lines_keep_part_of_their_level_gaps():
    from aidub import mix
    starts = [0.0, 5.0, 10.0, 200.0]
    levels = [0.1, 0.1, 0.4, 0.8]   # the third is 12 dB over its scene; the last is alone in another
    half = mix.even_levels(starts, levels, 0.5)
    assert half[0] == pytest.approx(0.1) and half[1] == pytest.approx(0.1)
    assert half[2] == pytest.approx(0.2, rel=0.01)        # 6 dB over now: still the loud one
    assert half[3] == pytest.approx(0.8)                  # nothing near to compare with
    assert mix.even_levels(starts, levels, 1.0) == levels
    assert mix.even_levels([], [], 0.5) == []


def test_a_line_can_be_levelled_within_itself():
    from aidub import polish as P
    sr = 48000
    t = np.arange(sr * 2) / sr
    x = np.sin(2 * np.pi * 200 * t).astype(np.float32) * 0.05
    x[sr:] *= 8                                     # the second half 18 dB louder
    y = P.even_within(x, sr, 4.0)
    rise = lambda z: 20 * np.log10(np.sqrt((z[int(sr * 1.5):] ** 2).mean()) / np.sqrt((z[: int(sr * 0.8)] ** 2).mean()))
    assert rise(x) == pytest.approx(18, abs=0.5) and rise(y) < rise(x) - 6


def test_voice_again_uses_the_speaker_chosen_on_the_review(tmp_path):
    import json as J
    from aidub import review
    work = tmp_path / "job-1"
    (work / "lines").mkdir(parents=True)
    prof = profiles.Profiles(tmp_path / "app").get_or_create("Show")
    a, b = prof.new_character("Character 1"), prof.new_character("Man")
    L.save([L.Line(7, 1.0, 2.5, kind=L.SPEECH, en="Over there!", character=a, voice="/x/a.flac", take="/x/t.wav"),
            L.Line(8, 3.0, 4.5, kind=L.SPEECH, en="Hello.", character=a, take="/x/t2.wav")], work / "lines.json")
    (work / "state.json").write_text(J.dumps({"done": ["ingest", "speak", "mix"], "data": {}}))
    review.reset_for_lines(work, tmp_path / "film.mkv", {7: {}, 8: {}}, prof, {"7": {"character": b}, "8": {"character": a}})
    ln = {x.id: x for x in L.load(work / "lines.json")}
    assert ln[7].character == b and ln[7].take == ""      # the review's speaker, voiced again
    assert ln[8].character == a and ln[8].voice == ""     # already theirs: left as it was


def test_an_echoey_clip_is_not_lent_as_a_voice(tmp_path):
    import soundfile as sf
    from aidub import voicebank
    prof = profiles.Profiles(tmp_path).get_or_create("Show")
    rnd = np.random.default_rng(2)
    sr = 16000
    speech = np.zeros(sr * 3, dtype=np.float32)
    for k in range(0, 3 * sr, sr // 2):
        speech[k: k + sr // 4] = rnd.normal(0, 0.2, sr // 4)
    clean, echo = speech + rnd.normal(0, 0.001, len(speech)).astype(np.float32), speech + rnd.normal(0, 0.08, len(speech)).astype(np.float32)
    for name, x in (("clean", clean), ("echo", echo)):
        cid = prof.new_character(name)
        (prof.root / "refs" / cid).mkdir(parents=True)
        sf.write(str(prof.root / "refs" / cid / "a.flac"), x, sr)
        prof.characters[cid]["refs"] = [{"file": f"refs/{cid}/a.flac", "score": 1.0}]
    ids = list(prof.characters)
    assert voicebank.clear_ref(prof, ids[0]) and not voicebank.clear_ref(prof, ids[1])


def test_a_borrowed_voice_follows_how_the_speaker_sounds_not_a_guessed_gender(tmp_path):
    from aidub import voicebank
    prof = profiles.Profiles(tmp_path).get_or_create("Show")
    man_v, woman_v = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    man = prof.new_character("Editor", gender="male", age="adult")
    woman = prof.new_character("Typist", gender="female", age="adult")
    for cid, v in ((man, man_v), (woman, woman_v)):
        prof.characters[cid]["refs"] = [{"file": f"refs/{cid}/a.flac", "score": 1.0, "embedding": v, "duration": 9.0}]
        prof.characters[cid]["embedding"] = v
    tiny = prof.new_character("Character 9", gender="female")      # a few seconds, guessed wrong
    lines = [L.Line(1, 0, 0.3, kind=L.SPEECH, en="Over there!", character=tiny, speaker="S9", overlap=True, scene=5),
             L.Line(2, 1, 2.5, kind=L.SPEECH, en="Got it.", character=man, speaker="S1", scene=1),
             L.Line(3, 3, 4.5, kind=L.SPEECH, en="Yes.", character=woman, speaker="S2", scene=2),
             L.Line(4, 5, 6.5, kind=L.SPEECH, en="Right away.", character=tiny, speaker="S9", overlap=True, scene=5)]
    voices = {4: {"emb": [0.9, 0.1, 0.0]}}                        # its one longer line sounds like a man
    job = _StubJob({}, {})
    job.qc, job.cfg_get = {}, None
    job.cfg = type("C", (), {"get": lambda s, k, d=None: d})()
    job.path = lambda *a: tmp_path / "x.wav"
    import aidub.voicebank as VB
    VB.generic_voices, real = (lambda job: []), VB.generic_voices
    VB.clear_ref, real_clear = (lambda prof, cid: True), VB.clear_ref
    VB.own_voice_clips, real_own = (lambda *a, **k: {}), VB.own_voice_clips
    try:
        prof.characters[man]["clean_s"] = prof.characters[woman]["clean_s"] = 9.0
        prof.clean_seconds = lambda cid: 9.0 if cid in (man, woman) else 0.5
        voicebank.assign_voices(job, prof, lines, voices)
    finally:
        VB.generic_voices, VB.clear_ref, VB.own_voice_clips = real, real_clear, real_own
    assert lines[0].voice_note == "voice of Editor"                 # not the woman's, despite the guess
    assert job.get("voice_suggestions")[tiny][0][1] == "Editor"


def test_initials_are_said_letter_by_letter():
    from aidub import speak
    assert speak.spoken("The initials were G.I., right?") == "The initials were G I, right?"
    assert speak.spoken('Signed only "G.I."') == 'Signed only "G I"'
    assert speak.spoken("Mr. Tanaka.") == "Mr. Tanaka."


def test_backcheck_contraction_heard_as_the_word_it_sounds_like():
    from aidub import speak
    assert speak.wer("You're centurions.", "Your centurions.") == 0
    assert speak.wer("They're here.", "There here.") == 0
    assert speak.wer("Your centurions.", "You're centurions.") == 0
    assert speak.wer("Your house.", "Yes house.") > 0


def test_a_reviewed_section_keeps_its_lines_through_a_redub(tmp_path):
    from aidub import review
    work = tmp_path / "job-1"
    (work / "lines" / "00002").mkdir(parents=True)
    (work / "lines" / "00002" / "take_0.wav").write_bytes(b"x")
    L.save([L.Line(1, 1.0, 2.0, kind=L.SPEECH, en="Before."), L.Line(2, 10.0, 12.0, kind=L.SPEECH, en="Heard and liked.", take="/t.wav"),
            L.Line(3, 13.0, 14.0, kind=L.SPEECH, en="After.")], work / "lines.json")
    review.save_decisions(work, {"regions": [], "locked": [{"start": 9.0, "end": 12.5}, {"start": 3, "end": 3.2}]})
    review.save_decisions(work, {"regions": [], "cues": {"x": "keep"}})          # a later Save leaves the sections alone
    assert review.load_decisions(work)["locked"] == [{"start": 9.0, "end": 12.5, "at": pytest.approx(review.load_decisions(work)["locked"][0]["at"])}]
    assert review.keep_locked(work) == 1
    import shutil
    shutil.rmtree(work / "lines")                                                # what the reset does
    # The redub's new lines: line 2 translated again, and (below) a split line that was given number 2.
    new = [L.Line(1, 1.0, 2.0, kind=L.SPEECH, en="Before, again."), L.Line(2, 10.0, 12.0, kind=L.SPEECH, en="Something else."),
           L.Line(3, 13.0, 14.0, kind=L.SPEECH, en="After, again.")]
    free, kept = review.locked_part(work, new)
    assert [l.id for l in free] == [1, 3] and kept[0].en == "Heard and liked." and kept[0].take == "/t.wav"
    assert (work / "lines" / "00002" / "take_0.wav").exists()                    # its takes are back
    free.append(L.Line(2, 15.0, 16.0, kind=L.SPEECH, en="Split off."))
    out = review.join_locked(free, kept)
    assert [(l.id, l.en) for l in out] == [(1, "Before, again."), (2, "Heard and liked."), (3, "After, again."), (4, "Split off.")]


def test_a_line_in_a_reviewed_section_is_not_voiced_again(tmp_path):
    import json as J
    from aidub import review
    work = tmp_path / "job-1"
    (work / "lines").mkdir(parents=True)
    L.save([L.Line(7, 1.0, 2.5, kind=L.SPEECH, en="Locked.", take="/x/t.wav"),
            L.Line(8, 3.0, 4.5, kind=L.SPEECH, en="Free.", take="/x/t2.wav")], work / "lines.json")
    (work / "state.json").write_text(J.dumps({"done": ["ingest", "speak", "mix"], "data": {}}))
    review.save_decisions(work, {"locked": [{"start": 0.0, "end": 2.6}]})
    assert review.reset_for_lines(work, tmp_path / "film.mkv", {7: {"en": "Changed"}, 8: {}}, None) == [8]
    ln = {x.id: x for x in L.load(work / "lines.json")}
    assert ln[7].en == "Locked." and ln[7].take == "/x/t.wav" and ln[8].take == ""
