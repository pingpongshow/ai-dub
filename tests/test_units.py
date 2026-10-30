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
    assert [x.kind for x in lines] == [L.SONG, L.SPEECH, L.SONG, L.SONG, L.SONG, L.SPEECH]


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
    job = _StubJob({}, {})
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
        assert pronounce.only_sounds(speak._words("Mm-hmm.")) and pronounce.only_sounds(speak._words("Shh."))
        assert not pronounce.only_sounds(speak._words("Oh, no."))
    finally:
        pronounce.load(tmp_path / "none.dict")  # back to no dictionary for the other tests
    assert speak.missed_words("Brakes!", "Breaks.") == ["brakes"]  # without it, only exact spellings


def test_voice_model_reads_hyphenated_words_as_two():
    from aidub import speak
    assert speak.spoken("Build self-esteem, Ga-yeong.") == "Build self esteem, Ga yeong."
    assert speak.spoken("Ten-20 at 3-4.") == "Ten-20 at 3-4."  # only between letters


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


def test_japanese_names_spelled_as_they_sound():
    from aidub import pronounce as P
    assert P.respell("Kikoru") == "kee koh roo" and P.respell("chan") == "chahn"
    assert P.respell("Kaiju") == "kye joo" and P.respell("Shinomiya") == "shee noh mee yah"
    assert P.respell("Kafka") is None and P.respell("klicks") is None      # not romanised Japanese


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
    assert T.covered(a2, cues) and not T.covered(roger, cues)
    assert T.short_reaction(roger) and not T.short_reaction(speech)
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
    assert not ok and "content" in why
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
