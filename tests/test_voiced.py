import numpy as np

from aidub import lines as L
from aidub import voiced as V


class FakeJob:
    def __init__(self):
        self.qc, self.logs = {}, []

    def log(self, m):
        self.logs.append(m)


def _voice(spans, total=10.0, hop=0.01):
    on = np.zeros(int(total / hop), bool)
    for a, b in spans:
        on[int(a / hop): int(b / hop)] = True
    return V.Voice(on, hop)


def _line(i, a, b, text="", kind=L.SPEECH):
    return L.Line(id=i, start=a, end=b, source_text=text, kind=kind)


def test_how_long_a_line_takes_to_say():
    assert 0.8 < V.needed_s("何するんだい") < 1.0       # 7 morae
    assert V.needed_s("ね") < 0.5
    assert 0.3 < V.needed_s("What are you doing?") < 1.2


def test_a_line_is_muted_over_the_voice_that_runs_on_from_it():
    v = _voice([(1.0, 3.2)])
    # The aligner gave 1.6-2.5: the voice runs from 1.0 to 3.2.
    assert v.joined(1.6, 2.5, 0.0, 9.0) == (1.0, 3.2)
    # Never into the lines either side.
    assert v.joined(1.6, 2.5, 1.3, 2.9) == (1.3, 2.9)
    # Voice after a real pause is someone else's (or nothing to do with the line).
    v = _voice([(0.2, 0.6), (1.6, 2.5)])
    assert v.joined(1.6, 2.5, 0.0, 9.0) == (1.6, 2.5)


def test_a_line_timed_far_too_short_takes_its_times_from_the_voice():
    # Woman in the Dunes: 何するんだい at 3857.24-3857.56, said from 3855.92 to 3856.96.
    v = _voice([(5.92, 6.96)])
    lines = [_line(1, 2.0, 3.0, "はい"), _line(2, 7.24, 7.56, "何するんだい"), _line(3, 8.5, 9.0, "うん")]
    lines[1].words = [L.Word(text="何するんだい", start=7.24, end=7.56)]
    job = FakeJob()
    assert V.retime(job, lines, v) == 1
    assert (lines[1].start, lines[1].end) == (5.92, 7.56)
    assert lines[1].words == []  # misplaced: sync timing must not follow them
    assert job.qc["lines_retimed_by_voice"] == 1
    # Already right: left alone, and asking again changes nothing.
    assert V.retime(job, lines, v) == 0 and lines[0].start == 2.0


def test_retiming_never_reaches_back_into_the_line_before():
    v = _voice([(1.0, 3.0)])
    lines = [_line(1, 1.0, 2.0, "そうです", kind=L.WORDLESS), _line(2, 2.6, 2.7, "何するんだい")]
    V.retime(FakeJob(), lines, v)
    assert lines[1].start >= 2.0 + V.GAP_S - 1e-9


def test_a_line_far_shorter_than_the_voice_under_its_own_subtitle_takes_that_voice():
    # Woman in the Dunes: ワン (0.2 s) under "You idiot!", said from 7476.99 to 7479.33.
    v = _voice([(1.0, 3.3)])
    lines = [_line(1, 2.1, 2.3, "ワン"), _line(2, 5.0, 6.0, "はい")]
    job = FakeJob()
    assert V.retime(job, lines, v, cues=[(0.9, 3.5)]) == 1
    assert (lines[0].start, lines[0].end) == (1.0, 3.3)
    # A subtitle over two lines says nothing of either's voice.
    lines = [_line(1, 2.1, 2.3, "ワン"), _line(2, 3.0, 3.2, "うん")]
    assert V.retime(FakeJob(), lines, v, cues=[(0.9, 3.5)]) == 0


def test_a_shout_with_no_pitch_counts_as_voice():
    # バカバカしい at 53:54: loud, but the pitch tracker found no pitch in it.
    sr, hop = 16000, 0.01
    x = np.zeros(sr * 4, np.float32)
    x[int(1.0 * sr): int(1.6 * sr)] = 0.3 * np.random.default_rng(0).standard_normal(int(0.6 * sr))  # the shout
    x[int(1.6 * sr): int(2.4 * sr)] = 0.3 * np.sin(np.arange(int(0.8 * sr)) / sr * 2 * np.pi * 150)  # its line
    x[int(3.0 * sr): int(3.4 * sr)] = 0.003 * np.random.default_rng(1).standard_normal(int(0.4 * sr))  # quiet noise
    pitched = np.zeros(400, bool)
    pitched[160:240] = True
    v = V.Voice.from_signal(x, sr, pitched, hop, [_line(1, 1.6, 2.4, "バカバカしい")])
    assert v.joined(1.6, 2.4, 0.0, 9.0) == (1.0, 2.4)
    assert not v.runs(2.9, 3.5)  # quiet and unpitched: not voice
