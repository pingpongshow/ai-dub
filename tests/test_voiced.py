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
