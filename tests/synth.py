"""Synthetic test media: two "speakers" (different pitches) over a quiet bed,
a shot change half way, a chapter, a font attachment, and optional subtitles."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 48000
# (start, duration, pitch Hz): speaker A is low (male), speaker B high (female)
LINES = [(1.0, 2.2, 180), (4.0, 1.8, 300), (6.6, 2.5, 180), (10.0, 2.0, 300),
         (13.0, 2.4, 180), (21.0, 2.0, 300), (24.0, 2.6, 180), (28.0, 1.6, 300), (31.0, 2.2, 180)]
DURATION = 36.0


def audio(path: Path, english_lines: bool = False) -> Path:
    t = np.arange(int(DURATION * SR)) / SR
    rng = np.random.default_rng(1)
    bed = rng.normal(0, 0.004, len(t)).astype(np.float32)
    voice = np.zeros_like(t, dtype=np.float32)
    for start, dur, f in LINES:
        a, b = int(start * SR), int((start + dur) * SR)
        tt = t[a:b] - start
        env = np.minimum(1, np.minimum(tt, dur - tt) / 0.05)
        sig = 0.25 * np.sin(2 * np.pi * f * tt) + 0.08 * np.sin(2 * np.pi * 2 * f * tt)
        voice[a:b] += (sig * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * tt)) * env).astype(np.float32)
    stereo = np.stack([voice + bed, voice * 0.9 + bed], axis=1)
    sf.write(str(path), stereo, SR, subtype="PCM_16")
    return path


def srt(path: Path, offset: float = 0.0, text: str = "Hello there") -> Path:
    def ts(x):
        ms = int(round(x * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
    blocks = [f"{i}\n{ts(s + offset)} --> {ts(s + d + offset)}\n{text} {i}\n" for i, (s, d, _) in enumerate(LINES, 1)]
    path.write_text("\n".join(blocks), encoding="utf-8")
    return path


def episode(path: Path, *, container: str = "mkv", english_audio: bool = False) -> Path:
    work = path.parent / f".synth-{path.stem}"
    work.mkdir(parents=True, exist_ok=True)
    wav = audio(work / "a.wav")
    font = work / "font.ttf"
    font.write_bytes(b"\x00\x01\x00\x00fake-font-for-tests")
    chapters = work / "ch.txt"
    chapters.write_text(";FFMETADATA1\n[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=18000\ntitle=Part A\n"
                        "[CHAPTER]\nTIMEBASE=1/1000\nSTART=18000\nEND=36000\ntitle=Part B\n")
    vf = ("color=c=red:s=160x90:r=10:d=18[a];color=c=blue:s=160x90:r=10:d=18[b];[a][b]concat=n=2:v=1:a=0[v]")
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-filter_complex", vf, "-i", str(wav), "-i", str(chapters)]
    if english_audio:
        cmd += ["-i", str(wav)]
    cmd += ["-map", "[v]", "-map", "0:a"]
    if english_audio:
        cmd += ["-map", "2:a"]
    cmd += ["-map_metadata", "1", "-map_chapters", "1", "-c:v", "libx264" if container == "mp4" else "mpeg4",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-metadata:s:a:0", "language=jpn"]
    if english_audio:
        cmd += ["-metadata:s:a:1", "language=eng"]
    if container == "mkv":
        cmd += ["-attach", str(font), "-metadata:s:t", "mimetype=application/x-truetype-font"]
    cmd += ["-t", str(DURATION), str(path)]
    subprocess.run(cmd, check=True, capture_output=True)
    for f in work.iterdir():
        f.unlink()
    work.rmdir()
    return path
