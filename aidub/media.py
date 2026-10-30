"""ffmpeg/ffprobe helpers and WAV input/output."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

ENGLISH = {"en", "eng", "english"}
VIDEO_EXT = {".mkv", ".mp4", ".m4v", ".avi", ".ts", ".m2ts", ".mov", ".wmv", ".webm"}
NOT_DIALOGUE_WORDS = ("commentary", "description", "descriptive", "audio description", "director")


class MediaError(RuntimeError):
    pass


def run(cmd: list[str], *, capture: bool = True, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=capture, text=True)
    if check and proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-12:]
        raise MediaError(f"{Path(cmd[0]).name} failed ({proc.returncode}): " + "\n".join(tail))
    return proc


def have(tool: str) -> bool:
    return shutil.which(tool) is not None


@dataclass
class Stream:
    index: int
    type: str
    codec: str
    language: str = ""
    title: str = ""
    channels: int = 0
    layout: str = ""
    sample_rate: int = 0
    duration: float = 0.0
    default: bool = False
    forced: bool = False
    comment: bool = False
    hearing_impaired: bool = False
    visual_impaired: bool = False
    tags: dict = field(default_factory=dict)

    @property
    def is_text_sub(self) -> bool:
        return self.type == "subtitle" and self.codec in {"subrip", "ass", "ssa", "webvtt", "mov_text", "text"}

    @property
    def is_bitmap_sub(self) -> bool:
        return self.type == "subtitle" and self.codec in {"hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle"}

    @property
    def is_aidub(self) -> bool:
        """A track Ai Dub added: version-stamp tags (MKV) or its title (MP4 has no custom tags)."""
        return any(k.upper().startswith("AIDUB_") for k in self.tags) or \
            any(t in self.title for t in ("(AI dub)", "(AI transcript)", "(AI translation)"))


@dataclass
class Probe:
    path: Path
    format: str
    duration: float
    streams: list[Stream]
    chapters: list[dict]
    raw: dict

    def of_type(self, kind: str) -> list[Stream]:
        return [s for s in self.streams if s.type == kind]

    @property
    def audio(self) -> list[Stream]:
        return self.of_type("audio")

    @property
    def subtitles(self) -> list[Stream]:
        return self.of_type("subtitle")

    @property
    def attachments(self) -> list[Stream]:
        return self.of_type("attachment")

    @property
    def container(self) -> str:
        names = self.format.split(",")
        if "matroska" in names:
            return "mkv"
        if "mp4" in names or "mov" in names:
            return "mp4"
        return names[0]


# Two-letter code: its ISO 639-2 code as Matroska writes it, and its English name.
LANGUAGES = {
    "en": ("eng", "english"), "ja": ("jpn", "japanese"), "ko": ("kor", "korean"), "zh": ("chi", "chinese"),
    "yue": ("chi", "cantonese"), "es": ("spa", "spanish"), "fr": ("fre", "french"), "de": ("ger", "german"),
    "ar": ("ara", "arabic"), "it": ("ita", "italian"), "pt": ("por", "portuguese"), "ru": ("rus", "russian"),
    "th": ("tha", "thai"), "vi": ("vie", "vietnamese"), "hi": ("hin", "hindi"), "id": ("ind", "indonesian"),
    "ms": ("may", "malay"), "tl": ("tgl", "tagalog"), "tr": ("tur", "turkish"), "pl": ("pol", "polish"),
    "nl": ("dut", "dutch"), "sv": ("swe", "swedish"), "no": ("nor", "norwegian"), "da": ("dan", "danish"),
    "fi": ("fin", "finnish"), "uk": ("ukr", "ukrainian"), "he": ("heb", "hebrew"), "el": ("gre", "greek"),
    "cs": ("cze", "czech"), "hu": ("hun", "hungarian"), "ro": ("rum", "romanian"),
}
_ALIASES = {"zho": "zh", "cmn": "zh", "yue": "yue", "fra": "fr", "deu": "de", "msa": "ms", "fil": "tl",
            "nld": "nl", "ell": "el", "ces": "cs", "ron": "ro", "nob": "no", "und": ""}
_TO_CODE = {**{a: c for c, (b, n) in LANGUAGES.items() for a in (b, n) if c != "yue" or a != "chi"}, **_ALIASES}


def lang_code(value: str) -> str:
    v = (value or "").strip().lower()
    return _TO_CODE.get(v, v)


def iso3(code: str) -> str:
    return LANGUAGES[code][0] if code in LANGUAGES else (code or "und")


def probe(path: str | Path) -> Probe:
    path = Path(path)
    proc = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format",
                "-show_streams", "-show_chapters", str(path)])
    data = json.loads(proc.stdout or "{}")
    streams = []
    for s in data.get("streams", []):
        tags = {k: v for k, v in (s.get("tags") or {}).items()}
        low = {k.lower(): v for k, v in tags.items()}
        disp = s.get("disposition") or {}
        dur = s.get("duration") or low.get("duration") or 0
        if isinstance(dur, str) and ":" in dur:
            h, m, sec = dur.split(":")
            dur = int(h) * 3600 + int(m) * 60 + float(sec)
        streams.append(Stream(
            index=int(s["index"]),
            type=s.get("codec_type", ""),
            codec=s.get("codec_name", "") or low.get("mimetype", ""),
            language=lang_code(low.get("language", "")),
            title=low.get("title", "") or low.get("handler_name", "") if s.get("codec_type") != "attachment" else low.get("filename", ""),
            channels=int(s.get("channels") or 0),
            layout=s.get("channel_layout", "") or "",
            sample_rate=int(s.get("sample_rate") or 0),
            duration=float(dur or 0),
            default=bool(disp.get("default")),
            forced=bool(disp.get("forced")),
            comment=bool(disp.get("comment")),
            hearing_impaired=bool(disp.get("hearing_impaired")),
            visual_impaired=bool(disp.get("visual_impaired")),
            tags=tags,
        ))
    fmt = data.get("format", {})
    return Probe(path=path, format=fmt.get("format_name", ""), duration=float(fmt.get("duration") or 0),
                 streams=streams, chapters=data.get("chapters", []), raw=data)


def is_dialogue_audio(s: Stream) -> bool:
    title = s.title.lower()
    if s.comment or s.visual_impaired:
        return False
    return not any(w in title for w in NOT_DIALOGUE_WORDS)


def extract_audio(src: Path, stream_index: int, dst: Path, *, rate: int = 48000, channels: int | None = None,
                  start: float | None = None, duration: float | None = None, pan: str | None = None) -> Path:
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(src)]
    if duration:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-map", f"0:{stream_index}", "-vn", "-sn", "-dn"]
    if pan:
        cmd += ["-af", pan]
    if channels:
        cmd += ["-ac", str(channels)]
    cmd += ["-ar", str(rate), "-c:a", "pcm_f32le", str(dst)]
    run(cmd)
    return dst


def read_wav(path: str | Path, *, mono: bool = False) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if mono:
        data = data.mean(axis=1, keepdims=True)
    return data, sr


def write_wav(path: str | Path, data: np.ndarray, sr: int, subtype: str = "FLOAT") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sr, subtype=subtype)
    return path


def resample(data: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return data
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(sr, target)
    return resample_poly(data, target // g, sr // g, axis=0).astype(np.float32)


def duration(path: str | Path) -> float:
    info = sf.info(str(path))
    return info.frames / float(info.samplerate)
