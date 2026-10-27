"""Stand-in model workers for tests: same protocol, no models.

The synthetic test media uses a different pitch per speaker, so "voices" are
told apart by their dominant frequency. The fake voice model writes a tone at
the reference's pitch and records what it was asked to say next to the WAV,
which the fake recogniser reads back for the back-check.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from aidub_worker import read_mono, serve

SEC_PER_SYL = 0.26


def dominant_freq(x: np.ndarray, sr: int) -> float:
    if len(x) < 256:
        return 0.0
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1 / sr)
    band = (f > 80) & (f < 600)
    return float(f[band][np.argmax(spec[band])]) if band.any() else 0.0


def voiced(x: np.ndarray, sr: int, hop: float = 0.02, gap: float = 0.3) -> list[tuple[float, float]]:
    n = int(sr * hop)
    if len(x) < n:
        return []
    e = np.sqrt(np.mean(x[: len(x) // n * n].reshape(-1, n) ** 2, axis=1))
    on = e > max(0.01, e.max() * 0.15)
    out, start = [], None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append([start * hop, i * hop])
            start = None
    if start is not None:
        out.append([start * hop, len(on) * hop])
    merged = []
    for a, b in out:
        if merged and a - merged[-1][1] < gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if b - a >= 0.2]


def embedding(f: float) -> list[float]:
    bins = np.arange(32)
    v = np.exp(-0.5 * ((bins - f / 20.0) / 1.5) ** 2) + 1e-3
    return (v / np.linalg.norm(v)).tolist()


def syllables(text: str) -> int:
    return max(1, len(re.findall(r"[a-zA-Z]+", text)))


# --- ops per fake worker -------------------------------------------------------------
def separate_ops():
    def separate(a):
        data, sr = sf.read(a["input"], dtype="float32", always_2d=True)
        out = Path(a["out_dir"])
        out.mkdir(parents=True, exist_ok=True)
        # The test mix puts "music" in the right channel only below 100 Hz; keep it simple:
        sf.write(out / "dialogue.wav", data, sr, subtype="FLOAT")
        sf.write(out / "mne.wav", data * 0.05, sr, subtype="FLOAT")
        return {"vocals": str(out / "dialogue.wav"), "instrumental": str(out / "mne.wav")}
    return {"separate": separate}


def diarize_ops():
    def diarize(a):
        x = read_mono(a["input"], 16000)
        segs = []
        for s, e in voiced(x, 16000):
            f = dominant_freq(x[int(s * 16000): int(e * 16000)], 16000)
            segs.append({"start": s, "end": e, "speaker": "spk0" if f < 240 else "spk1"})
        return {"segments": segs}
    return {"diarize": diarize}


def speaker_ops():
    def embed(a):
        x = read_mono(a["input"], 16000)
        return {"embeddings": [embedding(dominant_freq(x[int(s * 16000): int(e * 16000)], 16000))
                               for s, e in a["segments"]]}

    def embed_files(a):
        return {"embeddings": [embedding(dominant_freq(read_mono(p, 16000), 16000)) for p in a["paths"]]}

    def gender_age(a):
        x = read_mono(a["input"], 16000)
        out = {}
        for spk in a["speakers"]:
            fs = [dominant_freq(x[int(s * 16000): int(e * 16000)], 16000) for s, e in spk["segments"]]
            f = float(np.median(fs)) if fs else 0
            out[spk["speaker"]] = {"gender": "female" if f >= 240 else "male", "age": "adult"}
        return {"speakers": out, "model": "fake"}

    def text_similarity(a):
        # Ai Dub sends true pairs followed by the same lines paired with speech from
        # elsewhere; a matching subtitle scores higher on the first half.
        n = len(a["pairs"]) // 2
        return {"similarity": [0.8] * n + [0.4] * (len(a["pairs"]) - n)}

    return {"info": lambda a: {"embedding_model": "fake-pitch"}, "embed": embed, "embed_files": embed_files,
            "gender_age": gender_age, "text_similarity": text_similarity}


def asr_ops():
    def transcribe(a):
        out = []
        for it in a["items"]:
            side = Path(it["path"] + ".json")
            if side.exists():
                out.append({"text": json.loads(side.read_text())["text"], "language": "en", "words": []})
                continue
            x = read_mono(it["path"], 16000)
            words = []
            for s, e in voiced(x, 16000):
                t = s
                while t < e - 0.05:
                    words.append({"text": "ことば", "start": round(t, 3), "end": round(min(e, t + 0.45), 3)})
                    t += 0.5
            out.append({"text": "".join(w["text"] for w in words), "language": "ja", "words": words})
        return {"results": out}

    def align(a):
        out = []
        for it in a["items"]:
            x = read_mono(it["path"], 16000)
            v = voiced(x, 16000)
            if not v:
                out.append({"words": []})
                continue
            s, e = v[0][0], v[-1][1]
            chars = [c for c in it["text"] if c.strip()]
            step = (e - s) / max(1, len(chars))
            out.append({"words": [{"text": c, "start": round(s + i * step, 3), "end": round(s + (i + 1) * step, 3)}
                                  for i, c in enumerate(chars)]})
        return {"results": out}

    return {"transcribe": transcribe, "align": align}


def events_ops():
    return {"tag": lambda a: {"results": [{"language": "ja", "emotion": "NEUTRAL", "events": [], "text": ""}
                                          for _ in a["segments"]]}}


def llm_ops():
    def chat(a):
        outs = []
        for conv in a["conversations"]:
            user = conv[-1]["content"]
            if '"notes"' in user:
                outs.append(json.dumps({"notes": []}))
                continue
            if '"terms"' in user:
                outs.append(json.dumps({"terms": [{"source": "東京", "en": "Tokyo", "type": "place"}]}))
                continue
            items = re.findall(r'"id":\s*(\d+).*?"target_syllables":\s*(\d+)', user, re.S)
            lines = [{"id": int(i), "en": " ".join(["word"] * int(n)),
                      "short": " ".join(["word"] * max(1, int(int(n) * 0.75)))} for i, n in items]
            outs.append("```json\n" + json.dumps({"lines": lines}) + "\n```")
        return {"outputs": outs}
    return {"chat": chat}


def tts_ops():
    def synthesize(a):
        done = []
        for it in a["items"]:
            ref = read_mono(it["speaker_ref"], 16000)
            f = dominant_freq(ref, 16000) or 200.0
            dur = syllables(it["text"]) * SEC_PER_SYL * float(it.get("duration_factor", 1.0))
            sr = 22050
            t = np.arange(int(dur * sr)) / sr
            y = 0.3 * np.sin(2 * np.pi * f * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))
            y = np.concatenate([np.zeros(int(0.05 * sr)), y, np.zeros(int(0.08 * sr))]).astype(np.float32)
            Path(it["out"]).parent.mkdir(parents=True, exist_ok=True)
            sf.write(it["out"], y, sr)
            Path(it["out"] + ".json").write_text(json.dumps({"text": it["text"]}))
            done.append(it["out"])
        return {"done": done}
    return {"load": lambda a: {"model": "fake-tts", "duration_control": True, "emotion_ref": True,
                               "emotion_vector": True, "sample_rate": 22050},
            "synthesize": synthesize}


def bwe_ops():
    def upsample(a):
        for it in a["items"]:
            x = read_mono(it["input"], 48000)
            sf.write(it["output"], x, 48000, subtype="FLOAT")
        return {"done": [it["output"] for it in a["items"]]}
    return {"upsample": upsample}


def ocr_ops():
    return {"ocr": lambda a: {"texts": ["テスト"] * len(a["images"])}}


OPS = {"separate": separate_ops, "diarize": diarize_ops, "speaker": speaker_ops, "asr": asr_ops,
       "events": events_ops, "llm": llm_ops, "tts-indextts": tts_ops, "bwe-flashsr": bwe_ops, "ocr": ocr_ops}


def run(name: str) -> None:
    ops = {"load": lambda a: {"model": f"fake-{name}"}, "doctor": lambda a: {"fake": True},
           "prefetch": lambda a: {}}
    ops.update(OPS[name]())
    serve(ops)


if __name__ == "__main__":
    run(sys.argv[1])
