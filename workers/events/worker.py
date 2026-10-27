"""Sound events, emotion and language per speech turn with SenseVoice-Small.

SenseVoice labels one language, one emotion and one event per input, so it is
run on each diarized turn separately (turns longer than 30 s are cut).
"""

import re
import tempfile
from pathlib import Path

import soundfile as sf

from aidub_worker import device, read_mono, serve, torch_info

TAG = re.compile(r"<\|([^|]*)\|>")
EMOTIONS = {"HAPPY", "SAD", "ANGRY", "NEUTRAL", "FEARFUL", "DISGUSTED", "SURPRISED", "EMO_UNKNOWN"}
state = {}


def load(args):
    from funasr import AutoModel
    name = args.get("model") or "iic/SenseVoiceSmall"
    state["model"] = AutoModel(model=name, device="cuda:0" if device() == "cuda" else "cpu", disable_update=True)
    return {"model": name}


def tag(args):
    audio = read_mono(args["input"], 16000)
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i, (a, b) in enumerate(args["segments"]):
            clip = audio[int(a * 16000): int(min(b, a + 30) * 16000)]
            p = Path(tmp) / f"{i:05d}.wav"
            sf.write(str(p), clip if len(clip) >= 1600 else audio[int(a * 16000): int(a * 16000) + 1600], 16000)
            paths.append(str(p))
        res = state["model"].generate(input=paths, cache={}, language="auto", use_itn=True, batch_size=32,
                                      ban_emo_unk=False)
    for r in res:
        tags = TAG.findall(r.get("text", ""))
        text = TAG.sub("", r.get("text", "")).strip()
        lang = tags[0] if tags else ""
        emo = next((t for t in tags if t in EMOTIONS), "")
        events = [t for t in tags[1:] if t not in EMOTIONS and t not in ("withitn", "woitn", "Speech")]
        out.append({"language": "" if lang in ("nospeech", "auto") else lang,
                    "emotion": "" if emo == "EMO_UNKNOWN" else emo, "events": events, "text": text})
    return {"results": out}


def prefetch(args):
    load(args)
    return {}


def doctor(args):
    import funasr
    return {**torch_info(), "funasr": funasr.__version__}


if __name__ == "__main__":
    serve({"load": load, "tag": tag, "prefetch": prefetch, "doctor": doctor})
