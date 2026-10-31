"""Sound events, emotion and language per speech turn with SenseVoice-Small,
and the voice's pitch over time with FCPE.

SenseVoice labels one language, one emotion and one event per input, so it is
run on each diarized turn separately (turns longer than 30 s are cut). FCPE (a
pitch tracker made for singing voices, MIT) gives the pitch of the separated
dialogue every 10 ms: held notes tell singing from speech, and how high a line
sits tells whose voice it is.
"""

import os
import re
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from aidub_worker import device, read_mono, serve, torch_info

TAG = re.compile(r"<\|([^|]*)\|>")
EMOTIONS = {"HAPPY", "SAD", "ANGRY", "NEUTRAL", "FEARFUL", "DISGUSTED", "SURPRISED", "EMO_UNKNOWN"}
state = {}


def _local(name: str) -> str:
    """The downloaded copy of a ModelScope model, so loading doesn't depend on reaching
    the hub (a failed lookup there makes FunASR report the model as "not registered")."""
    cache = Path(os.environ.get("MODELSCOPE_CACHE", Path.home() / ".cache" / "modelscope"))
    org, _, model = name.partition("/")
    for cfg in [*sorted(cache.glob(f"models/{org}--{model}/snapshots/*/config.yaml")),
                cache / "models" / org / model / "config.yaml", cache / "hub" / org / model / "config.yaml"]:
        if cfg.exists():
            return str(cfg.parent)
    return name


def load(args):
    from funasr import AutoModel
    name = args.get("model") or "iic/SenseVoiceSmall"
    state["model"] = AutoModel(model=_local(name), device="cuda:0" if device() == "cuda" else "cpu",
                               disable_update=True)
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


def pitch(args):
    """Pitch of `input` every 10 ms (Hz, 0 where unvoiced), saved to `output` (.npy)."""
    import torch
    from torchfcpe import spawn_bundled_infer_model
    if "fcpe" not in state:
        state["fcpe"] = spawn_bundled_infer_model(device="cuda" if device() == "cuda" else "cpu")
    audio = read_mono(args["input"], 16000).astype(np.float32)
    hop, chunk = 160, 16000 * 60
    out = []
    with torch.no_grad():
        for a in range(0, len(audio), chunk):
            x = audio[a: a + chunk]
            n = max(1, len(x) // hop)
            f0 = state["fcpe"].infer(torch.from_numpy(x)[None, :, None].to(state["fcpe"].get_device()), sr=16000,
                                     decoder_mode="local_argmax", threshold=float(args.get("threshold", 0.006)),
                                     f0_min=50, interp_uv=False, output_interp_target_length=n)
            out.append(f0.squeeze().reshape(-1).float().cpu().numpy()[:n])
    f0 = np.concatenate(out) if out else np.zeros(0, np.float32)
    np.save(args["output"], f0.astype(np.float32))
    return {"output": args["output"], "hop": hop / 16000, "frames": int(f0.size),
            "voiced": round(float(np.mean(f0 > 0)) if f0.size else 0.0, 3)}


def prefetch(args):
    load(args)
    from torchfcpe import spawn_bundled_infer_model
    spawn_bundled_infer_model(device="cpu")
    return {}


def doctor(args):
    import funasr
    return {**torch_info(), "funasr": funasr.__version__}


if __name__ == "__main__":
    serve({"load": load, "tag": tag, "pitch": pitch, "prefetch": prefetch, "doctor": doctor})
