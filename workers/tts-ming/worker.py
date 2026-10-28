"""Voice cloning with Ming-3.0-tts 0.5B (in-context: a reference clip plus its
transcript, here the original line and its source text). 44.1 kHz output."""

import glob
import os
import sys
from pathlib import Path

import soundfile as sf

from aidub_worker import device, log, serve, torch_info

MODEL = "inclusionAI/Ming-3.0-tts-0.5B"
state = {}


def _src():
    env = Path(sys.executable).resolve().parent.parent
    for p in glob.glob(str(Path(os.environ.get("VIRTUAL_ENV", env)) / "src" / "Ming-3.0-tts")) + \
            glob.glob(str(Path(sys.prefix) / "src" / "Ming-3.0-tts")):
        return p
    raise RuntimeError("Ming-3.0-tts source not found in the environment")


def load(args):
    sys.path.insert(0, _src())
    from ming_tts import MingTTS
    state["m"] = MingTTS(args.get("model") or MODEL, device="cuda:0" if device() == "cuda" else "cpu",
                         text_normalization=False)
    return {"model": args.get("model") or MODEL, "duration_control": False, "emotion_ref": False,
            "emotion_vector": False, "sample_rate": 44100}


def synthesize(args):
    import torch
    m = state["m"]
    done = []
    for it in args["items"]:
        ref = it.get("emotion_ref") or it["speaker_ref"]
        if it.get("seed") is not None:
            torch.manual_seed(int(it["seed"]))
        try:
            wav = m.synthesize(task="TTS", language_id="English", text=it["text"], prompt_wav_path=ref,
                               prompt_text=it.get("prompt_text") or "", use_spk_emb=True, cfg=2.0, sigma=0.25,
                               temperature=0.0)
            Path(it["out"]).parent.mkdir(parents=True, exist_ok=True)
            sf.write(it["out"], wav.squeeze(0).float().cpu().numpy(), 44100)
            done.append(it["out"])
        except Exception as exc:
            log(f"failed {it['out']}: {exc}")
    return {"done": done}


def prefetch(args):
    from huggingface_hub import snapshot_download
    return {"path": snapshot_download(args.get("model") or MODEL)}


def doctor(args):
    return torch_info()


if __name__ == "__main__":
    serve({"load": load, "synthesize": synthesize, "prefetch": prefetch, "doctor": doctor})
