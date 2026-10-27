"""Designs generic voices from text descriptions (Qwen3-TTS VoiceDesign), so
minor speakers never borrow a real person's voice."""

import soundfile as sf

from aidub_worker import device, serve, torch_info

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
state = {}


def load(args):
    import torch
    from qwen_tts import Qwen3TTSModel
    name = args.get("model") or MODEL
    state["m"] = Qwen3TTSModel.from_pretrained(name, device_map="cuda:0" if device() == "cuda" else "cpu",
                                               dtype=torch.bfloat16 if device() == "cuda" else torch.float32)
    return {"model": name}


def design(args):
    wavs, sr = state["m"].generate_voice_design(text=args["text"], instruct=args["description"], language="English")
    sf.write(args["out"], wavs[0], sr)
    return {"path": args["out"], "sample_rate": sr}


def prefetch(args):
    from huggingface_hub import snapshot_download
    snapshot_download(args.get("model") or MODEL)
    snapshot_download("Qwen/Qwen3-TTS-Tokenizer-12Hz")
    return {}


def doctor(args):
    return torch_info()


if __name__ == "__main__":
    serve({"load": load, "design": design, "prefetch": prefetch, "doctor": doctor})
