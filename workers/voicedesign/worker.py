"""Voices of Ai Dub's own (Qwen3-TTS), so no real person's voice is used.

design: a voice made from a text description (the VoiceDesign model), saying a sample sentence.
synthesize: lines spoken in a designed voice (the Base model clones the designed sample, given
its words), the same person in every line. No emotion or duration control.
Output: 24 kHz mono WAV.
"""

import os

import numpy as np
import soundfile as sf

from aidub_worker import device, log, serve, torch_info

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
SPEAKER_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
PEAK = 0.95      # a take's loudest moment, at most
TOKENS_PER_S = 12.5
state = {"prompts": {}}


def load(args):
    import torch
    from qwen_tts import Qwen3TTSModel
    name = args.get("model") or MODEL
    state["m"] = Qwen3TTSModel.from_pretrained(name, device_map="cuda:0" if device() == "cuda" else "cpu",
                                               dtype=torch.bfloat16 if device() == "cuda" else torch.float32)
    state["prompts"] = {}
    return {"model": name, "duration_control": False, "emotion_ref": False, "emotion_vector": False,
            "sample_rate": 24000}


def design(args):
    wavs, sr = state["m"].generate_voice_design(text=args["text"], instruct=args["description"], language="English")
    sf.write(args["out"], _under_full_scale(wavs[0]), sr)
    return {"path": args["out"], "sample_rate": sr}


def _under_full_scale(wav):
    wav = np.asarray(wav, dtype=np.float32)
    peak = float(np.abs(wav).max()) if wav.size else 0.0
    return wav * (PEAK / peak) if peak > PEAK else wav


def _prompt(ref: str, text: str):
    """The designed voice, analysed once per file (and again if the file changes)."""
    key = (ref, os.path.getmtime(ref), text)
    if key not in state["prompts"]:
        state["prompts"][key] = state["m"].create_voice_clone_prompt(ref_audio=ref, ref_text=text or None,
                                                                     x_vector_only_mode=not text)[0]
    return state["prompts"][key]


def synthesize(args):
    """Items: text, speaker_ref (a designed voice), speaker_text (its words), seed, out. One batch:
    its first seed seeds it, so the same items give the same takes."""
    import torch
    items = args["items"]
    if not items:
        return {"done": []}
    if items[0].get("seed") is not None:
        torch.manual_seed(int(items[0]["seed"]))
    longest = max(len(it["text"]) for it in items)
    done = []
    try:
        prompts = [_prompt(it["speaker_ref"], it.get("speaker_text", "")) for it in items]
        wavs, sr = state["m"].generate_voice_clone(
            text=[it["text"] for it in items], language=["English"] * len(items), voice_clone_prompt=prompts,
            # A runaway take (it can babble past the line) stops near twice its natural length.
            max_new_tokens=int(TOKENS_PER_S * (4 + longest / 7)))
    except Exception as exc:
        log(f"synthesis failed for {len(items)} lines: {exc}")
        return {"done": done}
    for it, wav in zip(items, wavs):
        os.makedirs(os.path.dirname(it["out"]), exist_ok=True)
        sf.write(it["out"], _under_full_scale(wav), sr)
        done.append(it["out"])
    return {"done": done}


def prefetch(args):
    from huggingface_hub import snapshot_download
    snapshot_download(args.get("model") or MODEL)
    snapshot_download(SPEAKER_MODEL)
    snapshot_download("Qwen/Qwen3-TTS-Tokenizer-12Hz")
    return {}


def doctor(args):
    return torch_info()


if __name__ == "__main__":
    serve({"load": load, "design": design, "synthesize": synthesize, "prefetch": prefetch, "doctor": doctor})
