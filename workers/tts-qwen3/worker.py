"""Voice cloning with Qwen3-TTS Base (speaker-embedding mode: the reference
needs no transcript). No speed or duration control; 24 kHz output."""

from pathlib import Path

import soundfile as sf

from aidub_worker import device, log, serve, torch_info

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
state = {"prompts": {}}


def load(args):
    import torch
    from qwen_tts import Qwen3TTSModel
    state["m"] = Qwen3TTSModel.from_pretrained(args.get("model") or MODEL, device_map="cuda:0" if device() == "cuda" else "cpu",
                                               dtype=torch.bfloat16, attn_implementation="sdpa")
    return {"model": args.get("model") or MODEL, "duration_control": False, "emotion_ref": False,
            "emotion_vector": False, "sample_rate": 24000}


def synthesize(args):
    import torch
    m = state["m"]
    done = []
    for it in args["items"]:
        ref = it["speaker_ref"]
        if ref not in state["prompts"]:
            state["prompts"][ref] = m.create_voice_clone_prompt(ref_audio=ref, ref_text=None, x_vector_only_mode=True)
        if it.get("seed") is not None:
            torch.manual_seed(int(it["seed"]))
        try:
            wavs, sr = m.generate_voice_clone(text=it["text"], language="English", voice_clone_prompt=state["prompts"][ref],
                                              non_streaming_mode=True, max_new_tokens=1024)
            Path(it["out"]).parent.mkdir(parents=True, exist_ok=True)
            sf.write(it["out"], wavs[0], sr)
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
