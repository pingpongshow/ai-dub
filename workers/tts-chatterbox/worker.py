"""Voice cloning with Chatterbox Multilingual (V3). For references in another
language, cfg_weight 0 avoids carrying the reference's accent. 24 kHz output."""

from pathlib import Path

import soundfile as sf

from aidub_worker import device, log, serve, torch_info

state = {}


def load(args):
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    state["tts"] = ChatterboxMultilingualTTS.from_pretrained(device=device(), t3_model=args.get("t3_model", "v3"))
    return {"model": "Chatterbox Multilingual v3", "duration_control": False, "emotion_ref": False,
            "emotion_vector": False, "sample_rate": int(state["tts"].sr)}


def synthesize(args):
    import torch
    tts = state["tts"]
    done = []
    for it in args["items"]:
        if it.get("seed") is not None:
            torch.manual_seed(int(it["seed"]))
        try:
            wav = tts.generate(it["text"], language_id="en", audio_prompt_path=it["speaker_ref"],
                               exaggeration=float(it.get("exaggeration", 0.5)), cfg_weight=float(it.get("cfg_weight", 0.0)))
            Path(it["out"]).parent.mkdir(parents=True, exist_ok=True)
            sf.write(it["out"], wav.squeeze(0).cpu().numpy(), tts.sr)
            done.append(it["out"])
        except Exception as exc:
            log(f"failed {it['out']}: {exc}")
    return {"done": done}


def prefetch(args):
    load(args)
    return {}


def doctor(args):
    return torch_info()


if __name__ == "__main__":
    serve({"load": load, "synthesize": synthesize, "prefetch": prefetch, "doctor": doctor})
