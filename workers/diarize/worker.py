"""Speaker diarization with NVIDIA Nemotron 3 Diarization (up to 8 speakers per pass).

Ai Dub calls it once per scene window and links speakers across windows itself.
"""

from aidub_worker import device, log, read_mono, serve, torch_info

state = {}
MODEL = "nvidia/Nemotron-3-Diarization"


def load(args):
    import torch
    from transformers import AutoModelForAudioFrameClassification, AutoProcessor
    name = args.get("model") or MODEL
    state["processor"] = AutoProcessor.from_pretrained(name)
    state["model"] = AutoModelForAudioFrameClassification.from_pretrained(
        name, device_map=device(), dtype=torch.bfloat16 if device() == "cuda" else torch.float32).eval()
    state["name"] = name
    return {"model": name}


def diarize(args):
    import torch
    proc, model = state["processor"], state["model"]
    sr = proc.feature_extractor.sampling_rate
    audio = read_mono(args["input"], sr)
    inputs = proc(audio, sampling_rate=sr, return_tensors="pt").to(model.device, dtype=model.dtype)
    with torch.inference_mode():
        logits = model(**inputs).logits
    segs = proc.extract_speaker_dict(logits, inputs.get("attention_mask"))[0]
    out = [{"start": round(float(s["Start"]), 3), "end": round(float(s["End"]), 3), "speaker": f"spk{int(s['Speaker'])}"}
           for s in segs if float(s["End"]) > float(s["Start"])]
    log(f"{len(out)} turns, {len({s['speaker'] for s in out})} speakers")
    return {"segments": out}


def prefetch(args):
    from huggingface_hub import snapshot_download
    return {"path": snapshot_download(args.get("model") or MODEL)}


def doctor(args):
    import transformers
    return {**torch_info(), "transformers": transformers.__version__}


if __name__ == "__main__":
    serve({"load": load, "diarize": diarize, "prefetch": prefetch, "doctor": doctor})
