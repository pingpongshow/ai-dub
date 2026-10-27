"""Dialogue / music & effects separation with audio-separator (RoFormer vocal models)."""

import os
from pathlib import Path

from aidub_worker import device, log, model_dir, serve, torch_info

state = {"sep": None, "model": None}
MODELS = Path(model_dir()) / "audio-separator"


def _separator(out_dir: str):
    from audio_separator.separator import Separator
    MODELS.mkdir(parents=True, exist_ok=True)
    return Separator(model_file_dir=str(MODELS), output_dir=out_dir, output_format="WAV", sample_rate=48000,
                     use_autocast=device() == "cuda", chunk_duration=600, use_soundfile=True)


def load(args):
    state["model"] = args.get("model") or "vocals_mel_band_roformer.ckpt"
    state["sep"] = _separator(os.getcwd())
    state["sep"].load_model(model_filename=state["model"])
    return {"model": state["model"]}


def separate(args):
    sep = state["sep"]
    out = Path(args["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    sep.output_dir = str(out)
    if getattr(sep, "model_instance", None) is not None:
        sep.model_instance.output_dir = str(out)
    files = sep.separate(args["input"])
    paths = [Path(f) if Path(f).is_absolute() else out / f for f in files]
    vocals = next((p for p in paths if "(vocals)" in p.name.lower()), None)
    other = next((p for p in paths if p is not vocals), None)
    if not vocals or not other:
        raise RuntimeError(f"unexpected separator output: {[p.name for p in paths]}")
    v, i = out / "dialogue.wav", out / "mne.wav"
    vocals.replace(v)
    other.replace(i)
    log(f"separated {args['input']} -> {v.name}, {i.name}")
    return {"vocals": str(v), "instrumental": str(i)}


def prefetch(args):
    sep = _separator(os.getcwd())
    sep.load_model(model_filename=args.get("model") or "vocals_mel_band_roformer.ckpt")
    return {"model": args.get("model")}


def doctor(args):
    import audio_separator
    return {**torch_info(), "audio_separator": getattr(audio_separator, "__version__", "?")}


if __name__ == "__main__":
    serve({"load": load, "separate": separate, "prefetch": prefetch, "doctor": doctor})
