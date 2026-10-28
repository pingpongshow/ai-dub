"""Speech bandwidth extension with the tiny FlashSR (16 kHz in, 48 kHz out).
Its output is peak-normalised, so the input loudness is restored afterwards."""

from pathlib import Path

import numpy as np
import soundfile as sf

from aidub_worker import model_dir, read_mono, serve, torch_info

state = {}


def load(args):
    from FastAudioSR import FASR
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("YatharthS/FlashSR", "upsampler.pth", local_dir=str(Path(model_dir()) / "flashsr-tiny"))
    state["m"] = FASR(p)
    return {"model": "FlashSR tiny (YatharthS)"}


def upsample(args):
    import torch
    done = []
    for it in args["items"]:
        x = read_mono(it["input"], 16000)
        y = state["m"].run(torch.from_numpy(x).unsqueeze(0)).float().cpu().numpy().reshape(-1)
        rms_in = float(np.sqrt(np.mean(x ** 2)) + 1e-9)
        y = y * (rms_in / float(np.sqrt(np.mean(y ** 2)) + 1e-9))
        sf.write(it["output"], np.clip(y, -0.99, 0.99).astype(np.float32), 48000, subtype="FLOAT")
        done.append(it["output"])
    return {"done": done}


def prefetch(args):
    load(args)
    return {}


def doctor(args):
    return torch_info()


if __name__ == "__main__":
    serve({"load": load, "upsample": upsample, "prefetch": prefetch, "doctor": doctor})
