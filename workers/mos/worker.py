"""Reference-free naturalness score (UTMOS22 strong, 1-5) for comparing voices."""

from aidub_worker import device, read_mono, serve, torch_info

state = {}


def load(args):
    import torch
    state["m"] = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device()).eval()
    return {"model": "UTMOS22 strong"}


def score(args):
    import torch
    out = []
    for p in args["paths"]:
        x = read_mono(p, 16000)
        if len(x) < 4000:
            out.append(None)
            continue
        with torch.no_grad():
            out.append(round(float(state["m"](torch.from_numpy(x)[None].to(device()), 16000)[0]), 3))
    return {"mos": out}


def prefetch(args):
    load(args)
    return {}


def doctor(args):
    return torch_info()


if __name__ == "__main__":
    serve({"load": load, "score": score, "prefetch": prefetch, "doctor": doctor})
