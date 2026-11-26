"""Japanese speech recognition with Haas-ja-v1.0 (Parakeet-TDT based, Apache-2.0).

Text only: on Woman in the Dunes its words matched the English subtitles' meaning more
closely than Qwen3-ASR's (0.53 against 0.49 LaBSE; closer on 33 % of lines, further on 12 %).
Word timing comes from Qwen3-ForcedAligner, given this text. Trained on up to 30 s clips.
"""

from aidub_worker import read_mono, serve, torch_info

MODEL = "third-intelligence/Haas-ja-v1.0"
state = {}


def load(args):
    import haas_asr
    name = args.get("model") or MODEL
    state["m"] = haas_asr.load_model(name)
    return {"model": name}


def transcribe(args):
    audio = [read_mono(it["path"], 16000) for it in args["items"]]
    return {"results": [{"text": o.text, "language": "ja", "words": []} for o in state["m"].transcribe(audio)]}


def prefetch(args):
    from huggingface_hub import snapshot_download
    snapshot_download(args.get("model") or MODEL)
    return {}


def doctor(args):
    import haas_asr  # noqa: F401
    return torch_info()


if __name__ == "__main__":
    serve({"load": load, "transcribe": transcribe, "prefetch": prefetch, "doctor": doctor})
