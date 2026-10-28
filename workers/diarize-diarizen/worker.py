"""Speaker diarization with DiariZen (comparison only: non-commercial weights)."""

import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")  # pyannote/Lightning checkpoints from the authors

from aidub_worker import serve, torch_info  # noqa: E402

MODEL = "BUT-FIT/diarizen-wavlm-large-s80-md-v2"
state = {}


def load(args):
    from diarizen.pipelines.inference import DiariZenPipeline
    state["p"] = DiariZenPipeline.from_pretrained(args.get("model") or MODEL, rttm_out_dir=None)
    return {"model": args.get("model") or MODEL}


def diarize(args):
    ann = state["p"](args["input"], sess_name="clip")
    return {"segments": [{"start": round(float(seg.start), 3), "end": round(float(seg.end), 3), "speaker": f"spk{spk}"}
                         for seg, _, spk in ann.itertracks(yield_label=True)]}


def prefetch(args):
    from huggingface_hub import snapshot_download
    snapshot_download(args.get("model") or MODEL)
    snapshot_download("pyannote/wespeaker-voxceleb-resnet34-LM")
    return {}


def doctor(args):
    return torch_info()


if __name__ == "__main__":
    serve({"load": load, "diarize": diarize, "prefetch": prefetch, "doctor": doctor})
