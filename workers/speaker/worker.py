"""Voice embeddings (SpeechBrain ECAPA), gender and age range (audeering
wav2vec2 age-gender), and cross-lingual sentence similarity (LaBSE)."""

from pathlib import Path

import numpy as np

from aidub_worker import device, log, model_dir, read_mono, serve, torch_info

ECAPA = "speechbrain/spkrec-ecapa-voxceleb"
AGE_GENDER = "audeering/wav2vec2-large-robust-24-ft-age-gender"
LABSE = "sentence-transformers/LaBSE"
state = {}


def _ecapa():
    if "ecapa" not in state:
        from speechbrain.inference.speaker import EncoderClassifier
        state["ecapa"] = EncoderClassifier.from_hparams(
            source=ECAPA, savedir=str(Path(model_dir()) / "speechbrain" / "spkrec-ecapa-voxceleb"),
            run_opts={"device": device()})
    return state["ecapa"]


def _age_gender():
    if "ag" not in state:
        import torch
        import torch.nn as nn
        from transformers import Wav2Vec2Processor
        from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

        class ModelHead(nn.Module):  # as published on the model card
            def __init__(self, config, num_labels):
                super().__init__()
                self.dense = nn.Linear(config.hidden_size, config.hidden_size)
                self.dropout = nn.Dropout(config.final_dropout)
                self.out_proj = nn.Linear(config.hidden_size, num_labels)

            def forward(self, features, **kwargs):
                x = self.dropout(features)
                x = torch.tanh(self.dense(x))
                return self.out_proj(self.dropout(x))

        class AgeGenderModel(Wav2Vec2PreTrainedModel):
            def __init__(self, config):
                super().__init__(config)
                self.config = config
                self.wav2vec2 = Wav2Vec2Model(config)
                self.age = ModelHead(config, 1)
                self.gender = ModelHead(config, 3)
                self.init_weights()

            def forward(self, input_values):
                hs = self.wav2vec2(input_values)[0].mean(dim=1)
                return hs, self.age(hs), torch.softmax(self.gender(hs), dim=1)

        state["ag_proc"] = Wav2Vec2Processor.from_pretrained(AGE_GENDER)
        state["ag"] = AgeGenderModel.from_pretrained(AGE_GENDER).to(device()).eval()
    return state["ag_proc"], state["ag"]


def _labse():
    if "labse" not in state:
        from sentence_transformers import SentenceTransformer
        state["labse"] = SentenceTransformer(LABSE, device=device())
    return state["labse"]


def _embed_arrays(clips):
    import torch
    enc = _ecapa()
    out = []
    for c in clips:
        if len(c) < 1600:
            out.append([])
            continue
        with torch.inference_mode():
            e = enc.encode_batch(torch.from_numpy(c).unsqueeze(0).to(device())).squeeze().float().cpu().numpy()
        out.append((e / (np.linalg.norm(e) + 1e-9)).round(5).tolist())
    return out


def load(args):
    _ecapa()
    return {"model": ECAPA}


def info(args):
    return {"embedding_model": ECAPA, "gender_age_model": AGE_GENDER, "text_model": LABSE}


def embed(args):
    audio = read_mono(args["input"], 16000)
    clips = [audio[int(a * 16000): int(b * 16000)] for a, b in args["segments"]]
    return {"embeddings": _embed_arrays(clips)}


def embed_files(args):
    return {"embeddings": _embed_arrays([read_mono(p, 16000) for p in args["paths"]])}


def _age_range(years: float, child: float) -> str:
    if child > 0.5 or years < 13:
        return "child"
    if years < 26:
        return "young"
    if years < 56:
        return "adult"
    return "senior"


def gender_age(args):
    import torch
    proc, model = _age_gender()
    audio = read_mono(args["input"], 16000)
    out = {}
    for spk in args["speakers"]:
        ages, probs = [], []
        for a, b in spk["segments"]:
            clip = audio[int(a * 16000): int(b * 16000)]
            if len(clip) < 16000:
                continue
            x = proc(clip, sampling_rate=16000)["input_values"][0].reshape(1, -1)
            with torch.inference_mode():
                _, age, g = model(torch.from_numpy(x).to(device()))
            ages.append(float(age[0, 0]) * 100)
            probs.append(g[0].float().cpu().numpy())
        if not ages:
            out[spk["speaker"]] = {}
            continue
        p = np.mean(probs, axis=0)  # female, male, child
        years = float(np.median(ages))
        gender = "female" if p[0] >= p[1] else "male"
        out[spk["speaker"]] = {"gender": gender, "age": _age_range(years, float(p[2])), "age_years": round(years)}
    return {"speakers": out, "model": AGE_GENDER}


def text_similarity(args):
    model = _labse()
    a = [x[0] for x in args["pairs"]]
    b = [x[1] for x in args["pairs"]]
    ea = model.encode(a, normalize_embeddings=True, batch_size=64)
    eb = model.encode(b, normalize_embeddings=True, batch_size=64)
    return {"similarity": [round(float(x), 4) for x in (ea * eb).sum(axis=1)]}


def prefetch(args):
    _ecapa()
    _age_gender()
    _labse()
    log("speaker models ready")
    return {}


def doctor(args):
    import speechbrain
    return {**torch_info(), "speechbrain": speechbrain.__version__}


if __name__ == "__main__":
    serve({"load": load, "info": info, "embed": embed, "embed_files": embed_files, "gender_age": gender_age,
           "text_similarity": text_similarity, "prefetch": prefetch, "doctor": doctor})
