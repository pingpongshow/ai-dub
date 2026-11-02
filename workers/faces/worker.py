"""Who is talking on screen: faces found with YuNet, followed through a line, checked
for talking against the sound with Light-ASD, and matched across the film with SFace."""

import math
from pathlib import Path

from aidub_worker import device, log, model_dir, serve, torch_info

DETECT_EVERY = 5     # frames between face searches; boxes are drawn in between
MAX_LINE_S = 12.0
MIN_FRAMES = 10
FACE_MIN = 40        # pixels, in the 640 wide search frame
WATCH_WIDTH = 960    # frames are read this wide: plenty for the model's 112 pixel faces
WEIGHTS = {
    "pretrain_AVA_CVPR.model": "https://github.com/Junhua-Liao/Light-ASD/raw/main/weight/pretrain_AVA_CVPR.model",
    "face_detection_yunet_2023mar.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}
state = {}


def _dir() -> Path:
    return Path(model_dir()) / "faces"


def load(args):
    import cv2
    import torch
    from lightasd.Model import ASD_Model
    model = ASD_Model()
    fc = torch.nn.Linear(128, 2)
    weights = torch.load(_dir() / "pretrain_AVA_CVPR.model", map_location="cpu", weights_only=True)
    model.load_state_dict({k[len("model."):]: v for k, v in weights.items() if k.startswith("model.")})
    fc.load_state_dict({k[len("lossAV.FC."):]: v for k, v in weights.items() if k.startswith("lossAV.FC.")})
    state.update(asd=model.to(device()).eval(), fc=fc.to(device()).eval(),
                 detect=cv2.FaceDetectorYN.create(str(_dir() / "face_detection_yunet_2023mar.onnx"), "", (640, 360), 0.7),
                 match=cv2.FaceRecognizerSF.create(str(_dir() / "face_recognition_sface_2021dec.onnx"), ""))
    return {"model": "Light-ASD + YuNet + SFace"}


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / max(1e-6, a[2] * a[3] + b[2] * b[3] - inter)


def _tracks(frames, scale):
    """Face tracks through a list of frames: {start frame: box} searched every few frames,
    joined by overlap, with the boxes in between drawn from their neighbours."""
    import cv2
    import numpy as np
    tracks = []
    for i in range(0, len(frames), DETECT_EVERY):
        small = cv2.resize(frames[i], None, fx=scale, fy=scale)
        state["detect"].setInputSize((small.shape[1], small.shape[0]))
        _, found = state["detect"].detect(small)
        for row in found if found is not None else []:
            if row[2] < FACE_MIN * 0.6 or row[3] < FACE_MIN * 0.6:
                continue
            best = max((t for t in tracks if i - t["last"] <= DETECT_EVERY * 2), key=lambda t: _iou(t["boxes"][t["last"]], row), default=None)
            if best is not None and _iou(best["boxes"][best["last"]], row) >= 0.3:
                best["boxes"][i], best["last"] = row, i
            else:
                tracks.append({"first": i, "last": i, "boxes": {i: row}})
    out = []
    for t in tracks:
        if t["last"] - t["first"] + 1 < MIN_FRAMES:
            continue
        keys = sorted(t["boxes"])
        ys = np.array([t["boxes"][k][:4] for k in keys])
        frames_in = np.arange(t["first"], t["last"] + 1)
        t["path"] = np.stack([np.interp(frames_in, keys, ys[:, c]) for c in range(4)], axis=1) / scale
        out.append(t)
    return out


def _crops(frames, t):
    """As the model was trained: a square around the face, grey, 224 px, the middle 112."""
    import cv2
    import numpy as np
    out = []
    for offset, (x, y, w, h) in enumerate(t["path"]):
        f = frames[t["first"] + offset]
        side = max(w, h) / 2
        pad = int(side * 2)
        padded = cv2.copyMakeBorder(f, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(110, 110, 110))
        cx, cy = x + w / 2 + pad, y + h / 2 + pad
        crop = padded[int(cy - side): int(cy + side * 1.8), int(cx - side * 1.4): int(cx + side * 1.4)]
        grey = cv2.cvtColor(cv2.resize(crop, (224, 224)), cv2.COLOR_BGR2GRAY)
        out.append(grey[56:168, 56:168])
    return np.array(out)


def _talking(faces, mfcc):
    """Mean talking score of a face through the sound (over 0: talking), at 25 and 100 per second."""
    import numpy as np
    import torch
    seconds = min((mfcc.shape[0] - mfcc.shape[0] % 4) / 100, faces.shape[0] / 25)
    if seconds < 0.4:
        return None
    a, v = mfcc[: int(round(seconds * 100))], faces[: int(round(seconds * 25))]
    scores = []
    with torch.no_grad():
        for dur in (1, 2, 3):
            part = []
            for i in range(int(math.ceil(seconds / dur))):
                ia = torch.FloatTensor(a[i * dur * 100: (i + 1) * dur * 100]).unsqueeze(0).to(device())
                iv = torch.FloatTensor(v[i * dur * 25: (i + 1) * dur * 25]).unsqueeze(0).to(device())
                if ia.shape[1] < 4 or iv.shape[1] < 1:
                    continue
                n = min(ia.shape[1] // 4, iv.shape[1])
                x = state["asd"].forward_audio_visual_backend(state["asd"].forward_audio_frontend(ia[:, : n * 4]),
                                                              state["asd"].forward_visual_frontend(iv[:, : n]))
                part.append(state["fc"](x)[:, 1].cpu().numpy())
            if part:
                scores.append(np.concatenate(part))
    size = min(len(s) for s in scores)
    return np.mean([s[:size] for s in scores], axis=0)


def _feature(frames, t):
    """Face print from the largest view of the face in a track."""
    import cv2
    best = max(t["boxes"], key=lambda k: t["boxes"][k][2] * t["boxes"][k][3])
    row = t["boxes"][best].copy()
    frame = frames[best]
    scale = t["scale"]
    small = cv2.resize(frame, None, fx=scale, fy=scale)
    return state["match"].feature(state["match"].alignCrop(small, row)).flatten().tolist()


def _read(cap, start, end):
    """Frames at 25 a second between two times, repeating or skipping to keep time."""
    import cv2
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    out, t = [], start
    ok, frame = cap.read()
    while ok and t < end:
        at = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        while ok and at < t - 0.02:
            ok, frame = cap.read()
            at = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if not ok:
            break
        out.append(cv2.resize(frame, (WATCH_WIDTH, frame.shape[0] * WATCH_WIDTH // frame.shape[1])))
        t += 0.04
    return out


def analyse(args):
    """For each line: the faces on screen and how much each is talking."""
    import cv2
    import numpy as np
    import python_speech_features
    import soundfile as sf
    audio, sr = sf.read(args["audio"], dtype="float32")  # (reading a float file as int16 gives zeros)
    audio = (audio[:, 0] if audio.ndim > 1 else audio) * 32767
    cap = cv2.VideoCapture(args["video"])
    scale = min(1.0, 640 / WATCH_WIDTH)
    out = []
    try:
        for ln in args["lines"]:
            a, b = float(ln["start"]), min(float(ln["end"]), float(ln["start"]) + MAX_LINE_S)
            faces = []
            frames = _read(cap, a, b)
            if len(frames) >= MIN_FRAMES:
                mfcc = python_speech_features.mfcc(audio[int(a * sr): int(b * sr)], sr, numcep=13, winlen=0.025, winstep=0.010)
                for t in _tracks(frames, scale):
                    t["scale"] = scale
                    score = _talking(_crops(frames, t), mfcc[int(t["first"] * 4):])
                    if score is None:
                        continue
                    faces.append({"score": round(float(np.mean(score)), 3), "share": round(float(np.mean(score > 0)), 3),
                                  "from": round(a + t["first"] * 0.04, 2), "to": round(a + t["last"] * 0.04, 2),
                                  "print": [round(x, 4) for x in _feature(frames, t)]})
            out.append({"id": ln["id"], "faces": faces})
    finally:
        cap.release()
    return {"lines": out}


def prefetch(args):
    import urllib.request
    d = _dir()
    d.mkdir(parents=True, exist_ok=True)
    for name, url in WEIGHTS.items():
        if not (d / name).exists():
            tmp = d / (name + ".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(d / name)
            log(f"downloaded {name}")
    return {}


def doctor(args):
    return torch_info()


if __name__ == "__main__":
    serve({"load": load, "analyse": analyse, "prefetch": prefetch, "doctor": doctor})
