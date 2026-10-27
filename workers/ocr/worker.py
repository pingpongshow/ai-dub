"""OCR for bitmap subtitles (PGS) with PaddleOCR 3.7 on ONNX Runtime (CPU).

Subtitle lines are small; CPU is fast enough and avoids GPU framework issues.
"""

import numpy as np
from PIL import Image

from aidub_worker import serve

LANG = {"ja": "japan", "zh": "ch", "yue": "chinese_cht", "ko": "korean", "en": "en"}
state = {}


def _pipeline(lang: str):
    key = LANG.get(lang, "ch")
    if key not in state:
        from paddleocr import PaddleOCR
        state[key] = PaddleOCR(lang=key, use_doc_orientation_classify=False, use_doc_unwarping=False,
                               use_textline_orientation=False, engine="onnxruntime")
    return state[key]


def _prepare(path: str) -> np.ndarray:
    """PGS captions are light text with a dark outline on transparency: put them on black, pad, upscale small ones."""
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, (0, 0, 0, 255))
    bg.alpha_composite(im)
    rgb = bg.convert("RGB")
    bbox = im.getchannel("A").getbbox()
    if bbox:
        rgb = rgb.crop(bbox)
    pad = Image.new("RGB", (rgb.width + 24, rgb.height + 24), (0, 0, 0))
    pad.paste(rgb, (12, 12))
    if pad.height < 48:
        pad = pad.resize((pad.width * 2, pad.height * 2), Image.LANCZOS)
    return np.asarray(pad)[:, :, ::-1].copy()  # BGR


def load(args):
    return {"model": "PaddleOCR 3.7 (onnxruntime)"}


def ocr(args):
    ocr = _pipeline(args.get("language") or "ja")
    texts = []
    for p in args["images"]:
        try:
            res = ocr.predict(_prepare(p))
            lines = []
            for r in res:
                polys = r.get("rec_polys") or r.get("dt_polys") or []
                items = list(zip(r.get("rec_texts", []), [float(np.asarray(b)[:, 1].mean()) for b in polys] or
                                 range(len(r.get("rec_texts", [])))))
                lines += [t for t, _ in sorted(items, key=lambda x: x[1]) if t.strip()]
            texts.append("\n".join(lines))
        except Exception:
            texts.append("")
    return {"texts": texts}


def prefetch(args):
    for lang in ("ja", "zh", "ko"):
        _pipeline(lang)
    return {}


def doctor(args):
    import paddleocr
    return {"paddleocr": paddleocr.__version__}


if __name__ == "__main__":
    serve({"load": load, "ocr": ocr, "prefetch": prefetch, "doctor": doctor})
