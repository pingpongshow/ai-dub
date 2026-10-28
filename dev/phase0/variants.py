"""Voice-setting variants for lines that sounded robotic or accented, to compare by ear.

For each chosen line of a finished check job, the same English line is voiced with:
  current   - the pipeline's chosen take (as dubbed; emotion strength 0.35 since round 2)
  more-emo  - the original line as emotion reference at the old strength (0.65)
  tag-emo   - no audio emotion reference; the detected emotion as a vector
  own-voice - the line's own audio as the voice reference
Writes /work/check/variants.json for review_page.py (which keeps a copy with the page).

    python variants.py /work/check/config.yaml clip:line_id[,line_id...] ...
"""

import json
import sys
from pathlib import Path

from aidub import config, media
from aidub import lines as L
from aidub import speak as S
from aidub.job import Job
from aidub.profiles import Profiles

SETTINGS = {"more-emo": {"emo_alpha": 0.65}, "tag-emo": {"emotion": "vector"}, "own-voice": {"voice": "own"}}


def main():
    cfg = config.load(sys.argv[1])
    root = Path(cfg.data["paths"]["work"]).parent
    root_json = root / "variants.json"
    out = json.loads(root_json.read_text()) if root_json.exists() else {}  # other clips' variants stay
    for arg in sys.argv[2:]:
        clip, ids = arg.split(":")
        ids = [int(x) for x in ids.split(",") if x]
        prof = Profiles(cfg.appdata).get_or_create(f"check {clip}")
        job = Job(cfg, f"chk-{clip}", root / f"{clip}.mkv", prof, {})
        lines = {ln.id: ln for ln in L.load(job.path("lines.json"))}
        names = {cid: ch["name"] for cid, ch in prof.characters.items()}
        dlg, sr = media.read_wav(job.path("audio", "dialogue.wav"), mono=True)
        reqs, rows = [], []
        for i in ids:
            ln = lines[i]
            if not (ln.take and ln.en):
                continue
            row = {"id": i, "start": ln.start, "end": ln.end, "en": ln.en, "who": names.get(ln.character, ""),
                   "takes": {"current": ln.take}}
            for label, opt in SETTINGS.items():
                r = S._requests(job, ln, ln.en, 1, f"var-{label}", dlg, sr)[0]
                if opt.get("emo_alpha") is not None and "emotion_ref" in r:
                    r["emo_alpha"] = opt["emo_alpha"]
                if opt.get("emotion") == "vector":
                    r.pop("emotion_ref", None)
                    r.pop("emo_alpha", None)
                    vec = S.emo_vector(ln.emotion)
                    if vec:
                        r["emo_vector"] = vec
                if opt.get("voice") == "own":
                    r["speaker_ref"] = str(S._orig_clip(job, ln, dlg, sr))
                reqs.append(r)
                row["takes"][label] = r["out"]
            rows.append(row)
        with job.worker("speak") as w:
            S._load_tts(job, w)
            S._synthesize(job, w, reqs, "variants")
        for row in rows:
            for label, p in list(row["takes"].items()):
                if label != "current":
                    if Path(p).exists():
                        S.F.trim(Path(p))
                    else:
                        row["takes"].pop(label)
        out[clip] = rows
        print(clip, len(rows), "lines,", len(reqs), "variant takes")
    root_json.write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
