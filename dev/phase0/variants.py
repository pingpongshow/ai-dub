"""Voice-setting variants for lines that sounded robotic or accented, to compare by ear.

For each chosen line of a finished check job, the same English line is voiced with:
  current   - the pipeline's chosen take (as dubbed; emotion strength 0.35 since round 2)
  more-emo  - the original line as emotion reference at the old strength (0.65)
  tag-emo   - no audio emotion reference; the detected emotion as a vector
  own-voice - the line's own audio as the voice reference
  own-happy - own voice, with a happy emotion vector (a short line has no emotion reference)
  own-child2/4 - the own-voice take with its formants raised 2 or 4 semitones, pitch kept
                 (a child's voice resonates higher than the adult voice the model leans to)
A version whose file already exists is reused, so a version judged before stays the same.
Writes /work/check/variants.json for review_page.py (which keeps a copy with the page).

    python variants.py /work/check/config.yaml clip:line_id[,line_id...][:label,label...] ...

Without labels: current, more-emo, tag-emo, own-voice.
"""

import json
import subprocess
import sys
from pathlib import Path

from aidub import config, media
from aidub import lines as L
from aidub import speak as S
from aidub.job import Job
from aidub.profiles import Profiles

SETTINGS = {"more-emo": {"emo_alpha": 0.65}, "tag-emo": {"emotion": "vector"}, "own-voice": {"voice": "own"},
            "own-happy": {"voice": "own", "emotion": "HAPPY", "strength": 0.6},
            "own-child2": {"from": "own-voice", "formant": 2}, "own-child4": {"from": "own-voice", "formant": 4}}
DEFAULT = ["current", "more-emo", "tag-emo", "own-voice"]


def raise_formants(src: Path, dst: Path, semitones: float) -> None:
    """Pitch up with the formants, then back down keeping them: same pitch, higher resonance."""
    r = 2 ** (semitones / 12)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-af",
                    f"rubberband=pitch={r}:formant=shifted,rubberband=pitch={1 / r}:formant=preserved", str(dst)],
                   check=True)


def main():
    cfg = config.load(sys.argv[1])
    root = Path(cfg.data["paths"]["work"]).parent
    root_json = root / "variants.json"
    out = json.loads(root_json.read_text()) if root_json.exists() else {}  # other clips' variants stay
    for arg in sys.argv[2:]:
        clip, ids, *labels = arg.split(":")
        ids = [int(x) for x in ids.split(",") if x]
        labels = labels[0].split(",") if labels else DEFAULT
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
                   "takes": {}}
            for label in labels:
                if label == "current":
                    row["takes"][label] = ln.take
                    continue
                opt = SETTINGS[label]
                r = S._requests(job, ln, ln.en, 1, f"var-{label}", dlg, sr)[0]
                row["takes"][label] = r["out"]
                if Path(r["out"]).exists() or "from" in opt:
                    continue  # judged before (kept as heard), or made from another version below
                if opt.get("emo_alpha") is not None and "emotion_ref" in r:
                    r["emo_alpha"] = opt["emo_alpha"]
                if opt.get("emotion"):
                    r.pop("emotion_ref", None)
                    r.pop("emo_alpha", None)
                    vec = S.emo_vector(ln.emotion if opt["emotion"] == "vector" else opt["emotion"], opt.get("strength", 0.7))
                    if vec:
                        r["emo_vector"] = vec
                if opt.get("voice") == "own":
                    r["speaker_ref"] = str(S._orig_clip(job, ln, dlg, sr))
                reqs.append(r)
            rows.append(row)
        if reqs:
            with job.worker("speak") as w:
                S._load_tts(job, w)
                S._synthesize(job, w, reqs, "variants")
            for r in reqs:
                if Path(r["out"]).exists():
                    S.F.trim(Path(r["out"]))
        for row in rows:
            for label, p in list(row["takes"].items()):
                opt = SETTINGS.get(label, {})
                if "from" in opt and not Path(p).exists() and Path(row["takes"].get(opt["from"], "-")).exists():
                    raise_formants(Path(row["takes"][opt["from"]]), Path(p), opt["formant"])
                if not Path(p).exists():
                    row["takes"].pop(label)
        out[clip] = rows
        print(clip, len(rows), "lines,", len(reqs), "variant takes")
    root_json.write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
