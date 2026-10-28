"""Web API and static UI. No login: meant for a trusted LAN."""

from __future__ import annotations

import collections
import html
from contextlib import asynccontextmanager
import os
import threading
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import __version__, gpu, originals, workers
from . import jobs as J
from .config import Config
from .models import Installer
from .pipeline import STEP_NAMES, TITLES
from .service import Service

VIDEO_EXT = {".mkv", ".mp4", ".m4v", ".avi", ".ts", ".m2ts", ".mov", ".wmv", ".webm"}
WEB_DIR = Path(os.environ.get("AIDUB_WEB", Path(__file__).resolve().parent.parent / "web"))
# Keys for features that exist. Direct subtitle providers (OpenSubtitles, Subdl,
# Jimaku) get theirs when those providers are built.
SECRET_NAMES = ["BAZARR_API_KEY", "HF_TOKEN"]


def create_app(cfg: Config, service: Service | None = None) -> FastAPI:
    svc = service or Service(cfg)
    installer = Installer(cfg)

    @asynccontextmanager
    async def lifespan(_app):
        if service is None:
            svc.start()
        if os.environ.get("AIDUB_INSTALL_ON_START", "").lower() in ("1", "true", "yes"):
            installer.request_missing()
        yield
        svc.stop()

    app = FastAPI(title="Ai Dub", version=__version__, lifespan=lifespan)
    app.state.service = svc
    app.state.installer = installer


    def bad(status: int, msg: str):
        raise HTTPException(status_code=status, detail=msg)

    # Status --------------------------------------------------------------------------------
    @app.get("/api/status")
    def status():
        g = gpu.query()
        names = sorted({cfg.get(f"stages.{s}.worker") for s in cfg.get("stages", {}) if cfg.get(f"stages.{s}.worker")})
        installed = {n: workers.env_ready(cfg, n) for n in names}
        return {"version": __version__, "gpu": {"present": g.present, "name": g.name, "total_mb": g.total_mb,
                                                "used_mb": g.used_mb, "processes": g.processes},
                "workers": installed, "steps": [{"name": s, "title": TITLES[s]} for s in STEP_NAMES]}

    # Browsing --------------------------------------------------------------------------------
    @app.get("/api/browse")
    def browse(path: str = ""):
        if not path:
            return {"path": "", "parent": None, "entries": [
                {"name": m["name"], "path": m["path"], "dir": True} for m in cfg.media_roots]}
        try:
            p = svc.check_path(path)
        except PermissionError:
            bad(403, "outside the media folders")
        if not p.is_dir():
            bad(404, "not a folder")
        queued = {j["path"] for j in svc.store.list(J.ACTIVE)}
        entries = []
        try:
            items = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        except OSError as exc:
            bad(500, str(exc))
        for e in items:
            if e.name.startswith("."):
                continue
            if e.is_dir():
                entries.append({"name": e.name, "path": str(e), "dir": True})
            elif e.suffix.lower() in VIDEO_EXT:
                entries.append({"name": e.name, "path": str(e), "dir": False, "size": e.stat().st_size,
                                "queued": str(e) in queued,
                                "set_aside": originals.aside_name(e).exists()})
        roots = [str(r.resolve()) for r in svc.media_roots()]
        parent = None if str(p) in roots else str(p.parent)
        return {"path": str(p), "parent": parent, "entries": entries}

    @app.post("/api/inspect")
    def inspect(path: str = Body(..., embed=True)):
        try:
            return svc.inspect(path)
        except PermissionError:
            bad(403, "outside the media folders")
        except Exception as exc:
            bad(400, str(exc))

    # Jobs ----------------------------------------------------------------------------------
    def _job_view(j: dict) -> dict:
        live = svc.live.get(j["id"], {})
        out = {**j, "live": live, "step_title": TITLES.get(j.get("step", ""), j.get("step", ""))}
        if j["status"] == J.FAILED:
            out["work_bytes"] = svc.work_size(j["id"])
        return out

    @app.get("/api/jobs")
    def list_jobs(view: str = "queue"):
        statuses = J.ACTIVE if view == "queue" else {J.DONE, J.SKIPPED, J.FAILED, J.CANCELLED}
        rows = svc.store.list(statuses)
        if view != "queue":
            rows.sort(key=lambda j: -(j.get("finished") or j.get("created") or 0))
        return [_job_view(j) for j in rows]

    @app.post("/api/jobs")
    def add_jobs(paths: list[str] = Body(...), profile: str | None = Body(None), audio_stream: int | None = Body(None),
                 language: str | None = Body(None), redo: bool = Body(False)):
        added, errors = [], []
        for p in paths:
            opts = {k: v for k, v in {"audio_stream": audio_stream, "language": language, "redo": redo}.items() if v}
            try:
                added.append(svc.add(p, profile or None, opts))
            except Exception as exc:
                errors.append({"path": p, "error": str(exc)})
        return {"added": added, "errors": errors}

    @app.post("/api/jobs/{jid}/cancel")
    def cancel(jid: int):
        svc.cancel(jid)
        return {"ok": True}

    @app.post("/api/jobs/{jid}/retry")
    def retry(jid: int):
        svc.retry(jid)
        return {"ok": True}

    @app.post("/api/jobs/{jid}/move")
    def move(jid: int, direction: int = Body(..., embed=True)):
        svc.store.move(jid, -1 if direction < 0 else 1)
        return {"ok": True}

    @app.delete("/api/jobs/{jid}")
    def remove(jid: int):
        svc.remove(jid)
        return {"ok": True}

    @app.get("/api/jobs/{jid}/log", response_class=PlainTextResponse)
    def job_log(jid: int, lines: int = 200):
        p = cfg.work / f"job-{jid}" / "job.log"
        if not p.exists():
            return ""
        return "\n".join(p.read_text(errors="replace").splitlines()[-lines:])

    # Profiles ------------------------------------------------------------------------------
    def _profile(slug: str):
        prof = svc.profiles.get(slug)
        if not prof:
            bad(404, "no such profile")
        return prof

    @app.get("/api/profiles")
    def profiles():
        return [{"slug": p.slug, "name": p.name, "size": p.size(), "characters": len(p.characters),
                 "glossary": len(p.glossary), "episodes": len(p.data.get("episodes", [])),
                 "updated": p.data.get("updated")} for p in svc.profiles.list()]

    @app.get("/api/profiles/{slug}")
    def profile(slug: str):
        p = _profile(slug)
        chars = []
        for cid, ch in p.characters.items():
            chars.append({"id": cid, **{k: v for k, v in ch.items() if k not in ("embedding",)},
                          "refs": [{k: v for k, v in r.items() if k != "embedding"} for r in ch.get("refs", [])],
                          "clean_seconds": round(p.clean_seconds(cid), 1)})
        chars.sort(key=lambda c: -c["clean_seconds"])
        return {"slug": p.slug, "name": p.name, "settings": p.settings, "glossary": p.glossary,
                "characters": chars, "episodes": p.data.get("episodes", []), "size": p.size()}

    @app.patch("/api/profiles/{slug}")
    def update_profile(slug: str, body: dict = Body(...)):
        p = _profile(slug)
        if "name" in body:
            p.data["name"] = str(body["name"]).strip() or p.name
        if "settings" in body and isinstance(body["settings"], dict):
            allowed = {"honorifics", "style", "reflect"}
            p.settings.update({k: v for k, v in body["settings"].items() if k in allowed})
        p.save()
        return {"ok": True}

    @app.delete("/api/profiles/{slug}")
    def delete_profile(slug: str, confirm: bool = Query(False)):
        if not confirm:
            bad(400, "confirmation required")
        _profile(slug)
        svc.profiles.delete(slug)
        return {"ok": True}

    @app.post("/api/profiles/{slug}/characters/{cid}")
    def character_action(slug: str, cid: str, body: dict = Body(...)):
        p = _profile(slug)
        if cid not in p.characters:
            bad(404, "no such character")
        action = body.get("action")
        if action == "rename":
            p.rename(cid, str(body.get("name", "")))
        elif action == "merge":
            other = body.get("other")
            if other not in p.characters or other == cid:
                bad(400, "pick another character to merge")
            p.merge(cid, other)
        elif action == "split":
            p.split(cid, list(body.get("refs", [])), str(body.get("name", "")))
        elif action == "delete":
            if not body.get("confirm"):
                bad(400, "confirmation required")
            p.delete_character(cid)
        elif action == "remove_ref":
            p.remove_ref(cid, str(body.get("ref")))
        elif action == "lock_ref":
            p.lock_ref(cid, str(body.get("ref")), bool(body.get("locked", True)))
        elif action == "set":
            for k in ("gender", "age"):
                if k in body:
                    p.characters[cid][k] = str(body[k])
        else:
            bad(400, "unknown action")
        p.save()
        return {"ok": True}

    @app.get("/api/profiles/{slug}/ref")
    def ref_audio(slug: str, file: str):
        p = _profile(slug)
        target = (p.root / file).resolve()
        if p.root.resolve() not in target.parents or not target.exists():
            bad(404, "not found")
        return FileResponse(target, media_type="audio/flac")

    @app.put("/api/profiles/{slug}/glossary")
    def glossary_put(slug: str, source: str = Body(...), en: str = Body(...)):
        p = _profile(slug)
        e = p.glossary.setdefault(source.strip(), {"count": 0, "type": "term"})
        e.update({"en": en.strip(), "edited": True})
        p.save()
        return {"ok": True}

    @app.delete("/api/profiles/{slug}/glossary")
    def glossary_delete(slug: str, source: str = Query(...)):
        p = _profile(slug)
        p.glossary.pop(source, None)
        p.save()
        return {"ok": True}

    # Originals pending deletion ---------------------------------------------------------------
    @app.get("/api/originals")
    def originals_list():
        return originals.listing(svc.store)

    @app.post("/api/originals/scan")
    def originals_scan():
        return {"added": originals.scan(svc.store, svc.media_roots())}

    @app.post("/api/originals/{oid}/restore")
    def originals_restore(oid: int):
        try:
            return {"message": originals.restore(svc.store, oid)}
        except Exception as exc:
            bad(400, str(exc))

    @app.post("/api/originals/delete")
    def originals_delete(ids: list[int] = Body(...), confirm: bool = Body(False)):
        if not confirm:
            bad(400, "confirmation required")
        n, freed = originals.delete(svc.store, ids)
        return {"deleted": n, "freed": freed}

    # Models --------------------------------------------------------------------------------
    @app.get("/api/models")
    def models():
        return installer.listing()

    @app.post("/api/models/install")
    def models_install(names: list[str] = Body(default_factory=list, embed=True)):
        try:
            started = [installer.request(n) or n for n in names] if names else installer.request_missing()
        except KeyError as exc:
            bad(404, f"unknown model {exc}")
        return {"queued": started}

    @app.get("/api/models/{name}/log", response_class=PlainTextResponse)
    def models_log(name: str):
        return installer.log(name)

    @app.post("/api/voices/design")
    def voices_design(force: bool = Body(False, embed=True)):
        from .voices import design_library
        installer.logs.setdefault("voicedesign", collections.deque(maxlen=400))
        threading.Thread(target=lambda: _safe(design_library, cfg, force=force,
                                              log=installer.logs["voicedesign"].append), daemon=True).start()
        return {"started": True}

    # Settings --------------------------------------------------------------------------------
    @app.get("/api/settings")
    def settings():
        return {"overrides": cfg.user_overrides(), "effective": _redact(cfg.data),
                "secrets": {n: n in cfg.secret_names() for n in SECRET_NAMES}}

    @app.post("/api/settings")
    def save_settings(overrides: dict = Body(..., embed=True)):
        cfg.save_overrides(overrides)
        return {"ok": True, "note": "saved; restart the container to apply"}

    @app.post("/api/secrets")
    def save_secret(name: str = Body(...), value: str = Body("")):
        if name not in SECRET_NAMES:
            bad(400, "unknown secret")
        cfg.save_secret(name, value)
        return {"ok": True}

    # Listening pages -------------------------------------------------------------------------
    # Review pages (comparison tests, preview dubs) are written to <work>/listen on the server
    # and opened from there; they are deleted when the review is over.
    @app.get("/listen/{rel:path}")
    def listen(request: Request, rel: str = ""):
        root = (cfg.work / "listen").resolve()
        target = (root / rel).resolve()
        if not target.is_relative_to(root) or not target.exists():
            bad(404, "not found")
        if target.is_file():
            return FileResponse(target)
        if not request.url.path.endswith("/"):
            return RedirectResponse(request.url.path + "/")  # relative links need the slash
        if (target / "index.html").is_file():
            return FileResponse(target / "index.html")
        entries = sorted(p for p in target.iterdir() if not p.name.startswith("."))
        items = "".join(f'<li><a href="{html.escape(p.name)}{"/" if p.is_dir() else ""}">{html.escape(p.name)}</a></li>'
                        for p in entries)
        title = html.escape("listen/" + rel)
        return HTMLResponse(f"<!doctype html><meta charset=utf-8><title>{title}</title>"
                            f"<style>body{{font:15px system-ui;margin:20px}}</style><h1>{title}</h1>"
                            f"<ul>{items or '<li>nothing here</li>'}</ul>")

    @app.middleware("http")
    async def _no_stale_ui(request, call_next):
        # The UI is plain files; make browsers re-check them so an update shows at once.
        resp = await call_next(request)
        if not request.url.path.startswith("/api/"):
            resp.headers["Cache-Control"] = "no-cache"
        return resp

    if WEB_DIR.exists():
        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
    return app


def _safe(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except Exception as exc:
        log = kwargs.get("log")
        if log:
            log(f"FAILED: {exc}")


def _redact(d):
    if isinstance(d, dict):
        return {k: ("••••" if any(s in k.lower() for s in ("key", "password", "token", "secret")) and v else _redact(v))
                for k, v in d.items()}
    if isinstance(d, list):
        return [_redact(v) for v in d]
    return d
