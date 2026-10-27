"""Installing models from inside the app.

Nothing is bundled: each model's Python environment is built and its weights
downloaded from the model's official source into the models folder, when you
ask (Models page, `aidub setup --download`, or AIDUB_INSTALL_ON_START).
Installs run one at a time in the background.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from collections import deque

from . import workers
from .config import Config


class Installer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.q: queue.Queue[str] = queue.Queue()
        self.state: dict[str, dict] = {}
        self.logs: dict[str, deque] = {}
        self._thread = threading.Thread(target=self._loop, name="aidub-installer", daemon=True)
        self._thread.start()

    def worker_names(self) -> list[str]:
        used = {self.cfg.get(f"stages.{s}.worker") for s in self.cfg.get("stages", {})}
        return [n for n in workers.available() if n in used]

    def stage_args(self, name: str) -> dict:
        for s in self.cfg.get("stages", {}):
            if self.cfg.get(f"stages.{s}.worker") == name:
                return dict(self.cfg.get(f"stages.{s}") or {})
        return {}

    def weights_ready(self, name: str) -> bool:
        return (workers.env_dir(self.cfg, name) / ".aidub-weights").exists()

    def listing(self) -> list[dict]:
        out = []
        for n in self.worker_names():
            meta = {}
            mf = workers.worker_dir(n) / "worker.json"
            if mf.exists():
                meta = json.loads(mf.read_text())
            stages = [s for s in self.cfg.get("stages", {}) if self.cfg.get(f"stages.{s}.worker") == n]
            out.append({"name": n, "title": meta.get("title", n), "models": meta.get("models", ""),
                        "licence": meta.get("licence", ""), "download": meta.get("download", ""),
                        "stages": stages, "model": self.stage_args(n).get("model", ""),
                        "installed": workers.env_ready(self.cfg, n), "weights": self.weights_ready(n),
                        **self.state.get(n, {"job": "idle"})})
        return out

    def request(self, name: str) -> None:
        if name not in self.worker_names():
            raise KeyError(name)
        if self.state.get(name, {}).get("job") in ("queued", "installing"):
            return
        self.state[name] = {"job": "queued", "since": time.time()}
        self.q.put(name)

    def request_missing(self) -> list[str]:
        names = [n for n in self.worker_names() if not (workers.env_ready(self.cfg, n) and self.weights_ready(n))]
        for n in names:
            self.request(n)
        return names

    def log(self, name: str) -> str:
        return "\n".join(self.logs.get(name, []))

    def _loop(self) -> None:
        while True:
            name = self.q.get()
            log = self.logs.setdefault(name, deque(maxlen=400))
            say = lambda line: log.append(f"{time.strftime('%H:%M:%S')} {line}")
            self.state[name] = {"job": "installing", "since": time.time()}
            try:
                workers.setup(self.cfg, name, log=say)
                say("downloading weights (this can take a while)")
                workers.prefetch(self.cfg, name, self.stage_args(name), log=say)
                self.state[name] = {"job": "done", "since": time.time()}
                say("ready")
            except Exception as exc:
                self.state[name] = {"job": "failed", "since": time.time(), "error": str(exc)[:500]}
                say(f"FAILED: {exc}")
