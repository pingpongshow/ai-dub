"""Starting model workers and talking to them.

Each worker lives in `workers/<name>/` with a `worker.py` and a
`requirements.txt`, and runs in its own virtual environment under
`<models>/envs/<name>`. `aidub setup` builds those environments.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Callable

from .config import Config

DEFAULT_WORKERS = Path(__file__).resolve().parent.parent / "workers"


def workers_root() -> Path:
    return Path(os.environ.get("AIDUB_WORKERS") or DEFAULT_WORKERS)


class WorkerError(RuntimeError):
    pass


def worker_dir(name: str) -> Path:
    return workers_root() / name


def env_dir(cfg: Config, name: str) -> Path:
    return cfg.models / "envs" / name


def requirements_hash(name: str) -> str:
    h = hashlib.sha256()
    for f in ("requirements.txt", "setup.sh"):
        p = worker_dir(name) / f
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def env_ready(cfg: Config, name: str) -> bool:
    stamp = env_dir(cfg, name) / ".aidub-ready"
    return stamp.exists() and stamp.read_text().strip() == requirements_hash(name)


def python_for(cfg: Config, name: str) -> str:
    override = os.environ.get("AIDUB_WORKER_PYTHON")  # tests and development
    if override:
        return override
    py = env_dir(cfg, name) / "bin" / "python"
    if not py.exists() or not env_ready(cfg, name):
        raise WorkerError(f"model '{name}' is not installed: install it on the Models page "
                          f"(or run `aidub setup --download {name}`)")
    return str(py)


class Worker:
    """One running worker process. Use as a context manager."""

    def __init__(self, cfg: Config, name: str, *, log: Callable[[str], None] | None = None,
                 env: dict | None = None):
        self.cfg = cfg
        self.name = name
        self.log = log or (lambda line: None)
        self.extra_env = env or {}
        self.proc: subprocess.Popen | None = None
        self._next = 0
        self._stderr_thread: threading.Thread | None = None

    def __enter__(self) -> "Worker":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self) -> None:
        script = worker_dir(self.name) / "worker.py"
        if not script.exists():
            raise WorkerError(f"worker '{self.name}' does not exist")
        env = dict(os.environ)
        common = workers_root() / "_common"
        if not common.exists():  # stand-in workers in tests share the real runtime
            common = DEFAULT_WORKERS / "_common"
        env.update({
            "PYTHONPATH": str(common),
            "PYTHONUNBUFFERED": "1",
            "AIDUB_MODELS": str(self.cfg.models),
            "HF_HOME": str(self.cfg.models / "hf"),
            "MODELSCOPE_CACHE": str(self.cfg.models / "modelscope"),
            "TORCH_HOME": str(self.cfg.models / "torch"),
            "XDG_CACHE_HOME": str(self.cfg.models / "cache"),
            "PADDLE_PDX_CACHE_HOME": str(self.cfg.models / "paddlex"),
            "AUDIO_SEPARATOR_MODEL_DIR": str(self.cfg.models / "audio-separator"),
        })
        if env.get("HOME", "/") in ("/", ""):
            env["HOME"] = str(self.cfg.appdata / "home")
        hf_token = self.cfg.secrets.get("HF_TOKEN") or os.environ.get("HF_TOKEN")
        if hf_token:
            env["HF_TOKEN"] = hf_token
        env.update(self.extra_env)
        self.proc = subprocess.Popen(
            [python_for(self.cfg, self.name), str(script)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, env=env, cwd=str(worker_dir(self.name)),
        )
        self._stderr_thread = threading.Thread(target=self._pump_stderr, daemon=True)
        self._stderr_thread.start()
        try:
            hello = self._read()
        except BaseException:
            self.kill()
            raise
        if not hello.get("ok"):
            self.kill()
            raise WorkerError(f"worker '{self.name}' failed to start: {hello.get('error')}")

    def _pump_stderr(self) -> None:
        assert self.proc and self.proc.stderr
        for line in self.proc.stderr:
            self.log(f"[{self.name}] {line.rstrip()}")

    def _read(self) -> dict:
        assert self.proc and self.proc.stdout
        line = self.proc.stdout.readline()
        if not line:
            code = self.proc.wait(timeout=10)
            raise WorkerError(f"worker '{self.name}' exited (code {code}); see the job log")
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            raise WorkerError(f"worker '{self.name}' sent a bad reply: {line[:200]!r}")

    def call(self, op: str, args: dict | None = None, *,
             on_progress: Callable[[float, str], None] | None = None) -> Any:
        if not self.proc or self.proc.poll() is not None:
            raise WorkerError(f"worker '{self.name}' is not running")
        self._next += 1
        req = {"id": self._next, "op": op, "args": args or {}}
        assert self.proc.stdin
        self.proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        while True:
            msg = self._read()
            if "progress" in msg:
                if on_progress:
                    on_progress(float(msg["progress"]), msg.get("message", ""))
                continue
            if msg.get("id") != self._next:
                continue
            if not msg.get("ok"):
                raise WorkerError(f"{self.name}.{op}: {msg.get('error')}")
            return msg.get("result")

    def stop(self) -> None:
        if not self.proc:
            return
        if self.proc.poll() is None:
            try:
                self.call("exit")
            except Exception:
                pass
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.proc = None

    def kill(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.kill()


def available() -> list[str]:
    return sorted(p.name for p in workers_root().iterdir()
                  if p.is_dir() and not p.name.startswith(("_", ".")) and (p / "worker.py").exists())


def _run(cmd: list[str], env: dict, cwd: str, log: Callable[[str], None]) -> None:
    """Run a setup command, streaming its output to `log`."""
    proc = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.stdout
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            log(line)
    if proc.wait() != 0:
        raise WorkerError(f"{Path(cmd[0]).name} exited with {proc.returncode}")


def setup(cfg: Config, name: str, *, log: Callable[[str], None] = print, force: bool = False) -> None:
    """Create or refresh one worker's environment with uv."""
    wdir = worker_dir(name)
    target = env_dir(cfg, name)
    if env_ready(cfg, name) and not force:
        log(f"{name}: up to date")
        return
    manifest = {}
    mf = wdir / "worker.json"
    if mf.exists():
        manifest = json.loads(mf.read_text())
    python = manifest.get("python", "3.11")
    uv = os.environ.get("UV", "uv")
    env = dict(os.environ)
    env.setdefault("UV_CACHE_DIR", str(cfg.models / "uv-cache"))
    env.setdefault("UV_PYTHON_INSTALL_DIR", str(cfg.models / "python"))
    env["UV_LINK_MODE"] = "hardlink"
    target.parent.mkdir(parents=True, exist_ok=True)
    log(f"{name}: creating environment (Python {python})")
    _run([uv, "venv", "--clear", "--python", python, str(target)], env, str(wdir), log)
    env["VIRTUAL_ENV"] = str(target)
    req = wdir / "requirements.txt"
    if req.exists():
        cmd = [uv, "pip", "install", "--python", str(target / "bin" / "python"), "-r", str(req)]
        for extra in manifest.get("index_urls", []):
            cmd += ["--extra-index-url", extra]
        if manifest.get("index_strategy"):
            cmd += ["--index-strategy", manifest["index_strategy"]]
        log(f"{name}: installing packages")
        _run(cmd, env, str(wdir), log)
    script = wdir / "setup.sh"
    if script.exists():
        log(f"{name}: running setup.sh")
        _run(["bash", str(script), str(target)], env, str(wdir), log)
    (target / ".aidub-ready").write_text(requirements_hash(name))
    log(f"{name}: ready")


def prefetch(cfg: Config, name: str, args: dict, log: Callable[[str], None] = print) -> Any:
    """Ask a worker to download its weights (op `prefetch`), without a GPU job."""
    with Worker(cfg, name, log=log) as w:
        result = w.call("prefetch", args)
    (env_dir(cfg, name) / ".aidub-weights").write_text(json.dumps({"model": args.get("model", "")}))
    return result


def python_exe() -> str:
    return sys.executable
