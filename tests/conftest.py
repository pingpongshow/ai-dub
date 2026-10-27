import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """A config pointing at temporary folders and the stand-in workers."""
    from aidub import config
    media = tmp_path / "media"
    (media / "Anime").mkdir(parents=True)
    conf = {
        "paths": {"appdata": str(tmp_path / "appdata"), "models": str(tmp_path / "models"),
                  "work": str(tmp_path / "work")},
        "media": [{"name": "Anime", "path": str(media / "Anime")}],
        "gpu": {"wait": False},
        "speak": {"takes": 2},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(conf))
    monkeypatch.setenv("AIDUB_WORKERS", str(Path(__file__).resolve().parent / "fakes"))
    monkeypatch.setenv("AIDUB_WORKER_PYTHON", sys.executable)
    return config.load(path)


@pytest.fixture
def media_root(cfg) -> Path:
    return Path(cfg.media_roots[0]["path"])
