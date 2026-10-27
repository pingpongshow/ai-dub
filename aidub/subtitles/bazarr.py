"""Bazarr as a subtitle source.

Ai Dub asks Bazarr which subtitles it has for a title, and can ask it to search
its own providers for a missing language. Bazarr saves what it downloads as a
sidecar file next to the video, so after a search the sidecar scan picks it up.

Bazarr sees the media under its own paths, so `path_map` translates:
  path_map: [{ aidub: /media/shows, bazarr: /tv }, ...]
"""

from __future__ import annotations

from pathlib import Path

import httpx


class Bazarr:
    def __init__(self, url: str, api_key: str, path_map: list[dict] | None = None, timeout: float = 300):
        if not url or not api_key:
            raise ValueError("Bazarr needs a URL and an API key")
        self.url = url.rstrip("/")
        self.client = httpx.Client(base_url=self.url + "/api", headers={"X-API-KEY": api_key}, timeout=timeout)
        self.path_map = path_map or []

    def close(self) -> None:
        self.client.close()

    def to_bazarr(self, path: str) -> str:
        for m in self.path_map:
            a, b = m.get("aidub", "").rstrip("/"), m.get("bazarr", "").rstrip("/")
            if a and path.startswith(a + "/"):
                return b + path[len(a):]
        return path

    def _get(self, path: str, **params) -> dict:
        r = self.client.get(path, params=params)
        r.raise_for_status()
        return r.json()

    def status(self) -> dict:
        return self._get("/system/status").get("data", {})

    def find(self, path: str) -> dict | None:
        """The Bazarr record for a file: {"type": "episode"|"movie", ids..., "subtitles": [...]}."""
        target = self.to_bazarr(path)
        name = Path(target).name
        parent = str(Path(target).parent)
        # Movies first: one call lists them all.
        for m in self._get("/movies", length=-1).get("data", []):
            if m.get("path") == target or (m.get("path") or "").endswith("/" + name):
                return {"type": "movie", "radarrid": m["radarrId"], "subtitles": m.get("subtitles", []),
                        "title": m.get("title", "")}
        for s in self._get("/series", length=-1).get("data", []):
            sp = (s.get("path") or "").rstrip("/")
            if not sp or not (parent == sp or parent.startswith(sp + "/")):
                continue
            eps = self._get("/episodes", **{"seriesid[]": s["sonarrSeriesId"]}).get("data", [])
            for e in eps:
                if e.get("path") == target or (e.get("path") or "").endswith("/" + name):
                    return {"type": "episode", "seriesid": s["sonarrSeriesId"], "episodeid": e["sonarrEpisodeId"],
                            "subtitles": e.get("subtitles", []), "title": f"{s.get('title')} {e.get('title', '')}"}
        return None

    def search(self, rec: dict, language: str, hi: bool = False) -> None:
        """Ask Bazarr to search its providers and download the best subtitle."""
        params = {"language": language, "forced": "False", "hi": "True" if hi else "False"}
        if rec["type"] == "movie":
            r = self.client.patch("/movies/subtitles", params={"radarrid": rec["radarrid"], **params})
        else:
            r = self.client.patch("/episodes/subtitles",
                                  params={"seriesid": rec["seriesid"], "episodeid": rec["episodeid"], **params})
        r.raise_for_status()
