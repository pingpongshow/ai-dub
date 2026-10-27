"""Speaker-embedding helpers: cosine similarity and agglomerative linking."""

from __future__ import annotations

import numpy as np


def normalize(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


def cosine(a, b) -> float:
    return float(np.dot(normalize(a), normalize(b)))


def mean_embedding(vectors: list, weights: list[float] | None = None) -> list[float]:
    if not vectors:
        return []
    m = np.asarray(vectors, dtype=np.float32)
    w = np.asarray(weights if weights else [1.0] * len(vectors), dtype=np.float32)[:, None]
    return normalize((normalize(m) * w).sum(axis=0)).tolist()


def link(vectors: list, threshold: float, cannot_link: set[tuple[int, int]] | None = None) -> list[int]:
    """Average-linkage clustering by cosine similarity. Items in `cannot_link`
    (for example two speakers diarized apart within one window) never merge.
    Returns a cluster number per item."""
    n = len(vectors)
    if n == 0:
        return []
    x = normalize(vectors)
    sim = x @ x.T
    clusters: dict[int, list[int]] = {i: [i] for i in range(n)}
    forbid = cannot_link or set()

    def blocked(a: list[int], b: list[int]) -> bool:
        return any((i, j) in forbid or (j, i) in forbid for i in a for j in b)

    while len(clusters) > 1:
        best, pair = -2.0, None
        keys = list(clusters)
        for ai in range(len(keys)):
            for bi in range(ai + 1, len(keys)):
                a, b = clusters[keys[ai]], clusters[keys[bi]]
                s = float(sim[np.ix_(a, b)].mean())
                if s > best and not blocked(a, b):
                    best, pair = s, (keys[ai], keys[bi])
        if pair is None or best < threshold:
            break
        a, b = pair
        clusters[a].extend(clusters.pop(b))
    labels = [0] * n
    for c, (_, members) in enumerate(sorted(clusters.items(), key=lambda kv: -len(kv[1]))):
        for i in members:
            labels[i] = c
    return labels
