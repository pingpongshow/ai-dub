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


def link(vectors: list, threshold: float, cannot_link: set[tuple[int, int]] | None = None,
         must_link: list[tuple[int, int]] | None = None) -> list[int]:
    """Average-linkage clustering by cosine similarity. Items in `cannot_link`
    (for example two speakers diarized apart within one window) never merge; items in
    `must_link` (the same face talking in both) start as one cluster, unless that would
    join a cannot-link pair. Returns a cluster number per item."""
    n = len(vectors)
    if n == 0:
        return []
    x = normalize(vectors)
    sim = x @ x.T
    clusters: dict[int, list[int]] = {i: [i] for i in range(n)}
    forbid = cannot_link or set()

    def blocked(a: list[int], b: list[int]) -> bool:
        return any((i, j) in forbid or (j, i) in forbid for i in a for j in b)

    home = list(range(n))
    for i, j in must_link or []:
        a, b = home[i], home[j]
        if a == b or blocked(clusters[a], clusters[b]):
            continue
        for k in clusters[b]:
            home[k] = a
        clusters[a].extend(clusters.pop(b))

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


def absorb_small(vectors: list, labels: list[int], seconds: list[float], genders: list[str],
                 cannot_link: set[tuple[int, int]], min_s: float, threshold: float) -> list[int]:
    """Clusters with less than `min_s` of speech join the main cluster they sound most like
    (average likeness at least `threshold`), smallest first. Never one of the other gender,
    nor one already talking in the same window (cannot-link). A few seconds of one person,
    recorded in another scene, rarely reach the linking threshold on their own: on two films
    this halved the speakers found, with no more people wrongly joined."""
    labels = list(labels)
    if not labels:
        return labels
    x = normalize(vectors)
    sim = x @ x.T
    members: dict[int, list[int]] = {}
    for i, lab in enumerate(labels):
        members.setdefault(lab, []).append(i)
    total = {lab: sum(seconds[i] for i in m) for lab, m in members.items()}

    def gender(m: list[int]) -> str:
        votes: dict[str, float] = {}
        for i in m:
            if genders[i]:
                votes[genders[i]] = votes.get(genders[i], 0.0) + seconds[i]
        return max(votes, key=votes.get) if votes else ""

    big = [lab for lab in members if total[lab] >= min_s]
    for lab in sorted(members, key=lambda k: total[k]):
        if total[lab] >= min_s or lab not in members:
            continue
        mine, g = members[lab], gender(members[lab])
        best, best_sim = None, -2.0
        for b in big:
            theirs = members[b]
            if any((i, j) in cannot_link or (j, i) in cannot_link for i in mine for j in theirs):
                continue
            gb = gender(theirs)
            if g and gb and g != gb:
                continue
            s = float(sim[np.ix_(mine, theirs)].mean())
            if s > best_sim:
                best, best_sim = b, s
        if best is not None and best_sim >= threshold:
            for i in mine:
                labels[i] = best
            members[best] = members[best] + members.pop(lab)
    return labels
