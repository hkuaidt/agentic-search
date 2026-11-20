"""Interest-point reduction funnel: raw points -> exact dedup -> embedding clustering -> effective points.

Reference scale (PROPOSAL §4.1(2)): 10M+ seed posts -> 30M+ raw points -> 4.1M+ unique (exact match)
-> 240K+ effective
(clustering merges synonyms such as 沈腾 / 沈腾电影 / 沈腾喜剧人; humans drop bad clusters).

``hierarchical`` (scipy average-linkage, cosine) is exact but O(n^2) memory -- use it per shard or per
L1 bucket. ``leader`` is a single-pass threshold clustering that scales linearly (blocked matrix ops)
and is what to run on millions of points before an optional hierarchical refinement.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from agentic_search.core.text import Embedder, normalize_text


@dataclass
class Cluster:
    representative: str
    members: list[str]
    count: int  # total raw occurrences across members


@dataclass
class FunnelResult:
    clusters: list[Cluster]
    stats: dict[str, int] = field(default_factory=dict)

    @property
    def effective_points(self) -> list[str]:
        return [c.representative for c in self.clusters]


def exact_dedup(points: Iterable[str]) -> Counter[str]:
    """Literal dedup after normalisation (what the data-warehouse job does with a GROUP BY)."""
    counts: Counter[str] = Counter()
    for p in points:
        key = normalize_text(p)
        if key:
            counts[key] += 1
    return counts


def hierarchical_clusters(emb: np.ndarray, distance_threshold: float) -> np.ndarray:
    if len(emb) == 1:
        return np.zeros(1, dtype=int)
    z = linkage(emb, method="average", metric="cosine")
    return fcluster(z, t=distance_threshold, criterion="distance") - 1


def leader_clusters(emb: np.ndarray, distance_threshold: float, block: int = 4096) -> np.ndarray:
    """Assign each point to the first leader within ``distance_threshold`` (cosine), else make it a leader.
    Points should be ordered by frequency so leaders are the most common surface forms."""
    labels = np.full(len(emb), -1, dtype=int)
    leaders: list[np.ndarray] = []
    leader_mat = np.zeros((0, emb.shape[1]), dtype=emb.dtype)
    sim_min = 1.0 - distance_threshold
    for start in range(0, len(emb), block):
        for i in range(start, min(start + block, len(emb))):
            if len(leaders):
                sims = leader_mat @ emb[i]
                j = int(np.argmax(sims))
                if sims[j] >= sim_min:
                    labels[i] = j
                    continue
            labels[i] = len(leaders)
            leaders.append(emb[i])
            leader_mat = np.vstack([leader_mat, emb[i][None]])
    return labels


def run_funnel(
    raw_points: Iterable[str],
    embedder: Embedder,
    distance_threshold: float = 0.35,
    method: Literal["hierarchical", "leader", "auto"] = "auto",
    max_exact: int = 20_000,
    reviewer: Callable[[Cluster], bool] | None = None,
) -> FunnelResult:
    raw = list(raw_points)
    counts = exact_dedup(raw)
    unique = [p for p, _ in counts.most_common()]
    if not unique:
        return FunnelResult([], {"raw": len(raw), "unique": 0, "clusters": 0, "effective": 0})
    emb = embedder.encode(unique)
    use = method if method != "auto" else ("hierarchical" if len(unique) <= max_exact else "leader")
    labels = (
        hierarchical_clusters(emb, distance_threshold)
        if use == "hierarchical"
        else leader_clusters(emb, distance_threshold)
    )

    groups: dict[int, list[str]] = {}
    for p, lab in zip(unique, labels):
        groups.setdefault(int(lab), []).append(p)
    clusters = []
    for members in groups.values():
        rep = max(members, key=lambda m: (counts[m], -len(m)))  # most frequent, then shortest
        clusters.append(Cluster(rep, members, sum(counts[m] for m in members)))
    clusters.sort(key=lambda c: -c.count)
    n_clusters = len(clusters)
    if reviewer is not None:  # human-in-the-loop removal of bad clusters
        clusters = [c for c in clusters if reviewer(c)]
    return FunnelResult(
        clusters, {"raw": len(raw), "unique": len(unique), "clusters": n_clusters, "effective": len(clusters)}
    )


def uniform_cluster_sample(
    points: Sequence[str], embedder: Embedder, budget: int, n_clusters: int | None = None, seed: int = 0
) -> list[str]:
    """Shrink a parent's points to ``budget`` (1200 for L2 induction, 300 for L3) while keeping topical
    coverage: cluster, then draw round-robin from every cluster so small topics are not crowded out."""
    if len(points) <= budget:
        return list(points)
    rng = np.random.default_rng(seed)
    k = n_clusters or max(2, min(budget // 4, len(points) // 2))
    emb = embedder.encode(list(points))
    labels = leader_clusters(emb, 0.5) if len(points) > 20_000 else _kmeans(emb, k, rng)
    buckets: dict[int, list[int]] = {}
    for i, lab in enumerate(labels):
        buckets.setdefault(int(lab), []).append(i)
    for b in buckets.values():
        rng.shuffle(b)
    out: list[str] = []
    while len(out) < budget:
        for b in list(buckets.values()):
            if b and len(out) < budget:
                out.append(points[b.pop()])
        if not any(buckets.values()):
            break
    return out


def _kmeans(emb: np.ndarray, k: int, rng: np.random.Generator, iters: int = 20) -> np.ndarray:
    centers = emb[rng.choice(len(emb), size=k, replace=False)]
    labels = np.zeros(len(emb), dtype=int)
    for _ in range(iters):
        labels = np.argmax(emb @ centers.T, axis=1)
        for j in range(k):
            sel = emb[labels == j]
            if len(sel):
                c = sel.mean(0)
                centers[j] = c / max(np.linalg.norm(c), 1e-12)
    return labels
