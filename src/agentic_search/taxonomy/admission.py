r"""Admission into the curated tag pool, and blank-tag driven taxonomy evolution.

Not everything is tagged. Over one configured statistics window::

    S = w_l * likes + w_f * favorites + w_s * shares + w_c * comments        (weights >= 0)

A post enters the tagging queue iff ``S > tau`` (``S == tau`` does not trigger). Weights and ``tau`` are
set by content operations, per vertical if needed. Stock content is scored in batch once; new content is
re-scored periodically and enqueued the first time it crosses the threshold, so a post that starts slow
is not locked out forever. Admitted posts are human-reviewed and never time-decayed; official pinned
posts are force-inserted. Posts below the threshold remain reachable via community search.

When the L3 tagger outputs the blank tag for enough new posts under one L1, that is both an alert about
interest drift and a trigger to extract new interest points and extend L3 (and retrain the L3 tagger).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from agentic_search.taxonomy.tree import BLANK_TAG

METRIC_KEYS = ("likes", "favorites", "shares", "comments")


@dataclass(frozen=True)
class InteractionWeights:
    likes: float = 1.0
    favorites: float = 2.0
    shares: float = 3.0
    comments: float = 1.5

    def __post_init__(self) -> None:
        if min(self.likes, self.favorites, self.shares, self.comments) < 0:
            raise ValueError("interaction weights must be non-negative")


def interaction_score(metrics: Mapping[str, float], w: InteractionWeights) -> float:
    return (
        w.likes * metrics.get("likes", 0)
        + w.favorites * metrics.get("favorites", 0)
        + w.shares * metrics.get("shares", 0)
        + w.comments * metrics.get("comments", 0)
    )


@dataclass
class AdmissionController:
    tau: float
    weights: InteractionWeights = field(default_factory=InteractionWeights)
    admitted: set[str] = field(default_factory=set)
    pinned: set[str] = field(default_factory=set)

    def admits(self, metrics: Mapping[str, float]) -> bool:
        return interaction_score(metrics, self.weights) > self.tau

    def batch_stock(self, posts: Iterable[tuple[str, Mapping[str, float]]]) -> list[str]:
        """One-off pass over existing content."""
        return self._enqueue(posts)

    def rescore_incremental(self, posts: Iterable[tuple[str, Mapping[str, float]]]) -> list[str]:
        """Periodic pass over recent content; returns posts crossing the threshold for the first time."""
        return self._enqueue(posts)

    def pin_official(self, post_ids: Iterable[str]) -> list[str]:
        new = [p for p in post_ids if p not in self.admitted]
        self.pinned.update(new)
        self.admitted.update(new)
        return new

    def _enqueue(self, posts: Iterable[tuple[str, Mapping[str, float]]]) -> list[str]:
        out = [pid for pid, m in posts if pid not in self.admitted and self.admits(m)]
        self.admitted.update(out)
        return out


@dataclass
class BlankTagMonitor:
    """Accumulates blank-tag predictions per L1 and fires once a bucket reaches ``trigger_count``."""

    trigger_count: int = 500
    buckets: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    def observe(self, post_id: str, l1: str, l3_prediction: str) -> list[str] | None:
        if l3_prediction != BLANK_TAG:
            return None
        bucket = self.buckets[l1]
        bucket.append(post_id)
        if len(bucket) >= self.trigger_count:
            batch, self.buckets[l1] = list(bucket), []
            return batch  # -> interest-point extraction -> extend L3 -> retrain L3 tagger
        return None
