"""Three-level taxonomy construction: top-down mounting + bottom-up multi-model induction.

    L1 (50+, defined by domain experts)
      | mount every effective interest point to an L1                      (top-down, LLM classifier)
      | per L1: cluster + uniform sample to ~1200 points
      | GPT / Claude / Gemini each propose L2 names; merge + dedupe         (bottom-up, cross-family)
      | judge reflection: covers the points? mutually exclusive? else retry
    L2 (700+)
      | same recursion per L2 with ~300 sampled points
    L3 (5000+)  -> granularity calibration (split too-coarse, merge too-fine) -> coverage test

This is the workflow version; a harness/self-evolving version (persistent memory of judge verdicts,
auto-learned split/merge rules) is the extension listed in PROPOSAL §7.2. All LLM roles are protocols
so the same code runs with real models or with the deterministic embedding stand-ins below.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from agentic_search.core.text import Embedder, normalize_text
from agentic_search.taxonomy.funnel import run_funnel, uniform_cluster_sample
from agentic_search.taxonomy.tree import TagTree


class Mounter(Protocol):
    def assign(self, point: str, candidates: Sequence[str]) -> str | None:
        """Pick the best candidate tag for an interest point, or ``None`` if nothing fits."""
        ...


class Inducer(Protocol):
    family: str

    def induce(self, parent: str, points: Sequence[str], max_children: int, level: int) -> list[str]:
        """Propose level-``level`` child tag names that abstract ``points`` under ``parent``."""
        ...


class Reflector(Protocol):
    def review(self, parent: str, points: Sequence[str], children: Sequence[str]) -> tuple[bool, str]:
        """Check coverage + mutual exclusivity; return (ok, feedback)."""
        ...


@dataclass
class LevelConfig:
    sample_budget: int  # 1200 for L2 induction, 300 for L3
    max_children: int
    max_reflection_rounds: int = 2


@dataclass
class BuildReport:
    mounted: dict[str, int] = field(default_factory=dict)
    unmounted: list[str] = field(default_factory=list)
    reflection_failures: list[str] = field(default_factory=list)


class TaxonomyBuilder:
    def __init__(
        self,
        embedder: Embedder,
        mounter: Mounter,
        inducers: Sequence[Inducer],
        reflector: Reflector | None = None,
        l2: LevelConfig | None = None,
        l3: LevelConfig | None = None,
        dedupe_threshold: float = 0.15,
    ):
        if not inducers:
            raise ValueError("need at least one inducer (ideally from different model families)")
        self.embedder, self.mounter, self.inducers, self.reflector = (
            embedder,
            mounter,
            list(inducers),
            reflector,
        )
        self.l2 = l2 or LevelConfig(sample_budget=1200, max_children=20)
        self.l3 = l3 or LevelConfig(sample_budget=300, max_children=12)
        self.dedupe_threshold = dedupe_threshold

    def build(self, l1_tags: Sequence[str], points: Sequence[str]) -> tuple[TagTree, BuildReport]:
        tree, report = TagTree(), BuildReport()
        for l1 in l1_tags:
            tree.add(l1, 1)
        by_l1 = self._mount(points, list(l1_tags), report)
        for l1, pts in by_l1.items():
            l2_children = self._induce_level(l1, pts, self.l2, report, taken=set(tree.nodes), level=2)
            for name in l2_children:
                tree.add(name, 2, l1)
            by_l2 = self._mount(pts, l2_children, report)
            for l2, pts2 in by_l2.items():
                for name in self._induce_level(l2, pts2, self.l3, report, taken=set(tree.nodes), level=3):
                    tree.add(name, 3, l2)
        return tree, report

    # -- steps --------------------------------------------------------------------------------
    def _mount(
        self, points: Sequence[str], candidates: list[str], report: BuildReport
    ) -> dict[str, list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for p in points:
            tag = self.mounter.assign(p, candidates) if candidates else None
            if tag is None:
                report.unmounted.append(p)
            else:
                out[tag].append(p)
        for tag, pts in out.items():
            report.mounted[tag] = len(pts)
        return dict(out)

    def _induce_level(
        self,
        parent: str,
        points: list[str],
        cfg: LevelConfig,
        report: BuildReport,
        taken: set[str],
        level: int,
    ) -> list[str]:
        sample = uniform_cluster_sample(points, self.embedder, cfg.sample_budget)
        feedback = ""
        children: list[str] = []
        for _ in range(cfg.max_reflection_rounds + 1):
            proposals = [
                n
                for ind in self.inducers
                for n in ind.induce(parent + feedback, sample, cfg.max_children, level)
            ]
            children = self._merge(proposals, taken, cfg.max_children)
            if self.reflector is None:
                return children
            ok, msg = self.reflector.review(parent, sample, children)
            if ok:
                return children
            feedback = f"（上轮问题：{msg}）"
        report.reflection_failures.append(parent)  # route to human review
        return children

    def _merge(self, proposals: list[str], taken: set[str], max_children: int) -> list[str]:
        """Cross-model merge: vote by normalised name, then collapse near-duplicates by embedding."""
        votes: dict[str, int] = defaultdict(int)
        surface: dict[str, str] = {}
        for name in proposals:
            key = normalize_text(name)
            if key and name not in taken:
                votes[key] += 1
                surface.setdefault(key, name.strip())
        ranked = sorted(votes, key=lambda k: (-votes[k], k))
        if not ranked:
            return []
        emb = self.embedder.encode(ranked)
        kept: list[int] = []
        for i in range(len(ranked)):
            if all(float(emb[i] @ emb[j]) < 1 - self.dedupe_threshold for j in kept):
                kept.append(i)
        return [surface[ranked[i]] for i in kept[:max_children]]


# -- deterministic stand-ins (tests / offline demos) ---------------------------------------------
class EmbeddingMounter:
    """Nearest candidate by cosine similarity of names (an LLM classifier in production)."""

    def __init__(self, embedder: Embedder, min_sim: float = 0.05, descriptions: dict[str, str] | None = None):
        self.embedder, self.min_sim, self.descriptions = embedder, min_sim, descriptions or {}

    def assign(self, point: str, candidates: Sequence[str]) -> str | None:
        texts = [f"{c} {self.descriptions.get(c, '')}" for c in candidates]
        sims = self.embedder.encode(texts) @ self.embedder.encode([point])[0]
        j = int(np.argmax(sims))
        return candidates[j] if sims[j] >= self.min_sim else None


class ClusterInducer:
    """Names each cluster of the sampled points by a member (an LLM in production). Uses a coarser
    clustering threshold for L2 than for L3, mirroring the per-level prompts of the LLM inducers."""

    def __init__(
        self, embedder: Embedder, family: str = "cluster", thresholds: dict[int, float] | None = None
    ):
        self.embedder, self.family = embedder, family
        self.thresholds = thresholds or {2: 0.6, 3: 0.2}

    def induce(self, parent: str, points: Sequence[str], max_children: int, level: int) -> list[str]:
        res = run_funnel(points, self.embedder, self.thresholds[level], method="hierarchical")
        parent_key = normalize_text(parent.split("（")[0])
        names = []
        for c in res.clusters:
            options = [m for m in sorted(c.members, key=len) if m != parent_key]
            if options:
                names.append(c.representative if c.representative != parent_key else options[0])
        return names[:max_children]


def coverage_test(old_tags: Sequence[str], tree: TagTree, mounter: Mounter, level: int = 3) -> float:
    """Mount a random sample of legacy hashtags (~100 in the PROPOSAL setup) onto the new tree; the share
    that finds a home estimates coverage of the old content-understanding system."""
    leaves = tree.level(level)
    if not old_tags or not leaves:
        return 0.0
    return sum(mounter.assign(t, leaves) is not None for t in old_tags) / len(old_tags)


def granularity_report(mounted: dict[str, int], min_points: int, max_points: int) -> dict[str, list[str]]:
    """Flag tags to split (too coarse) or merge (too fine) during calibration."""
    return {
        "split": sorted(t for t, n in mounted.items() if n > max_points),
        "merge": sorted(t for t, n in mounted.items() if n < min_points),
    }
