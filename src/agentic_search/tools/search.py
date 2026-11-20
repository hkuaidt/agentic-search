"""The two supply chains exposed to the agent with the same ``(query, limit)`` signature.

* ``community_search`` -- full-community retrieval (the existing search API; long tail, fresh content,
  targeted verification).
* ``tag_search`` -- curated pool behind the three-level taxonomy. Online, a short sub-query is routed by
  a cascaded multi-label classifier (L1 BERT -> L3 BERT restricted to that L1's subtree), possibly to
  several (L1, L3) clusters; posts in those clusters are then ranked by query relevance. The curated pool
  is admitted by interaction score + human review and is *not* time-decayed.

In-memory implementations are provided for tests/demos; :class:`HTTPSearchBackend` adapts a real API.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import httpx

from agentic_search.core.text import BM25, tokenize
from agentic_search.core.types import Post, ToolResult, ToolStatus
from agentic_search.taxonomy.tree import TagTree
from agentic_search.tools.base import Tool


class _BM25Index:
    def __init__(self, posts: Sequence[Post]):
        self.posts = list(posts)
        self.bm25 = BM25([tokenize(f"{p.title} {p.content}") for p in self.posts]) if self.posts else None

    def search(self, query: str, limit: int, subset: Sequence[int] | None = None) -> list[Post]:
        if not self.bm25:
            return []
        scores = self.bm25.scores(tokenize(query))
        idx = list(range(len(self.posts))) if subset is None else list(subset)
        ranked = sorted((i for i in idx if scores[i] > 0), key=lambda i: -scores[i])
        return [self.posts[i] for i in ranked[:limit]]


class CommunitySearch(Tool):
    name = "community_search"
    description = "检索全社区可搜索内容（含未进入标签池的内容），用于长尾问题、新话题与具体信息补查。"

    def __init__(self, backend: Callable[[str, int], Any]):
        self.backend = backend  # sync or async callable (query, limit) -> list[Post]

    async def __call__(self, arguments: dict[str, Any], call_id: str = "") -> ToolResult:
        t0 = time.perf_counter()
        try:
            out = self.backend(arguments["query"], int(arguments.get("limit", 5)))
            posts = await out if inspect.isawaitable(out) else out
        except Exception as e:  # surfaced as infra failure, never as an empty result
            return ToolResult(call_id, self.name, status="error", error=repr(e))
        status: ToolStatus = "ok" if posts else "empty"
        return ToolResult(
            call_id, self.name, list(posts), status, latency_ms=(time.perf_counter() - t0) * 1e3
        )


class InMemoryCommunityBackend:
    def __init__(self, posts: Sequence[Post]):
        self.index = _BM25Index(posts)

    def __call__(self, query: str, limit: int) -> list[Post]:
        return self.index.search(query, limit)


class HTTPSearchBackend:
    """POST ``{"query", "limit"}`` -> ``{"posts": [{post_id, title, content, ...}]}``."""

    def __init__(self, url: str, timeout_s: float = 3.0, headers: dict[str, str] | None = None):
        self.url, self.timeout_s, self.headers = url, timeout_s, headers or {}

    async def __call__(self, query: str, limit: int) -> list[Post]:
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.post(self.url, json={"query": query, "limit": limit}, headers=self.headers)
            resp.raise_for_status()
            return [Post(**p) for p in resp.json().get("posts", [])]


class QueryTagRouter(Protocol):
    def route(self, query: str) -> list[tuple[str, str]]:
        """Return the (L1, L3) clusters a query maps to (multi-label)."""
        ...


class CascadeTagRouter:
    """Cascaded multi-label router: L1 classifier, then L3 classifier restricted to each chosen L1.

    ``l1_scores`` / ``l3_scores`` are any ``query -> {label: prob}`` callables -- in production two
    fine-tuned multi-label BERT heads; in tests a keyword scorer.
    """

    def __init__(
        self,
        tree: TagTree,
        l1_scores: Callable[[str], dict[str, float]],
        l3_scores: Callable[[str], dict[str, float]],
        l1_threshold: float = 0.5,
        l3_threshold: float = 0.5,
        max_routes: int = 4,
    ):
        self.tree, self.l1_scores, self.l3_scores = tree, l1_scores, l3_scores
        self.l1_threshold, self.l3_threshold, self.max_routes = l1_threshold, l3_threshold, max_routes

    def route(self, query: str) -> list[tuple[str, str]]:
        l1_probs = self.l1_scores(query)
        l3_probs = self.l3_scores(query)
        routes: list[tuple[float, str, str]] = []
        for l1, p1 in l1_probs.items():
            if p1 < self.l1_threshold or l1 not in self.tree.nodes:
                continue
            for l3 in self.tree.l3_under_l1(l1):
                p3 = l3_probs.get(l3, 0.0)
                if p3 >= self.l3_threshold:
                    routes.append((p1 * p3, l1, l3))
        routes.sort(reverse=True)
        return [(l1, l3) for _, l1, l3 in routes[: self.max_routes]]


def keyword_scorer(lexicon: dict[str, list[str]]) -> Callable[[str], dict[str, float]]:
    """Toy stand-in for a BERT head: score 1.0 for any label whose keywords appear in the query."""

    def score(query: str) -> dict[str, float]:
        q = query.lower()
        return {label: 1.0 for label, kws in lexicon.items() if any(k.lower() in q for k in kws)}

    return score


class TagPool:
    """Curated posts keyed by L3 tag (L1/L2 recovered from the tree)."""

    def __init__(self, posts: Sequence[Post]):
        self.posts = list(posts)
        self.by_l3: dict[str, list[int]] = {}
        for i, p in enumerate(self.posts):
            if len(p.tag_path) == 3:
                self.by_l3.setdefault(p.tag_path[2], []).append(i)
        self.index = _BM25Index(self.posts)


class TagSearch(Tool):
    name = "tag_search"
    description = "在经过互动分准入、打标与人工复核的精选标签内容池中检索，优先服务高频、主题明确的需求。"

    def __init__(self, router: QueryTagRouter, pool: TagPool, fallback_to_topic: bool = True):
        self.router, self.pool, self.fallback_to_topic = router, pool, fallback_to_topic

    async def __call__(self, arguments: dict[str, Any], call_id: str = "") -> ToolResult:
        t0 = time.perf_counter()
        query, limit = arguments["query"], int(arguments.get("limit", 5))
        try:
            routes = self.router.route(query)
            subset = sorted({i for _, l3 in routes for i in self.pool.by_l3.get(l3, [])})
            posts = self.pool.index.search(query, limit, subset) if subset else []
            if not posts and subset and self.fallback_to_topic:
                # topic matched but no lexical overlap: return the cluster's top curated posts
                posts = [self.pool.posts[i] for i in subset[:limit]]
        except Exception as e:
            return ToolResult(call_id, self.name, status="error", error=repr(e))
        status: ToolStatus = "ok" if posts else "empty"
        return ToolResult(call_id, self.name, posts, status, latency_ms=(time.perf_counter() - t0) * 1e3)
