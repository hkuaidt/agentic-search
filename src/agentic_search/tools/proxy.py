"""Training-dedicated search proxy.

During online RL every prompt is rolled out G=8 times and each rollout fans out horizontally into
several ``tag_search`` / ``community_search`` calls, so rollout produces a request *burst* that would
queue, time out and fail against production search. The proxy sits between the VERL tool executor and
the search backends and implements the five mechanisms of PROPOSAL §5.2:

1. **Bounded queue + backpressure** -- callers block (or are rejected) when the queue is full; queued
   items carry ``(job_id, trajectory_id, call_id)`` and are dropped when their trajectory is cancelled
   or they exceed ``max_queue_wait_s``.
2. **Per-tool rate limits** -- token bucket (rate, burst) + ``max_inflight`` per tool. Run one proxy
   per backend cluster (e.g. as a single Ray actor / service) so limits are global across rollout
   workers rather than per worker.
3. **Exact cache + singleflight** -- identical in-flight requests are coalesced; results are cached under
   ``(tool, query, limit, filters, index_version)``. No semantic cache: two similar queries can differ in
   time/place/audience constraints.
4. **Timeout + bounded retry** -- exponential backoff, and retries go through the rate limiter too.
   Results are tagged ``ok`` / ``empty`` / ``error`` so infra failures are never scored as policy errors.
5. **Metrics** -- QPS, in-flight, queue wait, cache hit rate, coalescing rate, timeouts, retries.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from agentic_search.core.types import Post, ToolResult, ToolStatus

Backend = Callable[..., Any]  # (query, limit, **filters) -> list[Post] | Awaitable[list[Post]]


@dataclass(frozen=True)
class CacheKey:
    tool: str
    query: str
    limit: int
    filters: str  # canonical JSON
    index_version: str

    @classmethod
    def build(cls, tool: str, args: dict[str, Any], index_version: str) -> CacheKey:
        filters = {k: v for k, v in args.items() if k not in ("query", "limit")}
        return cls(
            tool,
            str(args.get("query", "")),
            int(args.get("limit", 5)),
            json.dumps(filters, sort_keys=True, ensure_ascii=False),
            index_version,
        )


@dataclass
class ToolLimits:
    rate_per_s: float = 50.0
    burst: int = 20
    max_inflight: int = 16
    timeout_s: float = 3.0
    max_retries: int = 2
    backoff_base_s: float = 0.2


class TokenBucket:
    def __init__(self, rate: float, burst: int):
        if rate <= 0 or burst < 1:
            raise ValueError("rate must be > 0 and burst >= 1")
        self.rate, self.capacity = rate, float(burst)
        self.tokens, self.ts = float(burst), time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:  # FIFO among waiters
            while True:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.ts) * self.rate)
                self.ts = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                await asyncio.sleep((1.0 - self.tokens) / self.rate)


@dataclass
class _Flight:
    key: CacheKey
    future: asyncio.Future
    waiters: set[tuple[str, str]] = field(default_factory=set)
    enqueued_at: float = field(default_factory=time.monotonic)
    started: bool = False


class QueueFullError(RuntimeError):
    pass


class SearchProxy:
    def __init__(
        self,
        backends: dict[str, Backend],
        limits: dict[str, ToolLimits] | None = None,
        *,
        queue_size: int = 1024,
        num_workers: int = 32,
        cache_ttl_s: float | None = 600.0,
        index_version: str = "live",
        max_queue_wait_s: float = 30.0,
        reject_when_full: bool = False,
        auto_start: bool = True,
    ):
        self.backends = backends
        self.limits = {name: (limits or {}).get(name, ToolLimits()) for name in backends}
        self.index_version = index_version
        # None => never expire; use with a pinned index snapshot for reproducible experiments
        self.cache_ttl_s = cache_ttl_s
        self.max_queue_wait_s, self.reject_when_full = max_queue_wait_s, reject_when_full
        self._queue: asyncio.Queue[_Flight] = asyncio.Queue(maxsize=queue_size)
        self._buckets = {n: TokenBucket(lim.rate_per_s, lim.burst) for n, lim in self.limits.items()}
        self._inflight_sem = {n: asyncio.Semaphore(lim.max_inflight) for n, lim in self.limits.items()}
        self._flights: dict[CacheKey, _Flight] = {}
        self._cache: dict[CacheKey, tuple[float, ToolResult]] = {}
        self._workers: list[asyncio.Task] = []
        self._num_workers = num_workers
        self.auto_start = auto_start
        self.counters: dict[str, float] = defaultdict(float)
        self._queue_wait_total = 0.0

    # -- lifecycle ---------------------------------------------------------------------------
    async def start(self) -> SearchProxy:
        if not self._workers:
            self._workers = [asyncio.create_task(self._worker()) for _ in range(self._num_workers)]
        return self

    async def close(self) -> None:
        for w in self._workers:
            w.cancel()
        for w in self._workers:
            with contextlib.suppress(asyncio.CancelledError):
                await w
        self._workers = []

    async def __aenter__(self) -> SearchProxy:
        return await self.start()

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # -- public API --------------------------------------------------------------------------
    async def call(
        self,
        tool: str,
        arguments: dict[str, Any],
        *,
        job_id: str = "",
        trajectory_id: str = "",
        call_id: str = "",
    ) -> ToolResult:
        if tool not in self.backends:
            return ToolResult(call_id, tool, status="error", error=f"unknown tool {tool}")
        if (
            self.auto_start and not self._workers
        ):  # lazy start, e.g. when built synchronously inside a Ray worker
            await self.start()
        self.counters[f"{tool}/requests"] += 1
        key = CacheKey.build(tool, arguments, self.index_version)

        cached = self._cache.get(key)
        if cached and (self.cache_ttl_s is None or time.monotonic() - cached[0] < self.cache_ttl_s):
            self.counters["cache_hits"] += 1
            return self._for_caller(cached[1], call_id)

        waiter = (job_id, trajectory_id)
        flight = self._flights.get(key)
        if flight is not None:
            self.counters["coalesced"] += 1
            flight.waiters.add(waiter)
        else:
            flight = _Flight(key, asyncio.get_running_loop().create_future(), {waiter})
            self._flights[key] = flight
            if self.reject_when_full and self._queue.full():
                del self._flights[key]
                self.counters["rejected"] += 1
                raise QueueFullError("search proxy queue is full")
            await self._queue.put(flight)  # blocks the rollout when full (backpressure)
        result: ToolResult = await asyncio.shield(flight.future)
        return self._for_caller(result, call_id)

    def cancel_trajectory(self, job_id: str, trajectory_id: str) -> None:
        """Drop queued work that only this trajectory is waiting for."""
        waiter = (job_id, trajectory_id)
        for flight in self._flights.values():
            flight.waiters.discard(waiter)

    def metrics(self) -> dict[str, float]:
        c = dict(self.counters)
        total = sum(v for k, v in c.items() if k.endswith("/requests"))
        executed = sum(v for k, v in c.items() if k.endswith("/backend_calls"))
        c.update(
            queue_depth=float(self._queue.qsize()),
            cache_hit_rate=c.get("cache_hits", 0.0) / total if total else 0.0,
            coalesce_rate=c.get("coalesced", 0.0) / total if total else 0.0,
            mean_queue_wait_s=self._queue_wait_total / executed if executed else 0.0,
        )
        return c

    # -- internals ---------------------------------------------------------------------------
    @staticmethod
    def _for_caller(res: ToolResult, call_id: str) -> ToolResult:
        return ToolResult(call_id, res.name, list(res.posts), res.status, res.error, res.latency_ms)

    async def _worker(self) -> None:
        while True:
            flight = await self._queue.get()
            try:
                await self._run(flight)
            finally:
                self._queue.task_done()

    async def _run(self, flight: _Flight) -> None:
        key = flight.key
        waited = time.monotonic() - flight.enqueued_at
        if not flight.waiters:
            self._finish(flight, ToolResult("", key.tool, status="error", error="cancelled"), cache=False)
            self.counters["cancelled"] += 1
            return
        if waited > self.max_queue_wait_s:
            self._finish(flight, ToolResult("", key.tool, status="error", error="queue_timeout"), cache=False)
            self.counters["queue_timeouts"] += 1
            return
        flight.started = True
        self._queue_wait_total += waited
        lim = self.limits[key.tool]
        filters = json.loads(key.filters)
        last_err = ""
        for attempt in range(lim.max_retries + 1):
            await self._buckets[key.tool].acquire()  # retries are rate limited as well
            async with self._inflight_sem[key.tool]:
                self.counters[f"{key.tool}/backend_calls"] += 1
                t0 = time.perf_counter()
                try:
                    posts = await asyncio.wait_for(self._invoke(key, filters), timeout=lim.timeout_s)
                except asyncio.TimeoutError:
                    last_err = "timeout"
                    self.counters[f"{key.tool}/timeouts"] += 1
                except Exception as e:
                    last_err = repr(e)
                    self.counters[f"{key.tool}/errors"] += 1
                else:
                    latency = (time.perf_counter() - t0) * 1e3
                    status: ToolStatus = "ok" if posts else "empty"
                    self._finish(
                        flight, ToolResult("", key.tool, list(posts), status, latency_ms=latency), cache=True
                    )
                    return
            if attempt < lim.max_retries:
                self.counters[f"{key.tool}/retries"] += 1
                await asyncio.sleep(lim.backoff_base_s * (2**attempt))
        self._finish(flight, ToolResult("", key.tool, status="error", error=last_err), cache=False)

    async def _invoke(self, key: CacheKey, filters: dict[str, Any]) -> Sequence[Post]:
        backend = self.backends[key.tool]
        if inspect.iscoroutinefunction(backend) or inspect.iscoroutinefunction(type(backend).__call__):
            return await backend(key.query, key.limit, **filters)
        return await asyncio.to_thread(backend, key.query, key.limit, **filters)

    def _finish(self, flight: _Flight, result: ToolResult, cache: bool) -> None:
        if cache:
            self._cache[flight.key] = (time.monotonic(), result)
        self._flights.pop(flight.key, None)
        if not flight.future.done():
            flight.future.set_result(result)
