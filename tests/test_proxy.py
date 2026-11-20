import asyncio
import time

import pytest

from agentic_search.core.types import Post
from agentic_search.tools import CacheKey, QueueFullError, SearchProxy, TokenBucket, ToolLimits


class FakeBackend:
    def __init__(self, delay=0.02, fail_times=0, empty=False):
        self.calls, self.delay, self.fail_times, self.empty = 0, delay, fail_times, empty

    async def __call__(self, query, limit, **filters):
        self.calls += 1
        await asyncio.sleep(self.delay)
        if self.calls <= self.fail_times:
            raise ConnectionError("backend 503")
        return [] if self.empty else [Post(f"{query}-{i}", query) for i in range(limit)]


async def test_singleflight_coalesces_group_burst():
    backend = FakeBackend(delay=0.05)
    async with SearchProxy({"tag_search": backend}) as proxy:
        # G = 8 rollouts of one prompt fire the same sub-query at the same time
        results = await asyncio.gather(
            *(
                proxy.call(
                    "tag_search", {"query": "成都两日游", "limit": 3}, trajectory_id=f"t{i}", call_id=f"c{i}"
                )
                for i in range(8)
            )
        )
    assert backend.calls == 1
    assert [r.call_id for r in results] == [f"c{i}" for i in range(8)]  # each caller keeps its own id
    assert all(r.status == "ok" and len(r.posts) == 3 for r in results)
    assert proxy.metrics()["coalesce_rate"] == pytest.approx(7 / 8)


async def test_exact_cache_key_never_merges_different_requests():
    backend = FakeBackend(delay=0.0)
    async with SearchProxy({"tag_search": backend}) as proxy:
        await proxy.call("tag_search", {"query": "成都", "limit": 3})
        await proxy.call("tag_search", {"query": "成都", "limit": 3})  # cache hit
        await proxy.call("tag_search", {"query": "成都", "limit": 5})  # different limit
        await proxy.call("tag_search", {"query": "成都 ", "limit": 3})  # different string
        await proxy.call("tag_search", {"query": "成都", "limit": 3, "city": "cd"})  # different filters
    assert backend.calls == 4
    assert proxy.metrics()["cache_hits"] == 1


def test_cache_key_includes_index_version():
    a = CacheKey.build("tag_search", {"query": "q", "limit": 5}, "snap-1")
    b = CacheKey.build("tag_search", {"query": "q", "limit": 5}, "snap-2")
    assert a != b


async def test_token_bucket_limits_rate():
    bucket = TokenBucket(rate=50.0, burst=1)
    t0 = time.monotonic()
    for _ in range(6):
        await bucket.acquire()
    assert time.monotonic() - t0 >= 5 / 50 * 0.9


async def test_retry_then_success_and_metrics():
    backend = FakeBackend(delay=0.0, fail_times=1)
    limits = {"community_search": ToolLimits(max_retries=2, backoff_base_s=0.0)}
    async with SearchProxy({"community_search": backend}, limits) as proxy:
        res = await proxy.call("community_search", {"query": "q"})
    assert res.status == "ok" and backend.calls == 2
    assert proxy.metrics()["community_search/retries"] == 1


async def test_infra_failure_is_distinguished_from_empty_result():
    limits = {"t": ToolLimits(max_retries=1, backoff_base_s=0.0, timeout_s=0.01)}
    async with SearchProxy({"t": FakeBackend(delay=0.0, fail_times=99)}, limits) as proxy:
        err = await proxy.call("t", {"query": "q"})
    async with SearchProxy({"t": FakeBackend(delay=0.0, empty=True)}, limits) as proxy:
        empty = await proxy.call("t", {"query": "q"})
    async with SearchProxy({"t": FakeBackend(delay=0.5)}, limits) as proxy:
        timeout = await proxy.call("t", {"query": "q"})
    assert err.status == "error" and err.is_infra_failure
    assert empty.status == "empty" and not empty.is_infra_failure
    assert timeout.status == "error" and timeout.error == "timeout"


async def test_errors_are_not_cached():
    backend = FakeBackend(delay=0.0, fail_times=1)
    limits = {"t": ToolLimits(max_retries=0)}
    async with SearchProxy({"t": backend}, limits) as proxy:
        first = await proxy.call("t", {"query": "q"})
        second = await proxy.call("t", {"query": "q"})
    assert first.status == "error" and second.status == "ok"


async def test_bounded_queue_rejects_when_configured():
    proxy = SearchProxy(
        {"t": FakeBackend()}, queue_size=1, reject_when_full=True, auto_start=False
    )  # workers not started
    first = asyncio.create_task(proxy.call("t", {"query": "a"}))
    await asyncio.sleep(0)
    with pytest.raises(QueueFullError):
        await proxy.call("t", {"query": "b"})
    await proxy.start()
    assert (await first).status == "ok"
    await proxy.close()


async def test_cancelled_trajectory_work_is_dropped():
    backend = FakeBackend()
    proxy = SearchProxy({"t": backend}, auto_start=False)  # queue first, start workers later
    task = asyncio.create_task(proxy.call("t", {"query": "q"}, job_id="j", trajectory_id="dead"))
    await asyncio.sleep(0)
    proxy.cancel_trajectory("j", "dead")
    await proxy.start()
    res = await task
    await proxy.close()
    assert res.status == "error" and res.error == "cancelled"
    assert backend.calls == 0


async def test_unknown_tool():
    async with SearchProxy({"t": FakeBackend()}) as proxy:
        assert (await proxy.call("nope", {"query": "q"})).status == "error"
