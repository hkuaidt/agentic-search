"""pass@k and the SFT-vs-RL routing rule for badcases.

For a C-end product the target is pass@1; pass@k only diagnoses whether a successful path is reachable.

| observation                              | likely issue                              | route        |
|------------------------------------------|-------------------------------------------|--------------|
| pass@1 low, pass@k clearly higher         | can succeed, but unreliably               | RL           |
| both low, teacher/human solves it        | missing method / tool pattern / knowledge | SFT, then RL |
| both low, teacher/human cannot solve it  | retrieval gap, tool gap, bad task/rubric  | fix system   |
| pass@1 already high                      | stable                                    | low priority |

Over-sampling SFT data for unstable-but-solvable cases distorts the global distribution and can lower
pass@k elsewhere; pure RL on cases with pass@k ~ 0 receives all-zero rewards and no gradient.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from math import comb


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator 1 - C(n-c, k) / C(n, k) (Chen et al., 2021) from ``n`` samples with ``c`` passes."""
    if not 0 <= c <= n or k < 1:
        raise ValueError("need 0 <= c <= n and k >= 1")
    if k > n:
        raise ValueError("k must not exceed the number of samples")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


class Route(str, Enum):
    RL = "rl"
    SFT = "sft"
    FIX_SYSTEM = "fix_system"
    LOW_PRIORITY = "low_priority"


@dataclass
class RoutingThresholds:
    pass1_high: float = 0.8
    passk_min_for_rl: float = 0.5
    min_gap: float = 0.25


def route_case(
    pass1: float, passk: float, teacher_solvable: bool | None, th: RoutingThresholds | None = None
) -> Route:
    th = th or RoutingThresholds()
    if pass1 >= th.pass1_high:
        return Route.LOW_PRIORITY
    if passk >= th.passk_min_for_rl and passk - pass1 >= th.min_gap:
        return Route.RL
    if teacher_solvable:
        return Route.SFT
    return Route.FIX_SYSTEM


@dataclass
class CaseDiagnosis:
    task_id: str
    n: int
    c: int
    pass_at: dict[int, float]
    route: Route


def diagnose(
    outcomes: Mapping[str, Sequence[bool]],
    teacher_solvable: Mapping[str, bool] | None = None,
    ks: Sequence[int] = (1, 8, 16),
    route_k: int = 16,
    th: RoutingThresholds | None = None,
) -> list[CaseDiagnosis]:
    """``outcomes[task_id]`` = pass/fail of each independent rollout (PROPOSAL §6.3 samples 32 per query)."""
    out = []
    for tid, res in outcomes.items():
        n, c = len(res), int(sum(res))
        pk = {k: pass_at_k(n, c, k) for k in ks if k <= n}
        rk = min(route_k, n)
        route = route_case(
            pk.get(1, c / n if n else 0.0), pass_at_k(n, c, rk), (teacher_solvable or {}).get(tid), th
        )
        out.append(CaseDiagnosis(tid, n, c, pk, route))
    return out
