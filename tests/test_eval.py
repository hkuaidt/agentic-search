import pytest

from agentic_search import demo
from agentic_search.eval import (
    P0Constraint,
    RaREvaluator,
    RaRTask,
    Route,
    diagnose,
    pass_at_k,
    route_case,
)
from agentic_search.rewards import CallableJudge
from agentic_search.runtime import AgentRunner

QUERY = "成都两日游，带老人，每天步行不超过 3 公里，不去爬山景点"


async def _traj(registry, prompts):
    return await AgentRunner(demo.scripted_policy(), registry, prompts).run_once(QUERY)


def _task(**kw):
    return RaRTask("t1", QUERY, p0=[P0Constraint("walk<=3km", "每天步行不超过 3 公里")], **kw)


async def test_rar_full_marks_and_pass(registry, prompts):
    judge = CallableJudge(lambda p: "0" if "硬性要求" in p else "1")  # P0 judge: "0" = not violated
    res = RaREvaluator(registry, judge).evaluate(_task(), await _traj(registry, prompts))
    assert (res.F, res.Q, res.A) == (10, 30, 60)
    assert res.S_quality == 100 and res.G_P0 == 1 and res.passed is True
    assert res.p0_details == {"no_fabricated_source": True, "walk<=3km": True}


async def test_rar_p0_violation_zeroes_final_but_keeps_quality(registry, prompts):
    judge = CallableJudge(lambda p: "1")  # P0 judge says "1" = violated
    res = RaREvaluator(registry, judge).evaluate(_task(), await _traj(registry, prompts))
    assert res.G_P0 == 0 and res.S_final == 0 and res.passed is False
    assert res.S_quality == 100  # pre-gate score retained for attribution


async def test_rar_undecidable_p0_goes_to_human_review(registry, prompts):
    task = RaRTask("t", QUERY, p0=[P0Constraint("no_hiking", "不去爬山景点", check=lambda t: None)])
    res = RaREvaluator(registry, CallableJudge(lambda p: "1")).evaluate(task, await _traj(registry, prompts))
    assert res.needs_human_review and res.passed is None and res.S_final is None


async def test_rar_fabricated_citation_is_p0(registry, prompts):
    traj = await _traj(registry, prompts)
    traj.steps[-1].response += "[假的](ref: nope)"
    res = RaREvaluator(registry, CallableJudge(lambda p: "0" if "硬性要求" in p else "1")).evaluate(
        _task(), traj
    )
    assert res.p0_details["no_fabricated_source"] is False and res.G_P0 == 0
    assert res.sub_scores["evidence_support"] == 0


async def test_rar_partial_quality_threshold(registry, prompts):
    judge = CallableJudge(lambda p: "0" if "硬性要求" in p else "0.5")
    res = RaREvaluator(registry, judge).evaluate(_task(), await _traj(registry, prompts))
    assert res.S_quality == pytest.approx(10 + 15 + 30) and res.passed is False


def test_rubric_must_sum_to_100(registry):
    from agentic_search.eval import DEFAULT_RUBRIC

    with pytest.raises(ValueError):
        RaREvaluator(registry, CallableJudge(lambda p: "1"), rubric=DEFAULT_RUBRIC[:-1])


def test_pass_at_k_estimator():
    assert pass_at_k(32, 0, 8) == 0.0
    assert pass_at_k(32, 32, 1) == 1.0
    assert pass_at_k(32, 8, 1) == pytest.approx(0.25)
    assert pass_at_k(4, 1, 2) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        pass_at_k(4, 5, 1)


@pytest.mark.parametrize(
    ("p1", "pk", "teacher", "route"),
    [
        (0.9, 1.0, None, Route.LOW_PRIORITY),
        (0.2, 0.9, None, Route.RL),
        (0.05, 0.1, True, Route.SFT),
        (0.05, 0.1, False, Route.FIX_SYSTEM),
    ],
)
def test_route_case(p1, pk, teacher, route):
    assert route_case(p1, pk, teacher) is route


def test_diagnose_32_rollouts():
    outcomes = {"unstable": [True] * 4 + [False] * 28, "never": [False] * 32}
    out = {d.task_id: d for d in diagnose(outcomes, teacher_solvable={"never": True})}
    assert out["unstable"].route is Route.RL and out["unstable"].pass_at[1] == pytest.approx(0.125)
    assert out["never"].route is Route.SFT
