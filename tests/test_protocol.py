"""``skygovc.protocol`` 的单元测试 + 模型检验回归测试。

覆盖题面要求的 8 项：

1. ``State`` 可哈希、可放入 ``set``；
2. ``reliable`` 通道下 ``successors`` 不产生任何 ``drop:*`` 动作；
3. ``check("reliable")`` 三条性质全部成立；
4. ``check("lossy")`` 的 P2 不成立、反例非空、末状态满足 ``is_deadlock_alert``；
5. ``check("lossy_retry", max_retries=2)`` 的 P2 成立；
6. 可达状态数在 ``(1, 200000)`` 内（并额外收紧到题面要求的 ``< 1e4``）；
7. ``retry_boundary()`` 的边界（实测见
   :func:`test_retry_boundary_recovers_at_first_retry`；
   与题面预期的差异在断言注释与 README 中说明）；
8. ``counterexample_text`` 返回非空字符串且包含动作标签。

另加若干结构性测试，把建模决策（P1 守卫、强制送达、resend 语义、
状态空间上限阀值、性质注册表）钉死，防止后续改动无意中破坏结论。

运行::

    & "python" -m pytest <本文件路径> -q
"""
from __future__ import annotations

import dataclasses
import sys
from dataclasses import replace
from pathlib import Path

# 让 ``import skygovc`` 在未安装包的仓库里也能工作（pytest prepend 导入模式
# 只会把 tests/ 目录放进 sys.path）。
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest

from skygovc.protocol import (  # noqa: E402
    INITIAL,
    KINDS,
    K_BOUND,
    PROPERTIES,
    PROPERTY_ORDER,
    CheckResult,
    State,
    StateSpaceTooLarge,
    bounded_liveness_ok,
    check,
    classify_channel,
    counterexample_text,
    is_deadlock_alert,
    reachable_graph,
    retry_boundary,
    safety_p1,
    successors,
)


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------
def _alert_pending_state(**overrides) -> State:
    """构造一个"ALERT 已发出"的 alert_pending 状态，便于定点测试。"""
    base = State(
        phase="alert_pending",
        b_applications=1,
        b_applied=True,
        alert_issued=True,
        alert_acked=False,
        retries_used=0,
        in_flight="ALERT",
        last_msg="ALERT",
        steps=5,
    )
    return replace(base, **overrides) if overrides else base


def _actions(state: State, channel: str, max_retries: int = 2) -> list[str]:
    return [action for _nxt, action in successors(state, channel, max_retries)]


# --------------------------------------------------------------------------
# 1. State 可哈希
# --------------------------------------------------------------------------
def test_state_is_frozen_and_hashable() -> None:
    assert hash(INITIAL) == hash(replace(INITIAL))
    assert len({INITIAL, replace(INITIAL)}) == 1

    as_dict_key = {INITIAL: "root"}
    assert as_dict_key[replace(INITIAL)] == "root"

    registered = replace(
        INITIAL,
        phase="registered",
        in_flight="REGISTER",
        last_msg="REGISTER",
        steps=1,
    )
    assert registered != INITIAL
    assert len({INITIAL, registered}) == 2

    # frozen：字段不可变，因此哈希在对象生命周期内稳定。
    with pytest.raises(dataclasses.FrozenInstanceError):
        INITIAL.phase = "closed"  # type: ignore[misc]


# --------------------------------------------------------------------------
# 2. reliable 通道没有通道丢失动作
# --------------------------------------------------------------------------
def test_reliable_channel_has_no_drop_or_resend() -> None:
    states, graph = reachable_graph("reliable")
    actions = {action for edges in graph.values() for _nxt, action in edges}

    assert not any(a.startswith("drop:") for a in actions)
    assert not any(a.startswith("resend:") for a in actions)
    # 阶段推进动作齐全
    assert {
        "reg_send",
        "verify_ok",
        "verify_fail",
        "delegate_send",
        "b_apply",
        "b_alert",
        "b_ack",
        "b_close",
    } <= actions
    assert INITIAL in states


# --------------------------------------------------------------------------
# 3. reliable 通道三条性质全部成立
# --------------------------------------------------------------------------
def test_reliable_satisfies_all_properties() -> None:
    results = check("reliable")
    assert set(results) == {"P1", "P2", "P3"}
    for name, result in results.items():
        assert isinstance(result, CheckResult)
        assert result.satisfied is True, f"{name} 在 reliable 通道下应成立：{result.detail}"
        assert result.counterexample is None
        assert result.explored > 1


# --------------------------------------------------------------------------
# 4. lossy 通道 P2 反例
# --------------------------------------------------------------------------
def test_lossy_violates_p2_with_real_counterexample() -> None:
    results = check("lossy")
    assert results["P1"].satisfied is True
    assert results["P3"].satisfied is True

    p2 = results["P2"]
    assert p2.satisfied is False
    assert p2.counterexample, "P2 被违反时必须给出反例路径"

    path = p2.counterexample
    assert path[0][0] == INITIAL
    assert path[0][1] == "init"

    last_state = path[-1][0]
    assert is_deadlock_alert(last_state)
    assert last_state.phase == "alert_pending"
    assert last_state.alert_issued and not last_state.alert_acked
    assert last_state.in_flight is None

    # lossy 通道下该状态确实是死状态：没有任何后继。
    assert successors(last_state, "lossy") == []

    # 反例必须包含 ALERT 被丢弃这一步。
    assert "drop:ALERT" in [a for _s, a in path]


# --------------------------------------------------------------------------
# 5. lossy_retry(max_retries=2) 恢复 P2
# --------------------------------------------------------------------------
def test_lossy_retry_restores_p2() -> None:
    results = check("lossy_retry", max_retries=2)
    assert results["P2"].satisfied is True, results["P2"].detail
    assert results["P2"].counterexample is None
    for name in PROPERTY_ORDER:
        assert results[name].satisfied is True, f"{name}: {results[name].detail}"


# --------------------------------------------------------------------------
# 6. 状态空间规模
# --------------------------------------------------------------------------
def test_state_space_size_is_small_and_non_trivial() -> None:
    for channel, max_retries in (
        ("reliable", 2),
        ("lossy", 2),
        ("lossy_retry", 1),
        ("lossy_retry", 2),
        ("lossy_retry", 5),
    ):
        explored = check(channel, max_retries=max_retries)["P1"].explored
        assert explored > 1
        assert explored < 200_000
        # 题面要求显式状态数控制在 1e4 以内，便于穷举与论文列报。
        assert explored < 10_000, f"{channel}/{max_retries} explored={explored}"

    # 三种通道（max_retries=2）的规模差异：重传引入更多中间状态。
    sizes = {ch: check(ch, max_retries=2)["P1"].explored for ch in ("reliable", "lossy", "lossy_retry")}
    assert sizes["reliable"] < sizes["lossy"] < sizes["lossy_retry"]


# --------------------------------------------------------------------------
# 7. 重传预算边界
# --------------------------------------------------------------------------
def test_retry_boundary_recovers_at_first_retry() -> None:
    """实测边界：P2 在"存在重传动作"时即恢复，即 max_retries == 1。

    题面预期是"m=1 时为 False、m>=2 时为 True"。该预期与本文件第 5 项
    （``check("lossy_retry", max_retries=2)`` 的 P2 必须为 True）在逻辑上
    不相容：``max_retries`` 单调放宽时转移关系只增不减（``drop`` 的守卫
    ``retries_used < max_retries`` 关于 ``max_retries`` 单调），故 m=1 的
    可达集 ⊆ m=2 的可达集；P2 是"不存在坏状态"的全称性质，因此
    P2(m=2) 蕴含 P2(m=1)，任何单调模型都不可能给出 m=1 违反、m=2 成立的
    组合。这里如实断言实测边界，并在 README「关键发现」中给出证明与解释。
    """
    rows = retry_boundary()
    assert rows, "retry_boundary() 不应返回空表"

    lookup = {(r["channel"], r["max_retries"]): r for r in rows}
    assert set(lookup) == {
        (ch, m) for ch in ("lossy", "lossy_retry") for m in (1, 2, 3, 4, 5)
    }

    for m in (1, 2, 3, 4, 5):
        lossy = lookup[("lossy", m)]
        retry = lookup[("lossy_retry", m)]
        # 无重传通道：任意"重传参数"都无法弥补机制缺失。
        assert lossy["P2"] is False
        # 有重传通道：只要存在重传动作（m>=1），ALERT 就不会永久滞留。
        assert retry["P2"] is True
        assert retry["P1"] is True and retry["P3"] is True
        assert lossy["P1"] is True and lossy["P3"] is True

    # 状态空间随重传预算线性增长：|Π|(m) = 16m + 14。
    for m in (1, 2, 3, 4, 5):
        assert lookup[("lossy_retry", m)]["explored"] == 16 * m + 14
    # lossy 的重传参数无效，规模恒定。
    assert {lookup[("lossy", m)]["explored"] for m in (1, 2, 3, 4, 5)} == {22}

    # 自定义通道 / 区间也应生效。
    custom = retry_boundary(channels=("lossy_retry",), max_retries_range=(1, 2))
    assert [r["max_retries"] for r in custom] == [1, 2]
    assert all(r["channel"] == "lossy_retry" for r in custom)


def test_p2_is_monotone_in_retry_budget() -> None:
    """重传预算单调放宽时 P2 不应变坏（转移关系单调包含的直接推论）。"""
    flags = [check("lossy_retry", max_retries=m)["P2"].satisfied for m in range(0, 7)]
    assert all(flags), flags


# --------------------------------------------------------------------------
# 8. 反例渲染
# --------------------------------------------------------------------------
def test_counterexample_text_renders_actions() -> None:
    p2 = check("lossy")["P2"]
    text = counterexample_text(p2.counterexample)
    assert isinstance(text, str) and text.strip()
    assert "reg_send" in text
    assert "drop:ALERT" in text
    assert "alert_pending" in text
    assert len(text.splitlines()) > 3

    # 也接受裸状态序列。
    bare = counterexample_text([INITIAL, _alert_pending_state(in_flight=None)])
    assert bare.strip() and "alert_pending" in bare


# --------------------------------------------------------------------------
# 附加：谓词、守卫与通道语义
# --------------------------------------------------------------------------
def test_p1_predicate_and_structural_guard() -> None:
    assert safety_p1(INITIAL) is True
    assert safety_p1(_alert_pending_state(b_applications=2)) is False

    # 任意可达状态都不超过 1 次申请：P1 由 b_applied 守卫结构性保证。
    for channel in ("reliable", "lossy", "lossy_retry"):
        states, _graph = reachable_graph(channel, max_retries=2)
        assert max(s.b_applications for s in states) == 1
        assert all(safety_p1(s) for s in states)

    # b_apply 只可能触发一次：在 delegated 且 b_applied=False 时可用。
    delegated = replace(
        INITIAL,
        phase="delegated",
        b_applications=0,
        b_applied=False,
        in_flight="DELEGATE",
        last_msg="DELEGATE",
        steps=3,
    )
    assert "b_apply" in _actions(delegated, "reliable")
    applied = replace(delegated, b_applications=1, b_applied=True)
    assert "b_apply" not in _actions(applied, "reliable")


def test_deadlock_alert_predicate_semantics() -> None:
    assert is_deadlock_alert(_alert_pending_state(in_flight=None)) is True
    # 仍有在途 ALERT：还有被确认的希望，不算死状态。
    assert is_deadlock_alert(_alert_pending_state(in_flight="ALERT")) is False
    # 已 ACK。
    assert is_deadlock_alert(_alert_pending_state(alert_acked=True, in_flight=None)) is False
    # 阶段不对。
    assert is_deadlock_alert(INITIAL) is False
    # ALERT 从未发出（构造式状态，仅验证谓词）。
    assert is_deadlock_alert(
        _alert_pending_state(alert_issued=False, alert_acked=False, in_flight=None)
    ) is False


def test_bounded_liveness_predicate() -> None:
    assert K_BOUND == 12
    closed_ok = replace(INITIAL, phase="closed", steps=K_BOUND)
    closed_bad = replace(INITIAL, phase="closed", steps=K_BOUND + 1)
    assert bounded_liveness_ok(closed_ok, K_BOUND) is True
    assert bounded_liveness_ok(closed_bad, K_BOUND) is False
    assert bounded_liveness_ok(closed_bad, K_BOUND + 1) is True
    # 非 closed 状态不承担界。
    assert bounded_liveness_ok(replace(INITIAL, steps=10_000), K_BOUND) is True

    # 所有通道的可达 CLOSE 状态都在界内，且最长完成路径为 7 个推进步。
    for channel in ("reliable", "lossy", "lossy_retry"):
        states, _graph = reachable_graph(channel, max_retries=2)
        closed = [s for s in states if s.phase == "closed"]
        assert closed and max(s.steps for s in closed) == 7
        assert all(bounded_liveness_ok(s, K_BOUND) for s in closed)


def test_forced_delivery_when_retry_budget_is_exhausted() -> None:
    # 预算还剩 1 次：可以再丢一次，也可以确认。
    assert _actions(_alert_pending_state(retries_used=1), "lossy_retry", 2) == [
        "b_ack",
        "drop:ALERT",
    ]
    # 预算耗尽（retries_used == max_retries）：禁止再丢 → 强制送达。
    assert _actions(_alert_pending_state(retries_used=2), "lossy_retry", 2) == ["b_ack"]


def test_resend_restores_last_message() -> None:
    dropped = _alert_pending_state(in_flight=None, retries_used=1)
    assert is_deadlock_alert(dropped) is True
    assert _actions(dropped, "lossy_retry", 2) == ["resend:ALERT"]

    # 预算耗尽后仍可重传（这正是 P2 得以恢复的机制）。
    exhausted = _alert_pending_state(in_flight=None, retries_used=2)
    assert _actions(exhausted, "lossy_retry", 2) == ["resend:ALERT"]

    (restored, action), = successors(exhausted, "lossy_retry", 2)
    assert action == "resend:ALERT"
    assert restored.in_flight == "ALERT"
    assert is_deadlock_alert(restored) is False
    # 恢复后只能确认（不能再丢）。
    assert _actions(restored, "lossy_retry", 2) == ["b_ack"]

    # lossy 通道没有重传动作，丢掉的 ALERT 无法恢复。
    assert _actions(dropped, "lossy") == []

    # 终止状态不再产生通道动作。
    closed = replace(INITIAL, phase="closed", steps=7)
    assert _actions(closed, "lossy_retry", 2) == []


def test_classify_channel() -> None:
    assert classify_channel("reliable") == "pass"
    assert classify_channel("lossy") == "P2_violated"
    assert classify_channel("lossy_retry", max_retries=2) == "pass"
    assert classify_channel("lossy_retry", max_retries=1) == "pass"


def test_properties_registry() -> None:
    assert PROPERTY_ORDER == ("P1", "P2", "P3")
    assert set(PROPERTIES) == set(PROPERTY_ORDER)
    assert PROPERTIES["P1"][0] == "invariant"
    assert PROPERTIES["P2"][0] == "liveness_no_dead_alert"
    assert PROPERTIES["P3"][0] == "liveness_bounded"
    for name, (kind, description) in PROPERTIES.items():
        assert kind in KINDS
        assert isinstance(description, str) and len(description) > 10
        assert name in description


def test_reachable_graph_structure() -> None:
    order, graph = reachable_graph("lossy_retry", max_retries=1)
    assert order[0] == INITIAL
    assert len(order) == len(set(order)) == len(graph)
    assert set(graph) == set(order)

    known = set(order)
    for src, edges in graph.items():
        for dst, action in edges:
            assert dst in known
            assert dst != src, f"{action} 产生自环"
            assert isinstance(action, str) and action

    # 终止阶段没有出边，且只出现在 closed / rejected 上。
    for src, edges in graph.items():
        if src.is_terminal:
            assert edges == []


def test_state_space_guard() -> None:
    results = check("lossy_retry", max_retries=2, max_states=5)
    assert all(r.satisfied is False for r in results.values())
    assert all("超出上限" in r.detail for r in results.values())
    assert all(r.explored == 5 for r in results.values())

    with pytest.raises(StateSpaceTooLarge):
        reachable_graph("lossy_retry", max_retries=2, max_states=5)


def test_reachable_graph_is_acyclic() -> None:
    """可达状态图无环——这是 P2 形式化（"无法到达 CLOSE"≈"永久循环"）的前提。

    每次转移都严格增大某个势函数（推进步数 / retries_used / 清空 in_flight），
    因此不存在真正的环；"无限循环中 ALERT 不再改变"在本模型中等价于
    "到达一个不含 CLOSE 的死锁汇"。若以后引入 reset 等可回退动作，需要回到
    modelcheck 的反向可达判定（该判定本身不依赖无环性）。
    """
    for channel, max_retries in (("reliable", 2), ("lossy", 2), ("lossy_retry", 3)):
        order, graph = reachable_graph(channel, max_retries=max_retries)
        indegree = {s: 0 for s in order}
        for _src, edges in graph.items():
            for dst, _action in edges:
                indegree[dst] += 1
        queue = [s for s in order if indegree[s] == 0]
        visited = 0
        while queue:
            state = queue.pop()
            visited += 1
            for dst, _action in graph[state]:
                indegree[dst] -= 1
                if indegree[dst] == 0:
                    queue.append(dst)
        assert visited == len(order) == len(graph), f"{channel} 的可达图含环"


def test_invalid_channel_rejected() -> None:
    for call in (
        lambda: successors(INITIAL, "semi_reliable"),
        lambda: reachable_graph("nope"),
        lambda: check("nope"),
        lambda: classify_channel("nope"),
    ):
        with pytest.raises(ValueError):
            call()


def test_verify_fail_path_is_terminal_and_clean() -> None:
    rejected = [s for s in reachable_graph("reliable")[0] if s.phase == "rejected"]
    assert len(rejected) == 1
    assert rejected[0].b_applications == 0
    assert rejected[0].is_terminal
    assert successors(rejected[0], "reliable") == []
