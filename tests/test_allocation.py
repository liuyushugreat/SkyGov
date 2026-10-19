"""``skygovc.allocation`` 的单元测试。

覆盖规范要求的 7 项：

1. ``all_feasible`` 宽松预算下可行解数 > 0 且全部满足底线；
2. 穷举真 Pareto 前沿 vs NSGA-II 前沿（不被严格支配 + 超体积覆盖率 ≥ 0.8）；
3. ``dominates`` 的自反性/反对称性；
4. ``hypervolume_2d`` 与手算结果一致；
5. 固定 seed 下 ``nsga2`` 完全可复现；
6. ``tiered`` 返回可行解且满足底线；
7. ``budget_sweep`` 条数正确、字段齐全。

另附模型常量、约束支配、拥挤距离、超体积单调性等稳健性检查。

运行方式::

    & "python" -m pytest tests/test_allocation.py -q
"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest

# 让测试既能从仓库根目录、也能从 modules/SkyGovC 目录直接运行。
_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from skygovc.allocation import (  # noqa: E402
    CAP_KEYS,
    COST,
    DEFAULT_BUDGETS,
    FLOOR,
    NSGA2Result,
    SPACE_SIZE,
    TIERS,
    W_INCIDENT,
    W_SERVICE,
    all_feasible,
    all_levels,
    budget_sweep,
    compare_policies,
    constraint_dominance_matrix,
    cost_of,
    crowding_distance,
    decode,
    dominates,
    evaluate,
    feasibility,
    floor_cost,
    hypervolume_2d,
    hypervolume_nd,
    knee_index,
    non_dominated_indices,
    normalized_hypervolume,
    nsga2,
    objectives,
    pareto_front,
    repair,
    tiered,
    uniform_open,
    uniform_strict,
)

#: 穷举真前沿的黄金比较用的宽松预算。
LOOSE_BUDGET = 3.0


def _objectives_of(levels_list, tier):
    return [objectives(tuple(lv), tier) for lv in levels_list]


def _hv_ratio(tier, budget, seed=0, samples=200000):
    """返回 ``(nsga2 结果, 真前沿目标, HV_真, HV_NSGA2)``。"""
    feasible = all_feasible(tier, budget)
    assert feasible, f"{tier}/{budget} 没有可行解，无法做前沿对比"
    true_front = [feasible[i] for i in pareto_front(feasible, tier)]
    true_obj = _objectives_of(true_front, tier)

    res = nsga2(tier, budget, seed=seed)
    assert isinstance(res, NSGA2Result)

    all_obj = np.asarray(_objectives_of(feasible, tier), dtype=float)
    ideal = np.min(np.asarray(true_obj + res.front_objectives, dtype=float), axis=0)
    nadir = all_obj.max(axis=0)
    span = nadir - ideal
    span = np.where(span > 1e-12, span, 1e-9)
    reference = ideal + span * 1.05

    hv_true = normalized_hypervolume(true_obj, reference, ideal, samples=samples, seed=1)
    hv_nsga = normalized_hypervolume(res.front_objectives, reference, ideal, samples=samples, seed=1)
    return res, true_obj, hv_true, hv_nsga


# --------------------------------------------------------------------------
# 0. 模型常量与基础函数
# --------------------------------------------------------------------------
def test_constants_match_specification():
    assert CAP_KEYS == ("rule", "spatial", "service", "sensing", "emergency", "data")
    assert SPACE_SIZE == 729
    assert len(all_levels()) == 729
    assert set(FLOOR) == set(TIERS)
    assert set(W_INCIDENT) == set(TIERS) and set(W_SERVICE) == set(TIERS)
    for tier in TIERS:
        assert len(FLOOR[tier]) == 6 and len(W_INCIDENT[tier]) == 6 and len(W_SERVICE[tier]) == 6

    # 成本表逐格核对（规范里的三张表）
    assert COST["rule"] == {0: 0.05, 1: 0.12, 2: 0.22}
    assert COST["spatial"] == {0: 0.08, 1: 0.20, 2: 0.38}
    assert COST["service"] == {0: 0.10, 1: 0.26, 2: 0.50}
    assert COST["sensing"] == {0: 0.12, 1: 0.34, 2: 0.68}
    assert COST["emergency"] == {0: 0.06, 1: 0.15, 2: 0.29}
    assert COST["data"] == {0: 0.04, 1: 0.10, 2: 0.19}
    assert cost_of((2, 2, 2, 2, 2, 2)) == pytest.approx(2.26)
    assert cost_of((0, 0, 0, 0, 0, 0)) == pytest.approx(0.45)
    assert floor_cost("capital_core") == pytest.approx(1.61)
    assert floor_cost("sparse_resource") == pytest.approx(0.95)


def test_objectives_match_closed_formula():
    # 全 0：incident = exp(0) = 1，completion = 0.60，cost = 0.45
    inc, neg_comp, cost = objectives((0, 0, 0, 0, 0, 0), "sparse_resource")
    assert inc == pytest.approx(1.0)
    assert -neg_comp == pytest.approx(0.60)
    assert cost == pytest.approx(0.45)

    # 手动按规范公式复算一个非平凡解
    levels = (2, 1, 2, 0, 1, 2)
    tier = "mountainous"
    s = [l / 2.0 for l in levels]
    exp_inc = math.exp(-sum(W_INCIDENT[tier][k] * s[k] for k in range(6)))
    exp_comp = min(1.0, 0.60 + 0.30 * sum(W_SERVICE[tier][k] * s[k] for k in range(6)))
    exp_cost = sum(COST[CAP_KEYS[k]][levels[k]] for k in range(6))
    got = objectives(levels, tier)
    assert got[0] == pytest.approx(exp_inc, abs=1e-12)
    assert -got[1] == pytest.approx(exp_comp, abs=1e-12)
    assert got[2] == pytest.approx(exp_cost, abs=1e-12)

    # 单调性：等级越高，事件率越低、完成率越高、成本越大
    low = objectives((0, 0, 0, 0, 0, 0), "dense_hub")
    high = objectives((2, 2, 2, 2, 2, 2), "dense_hub")
    assert high[0] < low[0]
    assert high[1] < low[1]          # f2 = -completion，越小越好
    assert high[2] > low[2]
    assert 0.6 <= -high[1] <= 0.9


def test_decode_and_repair_contract():
    assert decode([0, 1, 2, 1, 0, 2], "capital_core") == (0, 1, 2, 1, 0, 2)
    # 解码负责表示归一：四舍五入 + 截断到 [0, 2]
    assert decode([3, -1, 1.4, 1.6, 2, 0], "dense_hub") == (2, 0, 1, 2, 2, 0)
    with pytest.raises(ValueError):
        decode([0, 0, 0], "capital_core")
    with pytest.raises(KeyError):
        decode([0] * 6, "no_such_tier")

    # 修复算子保证底线，且不越界
    lv = repair([0, 0, 0, 0, 0, 0], "capital_core")
    assert lv == FLOOR["capital_core"]
    assert repair([5, 5, 5, 5, 5, 5], "sparse_resource") == (2, 2, 2, 2, 2, 2)
    assert all(lv[k] >= FLOOR["capital_core"][k] for k in range(6))


def test_evaluate_matches_objectives_and_feasibility():
    tier, budget = "mountainous", 2.2
    levels = (2, 2, 2, 1, 2, 1)
    objs, violation = evaluate(levels, tier, budget)
    assert objs == objectives(levels, tier)
    ok, v2 = feasibility(levels, tier, budget)
    assert violation == pytest.approx(v2)
    assert ok == (v2 == 0.0)
    # 底线违反会被计入违反量
    ok_bad, viol_bad = feasibility((0, 0, 0, 0, 0, 0), "mountainous", 3.0)
    assert ok_bad is False and viol_bad > 0


# --------------------------------------------------------------------------
# 1. 穷举可行解
# --------------------------------------------------------------------------
def test_all_feasible_returns_floor_respecting_solutions():
    for tier in TIERS:
        feasible = all_feasible(tier, LOOSE_BUDGET)
        assert len(feasible) > 0, f"{tier} 在宽松预算下应有可行解"
        assert len(feasible) <= SPACE_SIZE
        for lv in feasible:
            assert all(lv[k] >= FLOOR[tier][k] for k in range(6)), f"{tier} 底线被违反: {lv}"
            assert cost_of(lv) <= LOOSE_BUDGET + 1e-9
            assert feasibility(lv, tier, LOOSE_BUDGET)[0] is True

    # 与逐点暴力过滤的结果完全一致
    tier, budget = "mountainous", 1.8
    brute = [lv for lv in all_levels() if feasibility(lv, tier, budget)[0]]
    assert all_feasible(tier, budget) == brute
    assert 0 < len(brute) < SPACE_SIZE

    # 预算收紧时可行解单调减少
    counts = [len(all_feasible("sparse_resource", b)) for b in (0.9, 1.0, 1.4, 3.0)]
    assert counts == sorted(counts)
    assert all_feasible("capital_core", 0.5) == []      # 连底线成本都装不下


# --------------------------------------------------------------------------
# 2. 穷举真前沿 vs NSGA-II
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "tier,budget",
    [
        ("sparse_resource", LOOSE_BUDGET),
        ("mountainous", LOOSE_BUDGET),
        ("market_driven", LOOSE_BUDGET),
        ("dense_hub", LOOSE_BUDGET),
        ("sparse_resource", 1.4),
    ],
)
def test_nsga2_front_matches_exhaustive_pareto(tier, budget):
    res, true_obj, hv_true, hv_nsga = _hv_ratio(tier, budget, seed=0)

    assert res.front, "NSGA-II 第一前沿不应为空"
    assert len(res.front) == len(res.front_objectives)

    # (a) 前沿上每个点都必须可行（预算 + 底线）
    for lv in res.front:
        assert all(lv[k] >= FLOOR[tier][k] for k in range(6))
        assert feasibility(lv, tier, budget)[0] is True

    # (b) 没有任何 NSGA-II 前沿点被真前沿"严格支配"
    true_arr = np.asarray(true_obj, dtype=float)
    for obj in res.front_objectives:
        dominated = any(dominates(tuple(t), tuple(obj)) for t in true_arr)
        assert not dominated, f"NSGA-II 前沿点 {obj} 被真前沿严格支配"

    # (c) 覆盖率（超体积比值）≥ 0.8
    assert hv_true > 0.0
    coverage = hv_nsga / hv_true
    assert coverage >= 0.8, f"{tier}/{budget} 覆盖率仅 {coverage:.3f}"


def test_nsga2_hypervolume_is_normalized_and_population_is_valid():
    res = nsga2("sparse_resource", LOOSE_BUDGET, pop_size=40, generations=20, seed=3)
    assert res.generations == 20
    assert len(res.population) == 40
    assert len(res.objectives) == 40
    assert 0.0 <= res.hypervolume <= 1.0
    for lv, obj in zip(res.population, res.objectives):
        assert all(lv[k] >= FLOOR["sparse_resource"][k] for k in range(6))
        assert tuple(objectives(tuple(lv), "sparse_resource")) == obj
    # 第一前沿的目标应当在最终种群中出现
    pop_objs = {tuple(o) for o in res.objectives}
    for obj in res.front_objectives:
        assert tuple(obj) in pop_objs


# --------------------------------------------------------------------------
# 3. 支配关系
# --------------------------------------------------------------------------
def test_dominates_reflexive_and_antisymmetric():
    a = (0.1, -0.9, 1.0)
    b = (0.2, -0.8, 1.5)
    c = (0.1, -0.9, 1.0)

    # 非自反：任何点都不严格支配自己
    for p in (a, b, c, (0.0, 0.0, 0.0)):
        assert dominates(p, p) is False

    # 反对称：不能互相支配
    assert dominates(a, b) is True
    assert dominates(b, a) is False

    for x, y in [(a, b), (b, a), (a, c), (c, a)]:
        assert not (dominates(x, y) and dominates(y, x))

    # 相等点互不支配
    assert not dominates(a, c) and not dominates(c, a)

    # 仅在某个目标上更差 → 不支配
    assert dominates((1.0, 2.0, 3.0), (1.0, 3.0, 3.0)) is True
    assert dominates((1.0, 2.0, 3.0), (1.0, 2.0, 3.0)) is False
    assert dominates((1.0, 2.0, 3.0), (0.5, 3.0, 3.0)) is False

    with pytest.raises(ValueError):
        dominates((1.0, 2.0), (1.0, 2.0, 3.0))


def test_non_dominated_indices_and_pareto_front_agree():
    pts = [(1, 1, 1), (2, 0.5, 1), (0.5, 2, 1), (3, 3, 3), (1, 1, 1)]
    idx = non_dominated_indices(pts)
    assert idx == [0, 1, 2, 4]
    for i in idx:
        assert not any(dominates(pts[j], pts[i]) for j in range(len(pts)) if j != i)

    feasible = all_feasible("mountainous", LOOSE_BUDGET)
    front_idx = pareto_front(feasible, "mountainous")
    assert front_idx == sorted(front_idx)
    front = [feasible[i] for i in front_idx]
    for lv in front:
        # 真前沿上的点不被任何可行解支配
        assert not any(
            dominates(objectives(other, "mountainous"), objectives(lv, "mountainous"))
            for other in feasible
        )
    assert pareto_front([], "mountainous") == []


def test_constraint_domination_prefers_feasible():
    objs = [(0.1, -0.9, 5.0), (0.9, -0.6, 1.0)]   # 前者目标更好但超预算
    dom = constraint_dominance_matrix(objs, violations=[1.0, 0.0])
    assert dom[1, 0]                  # 可行解支配不可行解
    assert not dom[0, 1]
    assert not np.any(np.diag(dom))    # 对角线恒 False
    # 都不可行时，违反量小者胜
    dom2 = constraint_dominance_matrix(objs, violations=[0.5, 0.2])
    assert dom2[1, 0] and not dom2[0, 1]


def test_crowding_distance_boundaries_and_interior():
    front = [(0.0, 3.0, 1.0), (1.0, 1.0, 1.0), (2.0, 0.0, 1.0)]
    cd = crowding_distance(front)
    assert cd[0] == float("inf") and cd[2] == float("inf")
    assert 0.0 < cd[1] < float("inf")
    assert crowding_distance([(1.0, 1.0, 1.0)]) == [float("inf")]
    assert crowding_distance([]) == []


# --------------------------------------------------------------------------
# 4. 二维超体积
# --------------------------------------------------------------------------
def test_hypervolume_2d_known_example():
    # 手算：(1,1) 与 (2,0.5) 对参考点 (3,3) 支配区域面积 = 2 + 2.5 = 4.5
    assert hypervolume_2d([(1.0, 1.0), (2.0, 0.5)], (3.0, 3.0)) == pytest.approx(4.5)

    # 单点：盒子面积 (3-1)*(3-1) = 4
    assert hypervolume_2d([(1.0, 1.0)], (3.0, 3.0)) == pytest.approx(4.0)

    # 被支配的点不贡献额外体积
    assert hypervolume_2d([(1.0, 1.0), (2.0, 2.0)], (3.0, 3.0)) == pytest.approx(4.0)

    # 参考点之外 / 空集 → 0
    assert hypervolume_2d([(5.0, 5.0)], (3.0, 3.0)) == 0.0
    assert hypervolume_2d([], (3.0, 3.0)) == 0.0

    # 简单解析式：单个点 (0,0) 对 (a,b) 的盒体积
    assert hypervolume_2d([(0.0, 0.0)], (2.0, 5.0)) == pytest.approx(10.0)

    # 与逐格数值积分一致（粗网格一致性检查）
    pts = [(1.0, 1.0), (2.0, 0.5), (0.5, 2.0)]
    ref = (3.0, 3.0)
    grid = 600
    xs = np.linspace(0.0, 3.0, grid, endpoint=False) + 1.5 / grid
    ys = np.linspace(0.0, 3.0, grid, endpoint=False) + 1.5 / grid
    inside = np.zeros((grid, grid), dtype=bool)
    for px, py in pts:
        inside |= (xs[:, None] >= px) & (ys[None, :] >= py)
    approx = inside.mean() * 9.0
    assert hypervolume_2d(pts, ref) == pytest.approx(approx, rel=0.02)


def test_hypervolume_nd_and_normalized_properties():
    pts = [(1.0, 1.0), (2.0, 0.5)]
    ref = (3.0, 3.0)
    # 二维时与精确算法一致：返回被支配部分占 [ideal, ref] 包围盒的比例
    ideal = np.min(np.asarray(pts), axis=0)
    box = float(np.prod(np.asarray(ref) - ideal))
    assert hypervolume_nd(pts, ref) == pytest.approx(hypervolume_2d(pts, ref) / box)

    # 归一化超体积：盒内全被支配 → 1.0
    assert normalized_hypervolume([(0.0, 0.0, 0.0)], (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)) == pytest.approx(1.0)

    # 三目标：点越靠理想点，HV 越大；加支配点不减少
    ideal3 = np.array([0.3, -0.9, 1.0])
    ref3 = np.array([0.7, -0.7, 2.2])
    one = [(0.5, -0.8, 1.5)]
    two = [(0.5, -0.8, 1.5), (0.4, -0.85, 1.8)]
    hv1 = normalized_hypervolume(one, ref3, ideal3, samples=40000, seed=5)
    hv2 = normalized_hypervolume(two, ref3, ideal3, samples=40000, seed=5)
    assert 0.0 < hv1 <= hv2 <= 1.0
    # 固定 seed 可复现
    assert hv2 == normalized_hypervolume(two, ref3, ideal3, samples=40000, seed=5)

    with pytest.raises(ValueError):
        normalized_hypervolume(one, ideal3, ref3, samples=1000, seed=0)


# --------------------------------------------------------------------------
# 5. 可复现性
# --------------------------------------------------------------------------
def test_nsga2_is_reproducible_with_fixed_seed():
    kwargs = dict(pop_size=24, generations=12, seed=11)
    r1 = nsga2("sparse_resource", LOOSE_BUDGET, **kwargs)
    r2 = nsga2("sparse_resource", LOOSE_BUDGET, **kwargs)
    assert r1.population == r2.population
    assert r1.objectives == r2.objectives
    assert r1.front == r2.front
    assert r1.front_objectives == r2.front_objectives
    assert r1.generations == r2.generations == 12
    assert r1.hypervolume == r2.hypervolume

    # 不同 seed 至少应产生不同的初始轨迹（允许极小概率相同，故比较整段结果）
    r3 = nsga2("sparse_resource", LOOSE_BUDGET, pop_size=24, generations=12, seed=12)
    assert (r3.population, r3.front_objectives) != (r1.population, r1.front_objectives)

    # 输入参数校验
    with pytest.raises(ValueError):
        nsga2("sparse_resource", LOOSE_BUDGET, pop_size=1)
    with pytest.raises(ValueError):
        nsga2("sparse_resource", LOOSE_BUDGET, generations=-1)
    with pytest.raises(KeyError):
        nsga2("no_such_tier", LOOSE_BUDGET)


# --------------------------------------------------------------------------
# 6. 分级策略
# --------------------------------------------------------------------------
@pytest.mark.parametrize("tier,budget", [("capital_core", 2.2), ("sparse_resource", 3.0), ("dense_hub", 2.0)])
def test_tiered_returns_feasible_floor_respecting_solution(tier, budget):
    lv = tiered(tier, budget, seed=0)
    assert isinstance(lv, tuple) and len(lv) == 6
    assert all(isinstance(v, int) and 0 <= v <= 2 for v in lv)
    assert all(lv[k] >= FLOOR[tier][k] for k in range(6))
    ok, violation = feasibility(lv, tier, budget)
    assert ok is True and violation == 0.0
    assert cost_of(lv) <= budget + 1e-9
    # 确定性
    assert tiered(tier, budget, seed=0) == lv


def test_tiered_degrades_gracefully_when_budget_too_small():
    budget = 0.5           # 低于所有情境的底线成本
    for tier in ("capital_core", "sparse_resource"):
        assert all_feasible(tier, budget) == []
        lv = tiered(tier, budget, seed=0)
        # 仍返回合法且满足底线的等级向量，但如实报告不可行
        assert all(lv[k] >= FLOOR[tier][k] for k in range(6))
        ok, _ = feasibility(lv, tier, budget)
        assert ok is False
        row = compare_policies(tier, budget, seed=0)
        assert row["n_feasible"] == 0
        for name in ("uniform_open", "uniform_strict", "tiered"):
            assert row[name]["feasible"] is False


def test_reference_policies_semantics():
    assert uniform_open("sparse_resource") == FLOOR["sparse_resource"]
    assert uniform_open("capital_core") == (2, 1, 1, 2, 1, 1)

    # 不给预算：不计成本的一刀切，全 2
    assert uniform_strict("capital_core") == (2, 2, 2, 2, 2, 2)
    # 给预算：装得下就全 2，装不下退到统一等级 1（底线修复），再不行退到底线
    assert uniform_strict("capital_core", 3.0) == (2, 2, 2, 2, 2, 2)
    assert uniform_strict("capital_core", 2.2) == FLOOR["capital_core"]
    assert uniform_strict("sparse_resource", 2.3) == (2, 2, 2, 2, 2, 2)
    # 全 2 需要 2.26 > 2.2，退到统一等级 1
    assert uniform_strict("sparse_resource", 2.2) == (1, 1, 1, 1, 1, 1)
    assert uniform_strict("sparse_resource", 1.5) == (1, 1, 1, 1, 1, 1)
    assert uniform_strict("sparse_resource", 1.0) == FLOOR["sparse_resource"]

    # knee point 选中"折中"解，而不是任一极端
    front = [(0.0, 10.0), (10.0, 0.0), (5.0, 5.0)]
    assert knee_index(front) == 2
    assert knee_index([(1.0, 1.0)]) == 0


def test_compare_policies_structure_and_improvements():
    row = compare_policies("sparse_resource", 2.6, seed=0)
    assert row["tier"] == "sparse_resource"
    assert row["budget"] == pytest.approx(2.6)
    assert row["seed"] == 0
    assert row["n_feasible"] == len(all_feasible("sparse_resource", 2.6))
    assert row["floor_cost"] == pytest.approx(floor_cost("sparse_resource"))

    for name in ("uniform_open", "uniform_strict", "tiered"):
        entry = row[name]
        assert entry is row["policies"][name]
        assert set(entry) == {"name", "levels", "cost", "incident_mult", "completion", "feasible", "violation"}
        assert len(entry["levels"]) == 6
        obj = objectives(tuple(entry["levels"]), "sparse_resource")
        assert entry["cost"] == pytest.approx(obj[2])
        assert entry["incident_mult"] == pytest.approx(obj[0])
        assert entry["completion"] == pytest.approx(-obj[1])
        assert 0.0 < entry["incident_mult"] <= 1.0
        assert 0.6 <= entry["completion"] <= 1.0

    assert row["tiered"]["feasible"] is True
    for key in ("tiered_vs_uniform_strict", "tiered_vs_uniform_open"):
        imp = row["improvement"][key]
        assert set(imp) == {
            "incident_reduction",
            "incident_reduction_pct",
            "completion_gain",
            "completion_gain_pct",
            "cost_delta",
        }
        assert imp["incident_reduction_pct"] == pytest.approx(100.0 * imp["incident_reduction"])

    # 分级策略在该预算下应不劣于"仅底线"基准（预算足够时）
    imp = row["improvement"]["tiered_vs_uniform_open"]
    assert imp["incident_reduction"] > 0.0
    assert imp["completion_gain"] > 0.0


# --------------------------------------------------------------------------
# 7. 预算扫描
# --------------------------------------------------------------------------
def test_budget_sweep_shape_and_fields():
    rows = budget_sweep(tiers=["capital_core", "sparse_resource"], budgets=[1.0, 2.2, 3.0], seed=0)
    assert len(rows) == 2 * 3
    assert [(r["tier"], r["budget"]) for r in rows] == [
        (t, b) for t in ("capital_core", "sparse_resource") for b in (1.0, 2.2, 3.0)
    ]

    required = {
        "tier",
        "budget",
        "seed",
        "floor_cost",
        "n_feasible",
        "policies",
        "uniform_open",
        "uniform_strict",
        "tiered",
        "improvement",
    }
    for row in rows:
        assert required <= set(row)
        assert set(row["policies"]) == {"uniform_open", "uniform_strict", "tiered"}
        for name in ("uniform_open", "uniform_strict", "tiered"):
            entry = row[name]
            assert {"levels", "cost", "incident_mult", "completion", "feasible"} <= set(entry)
            assert 0.0 < entry["incident_mult"] <= 1.0
            assert 0.6 <= entry["completion"] <= 1.0
            if entry["feasible"]:
                assert entry["cost"] <= row["budget"] + 1e-9

    # 默认参数：6 个情境 × 6 个预算档位
    default_rows = budget_sweep(seed=0)
    assert len(default_rows) == len(TIERS) * len(DEFAULT_BUDGETS) == 36
    assert [r["budget"] for r in default_rows[:6]] == list(DEFAULT_BUDGETS)
    assert {r["tier"] for r in default_rows} == set(TIERS)

    with pytest.raises(KeyError):
        budget_sweep(tiers=["no_such_tier"], budgets=[2.0])

