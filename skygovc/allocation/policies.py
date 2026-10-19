"""三个参考策略 + 由 NSGA-II 前沿导出的分级（tiered）策略。

策略语义
--------
``uniform_open``（统一最低配置）
    每个能力族只做到该情境的底线 ``FLOOR[tier]``。这是"合规但不再投入"的
    基准，成本最低，通常也是三个策略里事件率最高的。

``uniform_strict``（统一强化配置）
    所有能力族一律拉到最高等级 2。当预算不足以支撑全 2 时，退化为
    "统一标准等级 1（并按底线修复）"，再不行退到仅满足底线的配置。
    它代表"不做差异化、按最高标准一刀切"的做法，在低预算下往往直接
    不可行——这正是论文中"分级配置相对统一强化更省预算"的对照。

``tiered``（分级配置）
    在 NSGA-II 第一前沿上按 **knee point** 选解：目标归一化到单位盒后，
    取到理想点欧氏距离最小的解。若前沿中不存在可行解（预算低于该情境的
    底线成本），则依次退化为"种群内可行解"、"违反量最小的解"，保证函数
    永远返回一个满足 ``FLOOR`` 的等级向量，并用 ``feasible`` 字段暴露它
    是否真的在预算内。

所有函数都是确定性的：给定 ``seed``，``tiered`` / ``compare_policies`` /
``budget_sweep`` 的输出完全可复现（候选集先按等级字典序排序，再取 knee）。
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .model import (
    BUDGET_TOL,
    FLOOR,
    N_CAP,
    LEVEL_MAX,
    TIERS,
    all_feasible,
    cost_of,
    feasibility,
    floor_cost,
    objectives,
    repair,
)
from .nsga2 import nsga2

__all__ = [
    "POLICY_NAMES",
    "DEFAULT_BUDGETS",
    "uniform_strict",
    "uniform_open",
    "tiered",
    "knee_index",
    "compare_policies",
    "budget_sweep",
]

#: 策略名（固定顺序，便于论文表格列对齐）。
POLICY_NAMES: Tuple[str, str, str] = ("uniform_open", "uniform_strict", "tiered")

#: ``budget_sweep`` 的默认预算档位（亿元/年）。
DEFAULT_BUDGETS: Tuple[float, ...] = (1.0, 1.4, 1.8, 2.2, 2.6, 3.0)


# --------------------------------------------------------------------------
# 两个"不做优化"的参考策略
# --------------------------------------------------------------------------
def uniform_open(tier: str) -> Tuple[int, ...]:
    """统一最低配置：每个能力族取该情境的底线 ``FLOOR[tier]``。"""
    if tier not in FLOOR:
        raise KeyError(f"未知治理情境 tier={tier!r}，可选 {TIERS}")
    return tuple(FLOOR[tier])


def uniform_strict(tier: str, budget: float | None = None) -> Tuple[int, ...]:
    """统一强化配置：全部能力族取等级 2。

    ``budget`` 省略（默认）时直接返回全 2 配置——这是"不计预算的一刀切
    上限"，``compare_policies`` 会把它是否超预算如实写进 ``feasible``。
    给出 ``budget`` 时按"全 2 → 统一等级 1（底线修复）→ 仅底线"的顺序
    退让，返回第一个装得下预算的配置；若三个都装不下则返回仅满足底线的
    配置（此时必然不可行，但仍是合法等级向量）。
    """
    if tier not in FLOOR:
        raise KeyError(f"未知治理情境 tier={tier!r}，可选 {TIERS}")
    full = tuple([LEVEL_MAX] * N_CAP)
    if budget is None:
        return full
    standard = repair([1] * N_CAP, tier)
    floor = tuple(FLOOR[tier])
    for candidate in (full, standard, floor):
        if cost_of(candidate) <= float(budget) + BUDGET_TOL:
            return candidate
    return floor


# --------------------------------------------------------------------------
# 分级策略
# --------------------------------------------------------------------------
def knee_index(points: Sequence[Sequence[float]]) -> int:
    """knee point：归一化后离理想点欧氏距离最小的下标（并列取靠前者）。"""
    if not points:
        raise ValueError("points 不能为空")
    A = np.asarray([[float(v) for v in p] for p in points], dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("points 必须是等长的目标向量序列")
    lo = A.min(axis=0)
    hi = A.max(axis=0)
    span = hi - lo
    span = np.where(span > 1e-12, span, 1.0)      # 退化维度不参与区分
    Z = (A - lo) / span
    dist = np.sqrt(np.sum(Z * Z, axis=1))
    return int(np.argmin(dist))


def _tiered_candidates(tier: str, budget: float, seed: int) -> List[Tuple[int, ...]]:
    """从一次 NSGA-II 运行中挑出候选等级向量（已排序、已去重）。"""
    result = nsga2(tier, budget, seed=seed)
    front = [tuple(int(v) for v in lv) for lv in result.front]
    population = [tuple(int(v) for v in lv) for lv in result.population]

    feasible_front = sorted({lv for lv in front if feasibility(lv, tier, budget)[0]})
    if feasible_front:
        return feasible_front

    feasible_pop = sorted({lv for lv in population if feasibility(lv, tier, budget)[0]})
    if feasible_pop:
        return feasible_pop

    # 预算低于该情境的底线成本：所有解都不可行，退化为"违反量最小"
    scored = sorted((feasibility(lv, tier, budget)[1], lv) for lv in set(population))
    if not scored:                                  # 理论上不可达
        return [tuple(FLOOR[tier])]
    vmin = scored[0][0]
    return [lv for v, lv in scored if v <= vmin + 1e-15]


def tiered(tier: str, budget: float, seed: int = 0) -> Tuple[int, ...]:
    """分级配置：NSGA-II 前沿 + knee point 选解。

    返回值恒为合法等级向量且满足 ``FLOOR[tier]``；是否满足预算请用
    ``feasibility(...)` 或 ``compare_policies(...)["tiered"]["feasible"]``
    判断（预算低于底线成本时不可能可行）。
    """
    if tier not in FLOOR:
        raise KeyError(f"未知治理情境 tier={tier!r}，可选 {TIERS}")
    candidates = _tiered_candidates(tier, float(budget), int(seed))
    if len(candidates) == 1:
        return candidates[0]
    objs = [objectives(lv, tier) for lv in candidates]
    return candidates[knee_index(objs)]


# --------------------------------------------------------------------------
# 比较与扫描
# --------------------------------------------------------------------------
def _describe(name: str, levels: Tuple[int, ...], tier: str, budget: float) -> Dict[str, object]:
    """把一个等级向量整理成论文表格用的一行。"""
    inc, neg_comp, cost = objectives(levels, tier)
    ok, violation = feasibility(levels, tier, budget)
    return {
        "name": name,
        "levels": [int(v) for v in levels],
        "cost": float(cost),
        "incident_mult": float(inc),
        "completion": float(-neg_comp),
        "feasible": bool(ok),
        "violation": float(violation),
    }


def _improvement(base: Dict[str, object], cur: Dict[str, object]) -> Dict[str, float]:
    """``cur`` 相对 ``base`` 的改进量（事件率下降为正好，完成率上升为正好）。"""
    base_inc = float(base["incident_mult"])          # type: ignore[arg-type]
    cur_inc = float(cur["incident_mult"])            # type: ignore[arg-type]
    base_comp = float(base["completion"])            # type: ignore[arg-type]
    cur_comp = float(cur["completion"])              # type: ignore[arg-type]
    incident_reduction = (base_inc - cur_inc) / base_inc if base_inc > 0 else 0.0
    completion_gain = cur_comp - base_comp
    completion_gain_pct = completion_gain / base_comp if base_comp > 0 else 0.0
    return {
        "incident_reduction": float(incident_reduction),
        "incident_reduction_pct": float(100.0 * incident_reduction),
        "completion_gain": float(completion_gain),
        "completion_gain_pct": float(100.0 * completion_gain_pct),
        "cost_delta": float(float(cur["cost"]) - float(base["cost"])),  # type: ignore[arg-type]
    }


def compare_policies(tier: str, budget: float, seed: int = 0) -> Dict[str, object]:
    """在同一 ``tier``/``budget`` 下比较三个策略。

    返回字典字段：

    ``tier`` / ``budget`` / ``seed`` / ``floor_cost`` / ``n_feasible``
        元信息；``n_feasible`` 是穷举 729 个解后得到的可行解个数（0 表示
        该预算连底线配置都装不下，三个策略都会是不可行）。
    ``policies``
        ``{"uniform_open": {...}, "uniform_strict": {...}, "tiered": {...}}``，
        每个子字典含 ``levels/cost/incident_mult/completion/feasible/violation``。
    ``uniform_open`` / ``uniform_strict`` / ``tiered``
        同上的扁平别名，方便直接取用。
    ``improvement``
        ``{"tiered_vs_uniform_strict": {...}, "tiered_vs_uniform_open": {...}}``，
        每项含 ``incident_reduction``（比例，正=事件更少）、``incident_reduction_pct``、
        ``completion_gain``（绝对差）、``completion_gain_pct``、``cost_delta``。
    """
    if tier not in FLOOR:
        raise KeyError(f"未知治理情境 tier={tier!r}，可选 {TIERS}")
    budget = float(budget)
    rows = {
        "uniform_open": _describe("uniform_open", uniform_open(tier), tier, budget),
        "uniform_strict": _describe("uniform_strict", uniform_strict(tier, budget), tier, budget),
        "tiered": _describe("tiered", tiered(tier, budget, seed=seed), tier, budget),
    }
    out: Dict[str, object] = {
        "tier": tier,
        "budget": budget,
        "seed": int(seed),
        "floor_cost": float(floor_cost(tier)),
        "n_feasible": len(all_feasible(tier, budget)),
        "policies": rows,
        "improvement": {
            "tiered_vs_uniform_strict": _improvement(rows["uniform_strict"], rows["tiered"]),
            "tiered_vs_uniform_open": _improvement(rows["uniform_open"], rows["tiered"]),
        },
    }
    for name in POLICY_NAMES:                       # 扁平别名
        out[name] = rows[name]
    return out


def budget_sweep(
    tiers: Iterable[str] | None = None,
    budgets: Iterable[float] | None = None,
    seed: int = 0,
) -> List[Dict[str, object]]:
    """``tier × budget`` 网格上的三策略对比，返回 ``len(tiers)*len(budgets)`` 行。

    每一行就是 :func:`compare_policies` 的输出，直接可 ``pandas.DataFrame``
    或写 CSV 供论文画图。
    """
    tier_list = list(TIERS) if tiers is None else list(tiers)
    budget_list = list(DEFAULT_BUDGETS) if budgets is None else [float(b) for b in budgets]
    for t in tier_list:
        if t not in FLOOR:
            raise KeyError(f"未知治理情境 tier={t!r}，可选 {TIERS}")
    return [compare_policies(t, b, seed=seed) for t in tier_list for b in budget_list]
