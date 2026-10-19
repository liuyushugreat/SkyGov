"""问题定义与效应模型（budget-constrained capability allocation）。

本模块是 ``allocation`` 子包中**唯一**承载"论文数字"的地方：成本表、
各治理情境的能力底线、事件率乘子与服务完成率的系数全部在此固定为常量，
``nsga2`` / ``policies`` / 测试都只调用这里的函数，不重复硬编码任何数字，
从而保证论文表格与代码完全同源。

建模约定
--------
1. 决策变量：6 个能力族（:data:`CAP_KEYS`，顺序 == ``skygovc.core.CAPABILITIES``）
   各取离散等级 ``ℓ_k ∈ {0,1,2}``，解空间 ``3**6 = 729``，可穷举，因此
   ``all_feasible`` / ``pareto_front`` 给出的**穷举真前沿**是本文件里 NSGA-II
   的黄金基准。
2. 可行性 = 底线约束 ``ℓ_k >= FLOOR[tier][k]`` 且 预算硬约束
   ``cost(ℓ) <= budget``（单位：亿元/年）。
3. 目标（全部最小化）：

   * ``f1 = incident_mult = exp(-Σ_k w_inc[tier][k] · s_k)``
   * ``f2 = -completion``，``completion = clip(0.60 + 0.30·Σ_k w_svc[tier][k]·s_k, 0, 1)``
   * ``f3 = Σ_k COST[k][ℓ_k]``

   其中能力强度 ``s_k = ℓ_k / 2 ∈ {0, 0.5, 1}``。``f1``、``f2`` 关于 ``ℓ``
   单调改进，``f3`` 关于 ``ℓ`` 单调增加，故三目标两两冲突。

数值细节
--------
* 所有等级向量在内部统一表示为 ``tuple[int, ...]``（长度 6）。
* 预算比较留出 ``BUDGET_TOL = 1e-9`` 的浮点容差，使得
  ``feasibility``、``all_feasible`` 与 ``evaluate`` 对"刚好等于预算"的解
  给出**一致**的判断。
* 违反量（用于 Deb 约束支配）是标量：

      violation = Σ_k max(0, FLOOR[tier][k] - ℓ_k)          # 单位：等级
                + max(0, cost - budget) / max(budget, 1.0)   # 预算相对超支

  可行解的违反量恒为 ``0.0``。
"""
from __future__ import annotations

import math
from itertools import product
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..core import CAPABILITIES, TIERS

__all__ = [
    # 常量
    "CAP_KEYS",
    "CAPABILITIES",
    "TIERS",
    "N_CAP",
    "LEVELS",
    "LEVEL_MAX",
    "SPACE_SIZE",
    "BUDGET_TOL",
    "COST",
    "FLOOR",
    "W_INCIDENT",
    "W_SERVICE",
    # 函数
    "decode",
    "repair",
    "cost_of",
    "floor_cost",
    "feasibility",
    "objectives",
    "evaluate",
    "all_levels",
    "all_feasible",
    "dominates",
    "non_dominated_indices",
    "pareto_front",
]

# --------------------------------------------------------------------------
# 常量：能力族与解空间
# --------------------------------------------------------------------------
#: 6 个能力族，顺序严格等于 ``skygovc.core.CAPABILITIES``。
CAP_KEYS: Tuple[str, ...] = tuple(CAPABILITIES)

#: 能力等级取值。
LEVELS: Tuple[int, int, int] = (0, 1, 2)

#: 最高等级。
LEVEL_MAX: int = 2

#: 能力族数量（6）。
N_CAP: int = len(CAP_KEYS)

#: 解空间大小（729）。
SPACE_SIZE: int = len(LEVELS) ** N_CAP

#: 预算比较的浮点容差（亿元/年）。
BUDGET_TOL: float = 1e-9

#: 目标维度。
N_OBJECTIVES: int = 3

# --------------------------------------------------------------------------
# 常量：成本表（亿元/年）
# --------------------------------------------------------------------------
#: ``COST[capability][level] -> 亿元/年``。
#:
#: 注意：规范里写的是 ``dict[str, dict[str, int]]``，但数值本身是小数
#: （0.05 亿元等），故实现取 ``dict[str, dict[int, float]]``——内层键为
#: **整数等级**、值为 **float**，便于 ``COST[cap][level]`` 直接索引。
COST: Dict[str, Dict[int, float]] = {
    "rule": {0: 0.05, 1: 0.12, 2: 0.22},
    "spatial": {0: 0.08, 1: 0.20, 2: 0.38},
    "service": {0: 0.10, 1: 0.26, 2: 0.50},
    "sensing": {0: 0.12, 1: 0.34, 2: 0.68},
    "emergency": {0: 0.06, 1: 0.15, 2: 0.29},
    "data": {0: 0.04, 1: 0.10, 2: 0.19},
}

# --------------------------------------------------------------------------
# 常量：各治理情境的最低能力底线
# --------------------------------------------------------------------------
#: ``FLOOR[tier][k]``：情境 ``tier`` 下第 ``k`` 个能力族的等级下限（0/1/2）。
FLOOR: Dict[str, Tuple[int, ...]] = {
    "capital_core":     (2, 1, 1, 2, 1, 1),
    "dense_hub":        (1, 2, 1, 2, 1, 1),
    "market_driven":    (1, 1, 1, 1, 1, 2),
    "mountainous":      (1, 1, 1, 1, 2, 1),
    "border_coastal":   (2, 1, 1, 2, 1, 1),
    "sparse_resource":  (1, 1, 1, 0, 1, 1),

}

# --------------------------------------------------------------------------
# 常量：效应模型系数
# --------------------------------------------------------------------------
#: 事件率乘子的指数系数 ``w_inc[tier][k]``。
W_INCIDENT: Dict[str, Tuple[float, ...]] = {
    "capital_core": (0.18, 0.22, 0.10, 0.30, 0.20, 0.12),
    "dense_hub": (0.12, 0.28, 0.10, 0.32, 0.22, 0.10),
    "market_driven": (0.14, 0.16, 0.26, 0.14, 0.14, 0.24),
    "mountainous": (0.12, 0.24, 0.12, 0.16, 0.26, 0.10),
    "border_coastal": (0.22, 0.24, 0.10, 0.26, 0.24, 0.18),
    "sparse_resource": (0.16, 0.14, 0.20, 0.10, 0.18, 0.22),
}

#: 服务完成率的线性系数 ``w_svc[tier][k]``。
W_SERVICE: Dict[str, Tuple[float, ...]] = {
    "capital_core": (0.10, 0.10, 0.26, 0.08, 0.10, 0.16),
    "dense_hub": (0.12, 0.14, 0.24, 0.10, 0.14, 0.16),
    "market_driven": (0.12, 0.10, 0.30, 0.08, 0.10, 0.30),
    "mountainous": (0.10, 0.14, 0.20, 0.16, 0.24, 0.10),
    "border_coastal": (0.14, 0.12, 0.22, 0.10, 0.20, 0.16),
    "sparse_resource": (0.16, 0.10, 0.26, 0.06, 0.16, 0.20),
}

#: 服务完成率的基线（等级全 0 时的完成率）。
COMPLETION_BASE: float = 0.60

#: 服务完成率的线性增益系数。
COMPLETION_GAIN: float = 0.30

#: 事件率乘子的支配比较容差（用于 :func:`dominates`）。
DOMINANCE_EPS: float = 1e-12


# --------------------------------------------------------------------------
# 内部：穷举表（一次性构建，之后所有目标/成本查询都是 O(1)）
# --------------------------------------------------------------------------
#: 全部 729 个等级向量，字典序（即 ``itertools.product`` 顺序）。
_ALL_LEVELS: List[Tuple[int, ...]] = [tuple(int(v) for v in row) for row in product(LEVELS, repeat=N_CAP)]

#: 全部等级向量的矩阵形式，形状 ``(729, 6)``。
_L_ALL = np.array(list(product(LEVELS, repeat=N_CAP)), dtype=np.int64)


def _index_of(levels: Sequence[int]) -> int:
    """把等级向量映射为 ``0..728`` 的 base-3 下标（并做合法性校验）。"""
    lv = list(levels)
    if len(lv) != N_CAP:
        raise ValueError(f"等级向量必须长度为 {N_CAP}，收到 {len(lv)}: {lv!r}")
    idx = 0
    for v in lv:
        iv = int(v)
        if iv != v:
            raise ValueError(f"等级必须是整数 0/1/2，收到 {v!r}")
        if iv < 0 or iv > LEVEL_MAX:
            raise ValueError(f"等级必须落在 [0, {LEVEL_MAX}]，收到 {iv}")
        idx = idx * len(LEVELS) + iv
    return idx


def _cost_vector() -> np.ndarray:
    """预计算全部 729 个解的成本（与 tier 无关）。"""
    out = np.empty(SPACE_SIZE, dtype=np.float64)
    for i, lv in enumerate(_ALL_LEVELS):
        out[i] = sum(COST[cap][lv[k]] for k, cap in enumerate(CAP_KEYS))
    return out


_COST_ALL: np.ndarray = _cost_vector()

#: 每个 tier 的 ``(729, 3)`` 目标矩阵缓存。
_OBJECTIVE_TABLE: Dict[str, np.ndarray] = {}


def _compute_objectives(levels: Sequence[int], tier: str) -> Tuple[float, float, float]:
    """按固定公式计算 ``(incident_mult, -completion, cost)``（无缓存、可审计）。"""
    if tier not in W_INCIDENT:
        raise KeyError(f"未知治理情境 tier={tier!r}，可选 {sorted(W_INCIDENT)}")
    lv = tuple(int(v) for v in levels)
    if len(lv) != N_CAP:
        raise ValueError(f"等级向量必须长度为 {N_CAP}，收到 {len(lv)}")
    strength = [lv[k] / 2.0 for k in range(N_CAP)]        # s_k ∈ {0, 0.5, 1}
    w_inc = W_INCIDENT[tier]
    w_svc = W_SERVICE[tier]
    exponent = sum(w_inc[k] * strength[k] for k in range(N_CAP))
    incident_mult = math.exp(-exponent)
    completion = COMPLETION_BASE + COMPLETION_GAIN * sum(w_svc[k] * strength[k] for k in range(N_CAP))
    completion = min(1.0, max(0.0, completion))
    cost = sum(COST[cap][lv[k]] for k, cap in enumerate(CAP_KEYS))
    return (float(incident_mult), float(-completion), float(cost))


def _objective_table(tier: str) -> np.ndarray:
    """返回 ``tier`` 对应的 ``(729, 3)`` 目标矩阵（惰性构建并缓存）。"""
    table = _OBJECTIVE_TABLE.get(tier)
    if table is None:
        if tier not in W_INCIDENT:
            raise KeyError(f"未知治理情境 tier={tier!r}，可选 {sorted(W_INCIDENT)}")
        rows = np.empty((SPACE_SIZE, N_OBJECTIVES), dtype=np.float64)
        for i, lv in enumerate(_ALL_LEVELS):
            rows[i] = _compute_objectives(lv, tier)
        _OBJECTIVE_TABLE[tier] = rows
        table = rows
    return table


# --------------------------------------------------------------------------
# 决策变量层
# --------------------------------------------------------------------------
def decode(x: Sequence[int], tier: str) -> Tuple[int, ...]:
    """把长度为 6 的整数序列解码成合法等级向量。

    仅做**表示层**归一：四舍五入到整数并截断到 ``[0, 2]``；**不**强制
    ``FLOOR`` 底线（底线由 :func:`repair` 负责），这样 NSGA-II 内部可以
    表达违反底线的个体并用约束支配处理。
    """
    if tier not in FLOOR:
        raise KeyError(f"未知治理情境 tier={tier!r}，可选 {TIERS}")
    vals = list(x)
    if len(vals) != N_CAP:
        raise ValueError(f"解码需要长度 {N_CAP} 的向量，收到 {len(vals)}: {vals!r}")
    out = []
    for v in vals:
        iv = int(round(float(v)))
        out.append(0 if iv < 0 else (LEVEL_MAX if iv > LEVEL_MAX else iv))
    return tuple(out)


def repair(levels: Sequence[int], tier: str) -> Tuple[int, ...]:
    """修复算子：把等级向量拉回 ``[0, 2]`` 且逐个不低于 ``FLOOR[tier]``。

    NSGA-II 的初始化、交叉与变异之后都会调用本函数，因此最终种群**必然**
    满足底线约束；只有预算约束可能被违反，交给 Deb 约束支配处理。
    """
    if tier not in FLOOR:
        raise KeyError(f"未知治理情境 tier={tier!r}，可选 {TIERS}")
    floor = FLOOR[tier]
    lv = list(levels)
    if len(lv) != N_CAP:
        raise ValueError(f"修复需要长度 {N_CAP} 的向量，收到 {len(lv)}")
    out = []
    for k in range(N_CAP):
        iv = int(round(float(lv[k])))
        ib = floor[k]
        if iv < ib:
            iv = ib
        if iv > LEVEL_MAX:
            iv = LEVEL_MAX
        out.append(iv)
    return tuple(out)


# --------------------------------------------------------------------------
# 评价层
# --------------------------------------------------------------------------
def cost_of(levels: Sequence[int]) -> float:
    """年度总成本 ``f3``（亿元/年）。"""
    return float(_COST_ALL[_index_of(levels)])


def floor_cost(tier: str) -> float:
    """该情境"仅满足底线"的配置成本（该 tier 的可行成本下界）。"""
    return cost_of(FLOOR[tier])


def feasibility(levels: Sequence[int], tier: str, budget: float) -> Tuple[bool, float]:
    """返回 ``(是否可行, 违反量)``。

    可行当且仅当 ``ℓ_k >= FLOOR[tier][k]``（全部 k）且 ``cost <= budget``。
    违反量定义见模块文档；可行解恒为 ``0.0``。
    """
    if tier not in FLOOR:
        raise KeyError(f"未知治理情境 tier={tier!r}，可选 {TIERS}")
    floor = FLOOR[tier]
    lv = tuple(int(v) for v in levels)
    if len(lv) != N_CAP:
        raise ValueError(f"需要长度 {N_CAP} 的向量，收到 {len(lv)}")
    cost = cost_of(lv)
    violation = 0.0
    for k in range(N_CAP):
        gap = floor[k] - lv[k]
        if gap > 0:
            violation += float(gap)
    over = cost - float(budget)
    if over > BUDGET_TOL:
        violation += over / max(float(budget), 1.0)
    return (violation == 0.0), float(violation)


def objectives(levels: Sequence[int], tier: str) -> Tuple[float, float, float]:
    """返回 ``(f1, f2, f3) = (incident_mult, -completion, cost)``。"""
    return tuple(  # type: ignore[return-value]
        float(v) for v in _objective_table(tier)[_index_of(levels)]
    )


def evaluate(x: Sequence[int], tier: str, budget: float) -> Tuple[Tuple[float, float, float], float]:
    """一步完成 解码 → 目标 → 可行性，返回 ``(objectives, violation)``。"""
    levels = decode(x, tier)
    objs = objectives(levels, tier)
    _, violation = feasibility(levels, tier, budget)
    return objs, violation


def all_levels() -> List[Tuple[int, ...]]:
    """返回全部 729 个等级向量（字典序），供穷举使用。"""
    return list(_ALL_LEVELS)


# --------------------------------------------------------------------------
# 穷举层
# --------------------------------------------------------------------------
def all_feasible(tier: str, budget: float) -> List[Tuple[int, ...]]:
    """穷举 729 个解，返回全部可行解（满足底线且不超预算），按字典序。"""
    if tier not in FLOOR:
        raise KeyError(f"未知治理情境 tier={tier!r}，可选 {TIERS}")
    floor = np.asarray(FLOOR[tier], dtype=np.int64)
    mask = np.all(_L_ALL >= floor[None, :], axis=1)          # 底线约束
    mask &= _COST_ALL <= float(budget) + BUDGET_TOL          # 预算硬约束
    idx = np.flatnonzero(mask)
    return [_ALL_LEVELS[int(i)] for i in idx]


# --------------------------------------------------------------------------
# Pareto 支配层
# --------------------------------------------------------------------------
def dominates(a: Sequence[float], b: Sequence[float], eps: float = DOMINANCE_EPS) -> bool:
    """严格 Pareto 支配（最小化语义）。

    ``a`` 支配 ``b`` 当且仅当 ``a_i <= b_i + eps``（所有 i）且存在
    ``a_j < b_j - eps``。因此：

    * 非自反：``dominates(a, a)`` 恒为 ``False``；
    * 反对称：若 ``dominates(a, b)`` 为真，则 ``dominates(b, a)`` 必为假。
    """
    fa = [float(v) for v in a]
    fb = [float(v) for v in b]
    if len(fa) != len(fb):
        raise ValueError(f"支配比较需要等长目标向量：{len(fa)} != {len(fb)}")
    strictly_better = False
    for x, y in zip(fa, fb):
        if x > y + eps:
            return False
        if x < y - eps:
            strictly_better = True
    return strictly_better


def non_dominated_indices(points: Sequence[Sequence[float]], eps: float = DOMINANCE_EPS) -> List[int]:
    """返回目标向量集合中非支配点的下标（升序）。"""
    if not points:
        return []
    F = np.asarray([[float(v) for v in p] for p in points], dtype=np.float64)
    if F.ndim != 2:
        raise ValueError("points 必须是等长的目标向量序列")
    le = np.all(F[:, None, :] <= F[None, :, :] + eps, axis=2)
    lt = np.any(F[:, None, :] < F[None, :, :] - eps, axis=2)
    dominated_by_other = np.any(le & lt, axis=0)   # 是否存在某个 j 支配 i
    return [int(i) for i in np.flatnonzero(~dominated_by_other)]


def pareto_front(solutions: List[Sequence[int]], tier: str) -> List[int]:
    """穷举支撑：给定等级向量列表，返回**非支配解**的下标。

    参数 ``solutions`` 是等级向量（长度 6），目标由 :func:`objectives` 计算；
    ``tier`` 决定效应模型。返回值可直接用于
    ``[solutions[i] for i in pareto_front(solutions, tier)]``。
    """
    if not solutions:
        return []
    pts = [objectives(tuple(s), tier) for s in solutions]
    return non_dominated_indices(pts)

