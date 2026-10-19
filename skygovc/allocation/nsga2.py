"""从零实现的 NSGA-II（快速非支配排序 + 拥挤距离 + 二元锦标赛）。

不依赖 pymoo / deap，仅用标准库 + numpy。实现要点：

============ ==========================================================
模块要素      本实现的选择
============ ==========================================================
编码          6 个整数等级 ``ℓ_k ∈ {0,1,2}``，个体即 ``tuple[int, ...]``
初始化        ``ℓ_k ~ U{FLOOR[tier][k], …, 2}``，**天然满足底线**
修复算子      交叉/变异后调用 :func:`skygovc.allocation.model.repair`
约束处理      Deb 约束支配原则（可行优先；同为不可行时违反量小者优先）
交叉          均匀交叉（整型友好，``crossover_rate`` 控制是否重组）
变异          阶跃变异（±1；撞到边界/底线时反向），``mutation_rate`` 为逐基因概率
选择          二元锦标赛（违反量 → 秩 → 拥挤距离）
环境选择      (μ+λ)：父代 ∪ 子代，按前沿逐层填充，末层用拥挤距离截断
============ ==========================================================

``seed`` 固定时结果**完全可复现**：唯一的随机源是 ``random.Random(seed)``，
所有排序与截断都有确定的次序（前沿内下标升序、拥挤距离并列时取靠前者）。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .hypervolume import normalized_hypervolume
from .model import (
    DOMINANCE_EPS,
    FLOOR,
    LEVEL_MAX,
    N_CAP,
    TIERS,
    evaluate,
    repair,
)

__all__ = [
    "NSGA2Result",
    "nsga2",
    "fast_non_dominated_sort",
    "crowding_distance",
    "constraint_dominance_matrix",
]

#: 违反量小于该阈值即视为可行解（与 ``model.BUDGET_TOL`` 语义一致）。
FEASIBILITY_TOL: float = 1e-12


@dataclass
class NSGA2Result:
    """NSGA-II 一次运行的完整产出。"""

    #: 最终种群（等级向量，每个元素为 6 个整数）。
    population: List[List[int]] = field(default_factory=list)
    #: 与 ``population`` 逐项对应的目标向量 ``(f1, f2, f3)``。
    objectives: List[Tuple[float, float, float]] = field(default_factory=list)
    #: 第一前沿的等级向量（已按等级去重，保持前沿内次序）。
    front: List[List[int]] = field(default_factory=list)
    #: 与 ``front`` 逐项对应的目标向量。
    front_objectives: List[Tuple[float, float, float]] = field(default_factory=list)
    #: 实际执行的代数。
    generations: int = 0
    #: 第一前沿在**归一化目标空间**中的超体积（以最终种群的理想点/nadir
    #: 为归一化基准，参考点取 nadir 外扩 5%，蒙特卡洛 10k 样本，seed 与运行
    #: 一致，故可复现）。
    hypervolume: float = 0.0


# --------------------------------------------------------------------------
# 非支配排序
# --------------------------------------------------------------------------
def constraint_dominance_matrix(
    objs: Sequence[Sequence[float]],
    violations: Optional[Sequence[float]] = None,
    eps: float = DOMINANCE_EPS,
) -> np.ndarray:
    """返回 ``(n, n)`` 布尔矩阵，``M[i, j]`` 表示"在约束支配意义下 i 支配 j"。

    规则（Deb 约束支配）：

    1. i 可行、j 不可行 → i 支配 j；
    2. 两者都可行 → 按 Pareto 支配；
    3. 两者都不可行 → 违反量小者支配大者；
    4. 其余（同为不可行且违反量相等 / 互不支配）→ 不支配。

    对角线恒为 ``False``（否决自反性）。
    """
    F = np.asarray([[float(v) for v in row] for row in objs], dtype=np.float64)
    n = F.shape[0]
    if violations is None:
        V = np.zeros(n, dtype=np.float64)
    else:
        V = np.asarray([float(v) for v in violations], dtype=np.float64)
        if V.shape != (n,):
            raise ValueError("violations 必须与 objs 等长")
    feasible = V <= FEASIBILITY_TOL

    le = np.all(F[:, None, :] <= F[None, :, :] + eps, axis=2)
    lt = np.any(F[:, None, :] < F[None, :, :] - eps, axis=2)
    obj_dom = le & lt

    fi = feasible[:, None]
    fj = feasible[None, :]
    both_infeasible = (~fi) & (~fj)
    both_feasible = fi & fj

    dom = np.zeros((n, n), dtype=bool)
    dom |= fi & (~fj)                                       # 可行压不可行
    dom |= both_feasible & obj_dom                          # 都可行 → Pareto
    dom |= both_infeasible & (V[:, None] < V[None, :])      # 都不可行 → 违反量
    np.fill_diagonal(dom, False)
    return dom


def fast_non_dominated_sort(
    objs: Sequence[Sequence[float]],
    violations: Optional[Sequence[float]] = None,
    eps: float = DOMINANCE_EPS,
) -> List[List[int]]:
    """快速非支配排序（Deb 2002），返回各前沿的下标列表（前沿内升序）。

    使用向量化的约束支配矩阵 + 逐层剥离；``violations`` 省略时退化为
    标准 Pareto 非支配排序。
    """
    n = len(objs)
    if n == 0:
        return []
    dom = constraint_dominance_matrix(objs, violations, eps=eps)
    remaining = np.ones(n, dtype=bool)
    fronts: List[List[int]] = []
    while remaining.any():
        idx = np.flatnonzero(remaining)
        # 当前前沿 = 在剩余集合内没有被任何点支配的点
        dominators = dom[np.ix_(idx, idx)].sum(axis=0)
        current = idx[dominators == 0]
        if current.size == 0:            # 数值退化保护（理论不可达）
            current = idx
        fronts.append([int(i) for i in current])
        remaining[current] = False
    return fronts


def crowding_distance(points: Sequence[Sequence[float]]) -> List[float]:
    """同一前沿内各点的拥挤距离（边界点取 ``inf``）。

    ``points`` 为该前沿的目标向量（最小化语义）。目标数 ≥ 1；点数 ≤ 2 时
    全部为 ``inf``。
    """
    n = len(points)
    if n == 0:
        return []
    F = np.asarray([[float(v) for v in row] for row in points], dtype=np.float64)
    if F.ndim != 2:
        raise ValueError("points 必须是等长的目标向量序列")
    m = F.shape[1]
    if n <= 2:
        return [float("inf")] * n
    dist = np.zeros(n, dtype=np.float64)
    for d in range(m):
        order = np.argsort(F[:, d], kind="stable")
        vals = F[order, d]
        span = float(vals[-1] - vals[0])
        dist[order[0]] = float("inf")
        dist[order[-1]] = float("inf")
        if span <= 0.0:
            continue
        for pos in range(1, n - 1):
            gap = float(vals[pos + 1] - vals[pos - 1])
            dist[order[pos]] += gap / span
    return [float(v) for v in dist]


# --------------------------------------------------------------------------
# 内部个体表示
# --------------------------------------------------------------------------
class _Individual:
    """轻量个体容器（等级向量 / 目标 / 违反量）。"""

    __slots__ = ("levels", "objectives", "violation")

    def __init__(self, levels: Tuple[int, ...], objectives: Tuple[float, ...], violation: float):
        self.levels = levels
        self.objectives = objectives
        self.violation = violation

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"_Individual(levels={self.levels}, obj={self.objectives}, viol={self.violation:.3g})"


def _evaluate(levels: Tuple[int, ...], tier: str, budget: float) -> _Individual:
    objs, violation = evaluate(levels, tier, budget)
    return _Individual(tuple(int(v) for v in levels), tuple(objs), float(violation))


def _random_levels(rng: random.Random, tier: str) -> Tuple[int, ...]:
    """底线内均匀初始化：``ℓ_k ~ U{FLOOR[tier][k], …, 2}``。"""
    floor = FLOOR[tier]
    return tuple(rng.randint(floor[k], LEVEL_MAX) for k in range(N_CAP))


# --------------------------------------------------------------------------
# 遗传算子
# --------------------------------------------------------------------------
def _uniform_crossover(
    p1: Tuple[int, ...], p2: Tuple[int, ...], tier: str, rng: random.Random
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """均匀交叉：逐基因以 1/2 概率交换，随后做底线修复。"""
    c1: List[int] = []
    c2: List[int] = []
    for k in range(N_CAP):
        if rng.random() < 0.5:
            c1.append(p1[k])
            c2.append(p2[k])
        else:
            c1.append(p2[k])
            c2.append(p1[k])
    return repair(c1, tier), repair(c2, tier)


def _step_mutation(
    levels: Tuple[int, ...], tier: str, rng: random.Random, mutation_rate: float
) -> Tuple[int, ...]:
    """阶跃变异：逐基因以 ``mutation_rate`` 概率 ``±1``；撞界时反向。"""
    floor = FLOOR[tier]
    out = list(levels)
    for k in range(N_CAP):
        if rng.random() >= mutation_rate:
            continue
        step = 1 if rng.random() < 0.5 else -1
        new = out[k] + step
        if new < floor[k] or new > LEVEL_MAX:      # 撞到底线/上界则反向
            new = out[k] - step
        if new < floor[k]:
            new = floor[k]
        if new > LEVEL_MAX:
            new = LEVEL_MAX
        out[k] = new
    return tuple(out)


# --------------------------------------------------------------------------
# 选择
# --------------------------------------------------------------------------
def _rank_population(
    pop: Sequence[_Individual],
) -> Tuple[List[List[int]], List[int], List[float]]:
    """返回 ``(fronts, rank, crowding)``，均与 ``pop`` 下标对齐。"""
    objs = [ind.objectives for ind in pop]
    viols = [ind.violation for ind in pop]
    fronts = fast_non_dominated_sort(objs, viols)
    rank = [0] * len(pop)
    crowd = [0.0] * len(pop)
    for r, front in enumerate(fronts):
        cd = crowding_distance([objs[i] for i in front])
        for pos, i in enumerate(front):
            rank[i] = r
            crowd[i] = cd[pos]
    return fronts, rank, crowd


def _tournament(
    n: int, rank: Sequence[int], crowd: Sequence[float], viols: Sequence[float], rng: random.Random
) -> int:
    """二元锦标赛：违反量小 > 秩低 > 拥挤距离大；完全并列时取先抽到的。"""
    a = rng.randrange(n)
    b = rng.randrange(n)
    if a == b:
        return a
    fa = viols[a] <= FEASIBILITY_TOL
    fb = viols[b] <= FEASIBILITY_TOL
    if fa != fb:
        return a if fa else b
    if (not fa) and (not fb) and (viols[a] != viols[b]):
        return a if viols[a] < viols[b] else b
    if rank[a] != rank[b]:
        return a if rank[a] < rank[b] else b
    if crowd[a] != crowd[b]:
        return a if crowd[a] > crowd[b] else b
    return a


def _environmental_selection(pool: Sequence[_Individual], pop_size: int) -> List[_Individual]:
    """(μ+λ) 环境选择：逐层填充前沿，末层按拥挤距离降序截断。"""
    objs = [ind.objectives for ind in pool]
    viols = [ind.violation for ind in pool]
    fronts = fast_non_dominated_sort(objs, viols)
    selected: List[_Individual] = []
    for front in fronts:
        if len(selected) + len(front) <= pop_size:
            selected.extend(pool[i] for i in front)
            continue
        cd = crowding_distance([objs[i] for i in front])
        order = sorted(range(len(front)), key=lambda t: (-cd[t], t))
        need = pop_size - len(selected)
        selected.extend(pool[front[t]] for t in order[:need])
        break
    return selected


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------
def nsga2(
    tier: str,
    budget: float,
    pop_size: int = 60,
    generations: int = 80,
    seed: int = 0,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.1,
) -> NSGA2Result:
    """在给定治理情境与预算下求解三目标能力配置问题。

    参数
    ----
    tier : 治理情境（``skygovc.core.TIERS`` 之一）。
    budget : 年度预算上限（亿元/年）。
    pop_size : 种群规模（≥ 2）。
    generations : 子代代数（0 表示只做随机初始化后排序）。
    seed : 随机种子；固定后结果完全可复现。
    crossover_rate : 一对父代发生均匀交叉的概率。
    mutation_rate : 每个基因发生阶跃变异的概率。

    返回
    ----
    :class:`NSGA2Result`

    说明
    ----
    底线 ``FLOOR`` 由初始化 + 修复算子**硬性保证**，故最终种群全部满足
    底线；预算约束用 Deb 约束支配处理，所以低预算下种群可能整体不可行，
    此时第一前沿给出的是"违反量最小"的一组解。
    """
    if tier not in TIERS:
        raise KeyError(f"未知治理情境 tier={tier!r}，可选 {TIERS}")
    if int(pop_size) < 2:
        raise ValueError(f"pop_size 必须 ≥ 2，收到 {pop_size!r}")
    if int(generations) < 0:
        raise ValueError(f"generations 必须 ≥ 0，收到 {generations!r}")
    pop_size = int(pop_size)
    generations = int(generations)
    rng = random.Random(int(seed))
    budget = float(budget)

    # ---- 初始化：全部满足底线 -------------------------------------------
    population: List[_Individual] = [
        _evaluate(_random_levels(rng, tier), tier, budget) for _ in range(pop_size)
    ]

    # ---- 迭代 -----------------------------------------------------------
    for _ in range(generations):
        fronts, rank, crowd = _rank_population(population)
        viols = [ind.violation for ind in population]
        offspring: List[_Individual] = []
        while len(offspring) < pop_size:
            i = _tournament(pop_size, rank, crowd, viols, rng)
            j = _tournament(pop_size, rank, crowd, viols, rng)
            if rng.random() < crossover_rate:
                c1, c2 = _uniform_crossover(population[i].levels, population[j].levels, tier, rng)
            else:
                c1, c2 = population[i].levels, population[j].levels
            for child in (c1, c2):
                mutated = _step_mutation(child, tier, rng, mutation_rate)
                offspring.append(_evaluate(mutated, tier, budget))
                if len(offspring) >= pop_size:
                    break
        population = _environmental_selection(population + offspring, pop_size)

    # ---- 收尾：第一前沿 -------------------------------------------------
    fronts, _, _ = _rank_population(population)
    first = fronts[0] if fronts else list(range(len(population)))
    front_levels: List[List[int]] = []
    front_objectives: List[Tuple[float, float, float]] = []
    seen = set()
    for i in first:
        lv = population[i].levels
        if lv in seen:               # 离散空间中不同个体可能重合，去重后更像"前沿"
            continue
        seen.add(lv)
        front_levels.append([int(v) for v in lv])
        front_objectives.append(tuple(population[i].objectives))  # type: ignore[arg-type]

    return NSGA2Result(
        population=[[int(v) for v in ind.levels] for ind in population],
        objectives=[tuple(ind.objectives) for ind in population],  # type: ignore[misc]
        front=front_levels,
        front_objectives=front_objectives,
        generations=generations,
        hypervolume=_front_hypervolume(population, front_objectives, seed=int(seed)),
    )


def _front_hypervolume(
    population: Sequence[_Individual],
    front_objectives: Sequence[Sequence[float]],
    seed: int,
    samples: int = 10000,
) -> float:
    """第一前沿的归一化超体积（基准取自最终种群，参考点 = nadir 外扩 5%）。"""
    if not front_objectives or not population:
        return 0.0
    F = np.asarray([[float(v) for v in ind.objectives] for ind in population], dtype=np.float64)
    ideal = F.min(axis=0)
    nadir = F.max(axis=0)
    span = nadir - ideal
    tiny = 1e-9
    span = np.where(span > tiny, span, tiny)
    reference = ideal + span * 1.05
    return float(
        normalized_hypervolume(
            front_objectives,
            reference_point=reference,
            ideal_point=ideal,
            samples=samples,
            seed=seed,
        )
    )
