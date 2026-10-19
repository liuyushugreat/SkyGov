"""预算约束下的分级能力配置（budget-constrained capability allocation）。

这是 SkyGov-C 流水线里 ``allocation`` 阶段的**自包含算法模块**：只有
``skygovc.core``（能力族 / 治理情境枚举）是共享依赖，其余一切都用标准库
+ numpy 从零实现，可独立复现论文中的三目标权衡数字。

模块划分
--------
``model``
    问题定义与效应模型：成本表、各情境能力底线、事件率乘子与服务完成率
    的固定公式，外加 729 个解的穷举工具（真 Pareto 前沿的黄金基准）。
``nsga2``
    从零实现的 NSGA-II（快速非支配排序 + 拥挤距离 + 二元锦标赛 +
    均匀交叉 + 阶跃变异 + Deb 约束支配），不依赖 pymoo/deap。
``hypervolume``
    二维精确扫描法 + 高维蒙特卡洛 + 归一化超体积。
``policies``
    三个参考策略（``uniform_open`` / ``uniform_strict`` / ``tiered``）
    的构造、对比与预算扫描。

典型用法
--------
>>> from skygovc.allocation import nsga2, compare_policies, budget_sweep
>>> res = nsga2("capital_core", 2.2, seed=0)
>>> res.front_objectives[0]                     # doctest: +SKIP
(0.43..., -0.85..., 2.02)
>>> compare_policies("sparse_resource", 3.0, seed=0)["tiered"]["feasible"]
True
>>> len(budget_sweep(tiers=["capital_core"], budgets=[2.2, 3.0]))
2

详见同目录 ``README.md``。
"""
from __future__ import annotations

from .hypervolume import (
    dominated_fraction,
    hypervolume_2d,
    hypervolume_nd,
    normalized_hypervolume,
)
from .model import (
    BUDGET_TOL,
    CAPABILITIES,
    CAP_KEYS,
    COST,
    FLOOR,
    LEVEL_MAX,
    LEVELS,
    N_CAP,
    SPACE_SIZE,
    TIERS,
    W_INCIDENT,
    W_SERVICE,
    all_feasible,
    all_levels,
    cost_of,
    decode,
    dominates,
    evaluate,
    feasibility,
    floor_cost,
    non_dominated_indices,
    objectives,
    pareto_front,
    repair,
)
from .nsga2 import (
    NSGA2Result,
    constraint_dominance_matrix,
    crowding_distance,
    fast_non_dominated_sort,
    nsga2,
)
from .policies import (
    DEFAULT_BUDGETS,
    POLICY_NAMES,
    budget_sweep,
    compare_policies,
    knee_index,
    tiered,
    uniform_open,
    uniform_strict,
)

__all__ = [
    # model
    "CAPABILITIES",
    "CAP_KEYS",
    "COST",
    "FLOOR",
    "LEVELS",
    "LEVEL_MAX",
    "N_CAP",
    "SPACE_SIZE",
    "TIERS",
    "BUDGET_TOL",
    "W_INCIDENT",
    "W_SERVICE",
    "all_feasible",
    "all_levels",
    "cost_of",
    "decode",
    "dominates",
    "evaluate",
    "feasibility",
    "floor_cost",
    "non_dominated_indices",
    "objectives",
    "pareto_front",
    "repair",
    # nsga2
    "NSGA2Result",
    "constraint_dominance_matrix",
    "crowding_distance",
    "fast_non_dominated_sort",
    "nsga2",
    # hypervolume
    "dominated_fraction",
    "hypervolume_2d",
    "hypervolume_nd",
    "normalized_hypervolume",
    # policies
    "DEFAULT_BUDGETS",
    "POLICY_NAMES",
    "budget_sweep",
    "compare_policies",
    "knee_index",
    "tiered",
    "uniform_open",
    "uniform_strict",
]

__version__ = "1.0.0"
