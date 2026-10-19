# `skygovc.allocation` — 预算约束下的分级能力配置

本目录是 SkyGov-C 流水线中 **allocation（能力配置）** 阶段的自包含算法模块，
对应论文中的三目标配置问题：在给定治理情境（tier）与年度预算下，为 6 个能力族
选择离散强化等级，使"事件率最低、服务完成率最高、成本最低"三个目标尽量同时达成。

* 唯一的外部共享依赖：`skygovc/core.py` 中的 `CAPABILITIES` 与 `TIERS`（只读导入）。
* 其余一切只用 **标准库 + numpy**：NSGA-II 从零实现，不依赖 pymoo / deap。
* 解空间只有 `3**6 = 729`，因此模块**同时提供穷举真前沿**，用于验证 NSGA-II 的正确性
  ——这是本模块的设计核心，而不是可选项。

```
allocation/
├── __init__.py        # 公开 API 汇总
├── model.py           # 问题定义与效应模型（成本表 / 底线 / 三个目标 / 穷举 / 支配）
├── nsga2.py           # 从零实现的 NSGA-II + Deb 约束支配
├── hypervolume.py     # 二维精确 / 高维蒙特卡洛 / 归一化超体积
├── policies.py        # uniform_open / uniform_strict / tiered + 对比与预算扫描
└── README.md
tests/test_allocation.py   # 25 个测试，含穷举 vs NSGA-II 的正确性校验
```

---

## 1. 问题定义

### 1.1 决策变量

6 个能力族各取一个离散等级 `ℓ_k ∈ {0, 1, 2}`（0 = 维持现状/基础，1 = 标准，2 = 强化），
即解向量 `ℓ = (ℓ_rule, ℓ_spatial, ℓ_service, ℓ_sensing, ℓ_emergency, ℓ_data)`，
解空间 `3^6 = 729`。

### 1.2 成本表（亿元/年，固定）

| 能力族 | level 0 | level 1 | level 2 |
|---|---|---|---|
| rule | 0.05 | 0.12 | 0.22 |
| spatial | 0.08 | 0.20 | 0.38 |
| service | 0.10 | 0.26 | 0.50 |
| sensing | 0.12 | 0.34 | 0.68 |
| emergency | 0.06 | 0.15 | 0.29 |
| data | 0.04 | 0.10 | 0.19 |

全 0 配置 0.45 亿元/年，全 2 配置 **2.26** 亿元/年。

### 1.3 各情境的最低能力底线 `FLOOR[tier]`

| tier | rule | spatial | service | sensing | emergency | data | 底线成本 |
|---|---|---|---|---|---|---|---|
| capital_core | 2 | 2 | 1 | 2 | 2 | 2 | **2.02** |
| dense_hub | 1 | 2 | 1 | 2 | 2 | 1 | **1.83** |
| market_driven | 1 | 1 | 2 | 1 | 1 | 2 | **1.50** |
| mountainous | 1 | 2 | 1 | 1 | 2 | 1 | **1.49** |
| border_coastal | 2 | 2 | 1 | 2 | 2 | 2 | **2.02** |
| sparse_resource | 1 | 1 | 1 | 0 | 1 | 1 | **0.95** |

可行性的硬前提：`ℓ_k >= FLOOR[tier][k]`（对所有 k）。表里的"底线成本"是该情境
在数学上可能的最小成本，也是预算可行性的下界——**预算低于它时该情境无任何可行解**。

### 1.4 效应模型（固定公式）

能力强度 `s_k = ℓ_k / 2 ∈ {0, 0.5, 1}`。

```
incident_mult = exp( -Σ_k w_inc[tier][k] · s_k )                     # 越小越好
completion    = clip( 0.60 + 0.30 · Σ_k w_svc[tier][k] · s_k, 0, 1 )  # 越大越好
```

`w_inc` / `w_svc` 见 `model.W_INCIDENT` / `model.W_SERVICE`（每情境各 6 个系数，
按 `CAPABILITIES` 顺序）。取值范围：`incident_mult ∈ (0, 1]`
（全 0 时恰为 1.0；实测最低 0.2894，出现在 `border_coastal` 全 2），
`completion ∈ [0.60, 0.90]`（`market_driven` 的 `w_svc` 之和恰为 1.0，全 2 时正好 0.90）。

### 1.5 三个目标与预算约束

```
f1 = incident_mult            （最小化）
f2 = -completion              （最小化 ⇔ 最大化完成率）
f3 = Σ_k COST[k][ℓ_k]         （最小化；无预算约束时与 f1/f2 单调冲突）
约束：f3 <= budget            （硬约束，Deb 约束支配处理）
```

f1、f2 关于 `ℓ` 单调改进，f3 关于 `ℓ` 单调增加，因此三目标两两冲突，
真前沿是一个离散曲面。

---

## 2. 参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `pop_size` | 60 | 种群规模（≥ 2） |
| `generations` | 80 | 子代代数（0 = 只做随机初始化后排序） |
| `seed` | 0 | 随机种子；固定后结果完全可复现 |
| `crossover_rate` | 0.9 | 一对父代发生均匀交叉的概率 |
| `mutation_rate` | 0.10 | 每个基因发生阶跃变异的概率 |
| `budget` | — | 年度预算（亿元/年） |
| `DEFAULT_BUDGETS` | `1.0, 1.4, 1.8, 2.2, 2.6, 3.0` | `budget_sweep` 默认档位 |
| `samples` | 200000 | 高维超体积的蒙特卡洛样本数 |
| `BUDGET_TOL` | `1e-9` | 预算比较的浮点容差（"刚好等于预算"判为可行） |

一次 `nsga2` 运行（60×80，729 点空间）约 **0.1 s**：目标值走 729 行预计算表（O(1) 查表），
非支配排序走向量化的约束支配矩阵。

---

## 3. API

### 3.1 `model.py`

| 名称 | 说明 |
|---|---|
| `CAP_KEYS` | `== CAPABILITIES`，6 个能力族的固定顺序 |
| `COST[tier][level]` | 成本表（`dict[str, dict[int, float]]`，亿元/年） |
| `FLOOR[tier]` | 6 元底线向量 |
| `W_INCIDENT` / `W_SERVICE` | 效应模型系数 |
| `decode(x, tier)` | 长度 6 的整数序列 → 等级向量（四舍五入 + 截断到 `[0,2]`，**不**强制底线） |
| `repair(levels, tier)` | 修复算子：拉回 `[0,2]` 且逐维不低于 `FLOOR` |
| `cost_of(levels)` | `f3` |
| `floor_cost(tier)` | 该情境底线配置的成本（可行成本下界） |
| `feasibility(levels, tier, budget)` | `(是否可行, 违反量)`；可行 ⟺ 底线满足且 `cost <= budget` |
| `objectives(levels, tier)` | `(f1, f2, f3)` |
| `evaluate(x, tier, budget)` | `(objectives, violation)` |
| `all_feasible(tier, budget)` | 穷举 729 个解中的全部可行解（字典序） |
| `pareto_front(solutions, tier)` | 给定等级向量列表，返回非支配解的**下标** |
| `dominates(a, b)` | 严格 Pareto 支配（最小化语义）：`a_i <= b_i + 1e-12` 且存在 `a_j < b_j - 1e-12` |
| `non_dominated_indices(points)` | 直接对目标向量做非支配筛选 |

违反量（供约束支配使用）定义为

```
violation = Σ_k max(0, FLOOR[tier][k] - ℓ_k)              # 单位：等级
          + max(0, cost - budget) / max(budget, 1.0)       # 预算相对超支
```

可行解恒为 `0.0`。

### 3.2 `nsga2.py`

```python
@dataclass
class NSGA2Result:
    population: list[list[int]]                    # 最终种群（等级向量）
    objectives: list[tuple[float, float, float]]   # 与 population 逐项对应
    front: list[list[int]]                         # 第一前沿（已去重）
    front_objectives: list[tuple[float, float, float]]
    generations: int
    hypervolume: float                             # 第一前沿的归一化超体积 ∈ [0,1]

def nsga2(tier, budget, pop_size=60, generations=80, seed=0,
          crossover_rate=0.9, mutation_rate=0.1) -> NSGA2Result
```

算法实现（全部从零）：

| 环节 | 做法 |
|---|---|
| 编码 | 6 个整数等级，个体即 `tuple[int, ...]` |
| 初始化 | `ℓ_k ~ U{FLOOR[tier][k], …, 2}`，**天然满足底线** |
| 修复 | 交叉/变异后调用 `repair`，底线是**不变量**（最终种群必然满足底线） |
| 约束处理 | Deb 约束支配：可行压不可行；都不可行时违反量小者优 |
| 排序 | 快速非支配排序（向量化支配矩阵 + 逐层剥离） |
| 多样性 | 拥挤距离（边界点 `inf`） |
| 父代选择 | 二元锦标赛（违反量 → 秩 → 拥挤距离） |
| 交叉 | 均匀交叉（整型友好；`crossover_rate` 控制是否重组） |
| 变异 | 阶跃变异 `±1`（撞到底线/上界时反向），逐基因概率 `mutation_rate` |
| 环境选择 | (μ+λ)：父代 ∪ 子代逐层填充，末层按拥挤距离截断 |

`seed` 固定时结果**完全可复现**：唯一随机源是 `random.Random(seed)`，所有排序/截断
都有确定次序（前沿内下标升序，拥挤距离并列取靠前者）。

> ⚠️ 底线由修复算子硬性保证，**预算**则通过约束支配软化处理。因此当
> `budget < floor_cost(tier)` 时，返回的第一前沿是"违反量最小"（即成本最低）的一组解，
> 它们必然不可行——请用 `feasibility(...)[0]` 显式判断，不要假设输出一定可行。

### 3.3 `hypervolume.py`

```python
hypervolume_2d(points, ref) -> float                 # 精确扫描法
hypervolume_nd(points, ref, samples=200000, seed=0) -> float
normalized_hypervolume(front_objectives, reference_point, ideal_point,
                       samples=200000, seed=0) -> float
dominated_fraction(points, samples=200000, seed=0)   # 单位盒被支配比例（内核）
```

* `hypervolume_2d`：按第一目标升序扫描，累加 `(ref_x - x)·(prev_y - y)` 横条面积。
  例：`points=[(1,1),(2,0.5)]`、`ref=(3,3)` → `2 + 2.5 = 4.5`。
* `hypervolume_nd`：返回**被支配部分占 `[ideal, ref]` 包围盒的比例**（∈ `[0,1]`），
  其中 `ideal` 取点的逐目标最小值。之所以返回比例：三目标下绝对体积强烈依赖单位
  （元 / 亿元），比例才可跨情境比较，且与 `normalized_hypervolume` 完全一致。
  `dim <= 2` 走精确路径，`dim >= 3` 走蒙特卡洛；固定 `seed` 可复现。
* `normalized_hypervolume`：先做 `(f - ideal)/(ref - ideal)` 仿射归一化到单位盒，
  再估计被支配比例。**两份前沿用同一个 `seed` 时蒙特卡洛样本完全相同**，因此
  `HV(A)/HV(B)` 是干净的覆盖率比值（测试 2 用到的就是这一点）。

### 3.4 `policies.py`

```python
uniform_open(tier) -> tuple[int,...]                 # 只做到 FLOOR
uniform_strict(tier, budget=None) -> tuple[int,...]  # 全 2；给预算时按 2 → 1 → 底线 退让
tiered(tier, budget, seed=0) -> tuple[int,...]       # NSGA-II 前沿 + knee point
compare_policies(tier, budget, seed=0) -> dict
budget_sweep(tiers=None, budgets=None, seed=0) -> list[dict]
```

**knee point**：把候选前沿的目标归一化到单位盒（逐目标 min–max），取到理想点欧氏
距离最小的解；并列时取等级字典序靠前者（保证确定性）。若第一前沿没有可行解，依次退化为
"种群内可行解"→"违反量最小解"，因此函数**永远**返回满足 `FLOOR` 的合法等级向量，
是否真的在预算内请查 `feasible` 字段。

`compare_policies` 返回结构：

```python
{
  "tier": "sparse_resource", "budget": 2.6, "seed": 0,
  "floor_cost": 0.95, "n_feasible": 96,          # 穷举得到的可行解个数
  "policies": {
      "uniform_open":   {"name":..., "levels":[...], "cost":..., "incident_mult":...,
                         "completion":..., "feasible": True, "violation": 0.0},
      "uniform_strict": {...},
      "tiered":         {...},
  },
  "uniform_open": {...}, "uniform_strict": {...}, "tiered": {...},   # 扁平别名
  "improvement": {
      "tiered_vs_uniform_strict": {"incident_reduction":0.03, "incident_reduction_pct":3.0,
                                   "completion_gain":0.02, "completion_gain_pct":2.5,
                                   "cost_delta":-0.34},
      "tiered_vs_uniform_open":   {...},
  },
}
```

* `incident_reduction = (base_inc - tiered_inc) / base_inc`，**正数 = tiered 事件更少**；
* `completion_gain = tiered_comp - base_comp`，**正数 = tiered 完成率更高**；
* `cost_delta = tiered_cost - base_cost`，负数表示更省预算。

`budget_sweep` 每个 `tier × budget` 输出一行（默认 6×6 = **36 行**），每行就是
`compare_policies` 的返回值，可直接 `pandas.DataFrame(rows)` 或写 CSV 画图。

---

## 4. 运行方式

```powershell
# 单元测试（25 个用例，约 15 s）
& "C:\Python314\python.exe" -m pytest modules\SkyGovC\tests\test_allocation.py -q

# 在模块目录内直接跑
cd modules\SkyGovC
& "C:\Python314\python.exe" -m pytest tests\test_allocation.py -q

# 快速冒烟
& "C:\Python314\python.exe" -c "from skygovc.allocation import *; print(len(budget_sweep()))"
```

在 Python 里使用：

```python
from skygovc.allocation import nsga2, compare_policies, budget_sweep

res = nsga2("sparse_resource", 2.6, seed=0)
print(len(res.front), res.front_objectives, res.hypervolume)

row = compare_policies("sparse_resource", 2.6, seed=0)
print(row["tiered"]["levels"], row["improvement"]["tiered_vs_uniform_open"])

rows = budget_sweep(tiers=["capital_core", "sparse_resource"], budgets=[2.2, 3.0])
```

---

## 5. 测试与正确性论证

`tests/test_allocation.py` 的 25 个用例覆盖：

1. `all_feasible` 在宽松预算（3.0）下对 6 个情境均返回非空集合、全部满足底线与预算，
   且与逐点暴力过滤结果逐项一致；
2. **穷举真前沿 vs NSGA-II**：在 5 个 `(tier, budget)` 组合上
   （`sparse_resource/3.0`、`mountainous/3.0`、`market_driven/3.0`、`dense_hub/3.0`、
   `sparse_resource/1.4`）验证——NSGA-II 前沿点全部可行、**没有任何点被真前沿严格支配**、
   归一化超体积覆盖率 `HV(NSGA-II)/HV(真前沿) ≥ 0.8`（实测 5 组均为 **1.000**）；
3. `dominates` 的非自反性与反对称性；
4. `hypervolume_2d` 与手算 `4.5` 一致，并与 600×600 网格数值积分交叉验证；
5. 固定 `seed` 下两次 `nsga2` 的 `population / objectives / front / hypervolume` 完全相同；
6. `tiered` 返回的向量合法、满足底线、在预算内且可复现；
7. `budget_sweep` 条数 = `len(tiers) × len(budgets)`（默认 36）且字段齐全。

实测（默认参数，`pop_size=60, generations=80`）：

| tier / budget | 穷举真前沿 | NSGA-II 前沿 | 覆盖率 | 被支配点 |
|---|---|---|---|---|
| sparse_resource / 3.0 | 13 | 13 | 1.000 | 0 |
| mountainous / 3.0 | 11 | 11 | 1.000 | 0 |
| market_driven / 3.0 | 7 | 7 | 1.000 | 0 |
| dense_hub / 3.0 | 7 | 7 | 1.000 | 0 |
| sparse_resource / 1.4 | 6 | 6 | 1.000 | 0 |

也就是说，在 729 点空间上 NSGA-II **完整复原了穷举真前沿**（因此正确性是可证的，
不是"看起来收敛"）。

---

## 6. 结果解读时务必注意的两点

1. **预算网格与可行域的关系**：由于底线成本差异很大（0.95–2.02），默认预算档位
   `1.0 / 1.4` 只能支撑 `sparse_resource`（0.95）等少数情境；`capital_core`（2.02）
   在 2.2 及以上才可行。`compare_policies(...)["n_feasible"]` 直接给出该格点的可行解
   个数，`0` 意味着三种策略全部不可行（`feasible=False`），这些格点**不应**进入论文
   的"策略优劣"讨论，只适合用来展示"预算不足"的边界。
2. **`capital_core` / `border_coastal` 的可行域极小**：底线 `(2,2,1,2,2,2)` 已把 6 个
   能力族中的 5 个钉死在上限，只有 `service` 可动（1→2 需 +0.24 亿元）。因此在
   `budget = 2.2` 时穷举可行解**只有 1 个**（就是底线配置），`budget >= 2.6` 时也
   只有 2 个。这两个情境上三种策略必然重合、改进量为 0——这是模型参数的结果，
   不是算法失败；论文里描述"分级配置优势"应主要引用 `dense_hub / mountainous /
   market_driven / sparse_resource`。
3. `f3`（成本）本身是目标之一，所以"预算更宽松 → 解更好"只体现在可行域变大，
   不会自动体现在 knee 解上（knee 解可能仍选便宜的解）。

---

## 7. 与规范的偏差说明

| 规范写法 | 实现 | 原因 |
|---|---|---|
| `COST: dict[str, dict[str, int]]` | `dict[str, dict[int, float]]`（内层键为**整数**等级，值为 `float`） | 表内数值是 0.05 这类小数，"int" 显然是笔误；整数键便于 `COST[cap][level]` 直接索引 |
| `uniform_strict(tier)` 中"统一取 2（若预算允许）否则 1" | 增加**可选** `budget` 形参：`uniform_strict(tier)` 返回全 2；`uniform_strict(tier, budget)` 按 2 → 1（底线修复）→ 底线 退让 | 原签名没有 budget，无法判断"预算是否允许"；加可选参数后两种语义都能表达，且 `compare_policies` 用的是预算可行的版本 |
| `hypervolume_nd(...)` 注释"返回（可行域比例估计）" | 返回**被支配体积占 `[ideal, ref]` 包围盒的比例**（∈ `[0,1]`） | 三目标绝对体积依赖单位；比例版本与 `normalized_hypervolume` 语义一致、可跨情境比较 |
| `decode(x, tier)` 未说明是否强制底线 | `decode` 只做表示归一（四舍五入 + 截断），底线由 `repair` 强制 | 否则 NSGA-II 无法表示违反底线的个体，Deb 约束支配失去意义；规范也要求"初始化 + 修复算子" |
| `pareto_front(solutions, tier)` 的 `solutions` 未明说 | 视为**等级向量**列表（长度 6），返回下标；另提供 `non_dominated_indices` 直接处理目标向量 | 长度 6 与目标三元组可区分，语义清晰 |
| `NSGA2Result.generations` | 返回实际执行的代数（等于入参） | 无提前终止 |
| `budget_sweep` 输出 | 每格点一行 `compare_policies` 结果（默认 36 行），含三策略子字典 | 一行内含三策略便于按 `tier×budget` 画分组柱状图；若需长表可自行展开 `row["policies"]` |

均未改动 `core.py`，也未触碰 `allocation/` 与 `tests/test_allocation.py` 之外的任何文件。
