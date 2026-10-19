# 统计校准与不确定性保证（`skygovc.calibration`）

SkyGov-C 流水线的 `calibration` 阶段：把"监测 → 反馈 → 校准 → 回退"的闭环做
成一个**自包含、可复现**的统计模块，只依赖标准库 + numpy（分位数另需 scipy）。

> **核心命题**
> 监测本身不产生校准；只有进入规则并形成纠错闭环的数据才改变判断。
> 因此 `L0`（无监测）与 `L1`（只监测、不反馈）的误差曲线**逐点严格相同**。

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | 公开 API（`GammaPrior`、`SplitConformal`、`HedgedCapitalCS`、`MECHANISMS`、`run_loop` …） |
| `bayes.py` | Poisson–Gamma 共轭更新、可信区间、相对 MAE、低估超额事件 |
| `conformal.py` | split conformal 区间（有限样本精确覆盖）+ 覆盖率核算 |
| `martingale.py` | hedged-capital 置信序列（Howard et al., 2021 风格），时间一致的序贯检验 |
| `loop.py` | L0/L1/L2/L3 闭环对照、回退判据、漂移注入反事实 |
| `tests/test_calibration.py` | 20 个单元测试（另含 3 个 doctest） |

---

## 1. 数据模型与四个机制

监管者对**三类风险任务**（低/中/高）的风险倍数 `θ = (θ_low, θ_medium, θ_high)`
有一个初始估计，真实值 `θ* = (1.0, 3.0, 10.0)`。默认先验刻意**低估**：

```
strength = 15,  prior_mult = (0.55, 1.7, 6.0)
alpha = prior_mult * strength,   beta = strength
⇒ θ̂_prior = (0.55, 1.7, 6.0),   相对 MAE = 0.4278
```

每天第 `c` 类飞 `n_c` 架次（默认 `(1200, 600, 180)`），基准率
`base_rate = 0.01`，监测覆盖 `coverage` 比例的架次：

```
e_{c,t} ~ Poisson( coverage · base_rate · n_c · θ*_c )
exposure_c = coverage · base_rate · n_c            # θ = 1 时的期望事件数
```

`exposure` 不依赖 `θ*`，所以 `α/β` 收敛到 `θ*`（而不是 `coverage·θ*`）。

| 机制 | `coverage` | `update_every` | `correct_rules` | 数据是否改变判断 |
| --- | --- | --- | --- | --- |
| `L0` 无监测 | 0.0 | 0（不更新） | False | 否 |
| `L1` 仅监测 | 0.6 | 0（不更新） | False | **否（只记录）** |
| `L2` 周报 | 0.6 | 7 | True | 是 |
| `L3` 日报 | 0.6 | 1 | True | 是 |

实现上 `L1` 的观测事件数**确实被生成并记进账本**（`pending_events`），只是既不做
贝叶斯更新、也不写回规则；`L0` 与 `L1` 走的是同一条 `theta = prior_theta` 代码
路径、同一个函数、同一 dtype，所以 `mae_series` 逐元素**位相同**（测试用 `==`
比较 Python list，不是近似）。

误差定义：`MAE = mean_c(|θ̂_c - θ*_c| / θ*_c)`；超额事件：

```
excess = Σ_days Σ_c  base_rate · θ*_c · n_c · max(0, θ*_c - θ̂_c)/θ*_c · 0.5
```

按**全部架次**（不只是被监测的 60%）累计——规则低估会让所有架次的风险被低估。

---

## 2. 方法

### 2.1 Poisson–Gamma（`bayes.py`）

```
events_c | θ_c ~ Poisson(θ_c · exposure_c)
θ_c ~ Gamma(α_c, β_c)
α_c ← α_c + events_c,  β_c ← β_c + exposure_c
θ̂_c = α_c / β_c
```

`credible_interval(level)` 用 `scipy.stats.gamma.ppf` 取等尾区间；
`relative_mae` / `underestimation_excess` 同时是模块级函数与估计器方法。

### 2.2 Split conformal（`conformal.py`）

`q = |R|_(k)`，`k = ceil((n+1)(1-alpha))`（`k > n` 时 `q = +inf`，因为有限样本
下无法达到目标覆盖）。区间 `[ŷ - q, ŷ + q]`，覆盖率 `mean(|y - ŷ| <= q)`。
`evaluate_split_conformal` 按 `calib_frac` 随机切分（固定 `seed` 完全可复现），
在另一半上核算经验覆盖率与平均宽度。

### 2.3 Hedged-capital 置信序列（`martingale.py`）

对取值于 `[a, b]`、均值 `μ` 的观测，设 `x_i - μ` 是 `σ`-次高斯的
（有界时 `σ = (b-a)/2` 由 Hoeffding 引理必然成立）。对一组下注 `Λ` 取混合：

```
K̄_t(μ) = (1/|Λ|) Σ_{λ∈Λ} exp( λ(S_t - tμ) - t λ² σ² / 2 ),   S_t = Σ x_i
CS_t    = { μ : K̄_t(μ) < 1/α }
```

每个 λ 分量的资本过程在 `H0: μ=μ0` 下是上鞅 ⇒ 混合仍是上鞅 ⇒ Ville 不等式给出
**时间一致**覆盖 `Pr(∀t: μ ∈ CS_t) ≥ 1-α`（可以随时看、随时停）。
实现要点：

* `Λ` 取**几何**网格（跨两个数量级），大 `n` 时也能命中接近最优的
  `λ* = (μ - μ̂)/σ²`；
* `log K̄_t(μ)` 是 `μ` 的**凸**函数 ⇒ `CS_t` 必为区间，用 `μ` 网格 + 线性插值求端点；
* 状态只有 `(n, S)`，`update` 为 O(1)；`is_reject(μ0)` 用 `log K̄_t(μ0) ≥ log(1/α)`；
* `lower()/upper()` 返回**至今所有区间之交**（仍是 `1-α` 置信序列，且宽度
  **单调不增**，便于画图/审计）；`lower_now()/upper_now()` 返回**当前时刻**区间，
  用于漂移检测（交集会"记住"历史，看不到下行）。

### 2.4 回退判据与反事实（`loop.py`）

`rollback_rule` 把"每万架次事件率 / 基准率"这类比值信号送进置信序列，**置信下界
连续 `min_days` 天低于 `threshold`** 即触发回退：

* 默认 `window=30`（滑窗型）。扩张型（`window=None`）在 35% 下行漂移下需要约
  60 天才能分辨，滑窗型约 10 天；重复观测带来的误触发由 `min_days=14` 的连续
  要求压住（阈值 0.85 距基线 1.0 有 0.15，约等于 16 个窗口均值的标准差）。
* `sigma=None` 时用预热期样本标准差估计次高斯尺度（无需外部噪声模型）；
  `evaluate_rollback` 里直接给 `sigma = 1/sqrt(expected_events)`（Poisson 抽样）。
* `evaluate_rollback` 在第 `true_drift_at` 天注入一次 `drift_size` 的真实漂移；
  负值走下界（风险下降 → 回退），正值走上界（风险上升 → 告警，阈值 ≤ 1 时
  自动取 `1 + |drift_size|/2`）。

---

## 3. 复现

```powershell
# 单元测试（20 项）
& "C:\Python314\python.exe" -m pytest modules/SkyGovC/tests/test_calibration.py -q

# 加 doctest
& "C:\Python314\python.exe" -m pytest modules/SkyGovC/skygovc/calibration `
    modules/SkyGovC/tests/test_calibration.py -q --doctest-modules

# 冒烟脚本（四机制 + 回退反事实）
& "C:\Python314\python.exe" -c @"
import sys; sys.path.insert(0, r'modules\SkyGovC')
from skygovc.calibration import MECHANISMS, run_loop, evaluate_rollback
for n in ('L0','L1','L2','L3'):
    r = run_loop(MECHANISMS[n], days=180, reps=200, seed=0)
    print(n, r['mae_at'], round(r['excess_events'],1), r['conformal'])
print(evaluate_rollback())
"@
```

---

## 4. 结果（本机实测：`days=180, reps=200, seed=0`）

### 4.1 四种反馈机制

| 机制 | MAE@30 | MAE@90 | MAE@180 | `excess_events` | conformal 覆盖 | `q` | 区间宽度 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `L0` 无监测 | 0.4278 | 0.4278 | 0.4278 | 1836.0 | — | — | — |
| `L1` 仅监测 | 0.4278 | 0.4278 | 0.4278 | 1836.0 | — | — | — |
| `L2` 周报 | 0.0839 | 0.0402 | 0.0246 | 270.6 | 0.901 | 9.888 | 19.776 |
| `L3` 日报 | 0.0805 | 0.0383 | 0.0243 | 233.7 | 0.899 | 9.362 | 18.724 |

* **`L0` 与 `L1` 的 `mae_series` 逐点严格相同**（`==` 为 `True`，180 个元素全等，
  `excess_events` 也完全相等：1836.0）。
* `L2`/`L3` 的 180 日 MAE 只有 `L0` 的 **5.75% / 5.68%**；超额事件降到
  **270.6 / 233.7**（−85% / −87%）。
* 终局估计：`L2` `θ̂ = (0.968, 2.993, 9.598)`、`L3` `θ̂ = (0.966, 2.998, 9.589)`
  （真值 `(1, 3, 10)`）。
* conformal 的 90% 区间实测覆盖 **0.901 / 0.899**（`L2`/`L3`），即"校准不仅点估计
  收敛，还带上了可审计的覆盖率保证"。
* `L0`/`L1` 的 `conformal` 为 `None`：没有反馈就没有可用的区间——这是论断的一部分
  （`L1` 手里有数据，但规则没变，区间只能围着先验点转）。

### 4.2 回退判据（第 90 天注入 −35% 真实风险下降）

| 指标 | 数值 |
| --- | --- |
| 检出率 `detection_rate` | **1.000**（200/200） |
| 误触发率 `false_alarm_rate` | **0.000** |
| 平均检出延迟 `mean_delay_days` | **22.9 天** |
| 中位 / p90 延迟 | 23.0 / 24.0 天 |
| 监测参数 | `threshold=0.85, alpha=0.05, min_days=14, window=30, sigma=0.05` |

延迟 ≈ 10 天（滑窗均值跌到阈值以下）+ 14 天（`min_days` 连续要求），与设计一致。
无漂移的反事实序列上，`min_days=14` 从未被满足，误触发率为 0。

---

## 5. 设计说明（与规范的偏差 / 取舍）

1. **`lower()/upper()` 是"交集包络"，`lower_now()/upper_now()` 是当前区间。**
   规范要求"区间宽度随 n 非增"，但赌注型/似然比型置信序列的**当前**区间会有
   数值抖动，且"取交集"才是既有单调性又保持 `1-α` 时间一致覆盖的唯一做法。
   漂移检测不能用在单调交集上（它永远不会往下走），故另设 `*_now()`，
   `rollback_rule` 用的是 `lower_now()`。
2. **`rollback_rule` 的默认滑窗 `window=30`**（规范未指定）。扩张型对 35% 的下行
   漂移要 ~60 天才能"确信均值已跌破阈值"，在 180 天窗口里检出率会被截断；
   滑窗型的代价是重复观测，用 `min_days=14` 的连续要求 + 0.15 的阈值余量抵消。
   `window=None` 可复现经典扩张型（同一序列实测第 102 天触发，滑窗型第 82 天）。
3. **`sigma`（次高斯尺度）**：`HedgedCapitalCS` 默认取 `(b-a)/2`（Hoeffding，
   永远合法但保守）；若已知监测噪声（本模块的 Poisson 设定下
   `sigma = 1/sqrt(expected_events)`），传入更小的 `sigma` 可显著收紧区间。
   `rollback_rule(sigma=None)` 用预热期样本标准差估计。
4. **`conformal` 只对闭环机制（`L2`/`L3`）给出**；`L0`/`L1` 返回 `None`。
   结构上仍然可以拿 `L1` 的数据做 conformal，但那条区间是围着**没写回规则的先验
   点**转的，不构成"校准产物"，所以不报。
5. **`run_loop` 的额外返回键**：除规范要求的 `mae_series` / `mae_at` /
   `excess_events` / `conformal` 外，另有 `mechanism`、`final_theta`（便于打印终局
   估计，`test_calibration.py` 也用它检查可复现性）。
6. **`mae_at` 的键固定为 `"30"/"90"/"180"`**；若 `days < 90`，后两个键退化为最后
   一天的值（测试里有断言）。
7. **额外关键字参数**（均带默认值，不改变规范签名）：
   `rollback_rule(..., window=30, sigma=None, scale=2.0)`、
   `evaluate_rollback(..., expected_events=400.0, min_days=14, window=30)`、
   `HedgedCapitalCS(..., sigma=None, window=None, mu_grid_size=None)`。
8. **`evaluate_rollback` 的均值口径**：`mae`/`excess` 用的是**写回规则的那个 θ̂**；
   对 `L0`/`L1` 它就等于先验均值，因此两条曲线严格相同。
9. **依赖**：`bayes.credible_interval` 用 `scipy.stats.gamma.ppf`（规范指定），
   其余模块只用 numpy；`HedgedCapitalCS` 的 `logsumexp` 是手写的（不引 scipy）。
10. **`CAPABILITY = "data"`**：与 `skygovc.core.CAPABILITIES` 的 `"data"` 能力族
    对齐，仅作标注用（未在运行时导入 `core`，保证模块可独立使用）。

---

## 6. 已知局限

* Poisson–Gamma 假设事件相互独立、暴露量精确已知；真实 UAM 运行里事件会聚集
  （同一运营人/同一空域），需要过散布（overdispersion）修正。
* conformal 的可交换性假设对"估计本身在变"的在线场景只是近似；本模块的做法是
  在**全部仿真日**上随机切分，因此保证的是对池化分布的边际覆盖。
* 置信序列的下界阈值余量（0.15）是按当前噪声尺度（`sigma=0.05`）设计的；若监测
  机队缩小一个量级（`sigma` 变为 0.16），需要同时放宽 `threshold` 或 `min_days`。
