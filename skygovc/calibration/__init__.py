"""统计校准与不确定性保证（statistical calibration & uncertainty guarantees）。

这是 SkyGov-C 流水线里 ``calibration`` 阶段的**自包含模块**：只依赖标准库 +
numpy（``bayes`` 另需 scipy 的 Gamma 分位数），可独立复现论文中"监测 → 反馈
→ 校准 → 回退"的闭环数字。

模块划分
--------
``bayes``
    Poisson–Gamma 共轭更新：风险倍数 ``θ = (θ_low, θ_medium, θ_high)`` 的
    后验均值 / 可信区间 / 相对 MAE / 低估超额事件。
``conformal``
    split conformal 区间：有限样本精确覆盖的 ``q``、区间、经验覆盖率核算。
``martingale``
    hedged-capital 置信序列（Howard et al., 2021 风格）：时间一致的序贯检验，
    可随时看、随时停。
``loop``
    L0/L1/L2/L3 四种反馈机制的闭环对照、回退判据与漂移注入反事实。

核心论断
--------
**监测本身不产生校准；只有进入规则并形成纠错闭环的数据才改变判断。**
``L0``（无监测）与 ``L1``（只监测不反馈）的 ``mae_series`` 严格逐点相同。

典型用法
--------
>>> from skygovc.calibration import MECHANISMS, run_loop, evaluate_rollback
>>> res = run_loop(MECHANISMS["L3"], days=180, reps=50, seed=0)
>>> round(res["mae_at"]["180"], 4) < 0.1                       # doctest: +SKIP
True
>>> evaluate_rollback(reps=20, seed=0)["detection_rate"] > 0.5  # doctest: +SKIP
True

详见同目录 ``README.md``。
"""
from __future__ import annotations

from .bayes import (
    DEFAULT_PRIOR_MULT,
    GammaPrior,
    PoissonGammaEstimator,
    default_prior,
    relative_mae,
    underestimation_excess,
)
from .conformal import ConformalReport, SplitConformal, evaluate_split_conformal
from .loop import (
    CAPABILITY,
    DEFAULT_N_BY_CLASS,
    DEFAULT_TRUTH,
    MECHANISMS,
    MechanismConfig,
    RollbackDecision,
    evaluate_rollback,
    rollback_rule,
    run_loop,
)
from .martingale import HedgedCapitalCS

__all__ = [
    # bayes
    "DEFAULT_PRIOR_MULT",
    "GammaPrior",
    "PoissonGammaEstimator",
    "default_prior",
    "relative_mae",
    "underestimation_excess",
    # conformal
    "ConformalReport",
    "SplitConformal",
    "evaluate_split_conformal",
    # martingale
    "HedgedCapitalCS",
    # loop
    "CAPABILITY",
    "DEFAULT_N_BY_CLASS",
    "DEFAULT_TRUTH",
    "MECHANISMS",
    "MechanismConfig",
    "RollbackDecision",
    "evaluate_rollback",
    "rollback_rule",
    "run_loop",
]

__version__ = "1.0.0"
