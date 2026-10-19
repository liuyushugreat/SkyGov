"""闭环校准器：L0/L1/L2/L3 四种反馈机制的对照实验 + 回退判据 + 反事实。

论文论断
--------
**监测本身不产生校准；只有进入规则并形成纠错闭环的数据才改变判断。**

四个机制（见 :data:`MECHANISMS`）：

===========  =========  ==============  ==============  ================
机制          监测覆盖率  更新周期        写回规则        数据是否改变判断
===========  =========  ==============  ==============  ================
``L0`` 无监测   0.0        不更新(0)       否              否
``L1`` 仅监测   0.6        不更新(0)       否              **否（只记录）**
``L2`` 周报     0.6        7 天            是              是
``L3`` 日报     0.6        1 天            是              是
===========  =========  ==============  ==============  ================

``L0`` 与 ``L1`` 的区别只在"有没有把数据记进账本"：两者都**不做**贝叶斯更新、
都不**把估计写回规则**，因此 ``θ̂`` 轨迹逐位相同，``mae_series`` 必须**严格
逐点相等**（测试用 ``==`` 检查，不是近似）。

模拟设定
--------
每天第 ``c`` 类飞 ``n_c`` 架次，真实风险倍数 ``θ* = (1, 3, 10)``，基准率
``base_rate``；监测覆盖 ``coverage`` 比例的架次，观测事件数

.. math::

    e_{c,t} \\sim \\mathrm{Poisson}\\left(
        \\mathrm{coverage} \\cdot \\mathrm{base\\_rate} \\cdot n_c \\cdot \\theta^*_c\\right)

暴露量（即 ``θ = 1`` 时的期望事件数）不依赖 ``θ*``，所以 ``α/β`` 收敛到
``θ*``。误差用相对 MAE ``mean(|θ̂_c - θ*_c| / θ*_c)``；超额事件用
:func:`skygovc.calibration.bayes.underestimation_excess` 按**全部架次**累计。

回退判据与反事实
----------------
:func:`rollback_rule` 用 :class:`~skygovc.calibration.martingale.HedgedCapitalCS`
对"每万架次事件率 / 基准率"这类比值信号做序贯监测；当置信下界连续
``min_days`` 天低于阈值时触发回退（保守、低误触发）。
:func:`evaluate_rollback` 在第 ``true_drift_at`` 天注入一次真实风险变化，
统计检出率 / 误触发率 / 平均检出延迟。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .bayes import (
    PoissonGammaEstimator,
    default_prior,
    relative_mae,
    underestimation_excess,
)
from .conformal import ConformalReport, evaluate_split_conformal
from .martingale import HedgedCapitalCS

__all__ = [
    "CAPABILITY",
    "MECHANISMS",
    "MechanismConfig",
    "RollbackDecision",
    "evaluate_rollback",
    "rollback_rule",
    "run_loop",
]

#: 校准闭环所属的能力族（与 ``skygovc.core.CAPABILITIES`` 中的 ``"data"`` 对齐）。
CAPABILITY: str = "data"

#: 默认风险倍数真值与每日架次（低/中/高）。
DEFAULT_TRUTH: Tuple[float, float, float] = (1.0, 3.0, 10.0)
DEFAULT_N_BY_CLASS: Tuple[float, float, float] = (1200.0, 600.0, 180.0)

#: 回退监测的默认参数。
DEFAULT_WINDOW: int = 30
DEFAULT_MIN_DAYS: int = 14
DEFAULT_SIGNAL_SCALE: float = 2.0
DEFAULT_EXPECTED_EVENTS: float = 400.0


@dataclass(frozen=True)
class MechanismConfig:
    """一种反馈机制的配置。

    Attributes
    ----------
    name:
        ``"L0"`` / ``"L1"`` / ``"L2"`` / ``"L3"``。
    coverage:
        监测覆盖率（0..1）。``0`` 表示完全不监测。
    update_every:
        贝叶斯更新周期（天）；``0`` 表示从不更新。
    correct_rules:
        是否把估计写回规则（决定"数据是否改变判断"）。
    """

    name: str
    coverage: float
    update_every: int
    correct_rules: bool

    def __post_init__(self) -> None:
        name = str(self.name)
        if not name:
            raise ValueError("name 不能为空")
        cov = float(self.coverage)
        if not (0.0 <= cov <= 1.0):
            raise ValueError(f"coverage 必须在 [0, 1] 内，得到 {cov}")
        upd = int(self.update_every)
        if upd < 0:
            raise ValueError(f"update_every 不能为负，得到 {upd}")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "coverage", cov)
        object.__setattr__(self, "update_every", upd)
        object.__setattr__(self, "correct_rules", bool(self.correct_rules))

    @property
    def closed_loop(self) -> bool:
        """数据是否真正进入规则（更新 + 写回都成立）。"""
        return self.update_every > 0 and self.correct_rules

    @property
    def monitors(self) -> bool:
        """是否有监测数据被记录（L0 为 False）。"""
        return self.coverage > 0.0


#: 四种机制的标准配置（论文对照实验用）。
MECHANISMS: Dict[str, MechanismConfig] = {
    "L0": MechanismConfig("L0", coverage=0.0, update_every=0, correct_rules=False),
    "L1": MechanismConfig("L1", coverage=0.6, update_every=0, correct_rules=False),
    "L2": MechanismConfig("L2", coverage=0.6, update_every=7, correct_rules=True),
    "L3": MechanismConfig("L3", coverage=0.6, update_every=1, correct_rules=True),
}


def _as_three(x, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size != 3:
        raise ValueError(f"{name} 必须是 3 元组（低/中/高），得到长度 {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} 必须全部有限")
    return arr


# --------------------------------------------------------------------------
# 闭环对照实验
# --------------------------------------------------------------------------
def run_loop(
    cfg: MechanismConfig,
    days: int = 180,
    reps: int = 200,
    seed: int = 0,
    base_rate: float = 0.01,
    truth: Tuple[float, float, float] = DEFAULT_TRUTH,
    n_by_class: Tuple[float, float, float] = DEFAULT_N_BY_CLASS,
) -> dict:
    """模拟一种反馈机制 ``days`` 天、重复 ``reps`` 次。

    Parameters
    ----------
    cfg:
        机制配置（一般取自 :data:`MECHANISMS`）。
    days, reps:
        模拟天数与重复次数；``reps`` 次的结果取平均。
    seed:
        随机种子。``reps`` 次模拟分别用 ``default_rng([seed, rep])``，
        因此固定 ``seed`` 时结果**完全可复现**。
    base_rate:
        基准事件率（每次飞行的期望事件数当 ``θ = 1``）。
    truth:
        真实风险倍数 ``(θ_low, θ_medium, θ_high)``。
    n_by_class:
        每日架次 ``(n_low, n_medium, n_high)``。

    Returns
    -------
    dict
        ``{"mae_series": list[float] (长度 days),
        "mae_at": {"30":…, "90":…, "180":…},
        "excess_events": float, "conformal": ConformalReport|None,
        "mechanism": str, "final_theta": np.ndarray}``

        其中 ``conformal`` 只在**闭环**机制（L2/L3）下非 ``None``：
        ``L0``/``L1`` 没有反馈就没有可用的区间，这正是论断的一部分。
    """
    if not isinstance(cfg, MechanismConfig):
        raise TypeError("cfg 必须是 MechanismConfig 实例")
    days = int(days)
    reps = int(reps)
    if days < 1:
        raise ValueError(f"days 至少为 1，得到 {days}")
    if reps < 1:
        raise ValueError(f"reps 至少为 1，得到 {reps}")
    base_rate = float(base_rate)
    if not np.isfinite(base_rate) or base_rate <= 0.0:
        raise ValueError(f"base_rate 必须为正的有限数，得到 {base_rate}")
    truth_vec = _as_three(truth, "truth")
    if np.any(truth_vec <= 0.0):
        raise ValueError("truth 必须严格为正")
    n_vec = _as_three(n_by_class, "n_by_class")
    if np.any(n_vec < 0.0):
        raise ValueError("n_by_class 不能为负")

    prior = default_prior()
    prior_theta = prior.mean

    update_every = cfg.update_every
    closed = cfg.closed_loop
    # 监测侧的"θ = 1 期望事件数"，只覆盖被监测的架次。
    exposure_mon = cfg.coverage * base_rate * n_vec
    lam_obs = exposure_mon * truth_vec

    mae_sum = np.zeros(days, dtype=float)
    excess_total = 0.0
    if closed:
        preds = np.empty(reps * days, dtype=float)
        obs = np.empty(reps * days, dtype=float)
    else:
        preds = obs = None

    for rep in range(reps):
        rng = np.random.default_rng([int(seed), rep])
        events = rng.poisson(lam_obs, size=(days, 3)).astype(float)
        est = PoissonGammaEstimator(prior)
        pending_events = np.zeros(3, dtype=float)
        pending_exposure = np.zeros(3, dtype=float)
        row = rep * days
        for d in range(days):
            # 1) 记录：无论是否反馈，监测数据都先进账本
            pending_events += events[d]
            pending_exposure += exposure_mon
            # 2) 更新：只有闭环机制才把账本写进后验
            if update_every > 0 and (d + 1) % update_every == 0:
                est.update(pending_events, pending_exposure)
                pending_events[:] = 0.0
                pending_exposure[:] = 0.0
            # 3) 判断：只有把估计写回规则，规则才会变
            theta = est.mean if closed else prior_theta
            mae_sum[d] += relative_mae(theta, truth_vec)
            excess_total += underestimation_excess(theta, truth_vec, base_rate, n_vec)
            if closed:
                preds[row + d] = float(np.dot(exposure_mon, theta))
                obs[row + d] = float(np.sum(events[d]))

    mae_series = (mae_sum / reps).tolist()
    mae_at = {str(k): float(mae_series[min(k, days) - 1]) for k in (30, 90, 180)}

    conformal: Optional[ConformalReport] = None
    if closed:
        conformal = evaluate_split_conformal(preds, obs, alpha=0.1, calib_frac=0.5, seed=int(seed))

    return {
        "mae_series": mae_series,
        "mae_at": mae_at,
        "excess_events": float(excess_total / reps),
        "conformal": conformal,
        "mechanism": cfg.name,
        "final_theta": np.asarray(est.mean, dtype=float),
    }


# --------------------------------------------------------------------------
# 回退判据
# --------------------------------------------------------------------------
@dataclass
class RollbackDecision:
    """回退判据的判定结果。

    Attributes
    ----------
    triggered:
        是否触发回退。
    day:
        触发的第几天（1 起算）；未触发为 ``None``。
    lower_bound:
        触发当天的置信界（下行监测为下界，上行监测为上界）；未触发时为最后
        一天的对应界。
    threshold:
        阈值（与 ``lower_bound`` 同量纲）。
    reason:
        人类可读的判定说明（进审计日志）。
    """

    triggered: bool
    day: Optional[int]
    lower_bound: float
    threshold: float
    reason: str


def _estimate_sigma(signal: np.ndarray, warmup: int, scale: float) -> float:
    """用预热期样本标准差估计次高斯尺度（下限防零方差，上限取有界尺度）。"""
    k = int(min(max(2, warmup), signal.size))
    warm = signal[:k]
    sd = float(np.std(warm, ddof=1)) if k >= 2 else 0.0
    floor = 1e-3 * max(scale, 1.0)
    cap = 0.5 * scale
    return float(min(max(sd, floor), cap))


def _monitor(
    signal: np.ndarray,
    level: float,
    alpha: float,
    min_days: int,
    window: Optional[int],
    sigma: Optional[float],
    scale: float,
    direction: str,
) -> RollbackDecision:
    """内部监测器：``direction="down"`` 盯下界，``"up"`` 盯上界。"""
    if direction not in ("down", "up"):
        raise ValueError("direction 只能是 'down' 或 'up'")
    min_days = int(min_days)
    if min_days < 1:
        raise ValueError(f"min_days 至少为 1，得到 {min_days}")
    alpha = float(alpha)
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha 必须在 (0, 1) 内，得到 {alpha}")
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"scale 必须为正的有限数，得到 {scale}")
    level = float(level)
    if not np.isfinite(level) or not (0.0 < level < scale):
        raise ValueError(f"阈值必须在 (0, {scale}) 内，得到 {level}")

    if sigma is None:
        sigma = _estimate_sigma(signal, DEFAULT_WINDOW if window is None else window, scale)
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"sigma 必须是正的有限数，得到 {sigma}")

    cs = HedgedCapitalCS(
        grid_size=51,
        alpha=alpha,
        bound=(0.0, scale),
        sigma=sigma,
        window=window,
    )

    streak = 0
    max_streak = 0
    last = float("nan")
    for t, x in enumerate(signal, start=1):
        cs.update(float(x))
        last = float(cs.upper_now() if direction == "up" else cs.lower_now())
        breach = last > level if direction == "up" else last < level
        if breach:
            streak += 1
            if streak > max_streak:
                max_streak = streak
            if streak >= min_days:
                side = "上界" if direction == "up" else "下界"
                cmp_word = "高于" if direction == "up" else "低于"
                return RollbackDecision(
                    triggered=True,
                    day=int(t),
                    lower_bound=last,
                    threshold=level,
                    reason=(
                        f"第 {t} 天：置信{side} {last:.4f} 连续 {min_days} 天"
                        f"{cmp_word}阈值 {level:.4f}，触发回退"
                    ),
                )
        else:
            streak = 0

    side = "上界" if direction == "up" else "下界"
    return RollbackDecision(
        triggered=False,
        day=None,
        lower_bound=last,
        threshold=level,
        reason=(
            f"未触发：最长连续越界 {max_streak} 天 < min_days={min_days}，"
            f"末次置信{side}={last:.4f}，阈值 {level:.4f}"
        ),
    )


def rollback_rule(
    signal_series: List[float],
    threshold: float,
    alpha: float = 0.05,
    min_days: int = DEFAULT_MIN_DAYS,
    window: Optional[int] = DEFAULT_WINDOW,
    sigma: Optional[float] = None,
    scale: float = DEFAULT_SIGNAL_SCALE,
) -> RollbackDecision:
    """对下行信号做序贯监测：置信下界连续 ``min_days`` 天低于阈值即回退。

    Parameters
    ----------
    signal_series:
        逐日信号，例如"每万架次事件率 / 基准率"（量纲约为 1.0 的比值），
        取值应在 ``(0, scale)`` 内。
    threshold:
        回退阈值（同量纲）。
    alpha:
        置信序列的错覆盖率。
    min_days:
        连续越界天数，越大越保守。
    window:
        置信序列的滑窗长度；``None`` 表示扩张型（用全部历史）。默认 30 天：
        扩张型对 35% 的下行漂移要 ~60 天才能分辨，滑窗型 ~10 天即可，而
        ``min_days`` 的连续要求足以压住重复观测带来的误触发。
    sigma:
        次高斯尺度（信号的日抽样标准差）。``None`` 时用前 ``window`` 天做
        预热估计（稳健、无需外部噪声模型）。
    scale:
        信号上界（``[0, scale]`` 有界），默认 2.0。

    Returns
    -------
    RollbackDecision
    """
    sig = np.asarray(signal_series, dtype=float).ravel()
    if sig.size == 0:
        raise ValueError("signal_series 不能为空")
    if not np.all(np.isfinite(sig)):
        raise ValueError("signal_series 必须全部有限")
    if window is not None:
        window = int(window)
        if window < 2:
            raise ValueError(f"window 至少为 2（或 None），得到 {window}")
    return _monitor(
        sig,
        level=threshold,
        alpha=alpha,
        min_days=min_days,
        window=window,
        sigma=sigma,
        scale=scale,
        direction="down",
    )


# --------------------------------------------------------------------------
# 反事实：注入一次真实漂移
# --------------------------------------------------------------------------
def evaluate_rollback(
    n_days: int = 180,
    true_drift_at: int = 90,
    drift_size: float = -0.35,
    threshold: float = 0.85,
    alpha: float = 0.05,
    reps: int = 200,
    seed: int = 0,
    expected_events: float = DEFAULT_EXPECTED_EVENTS,
    min_days: int = DEFAULT_MIN_DAYS,
    window: Optional[int] = DEFAULT_WINDOW,
) -> dict:
    """回退判据的反事实评估：检出率 / 误触发率 / 平均检出延迟。

    信号生成：监测机队每天有一个基线期望事件数 ``expected_events``，事件率
    比值 ``signal_t = e_t / expected_events``，``e_t ~ Poisson(mu_t *
    expected_events)``；``mu_t = 1``，从第 ``true_drift_at`` 天起变为
    ``1 + drift_size``（负值为风险下降，正是回退要捕捉的方向）。

    Parameters
    ----------
    drift_size:
        负值 → 监测下界跌破 ``threshold``；正值 → 风险上升，改为监测上界是否
        超过阈值（此时 ``threshold`` 应取 > 1 的告警线，若传入 ≤1 的值则用
        ``1 + |drift_size|/2``）。
    expected_events:
        基线期望事件数，决定信号的抽样噪声 ``sigma = 1/sqrt(expected)``。
    reps, seed:
        重复次数与种子（固定后完全可复现）。

    Returns
    -------
    dict
        ``{"detection_rate", "false_alarm_rate", "mean_delay_days",
        "median_delay_days", "n_detected", "reps", "n_days",
        "true_drift_at", "drift_size", "threshold", "alpha", "min_days",
        "window", "direction", "sigma"}``

        延迟按"注入后第几天检出"计（1 起算，注入当天记为 0）。
    """
    n_days = int(n_days)
    reps = int(reps)
    if n_days < 2:
        raise ValueError(f"n_days 至少为 2，得到 {n_days}")
    if reps < 1:
        raise ValueError(f"reps 至少为 1，得到 {reps}")
    true_drift_at = int(true_drift_at)
    if not (0 <= true_drift_at < n_days):
        raise ValueError(f"true_drift_at 必须在 [0, {n_days}) 内，得到 {true_drift_at}")
    drift_size = float(drift_size)
    if not np.isfinite(drift_size) or drift_size == 0.0:
        raise ValueError(f"drift_size 必须是非零有限数，得到 {drift_size}")
    expected_events = float(expected_events)
    if not np.isfinite(expected_events) or expected_events <= 0.0:
        raise ValueError(f"expected_events 必须为正的有限数，得到 {expected_events}")

    direction = "down" if drift_size < 0.0 else "up"
    drift_start_day = true_drift_at + 1  # 1 起算
    sigma = float(np.sqrt(1.0 / expected_events))

    level = float(threshold)
    if direction == "up" and level <= 1.0:
        level = 1.0 + 0.5 * abs(drift_size)
    if direction == "down" and level >= DEFAULT_SIGNAL_SCALE:
        raise ValueError(
            f"下行监测的阈值必须小于信号上界 {DEFAULT_SIGNAL_SCALE}，得到 {level}"
        )

    mu = np.ones(n_days, dtype=float)
    mu[true_drift_at:] = 1.0 + drift_size

    delays: List[float] = []
    n_detected = 0
    for rep in range(reps):
        rng = np.random.default_rng([int(seed), rep, 1])
        counts = rng.poisson(mu * expected_events)
        signal = counts / expected_events
        dec = _monitor(
            signal,
            level=level,
            alpha=alpha,
            min_days=min_days,
            window=window,
            sigma=sigma,
            scale=DEFAULT_SIGNAL_SCALE,
            direction=direction,
        )
        if dec.triggered:
            n_detected += 1
            delays.append(float(max(0, dec.day - drift_start_day)))

    n_false = 0
    for rep in range(reps):
        rng = np.random.default_rng([int(seed), rep, 2])
        counts = rng.poisson(np.full(n_days, expected_events, dtype=float))
        signal = counts / expected_events
        dec = _monitor(
            signal,
            level=level,
            alpha=alpha,
            min_days=min_days,
            window=window,
            sigma=sigma,
            scale=DEFAULT_SIGNAL_SCALE,
            direction=direction,
        )
        if dec.triggered:
            n_false += 1

    delay_arr = np.asarray(delays, dtype=float)
    if delay_arr.size:
        mean_delay = float(np.mean(delay_arr))
        median_delay = float(np.median(delay_arr))
        p90_delay = float(np.percentile(delay_arr, 90))
    else:
        mean_delay = float("nan")
        median_delay = float("nan")
        p90_delay = float("nan")

    return {
        "detection_rate": float(n_detected / reps),
        "false_alarm_rate": float(n_false / reps),
        "mean_delay_days": mean_delay,
        "median_delay_days": median_delay,
        "p90_delay_days": p90_delay,
        "n_detected": int(n_detected),
        "reps": int(reps),
        "n_days": int(n_days),
        "true_drift_at": int(true_drift_at),
        "drift_size": float(drift_size),
        "threshold": float(level),
        "alpha": float(alpha),
        "min_days": int(min_days),
        "window": None if window is None else int(window),
        "direction": direction,
        "sigma": sigma,
    }
