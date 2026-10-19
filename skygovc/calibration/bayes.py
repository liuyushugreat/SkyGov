"""Poisson–Gamma 共轭更新：三类风险任务的风险倍数 θ 的贝叶斯估计。

模型
----
对第 ``c`` 类风险任务（低/中/高），把"风险倍数" ``θ_c`` 视为未知参数，观测到
事件数 ``events_c`` 与风险暴露量 ``exposure_c``（架次 × 基准率，即在
``θ_c = 1`` 时的**期望事件数**）后，采用 Poisson 似然

.. math::

    \\mathrm{events}_c \\mid \\theta_c \\sim \\mathrm{Poisson}
        \\left(\\theta_c \\cdot \\mathrm{exposure}_c\\right)

与 Gamma 先验

.. math::

    \\theta_c \\sim \\mathrm{Gamma}(\\alpha_c, \\beta_c)

共轭更新为

.. math::

    \\alpha_c \\leftarrow \\alpha_c + \\mathrm{events}_c, \\qquad
    \\beta_c  \\leftarrow \\beta_c  + \\mathrm{exposure}_c

后验均值 ``α/β`` 即风险倍数估计 ``θ̂``；``α`` 可读作"伪事件数"、``β`` 可读作
"伪暴露量"，因此 ``strength`` 就是先验等价样本量（默认 15 天量级的暴露）。

先验默认值 ``prior_mult = (0.55, 1.7, 6.0)`` 使初始估计**系统性低估**真实风险
``θ* = (1.0, 3.0, 10.0)``——这正是论文里"未校准的规则会低估风险"的起点，
也是 ``underestimation_excess`` 衡量超额事件的依据。

备注
----
本模块只依赖标准库 + numpy（分位数另需 scipy，规范要求用
``scipy.stats.gamma.ppf``）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import stats

__all__ = [
    "DEFAULT_PRIOR_MULT",
    "GammaPrior",
    "PoissonGammaEstimator",
    "default_prior",
    "relative_mae",
    "underestimation_excess",
]

#: 默认先验均值倍数（低/中/高），刻意低于真值 ``(1.0, 3.0, 10.0)``。
DEFAULT_PRIOR_MULT: Tuple[float, float, float] = (0.55, 1.7, 6.0)


@dataclass
class GammaPrior:
    """Gamma 先验参数（逐风险类）。

    Attributes
    ----------
    alpha:
        形状参数，``shape (3,)``，全部为正。
    beta:
        速率参数（rate），``shape (3,)``，全部为正；后验均值为
        ``alpha / beta``。
    """

    alpha: np.ndarray
    beta: np.ndarray

    def __post_init__(self) -> None:
        a = np.asarray(self.alpha, dtype=float).ravel()
        b = np.asarray(self.beta, dtype=float).ravel()
        if a.shape != b.shape:
            raise ValueError(
                f"alpha 与 beta 形状必须一致，得到 {a.shape} 与 {b.shape}"
            )
        if a.size == 0:
            raise ValueError("先验不能为空")
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            raise ValueError("先验参数必须全部有限")
        if np.any(a <= 0.0) or np.any(b <= 0.0):
            raise ValueError("Gamma 先验的 alpha/beta 必须严格为正")
        self.alpha = a
        self.beta = b

    # -- 便利方法 ---------------------------------------------------------
    @property
    def mean(self) -> np.ndarray:
        """先验均值 ``alpha / beta``。"""
        return self.alpha / self.beta

    def copy(self) -> "GammaPrior":
        """独立副本（避免调用方被估计器内部更新污染）。"""
        return GammaPrior(alpha=self.alpha.copy(), beta=self.beta.copy())

    def __len__(self) -> int:
        return int(self.alpha.size)


def default_prior(
    strength: float = 15.0,
    prior_mult: Tuple[float, float, float] = DEFAULT_PRIOR_MULT,
) -> GammaPrior:
    """构造默认先验 ``alpha = prior_mult * strength``、``beta = strength``。

    Parameters
    ----------
    strength:
        先验等价样本量（伪暴露量天数），越大越"顽固"，闭环收敛越慢。
    prior_mult:
        三类任务的先验倍数均值。
    """
    strength = float(strength)
    if not np.isfinite(strength) or strength <= 0.0:
        raise ValueError(f"strength 必须为正的有限数，得到 {strength}")
    mult = np.asarray(prior_mult, dtype=float).ravel()
    if mult.size == 0 or not np.all(np.isfinite(mult)) or np.any(mult <= 0.0):
        raise ValueError("prior_mult 必须是全为正的有限序列")
    return GammaPrior(alpha=mult * strength, beta=np.full(mult.shape, strength, dtype=float))


def _as_vector(x, name: str, size: int | None = None) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    if size is not None and arr.size != size:
        raise ValueError(f"{name} 长度应为 {size}，得到 {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} 必须全部有限")
    return arr


def relative_mae(theta_hat, truth) -> float:
    """相对平均绝对误差 ``mean(|θ̂_c - θ_c| / θ_c)``。

    逐类相对误差再取平均，量纲无关；``truth`` 必须严格为正。
    """
    th = _as_vector(truth, "truth")
    est = _as_vector(theta_hat, "theta_hat")
    if th.size != est.size:
        raise ValueError(f"theta_hat 与 truth 长度不一致：{est.size} vs {th.size}")
    if np.any(th <= 0.0):
        raise ValueError("truth 必须严格为正（用于相对误差分母）")
    return float(np.mean(np.abs(est - th) / th))


def underestimation_excess(theta_hat, truth, base_rate: float, n_by_class) -> float:
    """低估导致的期望超额事件数（单日）。

    .. math::

        \\sum_c \\lambda_c \\cdot
            \\frac{\\max(0, \\theta_c - \\hat{\\theta}_c)}{\\theta_c} \\cdot 0.5,
        \\qquad \\lambda_c = \\text{base\\_rate} \\cdot \\theta_c \\cdot n_c

    其中 ``λ_c`` 是第 ``c`` 类在真实倍数下的期望事件数（全部架次，不只是监测
    覆盖部分——规则低估会让**所有**架次的风险被低估），``0.5`` 是"低估部分中
    转化为可预防超额事件"的折算系数。
    """
    th = _as_vector(truth, "truth")
    est = _as_vector(theta_hat, "theta_hat")
    if th.size != est.size:
        raise ValueError(f"theta_hat 与 truth 长度不一致：{est.size} vs {th.size}")
    if np.any(th <= 0.0):
        raise ValueError("truth 必须严格为正")
    lam = _as_vector(n_by_class, "n_by_class", size=th.size)
    if np.any(lam < 0.0):
        raise ValueError("n_by_class 不能为负")
    base_rate = float(base_rate)
    if not np.isfinite(base_rate) or base_rate <= 0.0:
        raise ValueError(f"base_rate 必须为正的有限数，得到 {base_rate}")
    expected_events = base_rate * th * lam
    shortfall = np.maximum(0.0, th - est) / th
    return float(np.sum(expected_events * shortfall * 0.5))


class PoissonGammaEstimator:
    """逐风险类的 Poisson–Gamma 共轭估计器。

    >>> est = PoissonGammaEstimator(default_prior())
    >>> est.update(np.array([7.0, 11.0, 12.0]), np.array([7.2, 3.6, 1.08]))
    >>> est.mean.shape
    (3,)
    """

    def __init__(self, prior: GammaPrior):
        if not isinstance(prior, GammaPrior):
            raise TypeError("prior 必须是 GammaPrior 实例")
        self._prior = prior.copy()
        self._alpha = self._prior.alpha.copy()
        self._beta = self._prior.beta.copy()
        self._total_events = np.zeros_like(self._alpha)
        self._total_exposure = np.zeros_like(self._alpha)
        self._n_updates = 0

    # -- 只读视图 ---------------------------------------------------------
    @property
    def prior(self) -> GammaPrior:
        """构造时使用的先验（副本）。"""
        return self._prior.copy()

    @property
    def alpha(self) -> np.ndarray:
        """当前后验形状参数（副本）。"""
        return self._alpha.copy()

    @property
    def beta(self) -> np.ndarray:
        """当前后验速率参数（副本）。"""
        return self._beta.copy()

    @property
    def n_updates(self) -> int:
        """已执行的更新次数。"""
        return self._n_updates

    @property
    def total_events(self) -> np.ndarray:
        """累计进入后验的事件数。"""
        return self._total_events.copy()

    @property
    def total_exposure(self) -> np.ndarray:
        """累计进入后验的暴露量。"""
        return self._total_exposure.copy()

    @property
    def mean(self) -> np.ndarray:
        """后验均值 ``alpha / beta``，即风险倍数估计 ``θ̂``。"""
        return self._alpha / self._beta

    # -- 更新 -------------------------------------------------------------
    def update(self, events, exposure) -> None:
        """共轭更新：``alpha += events``、``beta += exposure``。

        Parameters
        ----------
        events:
            观测事件数，``shape (3,)``，非负。
        exposure:
            风险暴露量（架次 × 基准率），``shape (3,)``，非负；即 ``θ = 1``
            时的期望事件数。
        """
        ev = _as_vector(events, "events")
        ex = _as_vector(exposure, "exposure")
        if ev.shape != self._alpha.shape:
            raise ValueError(
                f"events 长度应为 {self._alpha.size}，得到 {ev.size}"
            )
        if ex.shape != self._alpha.shape:
            raise ValueError(
                f"exposure 长度应为 {self._alpha.size}，得到 {ex.size}"
            )
        if np.any(ev < 0.0):
            raise ValueError("events 不能为负")
        if np.any(ex < 0.0):
            raise ValueError("exposure 不能为负")
        self._alpha = self._alpha + ev
        self._beta = self._beta + ex
        self._total_events = self._total_events + ev
        self._total_exposure = self._total_exposure + ex
        self._n_updates += 1

    # -- 不确定性 ---------------------------------------------------------
    def credible_interval(self, level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """等尾可信区间（Gamma 分位数）。

        返回 ``(lower, upper)``，各为 ``shape (3,)``；``level`` 为可信水平。
        """
        level = float(level)
        if not (0.0 < level < 1.0):
            raise ValueError(f"level 必须在 (0, 1) 内，得到 {level}")
        tail = (1.0 - level) / 2.0
        scale = 1.0 / self._beta
        lower = np.asarray(stats.gamma.ppf(tail, a=self._alpha, scale=scale), dtype=float)
        upper = np.asarray(
            stats.gamma.ppf(1.0 - tail, a=self._alpha, scale=scale), dtype=float
        )
        return lower, upper

    # -- 误差度量 ---------------------------------------------------------
    def relative_mae(self, truth) -> float:
        """``mean(|θ̂_c - θ_c| / θ_c)``，见 :func:`relative_mae`。"""
        return relative_mae(self.mean, truth)

    def underestimation_excess(
        self, truth, base_rate: float, n_by_class
    ) -> float:
        """低估导致的单日期望超额事件数，见 :func:`underestimation_excess`。"""
        return underestimation_excess(self.mean, truth, base_rate, n_by_class)

    # -- 展示 -------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - 仅用于交互排查
        theta = np.array2string(self.mean, precision=4)
        return (
            f"PoissonGammaEstimator(theta={theta}, "
            f"updates={self._n_updates})"
        )
