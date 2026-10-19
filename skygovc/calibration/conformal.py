"""Split conformal 区间预测：带有限样本精确覆盖的区间核算。

原理
----
给定校准残差 ``R_1..R_n``（通常取 ``|y - ŷ|``）与目标错覆盖率 ``alpha``，
取

.. math::

    q = R_{(k)}, \\qquad k = \\lceil (n + 1)(1 - \\alpha) \\rceil

（升序第 ``k`` 个顺序统计量；若 ``k > n`` 则 ``q = +inf``）。在**可交换性**
假设下，对新样本 ``q`` 满足

.. math::

    \\Pr\\left(|y_{n+1} - \\hat{y}_{n+1}| \\le q\\right) \\ge 1 - \\alpha

即覆盖率是**有限样本精确**的（不是渐近的），代价是区间宽度通常略大于
oracle 区间。这正是论文里"给校准结果配一个可审计的覆盖率保证"的工具。

接口
----
``SplitConformal`` 只负责"残差 → q → 区间 → 覆盖率"四件事，
``evaluate_split_conformal`` 把点预测序列按 ``calib_frac`` 随机对半切开，
在另一半上核算经验覆盖率与平均宽度，返回 :class:`ConformalReport`。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

__all__ = ["ConformalReport", "SplitConformal", "evaluate_split_conformal"]

#: 比较残差是否落入区间时的数值容差（防止 1 ulp 误差翻转计数）。
_TOL = 1e-12


def _as_1d(x, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} 不能为空")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} 必须全部有限")
    return arr


@dataclass(frozen=True)
class ConformalReport:
    """一次 split-conformal 核算的结果。

    Attributes
    ----------
    alpha:
        目标错覆盖率（``0.1`` 表示 90% 区间）。
    q:
        校准得到的残差分位数（区间半宽）；小样本时可能为 ``inf``。
    empirical_coverage:
        测试集上的经验覆盖率。
    mean_width:
        测试集上的平均区间宽度（``2q``）。
    n_calib:
        校准集样本量。
    """

    alpha: float
    q: float
    empirical_coverage: float
    mean_width: float
    n_calib: int

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "q": self.q,
            "empirical_coverage": self.empirical_coverage,
            "mean_width": self.mean_width,
            "n_calib": self.n_calib,
        }


class SplitConformal:
    """Split（inductive）conformal 区间的核算器。

    >>> sc = SplitConformal(alpha=0.2)
    >>> r = [-3.0, 1.0, -0.5, 2.0, -1.5, 0.25, -2.5, 1.75, -0.1, 0.9]
    >>> q = sc.calibrate(r)          # n=10, k=ceil(11*0.8)=9 ⇒ 次大绝对残差
    >>> q
    2.5
    >>> sc.interval(3.0, q)
    (0.5, 5.5)
    """

    def __init__(self, alpha: float = 0.1):
        alpha = float(alpha)
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha 必须在 (0, 1) 内，得到 {alpha}")
        self.alpha = alpha

    # -- 校准 -------------------------------------------------------------
    def calibrate(self, residuals) -> float:
        """返回 ``q = |R|`` 的第 ``ceil((n+1)(1-alpha))`` 个顺序统计量。

        传入带符号残差时会先取绝对值（conformal 用绝对残差）。
        """
        r = np.abs(_as_1d(residuals, "residuals"))
        n = int(r.size)
        # 用极小 epsilon 吸收 (n+1)(1-alpha) 的浮点表示误差。
        k = int(np.ceil((n + 1) * (1.0 - self.alpha) - 1e-12))
        if k < 1:
            k = 1
        if k > n:
            # 有限样本下无法达到目标覆盖：标准做法是返回 +inf。
            return float("inf")
        return float(np.sort(r)[k - 1])

    # -- 预测 -------------------------------------------------------------
    def interval(self, point: float, q: float) -> Tuple[float, float]:
        """对称区间 ``(point - q, point + q)``。"""
        point = float(point)
        q = float(q)
        if not np.isfinite(point):
            raise ValueError("point 必须有限")
        if not (q >= 0.0) or np.isnan(q):
            raise ValueError(f"q 必须为非负实数，得到 {q}")
        return point - q, point + q

    # -- 核算 -------------------------------------------------------------
    def coverage(self, y_true, y_hat, q: float) -> float:
        """经验覆盖率 ``mean(|y_true - y_hat| <= q)``。"""
        yt = _as_1d(y_true, "y_true")
        yh = _as_1d(y_hat, "y_hat")
        if yt.size != yh.size:
            raise ValueError(f"y_true 与 y_hat 长度不一致：{yt.size} vs {yh.size}")
        q = float(q)
        if np.isnan(q) or q < 0.0:
            raise ValueError(f"q 必须为非负实数，得到 {q}")
        if np.isinf(q):
            return 1.0
        covered = np.abs(yt - yh) <= q + _TOL * max(1.0, abs(q))
        return float(np.mean(covered))


def evaluate_split_conformal(
    points,
    y_true,
    alpha: float = 0.1,
    calib_frac: float = 0.5,
    seed: int = 0,
) -> ConformalReport:
    """随机对半切分并核算 split-conformal 区间。

    Parameters
    ----------
    points:
        点预测序列 ``ŷ``（1-D）。
    y_true:
        真实观测序列 ``y``（1-D，与 ``points`` 等长）。
    alpha:
        目标错覆盖率。
    calib_frac:
        校准集占比，``0 < calib_frac < 1``。
    seed:
        切分随机种子（固定后结果完全可复现）。

    Returns
    -------
    ConformalReport
    """
    p = _as_1d(points, "points")
    y = _as_1d(y_true, "y_true")
    if p.size != y.size:
        raise ValueError(f"points 与 y_true 长度不一致：{p.size} vs {y.size}")
    calib_frac = float(calib_frac)
    if not (0.0 < calib_frac < 1.0):
        raise ValueError(f"calib_frac 必须在 (0, 1) 内，得到 {calib_frac}")
    n = int(p.size)
    if n < 2:
        raise ValueError("至少需要 2 个样本才能做 split conformal")

    n_calib = int(round(n * calib_frac))
    n_calib = min(max(n_calib, 1), n - 1)

    rng = np.random.default_rng(int(seed))
    order = rng.permutation(n)
    calib_idx = order[:n_calib]
    test_idx = order[n_calib:]

    sc = SplitConformal(alpha=alpha)
    q = sc.calibrate(y[calib_idx] - p[calib_idx])
    cov = sc.coverage(y[test_idx], p[test_idx], q)
    width = float(2.0 * q) if np.isfinite(q) else float("inf")

    return ConformalReport(
        alpha=sc.alpha,
        q=float(q),
        empirical_coverage=float(cov),
        mean_width=width,
        n_calib=int(n_calib),
    )
