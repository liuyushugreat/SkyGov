"""Hedged-capital 置信序列（Howard et al., 2021 风格）用于序贯监测。

为什么要"时间一致"
------------------
监管闭环每天都要看一眼数据。若每天都用一个 95% 的**定点**区间，看 180 天
就会把误报概率放大到 ``1 - 0.95**180 ≈ 1``。"置信序列"（confidence sequence）
是**时间一致**的：以概率 ``1 - alpha`` 保证**整条区间带**在所有时刻同时覆盖
真值，因此可以随时看、随时停、随时判。

构造
----
对取值于 ``[a, b]`` 的观测 ``x_1, x_2, ...``（均值 ``mu``），假设
``x_i - mu`` 是 ``sigma``-次高斯的（``[a, b]`` 有界时取
``sigma = (b - a) / 2`` 由 Hoeffding 引理必然成立）。对固定备择
``mu_0`` 与下注 ``lam``，似然比资本过程

.. math::

    K_t(\\lambda) = \\prod_{i \\le t}
        \\exp\\left(\\lambda (x_i - \\mu_0) - \\frac{\\lambda^2 \\sigma^2}{2}\\right)

在 ``H_0: \\mu = \\mu_0`` 下是上鞅（``E[K_t] <= 1``）。对一组下注取混合
（即 *hedged capital*，"对冲"单个 lam 选得不好带来的损失）

.. math::

    \\bar K_t(\\mu_0) = \\frac{1}{|\\Lambda|} \\sum_{\\lambda \\in \\Lambda}
        \\exp\\left(\\lambda (S_t - t\\mu_0)
        - \\frac{t \\lambda^2 \\sigma^2}{2}\\right),
    \\qquad S_t = \\sum_{i \\le t} x_i

仍是上鞅（Ville 不等式），于是

.. math::

    \\mathrm{CS}_t = \\{\\mu : \\bar K_t(\\mu) < 1/\\alpha\\}

满足 ``Pr(forall t: mu in CS_t) >= 1 - alpha``。

实现要点
--------
* ``lam`` 网格取**几何**网格（覆盖多个数量级），因此大 ``n`` 时也能命中
  接近最优的下注 ``lam* = (mu - mu_hat)/sigma^2``；
* ``log K_t(mu)`` 是 ``mu`` 的凸函数，所以 ``CS_t`` 一定是区间，用
  ``mu`` 网格 + 线性插值求端点；
* ``lower()/upper()`` 返回**至今所有区间之交**（仍是 ``1-alpha`` 置信序列，
  且宽度单调不增，便于画图与审计）；``lower_now()/upper_now()`` 返回
  **当前时刻**的区间，用于变点/漂移检测（交集会"记住"历史，无法反映下行）。

状态只需 ``(n, S)`` 两个数，因此 ``update`` 是 O(1) 的（网格求值 O(|Λ||Μ|)
只在需要区间时发生），窗口监测也不需要重放历史。
"""
from __future__ import annotations

from collections import deque
from typing import Optional, Tuple

import numpy as np

__all__ = ["HedgedCapitalCS"]

#: 网格缓存（同参数重复构造时省掉网格生成），避免 evaluate_rollback 反复分配。
_GRID_CACHE: dict = {}
_GRID_CACHE_MAX = 64

#: lam 网格覆盖的下界相对宽度（相对 lam_max）。
_LAM_REL_MIN = 1e-2


def _log_mean_exp(mat: np.ndarray) -> np.ndarray:
    """按第 0 维求 ``log(mean_g exp(mat[g, :]))``（数值稳定）。"""
    m = np.max(mat, axis=0)
    return m + np.log(np.mean(np.exp(mat - m), axis=0))


def _make_grids(
    grid_size: int,
    alpha: float,
    sigma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """构造（并缓存）``lam`` 网格与 ``lam^2`` 半系数。"""
    key = (grid_size, round(alpha, 15), round(sigma, 15))
    hit = _GRID_CACHE.get(key)
    if hit is not None:
        return hit

    lam_max = float(np.sqrt(2.0 * (np.log(1.0 / alpha) + np.log(grid_size))) / sigma)
    half = grid_size // 2
    mags = lam_max * np.logspace(np.log10(_LAM_REL_MIN), 0.0, half)
    lam = np.concatenate((-mags[::-1], [0.0], mags))  # 长度 2*half+1 == grid_size(奇)
    lam_half_sq = 0.5 * (lam * lam)

    if len(_GRID_CACHE) >= _GRID_CACHE_MAX:
        _GRID_CACHE.clear()
    _GRID_CACHE[key] = (lam, lam_half_sq)
    return _GRID_CACHE[key]


class HedgedCapitalCS:
    """有界观测均值的 hedged-capital 置信序列。

    Parameters
    ----------
    grid_size:
        下注网格 ``|Λ|``（默认 101，取奇数；越大越接近最优下注，混合惩罚
        ``log|Λ|`` 也越大）。
    alpha:
        置信序列的错覆盖率（默认 0.05 ⇒ 95% 时间一致覆盖）。
    bound:
        观测的上下界 ``(a, b)``。
    sigma:
        次高斯尺度。``None`` 时取 ``(b - a) / 2``（Hoeffding 引理，对有界
        观测永远成立，但偏保守）；若已知观测噪声更小（例如日事件率比值的
        抽样标准差），传入更小的 ``sigma`` 可显著收紧区间。
    window:
        ``None`` 表示用全部历史（经典扩张型置信序列）；给定整数 ``w`` 则只
        用最近 ``w`` 个观测（滑窗型，用于跟踪漂移；见 ``loop.rollback_rule``）。
    mu_grid_size:
        求区间端点时的 ``mu`` 网格点数（默认自动）。

    Notes
    -----
    ``lower()/upper()`` 是"至今所有区间之交"，宽度单调不增；检测漂移请用
    ``lower_now()/upper_now()``。
    """

    def __init__(
        self,
        grid_size: int = 101,
        alpha: float = 0.05,
        bound: Tuple[float, float] = (0.0, 1.0),
        sigma: Optional[float] = None,
        window: Optional[int] = None,
        mu_grid_size: Optional[int] = None,
    ):
        grid_size = int(grid_size)
        if grid_size < 5:
            raise ValueError(f"grid_size 至少为 5，得到 {grid_size}")
        grid_size = grid_size if grid_size % 2 == 1 else grid_size + 1
        alpha = float(alpha)
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha 必须在 (0, 1) 内，得到 {alpha}")

        a, b = float(bound[0]), float(bound[1])
        if not (np.isfinite(a) and np.isfinite(b)) or not (a < b):
            raise ValueError(f"bound 必须是有限且 a < b 的二元组，得到 {bound}")

        if sigma is None:
            sigma = 0.5 * (b - a)
        sigma = float(sigma)
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError(f"sigma 必须是正的有限数，得到 {sigma}")

        if window is not None:
            window = int(window)
            if window < 2:
                raise ValueError(f"window 至少为 2（或 None），得到 {window}")

        if mu_grid_size is None:
            mu_grid_size = max(129, 2 * grid_size + 1)
        mu_grid_size = int(mu_grid_size)
        if mu_grid_size < 33:
            raise ValueError(f"mu_grid_size 至少为 33，得到 {mu_grid_size}")

        self.a = a
        self.b = b
        self.alpha = alpha
        self.grid_size = grid_size
        self.sigma = sigma
        self.window = window

        self._lam, self._lam_half_sq = _make_grids(grid_size, alpha, sigma)
        self._mu_grid_size = mu_grid_size
        self._log_alpha = float(np.log(1.0 / alpha))
        self._buf: deque = deque(maxlen=window)
        self._n_eff = 0
        self._t = 0
        self._sum = 0.0
        # 至今所有区间之交（单调不增宽度）
        self._lo = a
        self._hi = b
        # 当前时刻区间
        self._lo_now = a
        self._hi_now = b
        self._refreshed_n = -1

    # ------------------------------------------------------------------
    # 只读属性
    # ------------------------------------------------------------------
    @property
    def n(self) -> int:
        """已处理的观测个数（滑窗模式下仍是**总**个数）。"""
        return self._t

    @property
    def n_effective(self) -> int:
        """当前参与统计的观测个数（滑窗模式下为窗口长度）。"""
        return self._n_eff

    @property
    def mean(self) -> float:
        """当前（滑窗）样本均值；无观测时返回区间中点。"""
        if self._n_eff == 0:
            return 0.5 * (self.a + self.b)
        return self._sum / self._n_eff

    # ------------------------------------------------------------------
    # 更新
    # ------------------------------------------------------------------
    def update(self, x: float) -> None:
        """加入一个观测 ``x``（超出 ``bound`` 时截断到边界）。"""
        x = float(x)
        if not np.isfinite(x):
            raise ValueError(f"观测必须有限，得到 {x}")
        xc = min(max(x, self.a), self.b)
        if self.window is not None and len(self._buf) == self.window:
            self._sum -= self._buf[0]
        self._buf.append(xc)
        self._sum += xc
        self._n_eff = len(self._buf)
        self._t += 1
        self._refresh()

    # ------------------------------------------------------------------
    # 内部：由 (n, S) 求当前区间
    # ------------------------------------------------------------------
    def log_capital(self, mu: float) -> float:
        """混合资本过程在假设 ``mu`` 处的对数值 ``log bar K_t(mu)``。"""
        if self._n_eff == 0:
            return 0.0
        mu = float(mu)
        if not np.isfinite(mu):
            raise ValueError("mu 必须有限")
        z = self._lam * (self._sum - self._n_eff * mu) - (
            self._n_eff * self._lam_half_sq * self.sigma * self.sigma
        )
        m = float(np.max(z))
        return float(m + np.log(np.mean(np.exp(z - m))))

    def _capital_on(self, mu: np.ndarray) -> np.ndarray:
        """``log bar K_t(mu)`` 在给定 ``mu`` 数组上的值。"""
        n = self._n_eff
        z = np.multiply.outer(self._lam, self._sum - n * mu) - (
            n * self.sigma * self.sigma
        ) * self._lam_half_sq[:, None]
        return _log_mean_exp(z)

    def _refresh(self) -> None:
        """重算当前区间并更新"交集"包络。

        端点用**以样本均值中心的局部网格**求解（网格半宽取典型半宽的 4 倍），
        这样即使区间极窄（噪声很小、``n`` 很大）也不会"穿过"网格；网格解得
        不到接受点时回退到二分法。
        """
        n = self._n_eff
        if n == 0:
            self._lo_now, self._hi_now = self.a, self.b
            self._refreshed_n = 0
            return

        mu_hat = self._sum / n
        # 典型半宽（来自混合上限），局部网格覆盖其 4 倍；n 很小时直接用全界。
        typical = self.sigma * float(
            np.sqrt(2.0 * (self._log_alpha + np.log(self.grid_size)) / n)
        )
        if n >= 16 and typical > 0.0:
            radius = 4.0 * typical
            left_edge = max(self.a, mu_hat - radius)
            right_edge = min(self.b, mu_hat + radius)
        else:
            left_edge, right_edge = self.a, self.b
        if not (right_edge > left_edge):
            left_edge, right_edge = self.a, self.b

        mu = np.linspace(left_edge, right_edge, self._mu_grid_size)
        lse = self._capital_on(mu)
        ok = lse < self._log_alpha

        i0 = int(np.argmin(np.abs(mu - mu_hat)))
        if not ok[i0]:
            idx = np.flatnonzero(ok)
            if idx.size == 0:
                # 网格没有捕捉到区间（极窄情形）：直接对凸函数二分。
                lo = self._bisect(mu_hat, self.a)
                hi = self._bisect(mu_hat, self.b)
                self._set_bounds(lo, hi)
                return
            i0 = int(idx[np.argmin(np.abs(mu[idx] - mu_hat))])

        left = i0
        while left > 0 and ok[left - 1]:
            left -= 1
        right = i0
        last = mu.size - 1
        while right < last and ok[right + 1]:
            right += 1

        lo = self._cross(mu, lse, left, step=-1)
        hi = self._cross(mu, lse, right, step=+1)
        lo = min(max(lo, self.a), self.b)
        hi = min(max(hi, self.a), self.b)
        if lo > hi:
            lo, hi = hi, lo
        self._set_bounds(lo, hi)

    def _bisect(self, mu_in: float, mu_out: float) -> float:
        """在 ``[mu_in（在区间内）, mu_out（在区间外）]`` 上二分求端点。"""
        if self.log_capital(mu_out) < self._log_alpha:
            return float(mu_out)  # 该侧整段都被接受 ⇒ 端点就是界
        lo, hi = float(mu_in), float(mu_out)
        tol = 1e-13 * max(1.0, abs(mu_out))
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if self.log_capital(mid) < self._log_alpha:
                lo = mid
            else:
                hi = mid
            if abs(hi - lo) <= tol:
                break
        return float(lo)

    def _cross(self, mu: np.ndarray, lse: np.ndarray, i: int, step: int) -> float:
        """在 ``mu[i]`` 与相邻格点之间线性插值出 ``log K = log(1/alpha)``。"""
        j = i + step
        if 0 <= j < mu.size:
            f0, f1 = float(lse[i]), float(lse[j])
            if f1 != f0:
                t = (self._log_alpha - f0) / (f1 - f0)
                t = min(max(t, 0.0), 1.0)
                return float(mu[i] + t * (mu[j] - mu[i]))
        return float(mu[i])

    def _set_bounds(self, lo: float, hi: float) -> None:
        self._lo_now, self._hi_now = float(lo), float(hi)
        new_lo = max(self._lo, lo)
        new_hi = min(self._hi, hi)
        if new_lo <= new_hi:  # 交集非空才更新（宽度因此单调不增）
            self._lo, self._hi = float(new_lo), float(new_hi)
        self._refreshed_n = self._n_eff

    # ------------------------------------------------------------------
    # 区间
    # ------------------------------------------------------------------
    def lower(self) -> float:
        """至今所有区间之交的下界（宽度单调不增）。"""
        return float(self._lo)

    def upper(self) -> float:
        """至今所有区间之交的上界（宽度单调不增）。"""
        return float(self._hi)

    def lower_now(self) -> float:
        """**当前时刻**置信序列的下界（跟踪漂移用）。"""
        return float(self._lo_now)

    def upper_now(self) -> float:
        """**当前时刻**置信序列的上界（跟踪漂移用）。"""
        return float(self._hi_now)

    def interval(self) -> Tuple[float, float]:
        """``(lower(), upper())``，即单调包络区间。"""
        return self.lower(), self.upper()

    def interval_now(self) -> Tuple[float, float]:
        """当前时刻的置信区间。"""
        return self.lower_now(), self.upper_now()

    def width(self) -> float:
        """单调包络区间宽度（随 ``n`` 非增）。"""
        return float(self._hi - self._lo)

    def width_now(self) -> float:
        """当前时刻区间宽度。"""
        return float(self._hi_now - self._lo_now)

    # ------------------------------------------------------------------
    # 检验
    # ------------------------------------------------------------------
    def is_reject(self, mu0: float) -> bool:
        """是否拒绝 ``H_0: mean == mu0``（``bar K_t(mu0) >= 1/alpha``）。"""
        if self._n_eff == 0:
            return False
        return bool(self.log_capital(mu0) >= self._log_alpha)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - 交互排查用
        return (
            f"HedgedCapitalCS(n={self._t}, n_eff={self._n_eff}, "
            f"mean={self.mean:.5f}, "
            f"interval=({self.lower():.5f}, {self.upper():.5f}), "
            f"alpha={self.alpha})"
        )
