"""超体积（hypervolume）计算。

三个函数覆盖论文需要的两种用法：

* :func:`hypervolume_2d` —— 二维**精确**扫描法，用于单元测试与二维切片图；
* :func:`hypervolume_nd` —— 任意维的**蒙特卡洛**估计，返回被支配区域占
  ``[ideal, ref]`` 包围盒的比例（∈ ``[0, 1]``）；
* :func:`normalized_hypervolume` —— 先把目标按 ``ideal``/``reference_point``
  仿射归一化到单位盒，再估计被支配比例，因此可作为不同前沿之间的
  **覆盖率比值**（测试里用 ``HV(NSGA-II front) / HV(真前沿)``）。

全部函数采用最小化语义：点 ``y`` 支配的区域是
``{z : y_i <= z_i <= ref_i, ∀i}``，超体积即这些盒子的并集体积。
蒙特卡洛部分由 ``numpy.random.default_rng(seed)`` 驱动，固定 ``seed``
即可完全复现。
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

__all__ = [
    "hypervolume_2d",
    "hypervolume_nd",
    "normalized_hypervolume",
    "dominated_fraction",
]

#: 判定"点在参考点之内"的浮点容差。
_EPS = 1e-12


def hypervolume_2d(points: Sequence[Sequence[float]], ref: Sequence[float]) -> float:
    """二维精确超体积（扫描法），最小化语义。

    算法：按第一目标升序扫描，维护"已见过的最小第二目标" ``prev_y``。
    每遇到一个更低（更好）的 ``y``，就把宽度 ``ref_x - x``、高度
    ``prev_y - y`` 的横条面积累加进去；被前面点支配的点自然贡献 0。

    例：``points=[(1,1),(2,0.5)], ref=(3,3)``

    * ``x∈[1,2]``：``y`` 从 1 到 3，面积 ``1×2 = 2``
    * ``x∈[2,3]``：``y`` 从 0.5 到 3，面积 ``1×2.5 = 2.5``
    * 合计 ``4.5``
    """
    rx, ry = float(ref[0]), float(ref[1])
    pts: List[Tuple[float, float]] = []
    for p in points:
        x, y = float(p[0]), float(p[1])
        if x < rx and y < ry:                  # 完全落在参考点外的点体积为 0
            pts.append((x, y))
    if not pts:
        return 0.0
    pts.sort(key=lambda t: (t[0], t[1]))
    volume = 0.0
    prev_y = ry
    for x, y in pts:
        if y < prev_y - _EPS:
            volume += (rx - x) * (prev_y - y)
            prev_y = y
    return float(volume)


def dominated_fraction(points: Sequence[Sequence[float]], samples: int = 200000, seed: int = 0) -> float:
    """单位盒 ``[0,1]^d`` 中被 ``points`` 支配的体积比例。

    这是三个公开函数的公共内核：``d <= 2`` 时用精确算法，``d >= 3`` 时用
    蒙特卡洛。``points`` 的坐标允许略大于 1（此时该点在单位盒内没有投影，
    贡献 0）。
    """
    P = np.asarray(points, dtype=np.float64)
    if P.size == 0:
        return 0.0
    if P.ndim != 2:
        raise ValueError("points 必须是等长的目标向量序列")
    n, dim = P.shape
    if dim == 0 or n == 0:
        return 0.0
    if dim == 1:
        lo = float(np.min(P[:, 0]))
        return float(min(1.0, max(0.0, 1.0 - lo)))
    if dim == 2:
        # 单位盒面积 = 1，故超体积数值即比例。
        return hypervolume_2d(P, (1.0, 1.0))
    rng = np.random.default_rng(int(seed))
    U = rng.random((int(samples), dim))            # U ~ Uniform([0,1]^dim)
    covered = np.zeros(int(samples), dtype=bool)
    for row in P:
        if np.any(row > 1.0):                      # 参考点之外的点不可能支配盒内点
            continue
        covered |= np.all(U >= row[None, :], axis=1)
    return float(covered.mean())


def hypervolume_nd(
    points: Sequence[Sequence[float]],
    ref: Sequence[float],
    samples: int = 200000,
    seed: int = 0,
) -> float:
    """任意维超体积估计，返回被支配部分占 ``[ideal, ref]`` 包围盒的**比例**。

    其中 ``ideal`` 取 ``points`` 的逐目标最小值。之所以返回比例而不是绝对
    体积：三目标问题的绝对体积强烈依赖单位（元 vs 亿元），比例版本才可
    跨情境比较，也与 :func:`normalized_hypervolume` 完全一致。

    ``dim <= 2`` 走精确路径，``dim >= 3`` 走蒙特卡洛（``samples`` 个样本，
    固定 ``seed`` 可复现）。
    """
    P = np.asarray(points, dtype=np.float64)
    ref_a = np.asarray(ref, dtype=np.float64)
    if P.size == 0:
        return 0.0
    if P.ndim != 2:
        raise ValueError("points 必须是等长的目标向量序列")
    if ref_a.shape != (P.shape[1],):
        raise ValueError(f"参考点维度 {ref_a.shape} 与目标维度 {P.shape[1]} 不匹配")
    lo = P.min(axis=0)
    span = ref_a - lo
    if np.any(span <= 0.0):
        return 0.0
    Y = (P - lo) / span
    return dominated_fraction(Y, samples=samples, seed=seed)


def normalized_hypervolume(
    front_objectives: Sequence[Sequence[float]],
    reference_point: Sequence[float],
    ideal_point: Sequence[float],
    samples: int = 200000,
    seed: int = 0,
) -> float:
    """归一化超体积：``(f - ideal) / (ref - ideal)`` 后单位盒的被支配比例。

    返回值 ∈ ``[0, 1]``。若 ``ref`` 与 ``ideal`` 在某维退化（``ref <= ideal``）
    则抛 ``ValueError``——这通常意味着参考点选错了，而不是可以静默返回 0。
    """
    P = np.asarray([[float(v) for v in p] for p in front_objectives], dtype=np.float64)
    if P.size == 0:
        return 0.0
    ref_a = np.asarray([float(v) for v in reference_point], dtype=np.float64)
    ideal_a = np.asarray([float(v) for v in ideal_point], dtype=np.float64)
    if ref_a.shape != ideal_a.shape or ref_a.shape != (P.shape[1],):
        raise ValueError("ideal / reference / front 的维度必须一致")
    span = ref_a - ideal_a
    if np.any(span <= 0.0):
        raise ValueError(f"参考点必须在理想点之外（逐维严格大于）：span={span.tolist()}")
    Y = (P - ideal_a) / span
    Y = np.maximum(Y, 0.0)                        # 优于 ideal 的点按 ideal 处理
    return dominated_fraction(Y, samples=samples, seed=seed)
