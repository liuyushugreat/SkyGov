"""``skygovc.calibration`` 的单元测试。

覆盖规范要求的 7 项：

1. ``PoissonGammaEstimator`` 先验正确，且一次 ``update`` 后均值可手算核对；
2. ``relative_mae`` 对已知数组数值正确；
3. ``L0`` 与 ``L1`` 的 ``mae_series`` **严格逐元素相等**，``L2``/``L3`` 的
   180 日 MAE 显著小于 ``L0``（< 50%）；
4. ``SplitConformal.calibrate`` 的 ``q`` 与手算顺序统计量一致，
   ``evaluate_split_conformal`` 的经验覆盖率落在 ``[1-alpha-0.1, 1]``；
5. ``HedgedCapitalCS`` 对 1000 个 Bernoulli(0.3) 样本给出
   ``lower < 0.35``、``upper > 0.25``，且区间宽度随 ``n`` 非增；
6. ``rollback_rule`` 在无漂移序列上不触发、在明显下降序列上触发；
7. 固定 ``seed`` 时 ``run_loop`` 两次调用结果完全一致。

另附可信区间、低估超额事件、机制常量、conformal 切分可复现性等稳健性检查。

运行方式::

    & "python" -m pytest tests/test_calibration.py -q
"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest

# 让测试既能从仓库根目录、也能从 modules/SkyGovC 目录直接运行。
_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from skygovc.calibration import (  # noqa: E402
    DEFAULT_TRUTH,
    MECHANISMS,
    ConformalReport,
    GammaPrior,
    HedgedCapitalCS,
    MechanismConfig,
    PoissonGammaEstimator,
    RollbackDecision,
    SplitConformal,
    default_prior,
    evaluate_rollback,
    evaluate_split_conformal,
    relative_mae,
    rollback_rule,
    run_loop,
    underestimation_excess,
)

TRUTH = np.array(DEFAULT_TRUTH, dtype=float)


# --------------------------------------------------------------------------
# 1. 先验与一次更新
# --------------------------------------------------------------------------
def test_default_prior_values_and_shapes():
    prior = default_prior()
    assert prior.alpha.shape == (3,)
    assert prior.beta.shape == (3,)
    np.testing.assert_allclose(prior.alpha, np.array([0.55, 1.7, 6.0]) * 15.0)
    np.testing.assert_allclose(prior.beta, np.full(3, 15.0))
    np.testing.assert_allclose(prior.mean, [0.55, 1.7, 6.0])

    # 自定义 prior_mult / strength
    p2 = default_prior(strength=10.0, prior_mult=(1.0, 2.0, 4.0))
    np.testing.assert_allclose(p2.alpha, [10.0, 20.0, 40.0])
    np.testing.assert_allclose(p2.beta, [10.0, 10.0, 10.0])
    np.testing.assert_allclose(p2.mean, [1.0, 2.0, 4.0])


def test_prior_validation():
    with pytest.raises(ValueError):
        GammaPrior(alpha=np.array([1.0, -1.0, 2.0]), beta=np.ones(3))
    with pytest.raises(ValueError):
        GammaPrior(alpha=np.ones(3), beta=np.ones(2))
    with pytest.raises(ValueError):
        default_prior(strength=0.0)


def test_update_mean_hand_checked():
    prior = GammaPrior(alpha=np.array([2.0, 4.0, 6.0]), beta=np.array([2.0, 2.0, 2.0]))
    est = PoissonGammaEstimator(prior)
    np.testing.assert_allclose(est.mean, [1.0, 2.0, 3.0])

    est.update(events=np.array([1.0, 2.0, 3.0]), exposure=np.array([1.0, 2.0, 3.0]))
    # alpha = (3, 6, 9)，beta = (3, 4, 5) ⇒ mean = (1, 1.5, 1.8)
    np.testing.assert_allclose(est.alpha, [3.0, 6.0, 9.0])
    np.testing.assert_allclose(est.beta, [3.0, 4.0, 5.0])
    np.testing.assert_allclose(est.mean, [1.0, 1.5, 1.8])
    assert est.n_updates == 1
    np.testing.assert_allclose(est.total_events, [1.0, 2.0, 3.0])

    # 更新不污染先验
    np.testing.assert_allclose(prior.alpha, [2.0, 4.0, 6.0])
    np.testing.assert_allclose(est.prior.alpha, [2.0, 4.0, 6.0])


def test_update_validation_and_zero_update():
    est = PoissonGammaEstimator(default_prior())
    before = est.mean.copy()
    est.update(np.zeros(3), np.zeros(3))
    np.testing.assert_allclose(est.mean, before)

    with pytest.raises(ValueError):
        est.update(np.array([-1.0, 0.0, 0.0]), np.ones(3))
    with pytest.raises(ValueError):
        est.update(np.ones(3), np.array([-1.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        est.update(np.ones(2), np.ones(2))


def test_credible_interval_contains_mean_and_orders():
    est = PoissonGammaEstimator(default_prior())
    est.update(np.array([7.0, 11.0, 12.0]), np.array([7.2, 3.6, 1.08]))
    lo, hi = est.credible_interval(0.95)
    assert lo.shape == (3,) and hi.shape == (3,)
    assert np.all(lo < est.mean) and np.all(est.mean < hi)
    lo2, hi2 = est.credible_interval(0.5)
    # 水平越低区间越窄
    assert np.all(hi2 - lo2 < hi - lo)
    with pytest.raises(ValueError):
        est.credible_interval(1.5)


# --------------------------------------------------------------------------
# 2. 相对 MAE / 低估超额事件
# --------------------------------------------------------------------------
def test_relative_mae_numeric():
    theta_hat = np.array([1.1, 2.4, 12.0])
    truth = np.array([1.0, 3.0, 10.0])
    # 逐类相对误差 (0.1, 0.2, 0.2) ⇒ 平均 1/6
    assert relative_mae(theta_hat, truth) == pytest.approx((0.1 + 0.2 + 0.2) / 3.0, abs=1e-12)
    assert relative_mae(theta_hat, truth) == pytest.approx(1.0 / 6.0, abs=1e-12)
    assert relative_mae(truth, truth) == 0.0

    est = PoissonGammaEstimator(GammaPrior(alpha=np.array([1.1, 2.4, 12.0]),
                                          beta=np.array([1.0, 1.0, 1.0])))
    np.testing.assert_allclose(est.mean, theta_hat)
    assert est.relative_mae(truth) == pytest.approx(1.0 / 6.0, abs=1e-12)

    with pytest.raises(ValueError):
        relative_mae(theta_hat, np.array([1.0, 0.0, 10.0]))


def test_underestimation_excess_numeric():
    # λ = base_rate * θ * n = (12, 18, 18)；短路比 (0.45, 0.4333.., 0.4)
    excess = underestimation_excess(
        theta_hat=np.array([0.55, 1.7, 6.0]),
        truth=TRUTH,
        base_rate=0.01,
        n_by_class=(1200.0, 600.0, 180.0),
    )
    expected = 12.0 * 0.45 * 0.5 + 18.0 * (1.3 / 3.0) * 0.5 + 18.0 * 0.4 * 0.5
    assert excess == pytest.approx(expected, rel=1e-12)

    # 无低估 ⇒ 0；过度估计也不会产生"负超额"
    assert underestimation_excess(TRUTH, TRUTH, 0.01, (1200.0, 600.0, 180.0)) == 0.0
    assert underestimation_excess(TRUTH * 2.0, TRUTH, 0.01, (1200.0, 600.0, 180.0)) == 0.0


# --------------------------------------------------------------------------
# 3. L0/L1 严格相同 + L2/L3 显著收敛
# --------------------------------------------------------------------------
def test_mechanism_configs_match_spec():
    assert MECHANISMS["L0"] == MechanismConfig("L0", 0.0, 0, False)
    assert MECHANISMS["L1"] == MechanismConfig("L1", 0.6, 0, False)
    assert MECHANISMS["L2"] == MechanismConfig("L2", 0.6, 7, True)
    assert MECHANISMS["L3"] == MechanismConfig("L3", 0.6, 1, True)
    assert not MECHANISMS["L0"].closed_loop
    assert not MECHANISMS["L1"].closed_loop
    assert not MECHANISMS["L1"].correct_rules
    assert MECHANISMS["L2"].closed_loop and MECHANISMS["L3"].closed_loop
    with pytest.raises(ValueError):
        MechanismConfig("bad", 1.5, 0, False)


def test_l0_and_l1_mae_series_strictly_identical():
    l0 = run_loop(MECHANISMS["L0"], days=90, reps=8, seed=3)
    l1 = run_loop(MECHANISMS["L1"], days=90, reps=8, seed=3)

    # 列表逐元素严格相等（不是近似）
    assert l0["mae_series"] == l1["mae_series"]
    assert len(l0["mae_series"]) == 90
    assert l0["mae_at"] == l1["mae_at"]
    assert l0["excess_events"] == l1["excess_events"]

    # L1 确实记录了数据（覆盖率 0.6），但没有进入规则
    assert MECHANISMS["L1"].coverage > 0.0
    assert l1["conformal"] is None and l0["conformal"] is None

    # 误差恒定 = 先验的相对 MAE，且每一天都相同
    prior_mae = float(np.mean(np.abs(default_prior().mean - TRUTH) / TRUTH))
    assert l0["mae_at"]["180"] == pytest.approx(prior_mae, abs=1e-12)
    assert all(v == l0["mae_series"][0] for v in l0["mae_series"])


def test_l2_l3_converge_below_half_of_l0():
    l0 = run_loop(MECHANISMS["L0"], days=180, reps=20, seed=0)
    l2 = run_loop(MECHANISMS["L2"], days=180, reps=20, seed=0)
    l3 = run_loop(MECHANISMS["L3"], days=180, reps=20, seed=0)

    assert l0["mae_series"] == run_loop(MECHANISMS["L0"], days=180, reps=20, seed=0)["mae_series"]

    base = l0["mae_at"]["180"]
    assert l2["mae_at"]["180"] < 0.5 * base
    assert l3["mae_at"]["180"] < 0.5 * base
    # 闭环机制的误差单调下降（30 → 90 → 180 天）
    for res in (l2, l3):
        assert res["mae_at"]["180"] < res["mae_at"]["30"]
    # 日报不劣于周报（同样的信息量、更高的更新频率）
    assert l3["mae_at"]["30"] <= l2["mae_at"]["30"] + 1e-12
    # 超额事件同样显著下降
    assert l2["excess_events"] < 0.5 * l0["excess_events"]
    assert l3["excess_events"] < 0.5 * l0["excess_events"]

    # 闭环机制给出 conformal 报告，且经验覆盖率达标
    for res in (l2, l3):
        rep = res["conformal"]
        assert isinstance(rep, ConformalReport)
        assert rep.n_calib > 0
        assert 0.8 <= rep.empirical_coverage <= 1.0
        assert rep.q > 0.0
        assert rep.mean_width == pytest.approx(2.0 * rep.q, rel=1e-12)


# --------------------------------------------------------------------------
# 4. Split conformal
# --------------------------------------------------------------------------
def test_calibrate_matches_hand_computed_order_statistic():
    residuals = np.array([-3.0, 1.0, -0.5, 2.0, -1.5, 0.25, -2.5, 1.75, -0.1, 0.9])
    absr = np.sort(np.abs(residuals))
    n = absr.size

    for alpha in (0.1, 0.2, 0.5):
        k = math.ceil((n + 1) * (1.0 - alpha))
        expected = float(absr[k - 1]) if k <= n else float("inf")
        got = SplitConformal(alpha=alpha).calibrate(residuals)
        assert got == pytest.approx(expected, rel=1e-12), (alpha, k)

    # 具体数值：(n+1)(1-0.2)=8.8 ⇒ k=9 ⇒ 次大绝对残差 2.5
    assert SplitConformal(alpha=0.2).calibrate(residuals) == pytest.approx(2.5)
    # 样本过少时返回 +inf（无法达到目标覆盖）
    assert SplitConformal(alpha=0.1).calibrate(np.array([1.0, 2.0])) == float("inf")

    with pytest.raises(ValueError):
        SplitConformal(alpha=0.0)
    with pytest.raises(ValueError):
        SplitConformal(alpha=0.1).calibrate(np.array([]))


def test_conformal_interval_and_coverage():
    sc = SplitConformal(alpha=0.1)
    assert sc.interval(3.0, 2.0) == (1.0, 5.0)
    assert sc.coverage(
        y_true=np.array([1.0, 5.0, 5.1]), y_hat=np.array([3.0, 3.0, 3.0]), q=2.0
    ) == pytest.approx(2.0 / 3.0)
    assert sc.coverage(np.array([1.0]), np.array([1.0]), q=0.0) == 1.0
    with pytest.raises(ValueError):
        sc.interval(1.0, -1.0)


def test_evaluate_split_conformal_coverage_in_range():
    rng = np.random.default_rng(0)
    n = 2000
    y_hat = rng.normal(0.0, 1.0, size=n)
    y_true = y_hat + rng.normal(0.0, 1.0, size=n)

    for alpha in (0.05, 0.1, 0.2):
        rep = evaluate_split_conformal(y_hat, y_true, alpha=alpha, calib_frac=0.5, seed=0)
        assert isinstance(rep, ConformalReport)
        assert rep.alpha == alpha
        assert rep.n_calib == n // 2
        assert rep.q > 0.0
        assert 1.0 - alpha - 0.1 <= rep.empirical_coverage <= 1.0
        assert rep.mean_width == pytest.approx(2.0 * rep.q, rel=1e-12)

    # 同一 seed 完全可复现，换 seed 只是抽样波动
    a = evaluate_split_conformal(y_hat, y_true, alpha=0.1, seed=7)
    b = evaluate_split_conformal(y_hat, y_true, alpha=0.1, seed=7)
    assert a == b
    c = evaluate_split_conformal(y_hat, y_true, alpha=0.1, seed=8)
    assert 1.0 - 0.1 - 0.1 <= c.empirical_coverage <= 1.0

    with pytest.raises(ValueError):
        evaluate_split_conformal(y_hat[:1], y_true[:1])


# --------------------------------------------------------------------------
# 5. Hedged-capital 置信序列
# --------------------------------------------------------------------------
def test_hedged_capital_bernoulli_and_monotone_width():
    rng = np.random.default_rng(0)
    xs = rng.binomial(1, 0.3, size=1000).astype(float)

    cs = HedgedCapitalCS(grid_size=101, alpha=0.05, bound=(0.0, 1.0))
    checkpoints = (10, 50, 100, 300, 1000)
    widths = []
    for i, x in enumerate(xs, start=1):
        cs.update(x)
        if i in checkpoints:
            widths.append(cs.width())

    assert cs.n == 1000
    assert cs.lower() < 0.35, cs.lower()
    assert cs.upper() > 0.25, cs.upper()
    assert cs.lower() < cs.upper()

    # 宽度随 n 非增（允许 1e-9 数值抖动）
    for prev, cur in zip(widths, widths[1:]):
        assert cur <= prev + 1e-9, widths
    assert widths[-1] < widths[0]

    # 序贯检验：真值 0.3 不该被拒，明显偏离 0.5 应被拒
    assert not cs.is_reject(0.3)
    assert cs.is_reject(0.5)


def test_hedged_capital_behaves_on_shifted_mean():
    rng = np.random.default_rng(11)
    cs = HedgedCapitalCS(alpha=0.05, bound=(0.0, 1.0), sigma=0.1)
    for x in rng.normal(0.7, 0.1, size=400):
        cs.update(x)
    lo, hi = cs.interval_now()
    assert lo < 0.7 < hi
    assert hi - lo < 0.15  # 已知噪声尺度时区间明显收紧
    assert cs.is_reject(0.4)

    empty = HedgedCapitalCS()
    assert empty.n == 0
    assert empty.interval() == (0.0, 1.0)
    assert not empty.is_reject(0.5)

    with pytest.raises(ValueError):
        HedgedCapitalCS(alpha=1.5)
    with pytest.raises(ValueError):
        HedgedCapitalCS(bound=(1.0, 1.0))


# --------------------------------------------------------------------------
# 6. 回退判据
# --------------------------------------------------------------------------
def _signal(rng, mean, n):
    return rng.poisson(mean * 400.0, size=n) / 400.0


def test_rollback_no_false_trigger_without_drift():
    rng = np.random.default_rng(7)
    decision = rollback_rule(_signal(rng, 1.0, 180), threshold=0.85)
    assert isinstance(decision, RollbackDecision)
    assert decision.triggered is False
    assert decision.day is None
    assert decision.lower_bound > 0.85
    assert "未触发" in decision.reason

    # 常量序列同样不该触发
    assert rollback_rule([1.0] * 120, threshold=0.85).triggered is False


def test_rollback_detects_obvious_decline():
    rng = np.random.default_rng(7)
    series = np.concatenate([_signal(rng, 1.0, 60), _signal(rng, 0.55, 60)])
    decision = rollback_rule(series, threshold=0.85)
    assert decision.triggered is True
    assert decision.day is not None and 60 < decision.day <= len(series)
    assert decision.lower_bound < 0.85
    assert decision.threshold == 0.85
    assert "触发回退" in decision.reason

    # 扩张型置信序列（window=None）也能检出，只是更慢
    wide = rollback_rule(series, threshold=0.85, window=None)
    assert wide.triggered is True
    assert wide.day >= decision.day

    # 上行监测：风险上升时由上界触发
    up = rollback_rule(-series + 2.0, threshold=1.15)
    assert up.triggered is True

    with pytest.raises(ValueError):
        rollback_rule([], threshold=0.85)
    with pytest.raises(ValueError):
        rollback_rule([1.0] * 10, threshold=0.0)


def test_evaluate_rollback_counterfactual():
    res = evaluate_rollback(reps=30, seed=0)
    assert res["reps"] == 30 and res["n_days"] == 180
    assert 0.0 <= res["false_alarm_rate"] <= 1.0
    assert 0.0 <= res["detection_rate"] <= 1.0
    assert res["false_alarm_rate"] <= 0.1
    assert res["detection_rate"] >= 0.8
    assert 0.0 <= res["mean_delay_days"] <= 180
    assert res["n_detected"] == round(res["detection_rate"] * 30)
    # 固定 seed 可复现
    again = evaluate_rollback(reps=30, seed=0)
    assert res == again

    with pytest.raises(ValueError):
        evaluate_rollback(true_drift_at=200)
    with pytest.raises(ValueError):
        evaluate_rollback(drift_size=0.0)


# --------------------------------------------------------------------------
# 7. 可复现性
# --------------------------------------------------------------------------
def test_run_loop_is_deterministic_for_fixed_seed():
    cfg = MECHANISMS["L3"]
    a = run_loop(cfg, days=45, reps=6, seed=123)
    b = run_loop(cfg, days=45, reps=6, seed=123)

    assert a["mae_series"] == b["mae_series"]
    assert a["mae_at"] == b["mae_at"]
    assert a["excess_events"] == b["excess_events"]
    assert a["conformal"] == b["conformal"]
    np.testing.assert_array_equal(a["final_theta"], b["final_theta"])

    # 换 seed 会得到不同的抽样实现（但仍是同一量级）
    c = run_loop(cfg, days=45, reps=6, seed=124)
    assert c["mae_series"] != a["mae_series"]
    assert abs(c["mae_at"]["30"] - a["mae_at"]["30"]) < 0.05


def test_run_loop_validation_and_mae_at_keys():
    res = run_loop(MECHANISMS["L0"], days=45, reps=2, seed=0)
    assert set(res["mae_at"]) == {"30", "90", "180"}
    # days < 90 时后两个键退化为最后一天
    assert res["mae_at"]["90"] == res["mae_series"][-1]
    assert res["mae_at"]["180"] == res["mae_series"][-1]
    assert res["mae_at"]["30"] == res["mae_series"][29]
    assert res["mechanism"] == "L0"
    assert len(res["mae_series"]) == 45

    with pytest.raises(ValueError):
        run_loop(MECHANISMS["L0"], days=0)
    with pytest.raises(ValueError):
        run_loop(MECHANISMS["L0"], truth=(1.0, 2.0))
    with pytest.raises(TypeError):
        run_loop("L0")
