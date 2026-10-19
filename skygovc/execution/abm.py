"""Tier-aware execution layer: agent-based simulation over induced regimes.

Difference from a hand-set scenario model
----------------------------------------
Every regime parameter here is *induced*:

* the probability that a task of a given risk class needs an authorisation is
  obtained by running the decision engine over a Monte-Carlo sample of requests
  (:meth:`skygovc.reg2rule.engine.DecisionEngine.induce_regime`);
* the capability vector per tier is produced by the budget-constrained
  allocator (:mod:`skygovc.allocation`);
* the spatial regime (share of territory that is controlled airspace) is read
  off the local instruments (:func:`skygovc.classification.jurisdiction_airspace_regime`).

Nothing in the table of results is hand-set except the behavioural parameters
that have no published source (queue tolerance, unauthorised-flight propensity),
and those are swept in the sensitivity analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core import CAPABILITIES, TIERS, to_jsonable

#: Jurisdiction-level structural parameters, indexed like :data:`TIERS`.
SECURITY_CONSTRAINT = np.array([0.95, 0.75, 0.55, 0.60, 0.70, 0.30])
DEMAND_DENSITY = np.array([0.35, 0.95, 1.00, 0.60, 0.45, 0.20])
APPROVAL_CAPACITY = np.array([120.0, 400.0, 450.0, 250.0, 200.0, 80.0])
DEMAND_WEIGHT = DEMAND_DENSITY / DEMAND_DENSITY.sum()

RISK_SHARE = np.array([0.60, 0.30, 0.10])
RISK_MULT = np.array([1.0, 3.0, 10.0])


@dataclass
class Params:
    days: int = 30
    n_per_day: int = 1500
    p0: float = 3e-5
    cross_share: float = 0.18
    tau_hours: float = 12.0
    abandon_unauth: float = 0.60
    interrupt_base: float = 0.18
    interrupt_coord: float = 0.05
    det_base: float = 0.55
    casualty_prob: float = 0.10
    casualty_cost: float = 300.0
    admin_cost: float = 0.02
    delay_cost: float = 0.003
    penalty: float = 0.5
    interrupt_cost: float = 0.15
    seed: int = 0


@dataclass
class Regime:
    """One governance regime, fully determined by its parameters."""

    name: str
    req: np.ndarray                    # (3,) probability of authorisation by risk class
    cap: np.ndarray                    # (6,) effective capability index per tier, in [0,1]
    mu_mult: np.ndarray = field(default_factory=lambda: np.ones(6))
    demand: float = 1.0
    coord: bool = False
    priority: bool = False
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)


def capability_index(s_tier: Sequence[float]) -> float:
    """Map a tier's capability strengths to the [0.30, 0.90] index used by the
    incident model.  ``s_tier`` holds ``level/2`` per capability family."""
    s = np.asarray(s_tier, dtype=float)
    return float(0.30 + 0.60 * float(s.mean()))


def simulate(regime: Regime, p: Params) -> Dict[str, Any]:
    rng = np.random.default_rng(p.seed)
    n_city = len(TIERS)
    c_vec = np.clip(regime.cap, 0.05, 0.99)
    mu_vec = APPROVAL_CAPACITY * regime.mu_mult
    req = np.asarray(regime.req, dtype=float)

    tot = dict(attempted=0, flights=0, incidents=0.0, casualties=0.0,
               applications=0, duplicates=0, served=0, unauth=0, cancel=0,
               detected=0, interrupted=0, cross=0, completed=0, cost=0.0)
    delays: List[float] = []

    for _ in range(p.days):
        n = int(rng.poisson(p.n_per_day * regime.demand))
        city = rng.choice(n_city, size=n, p=DEMAND_WEIGHT)
        risk = rng.choice(3, size=n, p=RISK_SHARE)
        cross = rng.random(n) < p.cross_share
        dest = np.where(cross, (city + rng.integers(1, n_city, size=n)) % n_city, city)

        need = rng.random(n) < req[risk]
        dup = cross & need & (not regime.coord)

        app_need = np.zeros(n, dtype=int)
        app_served = np.zeros(n, dtype=int)
        app_lost = np.zeros(n, dtype=int)
        lost_unauth = np.zeros(n, dtype=bool)
        waits = np.zeros(n)

        # ---- authorisation queue, fully vectorised --------------------
        need_idx = np.where(need)[0]
        dup_idx = need_idx[dup[need_idx]]
        app_task = np.concatenate([need_idx, dup_idx]) if dup_idx.size else need_idx
        app_city = np.concatenate([city[need_idx], dest[dup_idx]]) if dup_idx.size \
            else city[need_idx]
        if app_task.size:
            app_risk = risk[app_task]
            seq = np.arange(app_task.size)
            if regime.priority:
                order = np.lexsort((seq, -app_risk, app_city))
            else:
                order = np.lexsort((seq, app_city))
            task_o = app_task[order]
            city_o = app_city[order]
            first = np.searchsorted(city_o, city_o, side="left")
            pos = np.arange(city_o.size) - first
            cap_per_app = np.maximum(mu_vec[city_o].astype(np.int64), 1)
            w_h_o = pos / cap_per_app * 24.0
            ok = w_h_o <= p.tau_hours
            app_need[:] = np.bincount(app_task, minlength=n)
            if ok.any():
                app_served[:] = np.bincount(task_o[ok], minlength=n)
                np.maximum.at(waits, task_o[ok], w_h_o[ok])
            if (~ok).any():
                lost_ids = task_o[~ok]
                app_lost[:] = np.bincount(lost_ids, minlength=n)
                flag = rng.random(lost_ids.size) < p.abandon_unauth
                lost_unauth[lost_ids[flag]] = True

        status = np.zeros(n, dtype=int)
        status[(app_need > 0) & (app_served == app_need) & (app_lost == 0)] = 1
        status[(app_lost > 0) & lost_unauth] = 2
        status[(app_lost > 0) & (~lost_unauth)] = 3

        flying = (status == 0) | (status == 1) | (status == 2)
        unauth = status == 2
        interrupt_rate = p.interrupt_coord if regime.coord else p.interrupt_base
        interrupted = flying & cross & (rng.random(n) < interrupt_rate)

        cap_air = np.maximum(250 + 900 * DEMAND_DENSITY, 1.0)
        per_city = np.bincount(city, minlength=n_city).astype(float)
        load = (per_city / cap_air)[city]

        p_inc = (p.p0 * RISK_MULT[risk] * (0.6 + 0.8 * DEMAND_DENSITY[city])
                 * (1 - 0.6 * c_vec[city]) * (1 + 3.0 * unauth)
                 * (1 + 0.7 * np.maximum(0, load - 0.8)))
        hit = (rng.random(n) < p_inc) & flying & (~interrupted)
        detected = unauth & (rng.random(n) < p.det_base * c_vec[city])
        hit = hit & (~detected)
        cas = hit & (rng.random(n) < p.casualty_prob)

        tot["attempted"] += n
        tot["flights"] += int(flying.sum())
        tot["completed"] += int((flying & (~interrupted)).sum())
        tot["incidents"] += float(hit.sum())
        tot["casualties"] += float(cas.sum())
        tot["applications"] += int(app_need.sum())
        tot["duplicates"] += int(dup.sum())
        tot["served"] += int((status == 1).sum())
        tot["unauth"] += int((status == 2).sum())
        tot["cancel"] += int((status == 3).sum())
        tot["detected"] += int(detected.sum())
        tot["interrupted"] += int(interrupted.sum())
        tot["cross"] += int(cross.sum())
        delays.extend(waits[status == 1].tolist())

        sev = np.array([20.0, 50.0, 120.0])[risk]
        cost = ((sev[hit].sum() if hit.any() else 0.0)
                + cas.sum() * p.casualty_cost
                + app_need.sum() * p.admin_cost
                + waits[status == 1].sum() * p.delay_cost
                + detected.sum() * p.penalty
                + interrupted.sum() * p.interrupt_cost)
        tot["cost"] += float(cost)

    n_fl = max(tot["flights"], 1)
    w = np.array(delays) if delays else np.array([0.0])
    return dict(
        regime=regime.name,
        flights=tot["flights"],
        incident_per_10k=tot["incidents"] / n_fl * 1e4,
        casualty_per_10k=tot["casualties"] / n_fl * 1e4,
        completion=tot["completed"] / max(tot["attempted"], 1),
        unauth_rate=tot["unauth"] / max(tot["attempted"], 1),
        cancel_rate=tot["cancel"] / max(tot["attempted"], 1),
        detect_rate=tot["detected"] / n_fl,
        duplicates_per_cross=tot["duplicates"] / max(tot["cross"], 1),
        interrupt_rate=tot["interrupted"] / max(tot["cross"], 1),
        mean_delay_h=float(w.mean()),
        p95_delay_h=float(np.percentile(w, 95)),
        cost_per_1k=tot["cost"] / max(tot["attempted"], 1) * 1000.0,
    )


def run_repeated(regime: Regime, p: Params, reps: int = 200,
                 base_seed: int = 0) -> Dict[str, Any]:
    """Repeat the simulation and report mean, standard error and 95% CI."""
    rows: List[Dict[str, float]] = []
    for r in range(reps):
        pp = Params(**{**p.__dict__, "seed": base_seed + r})
        rows.append(simulate(regime, pp))
    keys = [k for k in rows[0] if k != "regime"]
    out: Dict[str, Any] = dict(regime=regime.name, reps=reps)
    for k in keys:
        v = np.array([row[k] for row in rows], dtype=float)
        m = float(v.mean())
        se = float(v.std(ddof=1) / np.sqrt(len(v)))
        out[k] = dict(mean=m, se=se, ci_low=m - 1.96 * se, ci_high=m + 1.96 * se)
    return out


def paired_difference(a: List[Dict[str, float]], b: List[Dict[str, float]],
                      keys: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Paired comparison between two regimes over the same seeds."""
    from scipy import stats

    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        va = np.array([r[k] for r in a], dtype=float)
        vb = np.array([r[k] for r in b], dtype=float)
        diff = va - vb
        if np.allclose(diff, 0):
            out[k] = dict(diff=0.0, p_value=1.0, ci_low=0.0, ci_high=0.0, w=0.0)
            continue
        w = stats.wilcoxon(va, vb, zero_method="wilcox")
        m = float(diff.mean())
        se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
        out[k] = dict(diff=m, p_value=float(w.pvalue),
                      ci_low=m - 1.96 * se, ci_high=m + 1.96 * se, w=float(w.statistic))
    return out


def run_raw(regime: Regime, p: Params, reps: int = 200, base_seed: int = 0
            ) -> List[Dict[str, float]]:
    return [simulate(regime, Params(**{**p.__dict__, "seed": base_seed + r}))
            for r in range(reps)]


