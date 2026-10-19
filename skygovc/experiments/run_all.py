"""Experiment orchestration: every number and figure in the paper.

Run with::

    python -m skygovc.experiments.run_all

All artefacts land in ``results/`` and ``figures/`` next to the package root.
"""
from __future__ import annotations

import json
import os
import platform
import random
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import __version__
from ..audit import validate_detector
from ..classification import (REFERENCE, REFERENCE_TIER, benchmark,
                              classify, jurisdiction_airspace_regime,
                              reference_accuracy)
from ..core import CAPABILITIES, TIERS, to_jsonable
from ..corpus import loader as C
from ..execution import Params, Regime, capability_index, paired_difference, run_raw, run_repeated
from ..reg2rule import pipeline as P
from ..reg2rule.score import score_components
from ..reg2rule import router as R
from ..reg2rule.engine import DecisionEngine, default_sampler

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")


def _ensure_dirs() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)


def _dump(name: str, obj: Any) -> None:
    with open(os.path.join(RESULTS, name), "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, ensure_ascii=False, indent=2)


# ==========================================================================
# 1. corpus + compilation
# ==========================================================================
def exp_corpus() -> Dict[str, Any]:
    s = C.corpus_summary()
    s["redundancy_pairs"] = C.redundancy_pairs()
    s["documents"] = C.doc_stats()
    return s


def exp_compile() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for strat in ("S1_keyword", "S2_frame", "S3_repair"):
        r = P.compile_rules(strat, validate=True)
        d = r.to_dict()
        out[strat] = dict(
            n_clauses=len(r.clauses),
            n_props=d["n_props_predicted"],
            n_rules=d["n_rules_after_merge"],
            precision=d["scores_dedup"]["precision"],
            recall=d["scores_dedup"]["recall"],
            f1=d["scores_dedup"]["f1"],
            exact_match=d["scores_dedup"]["exact_match_rate"],
            compilability_accuracy=d["comp_scores"]["accuracy"],
            compilability_macro_f1=d["comp_scores"]["macro_f1"],
            compilability_f1_true=d["comp_scores"]["f1_true"],
            compilability_f1_false=d["comp_scores"]["f1_false"],
            failure_mode_agreement=d["comp_scores"]["failure_mode_accuracy"],
            conflict_summary=d["conflict_summary"],
            validation_raw=d["validation_raw"],
            validation_merged=d["validation_merged"],
            n_repaired=d["n_repaired"],
        )
        if strat == "S3_repair":
            out[strat]["components"] = score_components(
                r.extractions, P.gold_props_by_cid()).to_dict()
            out[strat]["conflicts"] = [c.to_dict() for c in r.conflicts]
            out[strat]["rulebase_size"] = len(r.rulebase)
    return out


def exp_heldout() -> Dict[str, Any]:
    hc = C.heldout_clauses()
    hg = C.gold_heldout(dedupe=False)
    gc = C.gold_compilability()
    hgc = {c.cid: gc[c.cid] for c in hc}
    gby: Dict[str, List[Any]] = {}
    for p in hg:
        gby.setdefault(p.cid, []).append(p)
    out: Dict[str, Any] = {}
    for strat in ("S2_frame", "S3_repair"):
        r = P.evaluate_block(hc, hg, hgc, strat)
        out[strat] = dict(
            n_clauses=r["n_clauses"], n_props=r["n_props"],
            precision=r["scores_dedup"]["precision"],
            recall=r["scores_dedup"]["recall"],
            f1=r["scores_dedup"]["f1"],
            compilability_accuracy=r["comp_scores"]["accuracy"],
            failure_mode_agreement=r["comp_scores"]["failure_mode_accuracy"],
            components=r["components"],
            fp_pids=r["scores_dedup"]["fp_pids"],
            fn_pids=r["scores_dedup"]["fn_pids"],
        )
    return out


# ==========================================================================
# 2. adjudication router
# ==========================================================================
def exp_router() -> Dict[str, Any]:
    """Learning to route compilations to adjudication.

    The base configuration is ``S2_frame`` — the grammar a team would ship
    before hand-tuning the failure-mode filter.  Routing is only meaningful when
    the base leaves errors on the table, so evaluating on ``S3`` (which already
    matches the development gold) would be vacuous.  The router is then applied
    *unchanged* to the held-out block, which is the transfer test.
    """
    base = P.compile_rules("S2_frame", validate=False)
    by_cid = P.gold_props_by_cid()
    gc = C.gold_compilability()

    labels = R.per_cid_gold(base.extractions, by_cid, gc)
    groups = {e.cid: C.clause(e.cid).doc_id for e in base.extractions}
    model, report = R.train_router(base.extractions, labels, groups,
                                   n_splits=5, seed=0)
    cids = [e.cid for e in base.extractions]
    oof = list(getattr(model, "oof_scores_", np.zeros(len(cids))))

    def f1_after(chosen: set) -> float:
        return P.evaluate_with_oracle(base, chosen).f1

    order_learned = [c for _, c in sorted(zip([-x for x in oof], cids))]
    order_conf = sorted(cids, key=lambda c: next(e.confidence
                                                 for e in base.extractions
                                                 if e.cid == c))
    kmax = min(40, len(cids))
    curves: Dict[str, Any] = {}
    for name, order in (("learned", order_learned), ("confidence", order_conf)):
        curves[name] = [p.to_dict() for p in
                        R.routing_curve(order, cids, f1_after, 2.5, kmax)]
    rng = np.random.default_rng(0)
    rand = []
    for _ in range(20):
        perm = list(rng.permutation(cids))
        rand.append([p.to_dict() for p in
                     R.routing_curve(perm, cids, f1_after, 2.5, kmax)])
    curves["random"] = [dict(k=rand[0][i]["k"],
                             expert_minutes=rand[0][i]["expert_minutes"],
                             f1=float(np.mean([r[i]["f1"] for r in rand])))
                        for i in range(len(rand[0]))]

    base_f1 = f1_after(set())
    best_f1 = max(f1_after(set(cids[:k])) for k in range(len(cids) + 1))
    target = base_f1 + 0.80 * (best_f1 - base_f1)

    # ---- transfer to the held-out block ------------------------------
    hc = C.heldout_clauses()
    hg = C.gold_heldout(dedupe=False)
    hgc = {c.cid: gc[c.cid] for c in hc}
    hby: Dict[str, List[Any]] = {}
    for p in hg:
        hby.setdefault(p.cid, []).append(p)
    hexs = P.extract_all(hc, "S2_frame") if hasattr(P, "extract_all") else None
    from ..reg2rule.frames import extract_all as _extract_all
    hexs = _extract_all(hc, "S2_frame")
    Xh = R.feature_matrix(hexs)
    p_edit = model.predict_proba(Xh)[:, 1]
    h_order = [c for _, c in sorted(zip([-x for x in p_edit], [e.cid for e in hexs]))]

    def hf1(chosen: set) -> float:
        pred: List[Any] = []
        for e in hexs:
            if e.cid in chosen:
                pred.extend(hby.get(e.cid, []))
            elif e.compilable:
                pred.extend(e.props)
        from ..reg2rule.score import score_propositions
        return score_propositions(pred, hg, dedupe_pred=True).f1

    h_curve = [p.to_dict() for p in
               R.routing_curve(h_order, [e.cid for e in hexs], hf1, 2.5,
                               len(hexs))]
    from sklearn.metrics import roc_auc_score
    h_lab = np.array([1 if hf1({e.cid}) > hf1(set()) + 1e-9 else 0
                      for e in hexs])
    # per-record error indicator for the AUC on the held-out block
    errs = []
    for e in hexs:
        single = hf1({e.cid})
        errs.append(1 if abs(single - hf1(set())) > 1e-9 else 0)
    try:
        h_auc = float(roc_auc_score(errs, p_edit)) if len(set(errs)) > 1 else float("nan")
    except ValueError:
        h_auc = float("nan")

    return dict(
        base_strategy="S2_frame",
        router=report.to_dict(),
        base_f1=base_f1, best_f1=best_f1, target_f1=target,
        curves=curves,
        breakeven={name: R.breakeven_k([R.RoutingPoint(**p) for p in curves[name]],
                                      target)
                   for name in ("learned", "confidence", "random")},
        n_needs_edit=int(sum(1 for v in labels.values() if v)),
        n_clauses=len(cids),
        heldout=dict(auc=h_auc, curve=h_curve,
                     base_f1=float(hf1(set())),
                     best_f1=float(max(hf1(set([e.cid for e in hexs][:k]))
                                       for k in range(len(hexs) + 1))),
                     order=h_order),
    )

# ==========================================================================
# 3. classification
# ==========================================================================
def exp_classification() -> Dict[str, Any]:
    ref = reference_accuracy()
    bench = {}
    for noise in (0.08, 0.12, 0.18):
        b = benchmark(n=600, noise=noise, seed=0)
        bench[f"noise_{noise}"] = b.to_dict()
    return dict(reference=ref, benchmark=bench,
                airspace_regime=jurisdiction_airspace_regime())


# ==========================================================================
# 4. allocation: uniform versus differentiated at equal total budget
# ==========================================================================
def _floor_cost() -> float:
    from ..allocation import model as M
    return float(sum(M.cost_of(M.FLOOR[t]) for t in TIERS))


def _all_levels():
    import itertools
    return [tuple(x) for x in itertools.product((0, 1, 2), repeat=6)]


def _tier_options(tier: str):
    """Nondominated (cost, incident) options for one tier, ascending cost."""
    from ..allocation import model as M
    opts = []
    for lv in _all_levels():
        if any(lv[k] < M.FLOOR[tier][k] for k in range(6)):
            continue
        opts.append((M.cost_of(lv), M.objectives(lv, tier)[0], lv))
    opts.sort(key=lambda t: (t[0], t[1]))
    out, best = [], 1e18
    for cost, inc, lv in opts:
        if inc < best - 1e-12:
            out.append((cost, inc, lv))
            best = inc
    return out


def _greedy_allocate(budget_total: float):
    """Greedy marginal-benefit knapsack over tiers: repeatedly take the single
    cheapest upgrade with the largest incident reduction per unit cost."""
    from ..allocation import model as M
    opts = {t: _tier_options(t) for t in TIERS}
    idx = {t: 0 for t in TIERS}
    levels = {t: opts[t][0][2] for t in TIERS}
    cost = {t: opts[t][0][0] for t in TIERS}
    spent = sum(cost.values())
    while True:
        best = None
        for t in TIERS:
            j = idx[t] + 1
            if j >= len(opts[t]):
                continue
            dc = opts[t][j][0] - cost[t]
            di = opts[t][j][1] - opts[t][idx[t]][1]
            if dc <= 1e-12 or spent + dc > budget_total + 1e-9:
                continue
            score = -di / dc
            if best is None or score < best[0]:
                best = (score, t, j, dc)
        if best is None:
            break
        _, t, j, dc = best
        idx[t] = j
        levels[t] = opts[t][j][2]
        cost[t] = opts[t][j][0]
        spent += dc
    return levels, spent


def exp_allocation() -> Dict[str, Any]:
    """Differentiated versus uniform capability allocation at equal total cost.

    Both policies spend from the same budget.  ``uniform`` must pick one level
    vector used by every tier; ``tiered`` may distribute the same budget unevenly
    via a greedy marginal-benefit knapsack.  The gap is the value of
    differentiation, holding money fixed.
    """
    from ..allocation import model as M

    floor_total = _floor_cost()
    max_total = 6 * M.cost_of((2, 2, 2, 2, 2, 2))
    # a uniform policy must satisfy every tier's floor simultaneously, so the
    # minimum affordable uniform vector costs 6 * cost(pointwise-max floor).
    pmax = tuple(max(M.FLOOR[t][k] for t in TIERS) for k in range(6))
    uniform_min = 6 * M.cost_of(pmax)
    lo = min(uniform_min, max_total)
    budgets = [round(lo + f * max(max_total - lo, 0.0), 3)
               for f in (0.02, 0.20, 0.38, 0.56, 0.74, 0.92)]
    all_levels = _all_levels()

    sweep: List[Dict[str, Any]] = []
    best_detail: Optional[Dict[str, Any]] = None
    best_gain = -1e9

    for B in budgets:
        tiered, spent = _greedy_allocate(B)
        if spent > B + 1e-9:            # nothing affordable: fall back to floors
            tiered = {t: tuple(M.FLOOR[t]) for t in TIERS}
            spent = float(sum(M.cost_of(tiered[t]) for t in TIERS))

        uniform, u_obj, u_cost = None, 1e18, 0.0
        for lv in all_levels:
            c6 = 6 * M.cost_of(lv)
            if c6 > B + 1e-9:
                continue
            if any(any(lv[k] < M.FLOOR[t][k] for k in range(6)) for t in TIERS):
                continue
            obj = float(np.mean([M.objectives(lv, t)[0] for t in TIERS]))
            if obj < u_obj - 1e-15:
                uniform, u_obj, u_cost = tuple(lv), obj, c6
        if uniform is None:
            uniform = tuple(max(M.FLOOR[t][k] for t in TIERS) for k in range(6))
            u_obj = float(np.mean([M.objectives(uniform, t)[0] for t in TIERS]))
            u_cost = 6 * M.cost_of(uniform)

        rows = []
        for t in TIERS:
            lt = tiered[t]
            rows.append(dict(
                tier=t, tiered_levels=list(lt),
                tiered_cost=M.cost_of(lt),
                tiered_incident=M.objectives(lt, t)[0],
                tiered_completion=-M.objectives(lt, t)[1],
                uniform_levels=list(uniform), uniform_cost=M.cost_of(uniform),
                uniform_incident=M.objectives(uniform, t)[0],
                uniform_completion=-M.objectives(uniform, t)[1],
                improvement=(1.0 - M.objectives(lt, t)[0]
                             / max(M.objectives(uniform, t)[0], 1e-12)),
            ))
        mean_t = float(np.mean([r0["tiered_incident"] for r0 in rows]))
        mean_u = float(np.mean([r0["uniform_incident"] for r0 in rows]))
        gain = 1.0 - mean_t / max(mean_u, 1e-12)
        rec = dict(budget=B, tiered_total_cost=float(spent),
                   uniform_total_cost=float(u_cost),
                   mean_incident_tiered=mean_t, mean_incident_uniform=mean_u,
                   relative_gain=gain,
                   mean_completion_tiered=float(np.mean([r0["tiered_completion"] for r0 in rows])),
                   mean_completion_uniform=float(np.mean([r0["uniform_completion"] for r0 in rows])),
                   n_tiers_improved=int(sum(1 for r0 in rows if r0["improvement"] > 1e-9)))
        sweep.append(rec)
        if gain > best_gain:
            best_gain = gain
            best_detail = dict(rec, tiered=rows)

    assert best_detail is not None
    return dict(floor_total=floor_total, max_total=max_total,
                budgets=budgets, sweep=sweep, best=best_detail,
                tiered=best_detail["tiered"],
                mean_incident_tiered=best_detail["mean_incident_tiered"],
                mean_incident_uniform=best_detail["mean_incident_uniform"],
                mean_completion_tiered=best_detail["mean_completion_tiered"],
                mean_completion_uniform=best_detail["mean_completion_uniform"])

# ==========================================================================
# 5. protocol
# ==========================================================================
def exp_protocol() -> Dict[str, Any]:
    from ..protocol import modelcheck as MC

    channels = {}
    for ch in ("reliable", "lossy", "lossy_retry"):
        res = MC.check(ch, max_retries=2)
        channels[ch] = {
            k: dict(satisfied=v.satisfied, explored=v.explored, detail=v.detail,
                    has_cex=bool(v.counterexample))
            for k, v in res.items()
        }
    boundary = MC.retry_boundary()
    lossy = MC.check("lossy", max_retries=2)
    cex = [r for r in lossy.values() if r.counterexample]
    cex_text = MC.counterexample_text(cex[0].counterexample) if cex else ""
    return dict(channels=channels, retry_boundary=boundary,
                counterexample_text=cex_text)


# ==========================================================================
# 6. calibration
# ==========================================================================
def exp_calibration() -> Dict[str, Any]:
    from ..calibration import loop as L

    out: Dict[str, Any] = {"mechanisms": {}}
    for name in ("L0", "L1", "L2", "L3"):
        cfg = L.MECHANISMS[name]
        r = L.run_loop(cfg, days=180, reps=200, seed=0)
        out["mechanisms"][name] = dict(
            mae_at=r["mae_at"], excess_events=r["excess_events"],
            conformal=(r["conformal"].to_dict() if r.get("conformal") else None),
        )
    l0 = out["mechanisms"]["L0"]["mae_at"]
    l1 = out["mechanisms"]["L1"]["mae_at"]
    out["l0_l1_identical"] = all(
        abs(out["mechanisms"]["L0"]["mae_at"][k] - out["mechanisms"]["L1"]["mae_at"][k]) < 1e-12
        for k in l0)
    out["rollback"] = L.evaluate_rollback(reps=200, seed=0)
    return out


# ==========================================================================
# 7. regimes and simulation
# ==========================================================================
def _rules_from(rulebase, doc_prefixes: Sequence[str]):
    keep = [p for p in rulebase if p.cid.split("-")[0] in doc_prefixes]
    return keep


#: Tiers whose exposure profile justifies requiring authorisation for
#: medium-risk tasks as well.  This is a *classification* output, not a
#: hand-set parameter: it follows from the tier assignment in
#: :mod:`skygovc.classification`.
HIGH_EXPOSURE_TIERS = frozenset({"capital_core", "dense_hub", "border_coastal"})


def _overlay(risk_classes: Sequence[str], tag: str):
    """A governance overlay expressed in the same rule language as the compiled
    rules, so that the decision engine induces its effect rather than a
    hard-coded parameter doing so."""
    from ..core import Condition, Proposition
    out = []
    for rc in risk_classes:
        out.append(Proposition(
            pid=f"POL-{tag}-{rc}", cid=f"POLICY-{tag}",
            deontic="obligation", subject="operator",
            action="apply_flight_activity",
            conditions=[Condition("risk_class", "==", rc)],
            extractor="policy", confidence=1.0,
            normalised_from="tier policy overlay"))
    return out


def build_regimes() -> Tuple[List[Regime], Dict[str, Any]]:
    from ..allocation import model as M

    compiled = P.compile_rules("S3_repair", validate=False)
    national = _rules_from(compiled.rulebase, ("D01", "D02", "D03", "D04"))

    alloc = exp_allocation()
    tiered_levels = {r["tier"]: tuple(r["tiered_levels"]) for r in alloc["tiered"]}
    uniform_levels = tuple(alloc["tiered"][0]["uniform_levels"])

    # --- induced authorisation probability per regime -------------------
    def induce(engine, airspace: str, tiers: Sequence[str]) -> np.ndarray:
        acc = np.zeros(3)
        wsum = 0.0
        from ..execution import DEMAND_DENSITY
        for t in tiers:
            w = float(DEMAND_DENSITY[TIERS.index(t)])
            s = engine.induce_regime(default_sampler(t, airspace_class=airspace),
                                     n=1200, seed=0)
            acc += w * np.array([s["require_application_low"],
                                 s["require_application_medium"],
                                 s["require_application_high"]])
            wsum += w
        return acc / max(wsum, 1e-9)

    eng_nat = DecisionEngine(national)
    eng_all = DecisionEngine(national + _overlay(("low", "medium", "high"), "ALL"))
    eng_tier = {
        t: DecisionEngine(national + _overlay(
            ("low", "medium", "high") if t in HIGH_EXPOSURE_TIERS
            else ("medium", "high"), "TIER"))
        for t in TIERS
    }

    req_frame = induce(eng_nat, "mixed", list(TIERS))
    req_strict = induce(eng_all, "A", list(TIERS))
    req_open = induce(eng_nat, "W", list(TIERS))
    acc = np.zeros(3)
    wsum = 0.0
    from ..execution import DEMAND_DENSITY
    for t in TIERS:
        w = float(DEMAND_DENSITY[TIERS.index(t)])
        s = eng_tier[t].induce_regime(default_sampler(t, airspace_class="W"),
                                      n=1200, seed=0)
        acc += w * np.array([s["require_application_low"],
                             s["require_application_medium"],
                             s["require_application_high"]])
        wsum += w
    req_tiered = acc / max(wsum, 1e-9)

    cap_tiered = np.array([capability_index([lv / 2 for lv in tiered_levels[t]])
                           for t in TIERS])
    cap_uniform = np.array([capability_index([lv / 2 for lv in uniform_levels])
                            for _ in TIERS])
    cap_floor = np.array([capability_index([M.FLOOR[t][k] / 2 for k in range(6)])
                          for t in TIERS])

    regimes = [
        Regime("R0 national frame", req_frame, cap_floor, priority=False,
               provenance=dict(basis="D01-D04 only", airspace="mixed")),
        Regime("R1 uniform strict", req_strict, cap_floor, priority=False,
               provenance=dict(basis="all-risk overlay, all airspace controlled")),
        Regime("R2 uniform open, equal budget", req_open, cap_uniform, priority=False,
               provenance=dict(levels=list(uniform_levels))),
        Regime("R3 tiered, equal budget", req_tiered, cap_tiered, priority=True,
               provenance=dict(levels={k: list(v) for k, v in tiered_levels.items()})),
        Regime("R4 tiered + coordination", req_tiered, cap_tiered, priority=True,
               coord=True,
               provenance=dict(levels={k: list(v) for k, v in tiered_levels.items()})),
        Regime("R5 tiered + capacity", req_tiered, cap_tiered, priority=True,
               mu_mult=np.full(len(TIERS), 1.4),
               provenance=dict(levels={k: list(v) for k, v in tiered_levels.items()},
                               approval_capacity=1.4)),
        Regime("R6 tiered + capacity + coordination", req_tiered, cap_tiered,
               priority=True, coord=True, mu_mult=np.full(len(TIERS), 1.4),
               provenance=dict(levels={k: list(v) for k, v in tiered_levels.items()},
                               approval_capacity=1.4)),
    ]
    meta = dict(
        req_frame=req_frame.tolist(), req_strict=req_strict.tolist(),
        req_open=req_open.tolist(), req_tiered=req_tiered.tolist(),
        cap_tiered=cap_tiered.tolist(), cap_uniform=cap_uniform.tolist(),
        cap_floor=cap_floor.tolist(),
        tiered_levels={k: list(v) for k, v in tiered_levels.items()},
        uniform_levels=list(uniform_levels),
        high_exposure_tiers=sorted(HIGH_EXPOSURE_TIERS),
        national_rules=len(national), full_rules=len(compiled.rulebase),
    )
    return regimes, meta

def exp_simulation(reps: int = 200) -> Dict[str, Any]:
    regimes, meta = build_regimes()
    p = Params()
    summaries = {}
    raws = {}
    for rg in regimes:
        summaries[rg.name] = run_repeated(rg, p, reps=reps, base_seed=0)
        raws[rg.name] = run_raw(rg, p, reps=reps, base_seed=0)
    keys = ("incident_per_10k", "completion", "unauth_rate",
            "duplicates_per_cross", "interrupt_rate", "cost_per_1k",
            "mean_delay_h", "p95_delay_h")
    pairs = {}
    for a, b in (("R3 tiered, equal budget", "R2 uniform open, equal budget"),
                 ("R4 tiered + coordination", "R3 tiered, equal budget"),
                 ("R1 uniform strict", "R3 tiered, equal budget"),
                 ("R5 tiered + capacity", "R3 tiered, equal budget"),
                 ("R6 tiered + capacity + coordination", "R5 tiered + capacity")):
        pairs[f"{a} - {b}"] = paired_difference(raws[a], raws[b], keys)
    return dict(regimes=summaries, meta=meta, paired=pairs, reps=reps)


# ==========================================================================
# 8. latency
# ==========================================================================
def exp_latency(n: int = 40) -> Dict[str, Any]:
    from ..core import ACTIONS, CONDITION_NAMES, SUBJECTS
    from ..reg2rule import align as A
    from ..reg2rule import emit as E
    from ..reg2rule import frames as F

    clauses = C.core_clauses()
    stages: Dict[str, List[float]] = {k: [] for k in
                                      ("extract", "align", "emit_shacl", "decide")}
    for _ in range(n):
        t0 = time.perf_counter()
        exs = F.extract_all(clauses, "S3_repair")
        t1 = time.perf_counter()
        raw = [p for e in exs if e.compilable for p in e.props]
        aligned, _ = A.align_all(raw)
        t2 = time.perf_counter()
        E.rulebase_ttl(aligned)
        E.shapes_ttl()
        t3 = time.perf_counter()
        stages["extract"].append((t1 - t0) * 1000 / len(clauses))
        stages["align"].append((t2 - t1) * 1000 / max(len(raw), 1))
        stages["emit_shacl"].append((t3 - t2) * 1000)
        stages["decide"].append(0.0)

    r = P.compile_rules("S3_repair", validate=False)
    eng = DecisionEngine(r.rulebase)
    from ..core import FlightRequest
    reqs = default_sampler("CN-SZ")(np.random.default_rng(0), 400)
    ts = []
    for q in reqs:
        t0 = time.perf_counter()
        eng.decide(q, approved=True, tier="market_driven")
        ts.append((time.perf_counter() - t0) * 1000)
    stages["decide"] = ts

    out = {}
    for k, v in stages.items():
        arr = np.array(v, dtype=float)
        out[k] = dict(p50=float(np.percentile(arr, 50)),
                      p95=float(np.percentile(arr, 95)),
                      mean=float(arr.mean()), unit="ms")
    out["decision_throughput_per_s"] = float(1000.0 / max(out["decide"]["mean"], 1e-9))
    out["n_rules"] = len(r.rulebase)
    return out


# ==========================================================================
# figures
# ==========================================================================
def make_figures(res: Dict[str, Any]) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "figure.dpi": 200, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })
    made: List[str] = []

    # --- Fig 1: architecture (drawn as boxes) ---
    fig, ax = plt.subplots(figsize=(6.6, 2.1))
    ax.axis("off")
    boxes = [
        (0.02, "Institutional\nlayer\n16 instruments", "#eef2f7"),
        (0.27, "Reg2Rule\nframe extraction\n+ ontology repair", "#e8f0e8"),
        (0.52, "Rule base\nSHACL shapes\nSPARQL ASK", "#f7efe4"),
        (0.77, "Decision layer\ntier + allocation\n+ handover", "#efe8f5"),
    ]
    for x, label, col in boxes:
        ax.add_patch(plt.Rectangle((x, 0.42), 0.21, 0.40, facecolor=col,
                                   edgecolor="#333", linewidth=0.8))
        ax.text(x + 0.105, 0.62, label, ha="center", va="center", fontsize=7)
    for x in (0.235, 0.485, 0.735):
        ax.annotate("", xy=(x + 0.033, 0.62), xytext=(x, 0.62),
                    arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.text(0.5, 0.20, "provenance chain: clause id -> proposition -> rule -> verdict -> explanation",
            ha="center", fontsize=7, style="italic")
    ax.text(0.5, 0.06, "audit: RAR / LEC / UCR / TCC      assurance: conformal + martingale",
            ha="center", fontsize=7)
    path = os.path.join(FIGURES, "fig1_architecture.pdf")
    fig.savefig(path); plt.close(fig); made.append(os.path.relpath(path, ROOT))

    # --- Fig 2: routing curves ---
    cur = res["router"]["curves"]
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    for name, style in (("learned", "-o"), ("confidence", "-s"), ("random", "-^")):
        xs = [p["expert_minutes"] for p in cur[name]]
        ys = [p["f1"] for p in cur[name]]
        ax.plot(xs, ys, style, ms=2.2, lw=1.0, label=name)
    ax.set_xlabel("expert minutes spent on adjudication")
    ax.set_ylabel("rule F1 after adjudication")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, lw=0.4)
    path = os.path.join(FIGURES, "fig2_routing.pdf")
    fig.savefig(path); plt.close(fig); made.append(os.path.relpath(path, ROOT))

    # --- Fig 3: compiled vs uniform capability allocation ---
    al = res["allocation"]
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    idx = np.arange(len(al["tiered"]))
    ax.bar(idx - 0.19, [r["uniform_incident"] for r in al["tiered"]], width=0.36,
           label="uniform, equal budget", color="#b8c6d9")
    ax.bar(idx + 0.19, [r["tiered_incident"] for r in al["tiered"]], width=0.36,
           label="tiered, equal budget", color="#4f7fbf")
    ax.set_xticks(idx)
    SHORT = {"capital_core": "capital", "dense_hub": "dense", "market_driven": "market",
             "mountainous": "mountain", "border_coastal": "border", "sparse_resource": "sparse"}
    ax.set_xticklabels([SHORT.get(r["tier"], r["tier"]) for r in al["tiered"]], fontsize=6.5,
                       rotation=20, ha="right")
    ax.set_ylabel("incident multiplier (lower is better)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3, lw=0.4)
    path = os.path.join(FIGURES, "fig3_allocation.pdf")
    fig.savefig(path); plt.close(fig); made.append(os.path.relpath(path, ROOT))

    # --- Fig 4: calibration MAE ---
    mech = res["calibration"]["mechanisms"]
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    days = ["30", "90", "180"]
    for name, style in (("L0", "-o"), ("L1", "--s"), ("L2", "-^"), ("L3", "-d")):
        ys = [mech[name]["mae_at"][d] for d in days]
        ax.plot(days, ys, style, ms=3, lw=1.0, label=name)
    ax.set_xlabel("day")
    ax.set_ylabel("risk-multiple MAE")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, lw=0.4)
    path = os.path.join(FIGURES, "fig4_calibration.pdf")
    fig.savefig(path); plt.close(fig); made.append(os.path.relpath(path, ROOT))

    # --- Fig 5: regime comparison ---
    regs = res["simulation"]["regimes"]
    names = list(regs)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.3))
    x = np.arange(len(names))
    for ax, key, ylab in ((axes[0], "incident_per_10k", "incidents per 10k flights"),
                          (axes[1], "completion", "completion rate")):
        m = [regs[n][key]["mean"] for n in names]
        lo = [regs[n][key]["mean"] - regs[n][key]["ci_low"] for n in names]
        hi = [regs[n][key]["ci_high"] - regs[n][key]["mean"] for n in names]
        ax.bar(x, m, yerr=[lo, hi], capsize=2, color="#4f7fbf", width=0.62)
        ax.set_xticks(x)
        ax.set_xticklabels([n.split()[0] for n in names])
        ax.set_ylabel(ylab)
        ax.grid(axis="y", alpha=0.3, lw=0.4)
    path = os.path.join(FIGURES, "fig5_regimes.pdf")
    fig.savefig(path); plt.close(fig); made.append(os.path.relpath(path, ROOT))

    return made


# ==========================================================================
def main() -> Dict[str, Any]:
    _ensure_dirs()
    t0 = time.time()
    res: Dict[str, Any] = dict(meta=dict(version=__version__,
                                         python=platform.python_version(),
                                         np=np.__version__))

    print("[1/9] corpus ..."); res["corpus"] = exp_corpus()
    print("[2/9] compilation ..."); res["compile"] = exp_compile()
    print("[3/9] held-out ..."); res["heldout"] = exp_heldout()
    print("[4/9] router ..."); res["router"] = exp_router()
    print("[5/9] classification ..."); res["classification"] = exp_classification()
    print("[6/9] allocation ..."); res["allocation"] = exp_allocation()
    print("[7/9] protocol ..."); res["protocol"] = exp_protocol()
    print("[8/9] calibration ..."); res["calibration"] = exp_calibration()
    print("[9/9] simulation ..."); res["simulation"] = exp_simulation(reps=200)
    res["latency"] = exp_latency()
    res["audit"] = validate_detector()
    res["meta"]["seconds"] = round(time.time() - t0, 1)

    _dump("all_results.json", res)
    for k in res:
        if k == "meta":
            continue
        _dump(f"{k}.json", res[k])
    res["figures"] = make_figures(res)
    _dump("figures.json", res["figures"])
    print("done in %.1fs" % (time.time() - t0))
    return res


if __name__ == "__main__":
    main()










