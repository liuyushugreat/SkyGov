"""Governance-tier classification from an observable context fingerprint.

The tier is not a label that a planner asserts; it is the conclusion of a rule
system over observables, and the rule identifiers are part of the audit trail.
Alongside the rule system this module provides a supervised baseline, so the
paper can report the accuracy/auditability trade-off rather than asserting that
rules are "interpretable".

Tiers and their reference jurisdictions follow the schema in
:data:`skygovc.core.TIERS`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core import ContextFingerprint, TIERS, to_jsonable

#: Six reference jurisdictions.  ``airspace_control_share`` is read off the
#: local instruments: Beijing and Tianjin declare the whole municipal territory
#: controlled (D15-C001/D16-C001, D13-C001); Shenzhen promotes industry and
#: leaves airspace to the national frame; the mountainous, border-coastal and
#: sparse jurisdictions are parameterised from the case descriptions.
REFERENCE: Dict[str, ContextFingerprint] = {
    "CN-BJ": ContextFingerprint(
        jurisdiction="CN-BJ", law_constraints=["all_territory_controlled"],
        population_exposure=0.95, critical_facility=0.95, demand_density=0.55,
        terrain_complexity=0.15, border_coastal=False, seasonal=False,
        airspace_control_share=1.0),
    "CN-TJ": ContextFingerprint(
        jurisdiction="CN-TJ", law_constraints=["all_territory_controlled"],
        population_exposure=0.80, critical_facility=0.75, demand_density=0.45,
        terrain_complexity=0.20, border_coastal=True, seasonal=False,
        airspace_control_share=1.0),
    "CN-SZ": ContextFingerprint(
        jurisdiction="CN-SZ", law_constraints=["industry_promotion"],
        population_exposure=0.90, critical_facility=0.55, demand_density=0.98,
        terrain_complexity=0.10, border_coastal=True, seasonal=False,
        airspace_control_share=0.25),
    "CN-CD": ContextFingerprint(
        jurisdiction="CN-CD", law_constraints=[],
        population_exposure=0.55, critical_facility=0.30, demand_density=0.50,
        terrain_complexity=0.85, border_coastal=False, seasonal=True,
        airspace_control_share=0.40),
    "CN-HN": ContextFingerprint(
        jurisdiction="CN-HN", law_constraints=["free_trade_port"],
        population_exposure=0.40, critical_facility=0.45, demand_density=0.35,
        terrain_complexity=0.30, border_coastal=True, seasonal=True,
        airspace_control_share=0.50),
    "CN-JL": ContextFingerprint(
        jurisdiction="CN-JL", law_constraints=["provincial_permitted_airspace"],
        population_exposure=0.20, critical_facility=0.15, demand_density=0.18,
        terrain_complexity=0.60, border_coastal=True, seasonal=True,
        airspace_control_share=0.15),
}

#: Reference tier of each jurisdiction (the expert label used for scoring).
REFERENCE_TIER: Dict[str, str] = {
    "CN-BJ": "capital_core",
    "CN-TJ": "dense_hub",
    "CN-SZ": "market_driven",
    "CN-CD": "mountainous",
    "CN-HN": "border_coastal",
    "CN-JL": "sparse_resource",
}


# --------------------------------------------------------------------------
# Rule system
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TierRule:
    rid: str
    tier: str
    priority: int
    description: str
    predicate: Any = field(compare=False, repr=False)

    def matches(self, f: ContextFingerprint) -> bool:
        return bool(self.predicate(f))


#: Ordered rules.  Priority resolves overlaps; the highest-priority matching
#: rule wins, so the rule set is a total function on the fingerprint space.
#: Every rule is a conjunction of literal thresholds — no weights, no scores —
#: which is what makes the assignment audit-able.
RULES: List[TierRule] = [
    TierRule("T1", "capital_core", 100,
             "critical_facility >= 0.85 and airspace_control_share >= 0.90",
             lambda f: f.critical_facility >= 0.85 and f.airspace_control_share >= 0.90),
    TierRule("T2", "dense_hub", 90,
             "population_exposure >= 0.70 and 0.40 <= demand_density < 0.90",
             lambda f: f.population_exposure >= 0.70 and 0.40 <= f.demand_density < 0.90),
    TierRule("T3", "border_coastal", 80,
             "border_coastal and demand_density >= 0.30 and (terrain_complexity < 0.50 or seasonal)",
             lambda f: (f.border_coastal and f.demand_density >= 0.30
                        and (f.terrain_complexity < 0.50 or f.seasonal))),
    TierRule("T5b", "sparse_resource", 75,
             "demand_density < 0.25 and population_exposure < 0.30",
             lambda f: f.demand_density < 0.25 and f.population_exposure < 0.30),
    TierRule("T4", "mountainous", 70,
             "terrain_complexity >= 0.55",
             lambda f: f.terrain_complexity >= 0.55),
    TierRule("T5", "market_driven", 85,
             "demand_density >= 0.75",
             lambda f: f.demand_density >= 0.75),
    TierRule("T6", "sparse_resource", 10,
             "otherwise",
             lambda f: True),
]


def classify(f: ContextFingerprint) -> Tuple[str, List[str]]:
    """Return (tier, firing rule chain).  The chain is the audit trail."""
    hits = [r for r in RULES if r.matches(f)]
    hits.sort(key=lambda r: -r.priority)
    return hits[0].tier, [r.rid for r in hits]


def classify_batch(fps: Sequence[ContextFingerprint]) -> List[str]:
    return [classify(f)[0] for f in fps]


def reference_accuracy() -> Dict[str, Any]:
    out = {}
    for j, f in REFERENCE.items():
        tier, chain = classify(f)
        out[j] = dict(predicted=tier, gold=REFERENCE_TIER[j],
                      correct=tier == REFERENCE_TIER[j], chain=chain)
    n = len(out)
    acc = sum(1 for v in out.values() if v["correct"]) / n
    return dict(per_jurisdiction=out, accuracy=acc, n=n)


# --------------------------------------------------------------------------
# Synthetic benchmark
# --------------------------------------------------------------------------
#: Prototype of each tier in fingerprint space; noise is added around it.
TIER_PROTOTYPE: Dict[str, Dict[str, float]] = {
    "capital_core":   dict(population_exposure=0.95, critical_facility=0.95,
                           demand_density=0.55, terrain_complexity=0.15,
                           airspace_control_share=0.95, border_coastal=0.0, seasonal=0.0),
    "dense_hub":      dict(population_exposure=0.85, critical_facility=0.50,
                           demand_density=0.60, terrain_complexity=0.25,
                           airspace_control_share=0.60, border_coastal=0.3, seasonal=0.0),
    "market_driven":  dict(population_exposure=0.75, critical_facility=0.35,
                           demand_density=0.95, terrain_complexity=0.15,
                           airspace_control_share=0.25, border_coastal=0.4, seasonal=0.0),
    "mountainous":    dict(population_exposure=0.45, critical_facility=0.25,
                           demand_density=0.45, terrain_complexity=0.85,
                           airspace_control_share=0.40, border_coastal=0.0, seasonal=0.6),
    "border_coastal": dict(population_exposure=0.40, critical_facility=0.45,
                           demand_density=0.35, terrain_complexity=0.30,
                           airspace_control_share=0.50, border_coastal=1.0, seasonal=0.5),
    "sparse_resource": dict(population_exposure=0.20, critical_facility=0.15,
                            demand_density=0.18, terrain_complexity=0.45,
                            airspace_control_share=0.15, border_coastal=0.2, seasonal=0.7),
}

FEATURES: Tuple[str, ...] = ("population_exposure", "critical_facility",
                             "demand_density", "terrain_complexity",
                             "airspace_control_share", "border_coastal", "seasonal")


def make_benchmark(n: int = 600, noise: float = 0.12, seed: int = 0
                   ) -> Tuple[List[ContextFingerprint], np.ndarray]:
    rng = np.random.default_rng(seed)
    fps: List[ContextFingerprint] = []
    ys: List[int] = []
    for i in range(n):
        k = i % len(TIERS)
        tier = TIERS[k]
        proto = TIER_PROTOTYPE[tier]
        vals = {}
        for name in FEATURES:
            v = proto[name] + rng.normal(0, noise)
            vals[name] = float(np.clip(v, 0.0, 1.0)) if name not in (
                "border_coastal", "seasonal") else float(v > 0.5)
        fps.append(ContextFingerprint(
            jurisdiction=f"SYN-{i:04d}", law_constraints=[],
            population_exposure=vals["population_exposure"],
            critical_facility=vals["critical_facility"],
            demand_density=vals["demand_density"],
            terrain_complexity=vals["terrain_complexity"],
            border_coastal=bool(vals["border_coastal"]),
            seasonal=bool(vals["seasonal"]),
            airspace_control_share=vals["airspace_control_share"]))
        ys.append(k)
    return fps, np.array(ys, dtype=int)


def feature_matrix(fps: Sequence[ContextFingerprint]) -> np.ndarray:
    return np.array([[float(getattr(f, k)) for k in FEATURES] for f in fps])


@dataclass
class BenchmarkResult:
    rule_accuracy: float
    ml_accuracy: Dict[str, float]
    rule_chain_length: float
    n: int
    confusion: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)


def benchmark(n: int = 600, noise: float = 0.12, seed: int = 0,
              test_size: float = 0.4) -> BenchmarkResult:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    fps, y = make_benchmark(n=n, noise=noise, seed=seed)
    X = feature_matrix(fps)

    pred_rule = np.array([TIERS.index(classify(f)[0]) for f in fps])
    rule_acc = float(accuracy_score(y, pred_rule))
    chain_len = float(np.mean([len(classify(f)[1]) for f in fps]))

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size,
                                          random_state=seed, stratify=y)
    models = {
        "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=seed),
        "logistic": LogisticRegression(max_iter=2000, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=120, random_state=seed),
    }
    accs: Dict[str, float] = {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        accs[name] = float(accuracy_score(yte, m.predict(Xte)))

    conf: Dict[str, Dict[str, int]] = {}
    for t in TIERS:
        conf[t] = {}
    for g, p in zip(y, pred_rule):
        gname, pname = TIERS[int(g)], TIERS[int(p)]
        conf[gname][pname] = conf[gname].get(pname, 0) + 1

    return BenchmarkResult(rule_accuracy=rule_acc, ml_accuracy=accs,
                           rule_chain_length=chain_len, n=n, confusion=conf)


# --------------------------------------------------------------------------
# Spatial regime read off the instruments
# --------------------------------------------------------------------------
def jurisdiction_airspace_regime() -> Dict[str, Dict[str, Any]]:
    """Read each jurisdiction's airspace regime off the compiled clause set.

    This is the bridge from the institutional layer to the context fingerprint:
    the dominant form of inter-instrument tension in this corpus is not a
    contradiction between deontic modalities but a change in the *extension* of
    the term "permitted airspace".
    """
    return {
        "CN-NATIONAL": dict(basis="D01-C003, D04-C002",
                            permitted_fraction=0.75,
                            note="真高120米以下非管制空域为适飞空域"),
        "CN-JL": dict(basis="D12-C001", permitted_fraction=0.70,
                      note="省级公告沿用国家口径，适飞范围扩大"),
        "CN-TJ": dict(basis="D13-C001", permitted_fraction=0.00,
                      note="全市域划为管制空域"),
        "CN-BJ": dict(basis="D15-C001, D16-C001", permitted_fraction=0.00,
                      note="全市域划为管制空域，两版一致"),
        "CN-SZ": dict(basis="D14-C001..C006", permitted_fraction=0.75,
                      note="产业促进型法规，不改空域类别"),
    }


