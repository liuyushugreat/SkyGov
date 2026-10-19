"""Learning to route compilations to human adjudication.

The compiler is allowed to be uncertain.  What matters for an information
system is that the *cheap* signal — the feature vector of the extraction —
predicts which records a domain expert will have to rewrite.  A router that
predicts this well converts a fixed expert budget into more rule quality than
either uniform review or confidence-only review.

The label is defined without looking at prediction confidence: a record *needs
editing* iff the compiled output differs from the expert gold standard, either
in compilability or in the proposition set.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core import Clause, Proposition, to_jsonable
from .frames import Extraction

FEATURES: Tuple[str, ...] = (
    "clause_len", "n_sentences", "n_modal", "n_conditions", "has_proviso",
    "has_cross_ref", "has_discretionary", "has_sanction", "has_definitional",
    "has_enum", "subject_found", "action_found", "modal_found", "level_rank",
    "n_props", "confidence", "pred_compilable", "n_condition_names",
)


def featurize(ex: Extraction) -> Dict[str, float]:
    f = ex.flags
    names = set()
    for p in ex.props:
        for c in p.conditions:
            names.add(c.name)
    return {
        "clause_len": float(f.get("clause_len", 0)),
        "n_sentences": float(f.get("n_sentences", 0)),
        "n_modal": float(f.get("n_modal", 0)),
        "n_conditions": float(f.get("n_conditions", 0)),
        "has_proviso": float(bool(f.get("has_proviso"))),
        "has_cross_ref": float(bool(f.get("has_cross_ref"))),
        "has_discretionary": float(bool(f.get("has_discretionary"))),
        "has_sanction": float(bool(f.get("has_sanction"))),
        "has_definitional": float(bool(f.get("has_definitional"))),
        "has_enum": float(bool(f.get("has_enum"))),
        "subject_found": float(bool(f.get("subject_found"))),
        "action_found": float(bool(f.get("action_found"))),
        "modal_found": float(bool(f.get("modal_found"))),
        "level_rank": float(f.get("level_rank", 5)),
        "n_props": float(f.get("n_props", 0)),
        "confidence": float(ex.confidence),
        "pred_compilable": float(ex.compilable),
        "n_condition_names": float(len(names)),
    }


def feature_matrix(exs: Sequence[Extraction]) -> np.ndarray:
    return np.array([[featurize(e)[k] for k in FEATURES] for e in exs], dtype=float)


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------
def per_cid_gold(exs: Sequence[Extraction], gold_props_by_cid: Dict[str, List[Proposition]],
                 gold_compilability: Dict[str, Tuple[bool, Optional[str]]]
                 ) -> Dict[str, bool]:
    """``needs_edit`` label per clause."""
    out: Dict[str, bool] = {}
    for e in exs:
        g_comp = gold_compilability[e.cid][0]
        if e.compilable != g_comp:
            out[e.cid] = True
            continue
        if not g_comp:
            out[e.cid] = False
            continue
        pred_sigs = {p.signature() for p in e.props}
        gold_sigs = {p.signature() for p in gold_props_by_cid.get(e.cid, [])}
        out[e.cid] = pred_sigs != gold_sigs
    return out


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
@dataclass
class RouterReport:
    auc: float
    average_precision: float
    n_train: int
    n_test: int
    n_pos: int
    threshold: float
    precision_at_threshold: float
    recall_at_threshold: float
    importances: Dict[str, float] = field(default_factory=dict)
    fold_aucs: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)


def train_router(exs: Sequence[Extraction], labels: Dict[str, bool],
                 groups: Dict[str, str], n_splits: int = 5, seed: int = 0
                 ) -> Tuple[Any, RouterReport]:
    """Group-K-fold cross-validation by *instrument* so that no clause from a
    test instrument appears in training."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
    from sklearn.model_selection import GroupKFold

    X = feature_matrix(exs)
    y = np.array([1 if labels[e.cid] else 0 for e in exs], dtype=int)
    g = np.array([groups[e.cid] for e in exs])

    n_groups = len(set(g))
    n_splits = max(2, min(n_splits, n_groups))
    gkf = GroupKFold(n_splits=n_splits)

    oof = np.zeros(len(y), dtype=float)
    fold_aucs: List[float] = []
    for tr, te in gkf.split(X, y, groups=g):
        clf = GradientBoostingClassifier(random_state=seed, n_estimators=120,
                                         max_depth=2, learning_rate=0.08)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
        if len(set(y[te])) > 1:
            fold_aucs.append(float(roc_auc_score(y[te], oof[te])))

    auc = float(roc_auc_score(y, oof)) if len(set(y)) > 1 else float("nan")
    ap = float(average_precision_score(y, oof)) if len(set(y)) > 1 else float("nan")

    # threshold that maximises F1 of the routing decision
    prec, rec, thr = precision_recall_curve(y, oof)
    f1s = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec + 1e-12), 0.0)
    k = int(np.argmax(f1s[:-1])) if len(thr) else 0
    best_thr = float(thr[k]) if len(thr) else 0.5

    final = GradientBoostingClassifier(random_state=seed, n_estimators=120,
                                       max_depth=2, learning_rate=0.08)
    final.fit(X, y)
    imp = {f: float(v) for f, v in zip(FEATURES, final.feature_importances_)}

    report = RouterReport(
        auc=auc, average_precision=ap, n_train=len(y), n_test=len(y),
        n_pos=int(y.sum()), threshold=best_thr,
        precision_at_threshold=float(prec[k]) if len(prec) else 0.0,
        recall_at_threshold=float(rec[k]) if len(rec) else 0.0,
        importances=dict(sorted(imp.items(), key=lambda kv: -kv[1])),
        fold_aucs=fold_aucs,
    )
    final.oof_scores_ = oof
    return final, report


# --------------------------------------------------------------------------
# Routing economics
# --------------------------------------------------------------------------
@dataclass
class RoutingPoint:
    k: int
    expert_minutes: float
    f1: float

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)


def routing_curve(order: Sequence[str], cids: Sequence[str],
                  score_after: Any, theta_edit: float = 2.5,
                  kmax: Optional[int] = None) -> List[RoutingPoint]:
    """Sweep the number of adjudicated records and record the resulting F1.

    ``score_after(cid_set)`` must return the rule-F1 obtained when the records
    in ``cid_set`` are corrected to gold.
    """
    n = len(cids)
    kmax = min(n if kmax is None else kmax, n)
    pts: List[RoutingPoint] = []
    for k in range(0, kmax + 1):
        chosen = set(order[:k])
        pts.append(RoutingPoint(k=k, expert_minutes=k * theta_edit,
                                f1=float(score_after(chosen))))
    return pts


def breakeven_k(curve: Sequence[RoutingPoint], target: float) -> Optional[int]:
    for p in curve:
        if p.f1 >= target - 1e-12:
            return p.k
    return None
