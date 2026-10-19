"""Scoring of compiler output against the held-out expert gold standard."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core import Proposition, to_jsonable


@dataclass
class PropScores:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    exact_match_rate: float = 0.0
    matched_pids: List[str] = field(default_factory=list)
    fp_pids: List[str] = field(default_factory=list)
    fn_pids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)


@dataclass
class CompilabilityScores:
    n: int = 0
    accuracy: float = 0.0
    precision_true: float = 0.0
    recall_true: float = 0.0
    f1_true: float = 0.0
    precision_false: float = 0.0
    recall_false: float = 0.0
    f1_false: float = 0.0
    macro_f1: float = 0.0
    confusion: Dict[str, int] = field(default_factory=dict)
    failure_mode_accuracy: float = 0.0
    failure_mode_n: int = 0
    per_mode: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def score_propositions(pred: Sequence[Proposition], gold: Sequence[Proposition],
                       dedupe_pred: bool = False) -> PropScores:
    """Exact-signature matching.

    ``dedupe_pred=True`` collapses duplicate predictions first; this is the
    fair comparison for the extractor, because cross-instrument redundancy is
    removed by the merger rather than by extraction.
    """
    preds = list(pred)
    if dedupe_pred:
        seen = set()
        uniq: List[Proposition] = []
        for p in preds:
            s = p.signature()
            if s in seen:
                continue
            seen.add(s)
            uniq.append(p)
        preds = uniq

    gold_sigs = {g.signature(): g for g in gold}
    pred_sigs = {p.signature(): p for p in preds}

    matched = set(gold_sigs) & set(pred_sigs)
    tp = len(matched)
    fp = len(pred_sigs) - tp
    fn = len(gold_sigs) - tp
    p, r, f = _prf(tp, fp, fn)
    return PropScores(
        tp=tp, fp=fp, fn=fn, precision=p, recall=r, f1=f,
        exact_match_rate=tp / max(len(gold_sigs), 1),
        matched_pids=[pred_sigs[s].pid for s in matched],
        fp_pids=[pred_sigs[s].pid for s in set(pred_sigs) - matched],
        fn_pids=[gold_sigs[s].pid for s in set(gold_sigs) - matched],
    )


def score_compilability(pred: Dict[str, bool],
                        gold: Dict[str, Tuple[bool, Optional[str]]],
                        pred_mode: Optional[Dict[str, str]] = None
                        ) -> CompilabilityScores:
    cids = sorted(set(pred) & set(gold))
    tp = fp = tn = fn = 0
    for cid in cids:
        g = gold[cid][0]
        p = bool(pred[cid])
        if g and p:
            tp += 1
        elif (not g) and p:
            fp += 1
        elif (not g) and (not p):
            tn += 1
        else:
            fn += 1
    pt, rt, ft = _prf(tp, fp, fn)
    pf, rf, ff = _prf(tn, fn, fp)
    n = len(cids)
    acc = (tp + tn) / n if n else 0.0

    # failure-mode agreement on the records gold marks non-compilable
    per_mode: Dict[str, Dict[str, int]] = {}
    correct = 0
    total = 0
    if pred_mode:
        for cid in cids:
            if gold[cid][0]:
                continue
            gmode = gold[cid][1]
            pmode = pred_mode.get(cid)
            per_mode.setdefault(gmode or "unknown", {"correct": 0, "total": 0})
            per_mode[gmode or "unknown"]["total"] += 1
            total += 1
            if pmode == gmode:
                per_mode[gmode or "unknown"]["correct"] += 1
                correct += 1
    return CompilabilityScores(
        n=n, accuracy=acc,
        precision_true=pt, recall_true=rt, f1_true=ft,
        precision_false=pf, recall_false=rf, f1_false=ff,
        macro_f1=(ft + ff) / 2,
        confusion=dict(tp=tp, fp=fp, tn=tn, fn=fn),
        failure_mode_accuracy=(correct / total if total else 0.0),
        failure_mode_n=total, per_mode=per_mode,
    )


def rulebase_scores(merged: Sequence[Proposition], gold: Sequence[Proposition]
                    ) -> Dict[str, Any]:
    """Post-merge rule-base quality plus executability, which is what a
    deployment actually consumes."""
    s = score_propositions(merged, gold, dedupe_pred=False)
    return dict(**s.to_dict(),
                rules=len(merged),
                executable_share=_executable_share(merged))


def _executable_share(props: Sequence[Proposition]) -> float:
    if not props:
        return 0.0
    ok = sum(1 for p in props if p.conditions is not None)
    return ok / len(props)

# --------------------------------------------------------------------------
# Component-level evaluation
# --------------------------------------------------------------------------
@dataclass
class ComponentScores:
    """Field-level agreement, which is far more informative than the
    all-or-nothing signature match once the target tuple has five fields.

    For every clause that has at least one gold proposition, the predicted
    proposition sharing the most fields with a gold proposition is taken as the
    *aligned pair*; field agreement is then read off that pair.  A clause with
    no predicted proposition contributes zeros, so the metric cannot be inflated
    by abstaining.
    """

    n_clauses: int = 0
    deontic_acc: float = 0.0
    subject_acc: float = 0.0
    action_acc: float = 0.0
    condition_jaccard: float = 0.0
    exact_acc: float = 0.0
    field_mean: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)


def _cond_set(p: Proposition) -> set:
    return {c.key() for c in p.conditions}


def score_components(exs: Sequence[Any], gold_by_cid: Dict[str, List[Proposition]]
                     ) -> ComponentScores:
    n = 0
    d = s = a = j = e = 0.0
    for ex in exs:
        golds = gold_by_cid.get(ex.cid, [])
        if not golds:
            continue
        n += 1
        preds = list(ex.props)
        if not preds:
            continue
        best, best_overlap = None, -1
        for p in preds:
            for g in golds:
                ov = ((p.deontic == g.deontic) + (p.subject == g.subject)
                      + (p.action == g.action))
                ov += len(_cond_set(p) & _cond_set(g)) / max(
                    len(_cond_set(p) | _cond_set(g)), 1)
                if ov > best_overlap:
                    best, best_overlap = (p, g), ov
        if best is None:
            continue
        p, g = best
        d += float(p.deontic == g.deontic)
        s += float(p.subject == g.subject)
        a += float(p.action == g.action)
        union = _cond_set(p) | _cond_set(g)
        j += (len(_cond_set(p) & _cond_set(g)) / len(union)) if union else 1.0
        e += float(p.signature() == g.signature())
    denom = max(n, 1)
    out = ComponentScores(
        n_clauses=n,
        deontic_acc=d / denom, subject_acc=s / denom, action_acc=a / denom,
        condition_jaccard=j / denom, exact_acc=e / denom)
    out.field_mean = (out.deontic_acc + out.subject_acc + out.action_acc
                      + out.condition_jaccard) / 4
    return out
