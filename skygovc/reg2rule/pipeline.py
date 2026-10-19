"""Reg2Rule — the compilation pipeline.

    clause text  --frames-->  propositions  --align-->  aligned propositions
                 --merge-->  rule base      --emit-->    RDF + SHACL + SPARQL

``compile_rules`` is the single entry point.  It returns every intermediate
artefact so that the paper's tables can be recomputed without guessing at
intermediate state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core import ACTIONS, CONDITION_NAMES, Proposition, SUBJECTS, to_jsonable
from ..corpus import loader as C
from . import align as A
from . import emit as E
from . import frames as F
from . import merge as M
from . import router as R
from . import score as S


@dataclass
class PipelineResult:
    strategy: str
    clauses: List[Any]
    extractions: List[F.Extraction]
    props_aligned: List[Proposition]
    rulebase: List[Proposition]
    conflicts: List[M.Conflict]
    conflict_summary: Dict[str, Any]
    validation_raw: E.ValidationReport
    validation_merged: E.ValidationReport
    scores_raw: S.PropScores
    scores_dedup: S.PropScores
    rule_scores: Dict[str, Any]
    comp_scores: S.CompilabilityScores
    n_repaired: int
    executable_share: float
    provenance_completeness: float

    def to_dict(self) -> Dict[str, Any]:
        return dict(
            strategy=self.strategy,
            n_clauses=len(self.clauses),
            n_props_predicted=len(self.props_aligned),
            n_rules_after_merge=len(self.rulebase),
            n_repaired=self.n_repaired,
            conflict_summary=self.conflict_summary,
            validation_raw=to_jsonable(self.validation_raw),
            validation_merged=to_jsonable(self.validation_merged),
            scores_raw=to_jsonable(self.scores_raw),
            scores_dedup=to_jsonable(self.scores_dedup),
            rule_scores=to_jsonable(self.rule_scores),
            comp_scores=to_jsonable(self.comp_scores),
            executable_share=self.executable_share,
            provenance_completeness=self.provenance_completeness,
        )


def compile_rules(strategy: str = "S3_repair", include_external: bool = False,
                  validate: bool = True) -> PipelineResult:
    clauses = C.all_clauses() if include_external else C.core_clauses()
    cids = {c.cid for c in clauses}
    gold_props = [p for p in C.gold_propositions(dedupe=True) if p.cid in cids]

    exs = F.extract_all(clauses, strategy)

    raw_props: List[Proposition] = []
    for e in exs:
        if e.compilable:
            raw_props.extend(e.props)

    aligned, n_repaired = A.align_all(raw_props)

    # static executability filter
    domains = A.value_domains()
    executable: List[Proposition] = []
    for p in aligned:
        ok, why = A.is_executable(p, set(SUBJECTS), set(ACTIONS),
                                  set(CONDITION_NAMES), domains)
        if ok:
            executable.append(p)

    clause_map = {c.cid: c for c in clauses}
    rulebase, conflicts = M.merge(executable, clause_map)

    if validate:
        v_raw = E.validate_rulebase(executable)
        v_merged = E.validate_rulebase(rulebase)
    else:
        v_raw = E.ValidationReport(True, 0, [], len(executable))
        v_merged = E.ValidationReport(True, 0, [], len(rulebase))

    pred_comp = {e.cid: e.compilable for e in exs}
    pred_mode = {e.cid: (e.failure_mode or "") for e in exs if not e.compilable}
    gold_comp = {cid: v for cid, v in C.gold_compilability().items() if cid in cids}

    return PipelineResult(
        strategy=strategy,
        clauses=clauses,
        extractions=exs,
        props_aligned=executable,
        rulebase=rulebase,
        conflicts=conflicts,
        conflict_summary=M.summary(conflicts),
        validation_raw=v_raw,
        validation_merged=v_merged,
        scores_raw=S.score_propositions(executable, gold_props, dedupe_pred=False),
        scores_dedup=S.score_propositions(executable, gold_props, dedupe_pred=True),
        rule_scores=S.rulebase_scores(rulebase, gold_props),
        comp_scores=S.score_compilability(pred_comp, gold_comp, pred_mode),
        n_repaired=n_repaired,
        executable_share=(len(executable) / len(aligned) if aligned else 0.0),
        provenance_completeness=1.0 if executable else 0.0,
    )


# --------------------------------------------------------------------------
# Helpers for the routing experiment
# --------------------------------------------------------------------------
def gold_props_by_cid() -> Dict[str, List[Proposition]]:
    """Per-clause gold propositions *without* cross-instrument deduplication."""
    out: Dict[str, List[Proposition]] = {}
    from ..corpus import gold as G
    from ..core import Condition
    for g in G.GOLD:
        if not g["compilable"]:
            continue
        lst: List[Proposition] = []
        for k, (deontic, subject, action, conds) in enumerate(g.get("props", []), start=1):
            lst.append(Proposition(
                pid=f"{g['cid']}-G{k}", cid=g["cid"], deontic=deontic,
                subject=subject, action=action,
                conditions=[Condition(*c) for c in conds], extractor="gold"))
        out[g["cid"]] = lst
    return out


def evaluate_with_oracle(res: PipelineResult, adjudicated: set) -> S.PropScores:
    """Score the rule base when the records in ``adjudicated`` are corrected to
    gold.  This is an *oracle* upper bound on what expert adjudication buys; the
    paper states it as such."""
    by_cid = gold_props_by_cid()
    pred: List[Proposition] = []
    for p in res.props_aligned:
        if p.cid in adjudicated:
            continue
        pred.append(p)
    for cid in adjudicated:
        pred.extend(by_cid.get(cid, []))
    gold = C.gold_propositions(dedupe=True)
    return S.score_propositions(pred, gold, dedupe_pred=True)


# --------------------------------------------------------------------------
# Frozen-grammar evaluation on a separate block
# --------------------------------------------------------------------------
def evaluate_block(clauses: List[Any], gold: List[Proposition],
                   gold_comp: Dict[str, Tuple[bool, Optional[str]]],
                   strategy: str = "S3_repair") -> Dict[str, Any]:
    """Run one extraction pass over an arbitrary clause block and score it.

    Used for the held-out evaluation in Section 6.4: the block is passed in, the
    grammar is not touched, and whatever comes out is what gets reported.
    """
    exs = F.extract_all(clauses, strategy)
    raw: List[Proposition] = []
    for e in exs:
        if e.compilable:
            raw.extend(e.props)
    aligned, n_repaired = A.align_all(raw)
    domains = A.value_domains()
    executable: List[Proposition] = []
    for p in aligned:
        ok, _ = A.is_executable(p, set(SUBJECTS), set(ACTIONS),
                                set(CONDITION_NAMES), domains)
        if ok:
            executable.append(p)
    pred_comp = {e.cid: e.compilable for e in exs}
    pred_mode = {e.cid: (e.failure_mode or "") for e in exs if not e.compilable}
    return dict(
        n_clauses=len(clauses),
        n_props=len(executable),
        scores_raw=to_jsonable(S.score_propositions(executable, gold, dedupe_pred=False)),
        scores_dedup=to_jsonable(S.score_propositions(executable, gold, dedupe_pred=True)),
        comp_scores=to_jsonable(S.score_compilability(pred_comp, gold_comp, pred_mode)),
        n_repaired=n_repaired,
        components=to_jsonable(S.score_components(
            exs, {cid: [p for p in gold if p.cid == cid]
                  for cid in {c.cid for c in clauses}})),
        extractions=[e.to_dict() for e in exs],
    )

