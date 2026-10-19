"""Frame-based extraction of institutional propositions from clause text.

Four extraction strategies correspond to the ablations in Section 6.3:

``S1_keyword``
    Modal + first verb-phrase hit.  No subject disambiguation, no conditions,
    no proviso handling, no compilability filtering.

``S2_frame``
    Adds syntax-anchored subject/action selection (the deontic scopes over the
    verb phrase that *follows* it, and over the actor named in the clause
    segment that *precedes* it), condition patterns, and a minimal
    compilability test.

``S3_repair``
    Adds (i) proviso condition transfer, (ii) disjunctive-antecedent splitting,
    (iii) two *lifting* frames that recover rules which carry no modal particle
    but whose deontic follows from an ontology axiom (``X 为适飞空域`` ⇒
    permission; ``由 Y 依法处置`` ⇒ competence), and (iv) the failure-mode
    filter that yields the compilability verdict.

``S4_route`` is ``S3`` plus the learned adjudication router in
:mod:`skygovc.reg2rule.router`, applied by the pipeline.

Every extraction carries the feature vector the router consumes, so the router
trains without re-running the extractor.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..core import Clause, Condition, Proposition
from . import lexicon as LX

STRATEGIES = ("S1_keyword", "S2_frame", "S3_repair")

_SENT_SPLIT = re.compile(r"[。；;]+")
_ENUM_SPLIT = re.compile(r"(?=（[一二三四五六七八九十]+）)")
_CLAUSE_BOUNDARY = re.compile(r"[，,；;。]")
_DISJUNCTIVE = re.compile(r"或者|或")
_BOOLEAN_COND = ("night_operation", "bvlos", "cross_border", "over_populated_area",
                 "within_airport_buffer", "registered", "info_reported")


@dataclass
class Extraction:
    cid: str
    strategy: str
    props: List[Proposition] = field(default_factory=list)
    compilable: bool = True
    failure_mode: Optional[str] = None
    flags: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return dict(cid=self.cid, strategy=self.strategy,
                    props=[p.to_dict() for p in self.props],
                    compilable=self.compilable, failure_mode=self.failure_mode,
                    flags=self.flags, confidence=self.confidence,
                    notes=self.notes)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _sentences(text: str) -> List[str]:
    out: List[str] = []
    for p in [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]:
        out.extend([s.strip() for s in _ENUM_SPLIT.split(p) if s.strip()])
    return out


def _subject_before_modal(sentence: str, modal_pos: int) -> str:
    """The actor named in the clause segment immediately preceding the modal."""
    if modal_pos <= 0:
        return sentence
    head = sentence[:modal_pos]
    bounds = [m.end() for m in _CLAUSE_BOUNDARY.finditer(head)]
    seg = head[bounds[-1]:] if bounds else head
    return seg if len(seg.strip()) >= 3 else head


#: Conditions inherited from clause scope when the operative sentence carries
#: none (the altitude ceiling and the airport buffer are clause-level scoping
#: facts, not sentence-level ones).
INHERIT_FROM_CLAUSE = ("altitude_agl_m", "within_airport_buffer")

_PROVISO_ONLY = re.compile(r"^\s*(?:但|除)|除外")


def _action_after(sentence: str, pos: int) -> str:
    """The verb phrase governed by the modal.

    Candidates are ranked by ``length - 0.1 * distance_from_modal`` so that a
    long, specific phrase slightly further away beats a short generic one right
    after the modal (``提供空域信息`` over ``飞行活动``), while the phrase that
    immediately follows the modal still wins when it is specific
    (``取得操控员执照`` over ``安全操作知识`` further down the sentence).
    """
    tail = sentence[pos:] if pos >= 0 else sentence
    GENERIC_PENALTY = {"operate": 3.0}
    best_act, best_score = "", -1e9
    for act, pat in LX.ACTION_TERMS:
        for m in pat.finditer(tail):
            score = ((m.end() - m.start()) - 0.1 * m.start()
                     - GENERIC_PENALTY.get(act, 0.0))
            if score > best_score:
                best_act, best_score = act, score
    if best_act:
        return best_act
    act, _ = LX.find_action(sentence)
    return act


def _proviso_conditions(text: str) -> List[Tuple[str, str, Any]]:
    """Conditions induced by the *proviso substring*, never by the whole clause.

    Scoping this to the proviso is what stops ``夜间或者超视距……除外`` from
    stripping the very flags the main clause asserts.
    """
    m = LX.PROVISO.search(text)
    if not m:
        return []
    prov = text[m.start():]
    out: List[Tuple[str, str, Any]] = []
    if re.search(r"经营许可|经营性|从事经营", prov):
        out.append(("purpose", "!=", "for_hire"))
    if re.search(r"夜间", prov):
        out.append(("night_operation", "==", False))
    if re.search(r"超视距", prov):
        out.append(("bvlos", "==", False))
    return out


def _split_disjunction(sentence: str, conds: List[Tuple[str, str, Any]]
                       ) -> List[List[Tuple[str, str, Any]]]:
    """``夜间或者超视距`` scopes each flag separately: emit one proposition per
    disjunct, carrying the shared conditions."""
    if not _DISJUNCTIVE.search(sentence):
        return [conds]
    flags = [c for c in conds if c[0] in _BOOLEAN_COND and c[2] is True]
    if len(flags) < 2:
        return [conds]
    shared = [c for c in conds if c not in flags]
    return [[f] + shared for f in flags]


def failure_mode(text: str, has_modal: bool, has_action: bool,
                 lifted: bool = False) -> Optional[str]:
    """Ordered failure-mode decision procedure.

    The order encodes a priority: an instrument that delegates to another
    instrument, or attaches a sanction, is non-compilable *even if* it contains
    a modal particle, because the emitted rule would be a fragment.
    """
    if lifted:
        return None
    if LX.AUTHORISATION.search(text) and re.search(r"另行规定|决定就|授权", text):
        return "authorisation"
    if LX.META_VALIDITY.search(text):
        return "meta_validity"
    if LX.SANCTION.search(text):
        return "sanction"
    if LX.DELEGATION.search(text):
        return "cross_reference"
    if LX.CONSTITUTIVE_SCHEMA.search(text) and has_modal:
        return "constitutive_schema"
    if LX.METHODOLOGICAL.search(text):
        return "methodological"
    if LX.HEDGE.search(text):
        return "discretionary"
    if not has_modal:
        if LX.POLICY_VERBS.search(text):
            return "policy_non_normative"
        if LX.DEFINITIONAL.search(text):
            return "definitional"
        if not has_action:
            return "policy_non_normative"
        return "cross_reference"
    if not has_action:
        return "cross_reference"
    return None


def _confidence(flags: Dict[str, Any]) -> float:
    c = 0.30
    if flags.get("subject_found"):
        c += 0.18
    if flags.get("action_found"):
        c += 0.18
    if flags.get("modal_found"):
        c += 0.14
    c += min(0.12, 0.04 * flags.get("n_conditions", 0))
    if flags.get("has_proviso"):
        c -= 0.10
    if flags.get("has_cross_ref"):
        c -= 0.08
    if flags.get("has_enum"):
        c -= 0.05
    if flags.get("subject_inferred"):
        c -= 0.06
    if flags.get("clause_len", 0) > 70:
        c -= 0.05
    return float(round(min(0.99, max(0.05, c)), 4))


# --------------------------------------------------------------------------
# main entry
# --------------------------------------------------------------------------
def extract_clause(clause: Clause, strategy: str = "S3_repair") -> Extraction:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy}")
    text = clause.text
    sents = _sentences(text)
    ex = Extraction(cid=clause.cid, strategy=strategy)

    subj0, _, _ = LX.find_subject(text)
    mod0, _ = LX.find_modal(text)
    act0, _ = LX.find_action(text)
    conds_all = LX.extract_conditions(text)

    lifted = False
    lift_kind = ""
    if strategy == "S3_repair" and not mod0:
        if LX.LIFT_PERMITTED_AIRSPACE.search(text):
            lifted, lift_kind = True, "constitutive_permitted_airspace"
        elif LX.LIFT_COMPETENCE.search(text) and not LX.SANCTION.search(text):
            lifted, lift_kind = True, "competence_attribution"

    flags: Dict[str, Any] = dict(
        n_sentences=len(sents), clause_len=len(text),
        n_modal=sum(1 for s in sents if LX.find_modal(s)[0]),
        n_conditions=len(conds_all),
        has_proviso=bool(LX.PROVISO.search(text)),
        has_cross_ref=bool(LX.CROSS_REF.search(text)),
        has_hedge=bool(LX.HEDGE.search(text)),
        has_sanction=bool(LX.SANCTION.search(text)),
        has_definitional=bool(LX.DEFINITIONAL.search(text)),
        has_enum=bool(LX.ENUM_MARKER.search(text)),
        subject_found=bool(subj0), subject_inferred=False,
        action_found=bool(act0), modal_found=bool(mod0),
        lifted=lifted, lift_kind=lift_kind,
        level_rank=clause.rank, n_props=0,
    )

    props: List[Proposition] = []
    pid_n = 0

    def _emit(deontic: str, subject: str, action: str,
              conds: List[Tuple[str, str, Any]], src: str, conf: float) -> None:
        nonlocal pid_n
        pid_n += 1
        props.append(Proposition(
            pid=f"{clause.cid}-P{pid_n}", cid=clause.cid, deontic=deontic,
            subject=subject, action=action,
            conditions=[Condition(n, o, v) for (n, o, v) in conds],
            extractor=strategy, confidence=conf, compilable=True,
            normalised_from=src,
            unresolved_refs=(["cross_reference"] if LX.CROSS_REF.search(src) else []),
        ))

    # ---- lifting frames (S3 only) --------------------------------------
    if lifted and lift_kind == "constitutive_permitted_airspace":
        subject, inferred = LX.infer_subject(text)
        subject = subject or "operator"
        conds = [c for c in conds_all]
        _emit("permission", subject, "operate_in_permitted_airspace",
              conds, text, 0.82)
        flags["subject_inferred"] = inferred

    elif lifted and lift_kind == "competence_attribution":
        subject, inferred = LX.infer_subject(text)
        subject = subject or "public_security"
        _emit("permission", subject, "counter_uas", [], text, 0.70)
        flags["subject_inferred"] = inferred

    # ---- sentence frames ------------------------------------------------
    if not lifted:
        for s in sents:
            deontic, mpos = LX.find_modal(s)
            if strategy == "S1_keyword":
                if not deontic:
                    continue
                subject, inferred = LX.find_subject(s.lstrip()).__getitem__(0), False
                subject = subject or "operator"
                action = ""
                for act, pat in LX.ACTION_TERMS:
                    m = pat.search(s)
                    if m:
                        action = act
                        break
                action = action or "operate"
                conf = 0.40
                _emit(deontic, subject, action, [], s, conf)
                continue

            if not deontic:
                continue

            seg = _subject_before_modal(s, mpos)
            subject, inferred = LX.infer_subject(seg)
            if not subject:
                subject, inferred = LX.infer_subject(s)
            flags["subject_inferred"] = bool(inferred)
            subject = subject or "operator"

            action = _action_after(s, mpos)
            if not action:
                continue

            # waiver: "无需提出飞行活动申请" is a permission to operate
            if deontic == "permission" and LX.WAIVER_MARKERS.search(s) \
                    and action == "apply_flight_activity":
                action = ("operate_in_permitted_airspace"
                          if "适飞空域" in text else "operate")

            if _PROVISO_ONLY.search(s) and not re.search(r"应当|不得|必须|可以", s):
                continue

            conds = LX.extract_conditions(s)
            if strategy == "S3_repair":
                for c in conds_all:
                    if c[0] in INHERIT_FROM_CLAUSE and c not in conds:
                        conds.append(c)
                for c in _proviso_conditions(text):
                    if c not in conds:
                        conds.append(c)
                groups = _split_disjunction(s, conds)
            else:
                groups = [conds]

            conf = _confidence(flags) if strategy != "S1_keyword" else 0.4
            for g in groups:
                _emit(deontic, subject, action, g, s, conf)

    flags["n_props"] = len(props)

    # ---- compilability verdict -----------------------------------------
    if strategy == "S1_keyword":
        ex.compilable, ex.failure_mode = True, None
    elif strategy == "S2_frame":
        ex.compilable = bool(props) and flags["modal_found"] and flags["action_found"]
        ex.failure_mode = None if ex.compilable else "cross_reference"
    else:
        fm = failure_mode(text, flags["modal_found"], flags["action_found"], lifted)
        if fm is not None:
            ex.compilable, ex.failure_mode, props = False, fm, []
        else:
            ex.compilable = bool(props)
            if not ex.compilable:
                ex.failure_mode = "cross_reference"

    ex.props = props
    ex.flags = flags
    ex.confidence = _confidence(flags) if strategy != "S1_keyword" else 0.4
    if not ex.compilable and strategy == "S3_repair":
        ex.confidence = float(round(ex.confidence * 0.8, 4))
    return ex


def extract_all(clauses: List[Clause], strategy: str) -> List[Extraction]:
    return [extract_clause(c, strategy) for c in clauses]


