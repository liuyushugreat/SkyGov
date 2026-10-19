"""Conflict-aware merging of propositions across the legal hierarchy.

Motivation
----------
Concatenating rules from sixteen instruments produces a rule base that is
executable but not *coherent*: the same regulated action can carry an
obligation in one instrument and a prohibition in another, and numeric
thresholds can differ.  A compiler that silently keeps both produces
engines that deadlock.

This module detects candidate conflicts and resolves them with a documented
precedence algebra, and — importantly — reports the candidates it *rejects* as
spurious because the two propositions do not actually regulate the same object.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core import Clause, Condition, Proposition

#: Resolution reason codes.
RANK = "lex_superior"
POSTERIOR = "lex_posterior"
SPECIALIS = "lex_specialis"
SPURIOUS = "object_mismatch"
UNRESOLVED = "unresolved"

_CN_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6,
           "七": 7, "八": 8, "九": 9, "十": 10}

#: Object noun-phrases that follow the operative verb; used for the
#: object-mismatch test.  Extracted by longest-suffix match on the clause text.
_OBJECT_MARKERS = re.compile(
    r"飞行活动|飞行活动申请|实名登记|操控员执照|经营许可|适飞空域|管制空域|"
    r"通信链路|导航信号|身份伪造|起降设施|通信导航监视设施|低空飞行服务|告警等服务|"
    r"情报|数据安全管理制度|位置、影像等数据|国家秘密|商业秘密|个人信息|"
    r"事故|现场|台账|身份信息|财政资金|产业发展|场景|推广|空域类别|飞行计划"
)


@dataclass
class Conflict:
    kind: str                     # modality | threshold | duplication
    a: str                        # proposition / rule id
    b: str
    detail: str
    resolution: str = UNRESOLVED
    winner: Optional[str] = None
    rationale: str = ""
    spurious: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return dict(kind=self.kind, a=self.a, b=self.b, detail=self.detail,
                    resolution=self.resolution, winner=self.winner,
                    rationale=self.rationale, spurious=self.spurious)


def _cond_map(p: Proposition) -> Dict[str, Tuple[str, Any]]:
    return {c.name: (c.op, c.value) for c in p.conditions}


def _objects(text: str) -> set:
    return set(m.group(0) for m in _OBJECT_MARKERS.finditer(text or ""))


def object_signature(p: Proposition) -> str:
    """Canonical string of the object noun phrases of a proposition.  Two rules
    may only collide on modality when their object signatures coincide."""
    return "|".join(sorted(_objects(p.normalised_from))) or "unknown"


def objects_overlap(a: Proposition, b: Proposition) -> bool:
    """Two propositions regulate the same object only if the object noun
    phrases of their source sentences intersect (closed-world test)."""
    oa, ob = _objects(a.normalised_from), _objects(b.normalised_from)
    if not oa or not ob:
        return True       # cannot decide -> assume overlap (conservative)
    return bool(oa & ob)


def detect(props: Sequence[Proposition], clauses: Dict[str, Clause]) -> List[Conflict]:
    """Detect candidate conflicts among propositions."""
    out: List[Conflict] = []
    by_action: Dict[Tuple[str, str], List[Proposition]] = {}
    for p in props:
        by_action.setdefault((p.subject, p.action), []).append(p)

    for (subj, act), group in by_action.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                ca, cb = _cond_map(a), _cond_map(b)

                # ---- modality conflict on overlapping scopes --------------
                if {a.deontic, b.deontic} == {"obligation", "prohibition"}:
                    shared = set(ca) & set(cb)
                    same_scope = all(str(ca[k]) == str(cb[k]) for k in shared) or not shared
                    if same_scope:
                        out.append(Conflict(
                            kind="modality", a=a.pid, b=b.pid,
                            detail=f"{subj}/{act}: {a.deontic} vs {b.deontic}",
                        ))
                # ---- numeric threshold conflict --------------------------
                for name in ("altitude_agl_m", "payload_kg", "mass_kg"):
                    if name in ca and name in cb:
                        va, vb = ca[name][1], cb[name][1]
                        if va != vb:
                            out.append(Conflict(
                                kind="threshold", a=a.pid, b=b.pid,
                                detail=f"{subj}/{act}/{name}: {va} vs {vb}",
                            ))
    return out


def _rank(p: Proposition, clauses: Dict[str, Clause]) -> int:
    return clauses[p.cid].rank


def _date(p: Proposition, clauses: Dict[str, Clause]) -> str:
    return clauses[p.cid].instrument_date


def _specificity(p: Proposition) -> int:
    return len(p.conditions)


def resolve(c: Conflict, props: Dict[str, Proposition],
            clauses: Dict[str, Clause]) -> Conflict:
    a, b = props[c.a], props[c.b]

    # Step 0: do they regulate the same object at all?
    if not objects_overlap(a, b):
        c.resolution = SPURIOUS
        c.spurious = True
        c.rationale = (
            f"source objects '{sorted(_objects(a.normalised_from))}' vs "
            f"'{sorted(_objects(b.normalised_from))}' do not intersect")
        return c

    # Step 1: hierarchy
    ra, rb = _rank(a, clauses), _rank(b, clauses)
    if ra != rb:
        w = a if ra < rb else b
        c.resolution = RANK
        c.winner = w.pid
        c.rationale = (f"instrument rank {min(ra, rb)} < {max(ra, rb)} "
                       f"({clauses[w.cid].doc_id} prevails)")
        return c

    # Step 2: lex posterior within the same rank
    da, db = _date(a, clauses), _date(b, clauses)
    if da != db:
        w = a if da > db else b
        c.resolution = POSTERIOR
        c.winner = w.pid
        c.rationale = f"same rank; later instrument {clauses[w.cid].doc_id} ({da}) prevails"
        return c

    # Step 3: lex specialis (more conditions == narrower scope)
    sa, sb = _specificity(a), _specificity(b)
    if sa != sb:
        w = a if sa > sb else b
        c.resolution = SPECIALIS
        c.winner = w.pid
        c.rationale = f"same rank and date; narrower scope ({max(sa, sb)} conditions) prevails"
        return c

    c.resolution = UNRESOLVED
    c.rationale = "no precedence criterion applies; escalated to adjudication"
    return c


def merge(props: Sequence[Proposition], clauses: Dict[str, Clause]
          ) -> Tuple[List[Proposition], List[Conflict]]:
    """Return (kept propositions, conflict log).

    A proposition is dropped when it loses a resolved conflict whose winner is
    a prohibition (the stricter norm subsumes) or when it is a duplicate.  A
    proposition is kept when it is the winner, or when the conflict is
    unresolved or spurious (unresolved ones are flagged for adjudication).
    """
    pmap = {p.pid: p for p in props}
    conflicts = detect(props, clauses)

    dropped: set = set()
    for c in conflicts:
        resolve(c, pmap, clauses)
        if c.spurious or c.resolution == UNRESOLVED:
            continue
        loser = c.b if c.winner == c.a else c.a
        winner = pmap[c.winner]
        # a prohibition always suppresses a conflicting obligation/permission
        if winner.deontic == "prohibition":
            dropped.add(loser)

    # exact duplicates: keep the earliest instrument's copy
    seen: Dict[Any, Proposition] = {}
    for p in sorted(props, key=lambda q: (_rank(q, clauses), _date(q, clauses))):
        sig = p.signature()
        if sig in seen:
            dropped.add(p.pid)
        else:
            seen[sig] = p

    kept = [p for p in props if p.pid not in dropped]
    return kept, conflicts


def summary(conflicts: List[Conflict]) -> Dict[str, Any]:
    n = len(conflicts)
    res: Dict[str, int] = {}
    for c in conflicts:
        res[c.resolution] = res.get(c.resolution, 0) + 1
    return dict(
        candidates=n,
        resolved_by_rank=res.get(RANK, 0),
        resolved_by_posterior=res.get(POSTERIOR, 0),
        resolved_by_specialis=res.get(SPECIALIS, 0),
        rejected_spurious=res.get(SPURIOUS, 0),
        unresolved=res.get(UNRESOLVED, 0),
        by_kind={k: sum(1 for c in conflicts if c.kind == k)
                 for k in ("modality", "threshold", "duplication")},
    )

