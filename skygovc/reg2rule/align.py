"""Ontology-driven alignment and repair of extracted propositions.

The extractor produces *surface* conditions (``airspace_class != controlled``).
The ontology says what the surface term denotes: ``G`` and ``W`` are the only
non-controlled classes, and ``A``--``E`` are controlled.  Alignment therefore
turns a negation over an open set into a closed enumeration, which is what
makes the rule executable by a SHACL/SPARQL engine.

The repair step is also where the compiler refuses to guess: if a surface term
has no ontology match the proposition keeps an ``unresolved`` flag and is
routed to adjudication rather than silently coerced.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Tuple

from ..core import Condition, Proposition

#: Airspace taxonomy consumed from D04 (国家空域基础分类方法).
CONTROLLED_CLASSES: Tuple[str, ...] = ("A", "B", "C", "D", "E")
NON_CONTROLLED_CLASSES: Tuple[str, ...] = ("G", "W")
ALL_CLASSES: Tuple[str, ...] = CONTROLLED_CLASSES + NON_CONTROLLED_CLASSES

#: Repair rules: surface condition -> ontology-grounded condition.
REPAIRS: List[Tuple[Tuple[str, str, Any], Tuple[str, str, Any], str]] = [
    (("airspace_class", "!=", "controlled"),
     ("airspace_class", "in", NON_CONTROLLED_CLASSES),
     "D04-C001/D04-C002: 非管制空域仅含 G、W 两类"),
    (("airspace_class", "==", "permitted"),
     ("airspace_class", "in", NON_CONTROLLED_CLASSES),
     "D01-C003+D04-C002: 适飞空域限定于真高120米以下非管制空域"),
    (("airspace_class", "==", "controlled"),
     ("airspace_class", "in", CONTROLLED_CLASSES),
     "D04-C003: 管制空域为 A、B、C、D、E 五类"),
]

_REPAIR_INDEX: Dict[Tuple[str, str, str], Tuple[Tuple[str, str, Any], str]] = {
    (a, b, str(c)): (d, why) for (a, b, c), d, why in REPAIRS
}


def repair_condition(cond: Condition) -> Tuple[Condition, str | None]:
    key = (cond.name, cond.op, str(cond.value))
    if key in _REPAIR_INDEX:
        (name, op, val), why = _REPAIR_INDEX[key]
        return Condition(name, op, val), why
    return cond, None


#: Predicate the term ``适飞空域`` entails: D01-C003 fixes the altitude ceiling and
#: D04-C002 fixes the non-controlled classes, so a permitted-airspace condition
#: cannot be satisfied above 120 m AGL.
ALTITUDE_CEILING_M = 120


def align(p: Proposition) -> Tuple[Proposition, List[str]]:
    """Return the aligned proposition and the list of applied repair notes."""
    notes: List[str] = []
    new_conds: List[Condition] = []
    for c in p.conditions:
        nc, why = repair_condition(c)
        if why:
            notes.append(why)
        new_conds.append(nc)
    # entailment: permitted airspace => altitude ceiling
    if any(c.name == "airspace_class" and c.op == "in"
           and tuple(c.value) == NON_CONTROLLED_CLASSES for c in new_conds):
        if not any(c.name == "altitude_agl_m" for c in new_conds):
            new_conds.append(Condition("altitude_agl_m", "<=", ALTITUDE_CEILING_M))
            notes.append("D01-C003: 适飞空域蕴含真高120米以下")
    # de-duplicate after repair
    seen = set()
    dedup: List[Condition] = []
    for c in new_conds:
        if c.key() not in seen:
            seen.add(c.key())
            dedup.append(c)
    out = replace(p, conditions=dedup)
    return out, notes


def align_all(props: List[Proposition]) -> Tuple[List[Proposition], int]:
    out: List[Proposition] = []
    n_repaired = 0
    for p in props:
        np_, notes = align(p)
        if notes:
            n_repaired += 1
        out.append(np_)
    return out, n_repaired


def is_executable(p: Proposition, subject_vocab, action_vocab, cond_vocab,
                  allowed_values: Dict[str, Any] | None = None) -> Tuple[bool, str]:
    """Static executability check of one proposition against the schema."""
    if p.subject not in subject_vocab:
        return False, f"unknown subject '{p.subject}'"
    if p.action not in action_vocab:
        return False, f"unknown action '{p.action}'"
    for c in p.conditions:
        if c.name not in cond_vocab:
            return False, f"unknown condition '{c.name}'"
        if allowed_values and c.name in allowed_values:
            allowed = allowed_values[c.name]
            vals = c.value if isinstance(c.value, (tuple, list)) else (c.value,)
            for v in vals:
                if v not in allowed:
                    return False, f"value '{v}' outside ontology for '{c.name}'"
    if p.deontic not in ("obligation", "prohibition", "permission", "procedure"):
        return False, f"unknown deontic '{p.deontic}'"
    return True, "ok"


def value_domains() -> Dict[str, Any]:
    return {
        "airspace_class": set(ALL_CLASSES) | {"controlled", "permitted"},
        "uas_category": {"micro", "light", "small", "medium", "large"},
        "risk_class": {"low", "medium", "high"},
        "purpose": {"for_hire", "logistics", "survey", "emergency", "recreation"},
    }

