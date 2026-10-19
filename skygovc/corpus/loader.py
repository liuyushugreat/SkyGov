"""Institutional clause corpus with provenance.

Three blocks
------------
``dev``
    ``records_national`` + ``records_local``: the development set on which the
    frame grammar and the failure-mode procedure were authored.
``heldout``
    ``records_heldout``: authored after the grammar was frozen and evaluated in
    a single run (Section 6.4).  Never used to tune the grammar.
``external``
    ``E*`` records inside the dev module: EU/FAA/JARUS comparison standards,
    excluded from the domestic rule base.

Public API
----------
``loader.all_clauses()`` / ``core_clauses()`` / ``heldout_clauses()``
``loader.gold_propositions()`` / ``gold_heldout()``
``loader.gold_compilability()``
``loader.corpus_summary()``
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..core import Clause, Condition, Proposition
from . import gold as G
from . import records_heldout as RH
from . import records_local as RL
from . import records_national as RN

_DOCS: Dict[str, Dict[str, Any]] = {**RN.DOCS, **RL.DOCS, **RH.DOCS}
_RAW: List[Dict[str, Any]] = (
    [{**r, "split": "dev"} for r in RN.CLAUSES]
    + [{**r, "split": "dev"} for r in RL.CLAUSES]
    + [{**r, "split": "heldout"} for r in RH.CLAUSES]
)

CORE_DOC_IDS: Tuple[str, ...] = tuple(f"D{i:02d}" for i in range(1, 17))
EXTERNAL_DOC_IDS: Tuple[str, ...] = tuple(d for d in _DOCS if d.startswith("E"))
HELDOUT_DOC_IDS: Tuple[str, ...] = tuple(d for d in _DOCS if d.startswith("H"))


def _make_clause(raw: Dict[str, Any]) -> Clause:
    doc = _DOCS[raw["doc_id"]]
    return Clause(
        cid=raw["cid"], doc_id=raw["doc_id"], doc_title=doc["title"],
        issuer=doc["issuer"], level=doc["level"],
        instrument_date=doc["instrument_date"],
        effective_date=doc["effective_date"], provision=raw["provision"],
        text=raw["text"], verbatim=False, source_url=None,
        source_note=raw.get("note") or doc.get("source_note", ""),
        jurisdiction=doc["jurisdiction"], split=raw["split"],
    )


CLAUSES: List[Clause] = [_make_clause(r) for r in _RAW]
_BY_CID: Dict[str, Clause] = {c.cid: c for c in CLAUSES}


def all_clauses() -> List[Clause]:
    return list(CLAUSES)


def core_clauses() -> List[Clause]:
    """The sixteen core instruments (dev block, domestic)."""
    return [c for c in CLAUSES if c.doc_id in CORE_DOC_IDS]


def dev_clauses() -> List[Clause]:
    return [c for c in CLAUSES if c.split == "dev"]


def heldout_clauses() -> List[Clause]:
    return [c for c in CLAUSES if c.split == "heldout"]


def external_clauses() -> List[Clause]:
    return [c for c in CLAUSES if c.doc_id in EXTERNAL_DOC_IDS]


def clause(cid: str) -> Clause:
    return _BY_CID[cid]


def documents() -> Dict[str, Dict[str, Any]]:
    return dict(_DOCS)


def doc_stats() -> List[Dict[str, Any]]:
    out = []
    for did, d in sorted(_DOCS.items()):
        cs = [c for c in CLAUSES if c.doc_id == did]
        out.append(dict(doc_id=did, title=d["title"], level=d["level"],
                        jurisdiction=d["jurisdiction"], n_clauses=len(cs),
                        instrument_date=d["instrument_date"],
                        effective_date=d["effective_date"],
                        split=cs[0].split if cs else ""))
    return out


# --------------------------------------------------------------------------
# Gold standard
# --------------------------------------------------------------------------
GOLD_EQUIVALENCE: Dict[str, List[str]] = {}


def _build_gold(records: List[Dict[str, Any]], dedupe: bool) -> List[Proposition]:
    out: List[Proposition] = []
    seen: Dict[Any, str] = {}
    for g in records:
        if not g["compilable"]:
            continue
        for k, (deontic, subject, action, conds) in enumerate(g.get("props", []), start=1):
            p = Proposition(pid=f"{g['cid']}-G{k}", cid=g["cid"], deontic=deontic,
                            subject=subject, action=action,
                            conditions=[Condition(*c) for c in conds],
                            extractor="gold")
            if dedupe:
                sig = p.signature()
                if sig in seen:
                    GOLD_EQUIVALENCE.setdefault(seen[sig], []).append(g["cid"])
                    continue
                seen[sig] = g["cid"]
            out.append(p)
    return out


def gold_propositions(dedupe: bool = True) -> List[Proposition]:
    GOLD_EQUIVALENCE.clear()
    return _build_gold(G.GOLD, dedupe)


def gold_heldout(dedupe: bool = False) -> List[Proposition]:
    return _build_gold(RH.GOLD_HELDOUT, dedupe)


def gold_compilability() -> Dict[str, Tuple[bool, Optional[str]]]:
    d = {g["cid"]: (g["compilable"], g.get("reason")) for g in G.GOLD}
    d.update({g["cid"]: (g["compilable"], g.get("reason")) for g in RH.GOLD_HELDOUT})
    return d


def failure_taxonomy() -> Dict[str, str]:
    return dict(G.FAILURE_MODES)


def corpus_summary() -> Dict[str, Any]:
    s_g = G.summary()
    n_h = len(RH.GOLD_HELDOUT)
    nc_h = sum(1 for g in RH.GOLD_HELDOUT if g["compilable"])
    modes_h: Dict[str, int] = {}
    for g in RH.GOLD_HELDOUT:
        if not g["compilable"]:
            modes_h[g["reason"]] = modes_h.get(g["reason"], 0) + 1
    gold_propositions(dedupe=True)
    return dict(
        n_clauses=len(CLAUSES),
        n_dev_clauses=len(dev_clauses()),
        n_core_clauses=len(core_clauses()),
        n_heldout_clauses=len(heldout_clauses()),
        n_external_clauses=len(external_clauses()),
        n_documents=len(_DOCS),
        n_core_documents=len(CORE_DOC_IDS),
        n_heldout_documents=len(HELDOUT_DOC_IDS),
        dev_records=s_g["records"], dev_compilable=s_g["compilable"],
        dev_non_compilable=s_g["non_compilable"],
        dev_gold_propositions=s_g["gold_propositions"],
        dev_distinct_gold_props=len(gold_propositions(dedupe=True)),
        dev_redundant_cids=sum(len(v) for v in GOLD_EQUIVALENCE.values()),
        heldout_records=n_h, heldout_compilable=nc_h,
        heldout_non_compilable=n_h - nc_h,
        heldout_gold_propositions=sum(len(g.get("props", []))
                                      for g in RH.GOLD_HELDOUT),
        heldout_distinct_gold_props=len(gold_heldout(dedupe=True)),
        failure_modes_dev=s_g["failure_modes"],
        failure_modes_heldout=modes_h,
    )


def redundancy_pairs() -> Dict[str, List[str]]:
    gold_propositions(dedupe=True)
    return {k: list(v) for k, v in GOLD_EQUIVALENCE.items()}


__all__ = [
    "CLAUSES", "CORE_DOC_IDS", "EXTERNAL_DOC_IDS", "HELDOUT_DOC_IDS",
    "GOLD_EQUIVALENCE", "all_clauses", "core_clauses", "dev_clauses",
    "heldout_clauses", "external_clauses", "clause", "documents", "doc_stats",
    "gold_propositions", "gold_heldout", "gold_compilability",
    "failure_taxonomy", "corpus_summary", "redundancy_pairs",
]
