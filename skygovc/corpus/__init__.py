"""Institutional clause corpus with provenance (dev / heldout / external)."""
from . import gold, loader, records_heldout, records_local, records_national  # noqa: F401
from .loader import (  # noqa: F401
    CLAUSES,
    CORE_DOC_IDS,
    EXTERNAL_DOC_IDS,
    GOLD_EQUIVALENCE,
    HELDOUT_DOC_IDS,
    all_clauses,
    clause,
    core_clauses,
    corpus_summary,
    dev_clauses,
    doc_stats,
    documents,
    external_clauses,
    failure_taxonomy,
    gold_compilability,
    gold_heldout,
    gold_propositions,
    heldout_clauses,
    redundancy_pairs,
)

__all__ = [
    "gold", "loader", "records_heldout", "records_local", "records_national",
    "CLAUSES", "CORE_DOC_IDS", "EXTERNAL_DOC_IDS", "HELDOUT_DOC_IDS",
    "GOLD_EQUIVALENCE", "all_clauses", "clause", "core_clauses",
    "corpus_summary", "dev_clauses", "doc_stats", "documents",
    "external_clauses", "failure_taxonomy", "gold_compilability",
    "gold_heldout", "gold_propositions", "heldout_clauses", "redundancy_pairs",
]
