"""Reg2Rule: from institutional text to an executable, auditable rule base."""
from . import align, emit, engine, frames, lexicon, merge, pipeline, router, score  # noqa: F401
from .emit import (  # noqa: F401
    ValidationReport,
    rulebase_ttl,
    shapes_ttl,
    sparql_queries,
    to_sparql_ask,
    validate_rulebase,
)
from .engine import DecisionEngine, default_sampler  # noqa: F401
from .frames import STRATEGIES, Extraction, extract_all, extract_clause  # noqa: F401
from .merge import Conflict  # noqa: F401
from .pipeline import (  # noqa: F401
    PipelineResult,
    compile_rules,
    evaluate_with_oracle,
    gold_props_by_cid,
)
from .router import (  # noqa: F401
    FEATURES,
    RouterReport,
    RoutingPoint,
    breakeven_k,
    featurize,
    feature_matrix,
    per_cid_gold,
    routing_curve,
    train_router,
)
from .score import (  # noqa: F401
    CompilabilityScores,
    PropScores,
    rulebase_scores,
    score_compilability,
    score_propositions,
)

__all__ = [
    "align", "emit", "engine", "frames", "lexicon", "merge", "pipeline",
    "router", "score", "ValidationReport", "rulebase_ttl", "shapes_ttl",
    "sparql_queries", "to_sparql_ask", "validate_rulebase", "DecisionEngine",
    "default_sampler", "STRATEGIES", "Extraction", "extract_all",
    "extract_clause", "Conflict", "PipelineResult", "compile_rules",
    "evaluate_with_oracle", "gold_props_by_cid", "FEATURES", "RouterReport",
    "RoutingPoint", "breakeven_k", "featurize", "feature_matrix",
    "per_cid_gold", "routing_curve", "train_router", "CompilabilityScores",
    "PropScores", "rulebase_scores", "score_compilability",
    "score_propositions",
]
