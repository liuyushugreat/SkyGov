"""SkyGov-C: provenance-preserving compilation of institutional rules into
auditable tiered governance for urban low-altitude safety.

The package is organised as a pipeline:

    corpus      ->  clause-level institutional records with provenance
    reg2rule    ->  clause  -> normalised proposition -> SHACL/SPARQL rule
    classification ->  constraint fingerprint -> governance tier
    allocation  ->  budget-constrained multi-objective capability allocation
    protocol    ->  cross-jurisdiction handover state machine + model checking
    calibration ->  Bayesian + conformal + martingale closed loop
    audit       ->  explanation-quality metrics (RAR/LEC/UCR/TCC)
    execution   ->  tier-aware agent-based execution layer
    bench       ->  UAMGov-Bench artefact builder

Everything reported in the paper is reproducible from ``python -m
skygovc.experiments.run_all``.
"""

__version__ = "1.0.0"
