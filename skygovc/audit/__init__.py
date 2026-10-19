"""Explanation generation and audit metrics (RAR / LEC / UCR / TCC).

The audit layer exists because a verdict without a checkable explanation is not
governance.  Four metrics are defined over the *claim set* of a generated
explanation, each claim being tied to a clause identifier from the provenance
chain:

``RAR``  rule alignment rate — supported claims / total claims
``LEC``  legal entity coverage — fraction of chain clause ids that appear
``UCR``  unsupported claim rate — 1 - RAR, reported separately because a claim
         can be unsupported yet not contradicted
``TCC``  tier-capability consistency — fraction of the tier's floor
         capabilities that the verdict actually exercises

Validation
----------
The metrics are validated as *detectors*, not as system performance: three
explanation generators are constructed with known defects (a grounded
generator, a generator that silently drops chain elements, and one that adds
plausible but unsupported claims) and the metrics must separate them.  This is
stated as a detector validation in the paper.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core import CAPABILITIES, TIERS, Verdict, to_jsonable


#: Minimum capability floor per tier, mirrored from the allocation model so that
#: TCC can be computed without importing numpy.
TIER_FLOOR: Dict[str, Tuple[str, ...]] = {
    "capital_core": ("rule", "spatial", "sensing", "emergency", "data"),
    "dense_hub": ("spatial", "sensing", "emergency"),
    "market_driven": ("service", "data"),
    "mountainous": ("spatial", "emergency"),
    "border_coastal": ("rule", "spatial", "sensing", "emergency", "data"),
    "sparse_resource": ("rule", "service", "data"),
}


@dataclass
class Claim:
    """One atomic assertion in an explanation."""

    kind: str              # trigger | duty | prohibition | permission | capability
    text: str
    support: Optional[str]  # clause id that entails the claim, or None

    def to_dict(self) -> Dict[str, Any]:
        return dict(kind=self.kind, text=self.text, support=self.support)


_FILLER = [
    ("capability", "该决定与本市治理能力水平相适应"),
    ("permission", "运营人可继续按既定方案实施任务"),
    ("trigger", "本情形属于低风险常态运行"),
]


def explain(v: Verdict, style: str = "grounded", rng: Optional[random.Random] = None
            ) -> List[Claim]:
    """Generate the claim set of an explanation.

    ``grounded``
        One claim per fired rule, each carrying the clause id from the chain.
        This is what the system emits.
    ``truncated``
        Drops every second claim's support, emulating an explanation generator
        that paraphrases without carrying provenance.
    ``augmented``
        Emulates unsupported elaboration: the grounded claims plus plausible
        statements with no provenance.
    """
    rng = rng or random.Random(0)
    claims: List[Claim] = []
    chain = list(v.chain)
    for k, (reason, rule_id) in enumerate(zip(v.reasons, v.triggered)):
        cid = chain[k % len(chain)] if chain else None
        kind = "trigger"
        if reason.startswith("prohibition"):
            kind = "prohibition"
        elif reason.startswith("duty"):
            kind = "duty"
        elif reason.startswith("permission"):
            kind = "permission"
        elif reason.startswith("authorisation"):
            kind = "duty"
        claims.append(Claim(kind, reason, cid))
    for cap in v.required_capabilities:
        claims.append(Claim("capability", f"需要启用能力族 {cap}", chain[-1] if chain else None))

    if style == "grounded":
        return claims
    if style == "truncated":
        return [Claim(c.kind, c.text, None if i % 2 else c.support)
                for i, c in enumerate(claims)]
    if style == "augmented":
        extra = [Claim(k, t, None) for k, t in _FILLER]
        return claims + extra
    raise ValueError(f"unknown style {style}")


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
@dataclass
class AuditReport:
    rar: float
    lec: float
    ucr: float
    tcc: float
    n_claims: int
    n_supported: int

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)


def audit(claims: Sequence[Claim], v: Verdict, tier: str) -> AuditReport:
    n = len(claims)
    supported = sum(1 for c in claims if c.support is not None)
    rar = supported / n if n else 0.0
    chain = set(v.chain)
    mentioned = {c.support for c in claims if c.support in chain}
    lec = len(mentioned) / len(chain) if chain else 1.0
    floor = set(TIER_FLOOR.get(tier, ()))
    caps = set(v.required_capabilities)
    tcc = len(floor & caps) / len(floor) if floor else 1.0
    return AuditReport(rar=rar, lec=lec, ucr=1.0 - rar, tcc=tcc,
                       n_claims=n, n_supported=supported)


def validate_detector(n_verdicts: int = 200, seed: int = 0) -> Dict[str, Any]:
    """Construct verdicts across tiers and evaluate the three generators.

    A detector is useful if grounded explanations score strictly higher on RAR
    than truncated or augmented ones, with non-overlapping ranges.
    """
    from ..reg2rule.engine import DecisionEngine
    from ..reg2rule.pipeline import compile_rules
    from ..core import FlightRequest

    res = compile_rules("S3_repair", validate=False)
    engine = DecisionEngine(res.rulebase)
    rng = random.Random(seed)
    styles = ("grounded", "truncated", "augmented")
    acc: Dict[str, List[AuditReport]] = {s: [] for s in styles}

    tiers = list(TIERS)
    for i in range(n_verdicts):
        tier = tiers[i % len(tiers)]
        req = FlightRequest(
            rid=f"a{i}", origin_jurisdiction="X", dest_jurisdiction="Y",
            risk_class=("low", "medium", "high")[i % 3],
            airspace_class=("W", "G", "controlled")[i % 3],
            altitude_agl_m=float(60 + (i * 7) % 80),
            uas_category=("micro", "light", "small", "medium")[i % 4],
            purpose=("logistics", "for_hire", "survey")[i % 3],
            night_operation=bool(i % 5 == 0),
            bvlos=bool(i % 7 == 0),
            within_airport_buffer=bool(i % 11 == 0),
        )
        v = engine.decide(req, approved=bool(i % 2), tier=tier)
        for s in styles:
            claims = explain(v, s, random.Random(i))
            acc[s].append(audit(claims, v, tier))

    out: Dict[str, Any] = {}
    for s in styles:
        rs = acc[s]
        out[s] = dict(
            rar_mean=float(sum(r.rar for r in rs) / len(rs)),
            rar_min=float(min(r.rar for r in rs)),
            rar_max=float(max(r.rar for r in rs)),
            lec_mean=float(sum(r.lec for r in rs) / len(rs)),
            tcc_mean=float(sum(r.tcc for r in rs) / len(rs)),
            n_claims_mean=float(sum(r.n_claims for r in rs) / len(rs)),
        )
    out["separation"] = dict(
        grounded_minus_truncated=out["grounded"]["rar_mean"] - out["truncated"]["rar_mean"],
        grounded_minus_augmented=out["grounded"]["rar_mean"] - out["augmented"]["rar_mean"],
        ranges_disjoint=(
            out["grounded"]["rar_min"] > max(out["truncated"]["rar_max"],
                                             out["augmented"]["rar_max"])
        ),
    )
    return out
