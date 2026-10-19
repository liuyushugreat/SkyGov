"""The decision engine: forward chaining over the compiled rule base.

Two entry points matter.

:meth:`DecisionEngine.decide`
    Produces a :class:`~skygovc.core.Verdict` with a provenance chain, used
    for the auditable single-decision case (and for the SPARQL cross-check).

:meth:`DecisionEngine.induce_regime`
    Collapses the symbolic rule base into the tier-level parameters that the
    agent-based execution layer consumes.  This is the bridge that makes the
    simulated governance regimes *derived* rather than hand-set.

Evaluation semantics
--------------------
* A **prohibition** fires when its antecedent matches.  Any fired prohibition
  blocks the request.
* An **authorisation obligation** (``apply_flight_activity``) fires when its
  antecedent matches; the request is then allowed only if it carries a
  recorded approval.
* Any other **obligation** fires as a duty and is recorded, but does not block.
* A **permission** never blocks; it contributes to the audit trail.

Conditions whose fact is missing from the request do not fire the rule and are
recorded as ``unresolved`` so that the verdict is never silently optimistic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core import CAPABILITIES, FlightRequest, Proposition, Verdict

AUTHORISATION_ACTIONS = frozenset({"apply_flight_activity"})


def _eval_condition(cond, facts: Dict[str, Any]) -> Tuple[Optional[bool], str]:
    """Return (truth, note); ``None`` means the fact was unavailable."""
    if cond.name not in facts:
        return None, f"missing fact {cond.name}"
    v = facts[cond.name]
    try:
        if cond.op == "in":
            return (v in tuple(cond.value)), ""
        if cond.op == "not_in":
            return (v not in tuple(cond.value)), ""
        if cond.op == "==":
            return (v == cond.value), ""
        if cond.op == "!=":
            return (v != cond.value), ""
        if cond.op == "<=":
            return (v <= cond.value), ""
        if cond.op == ">=":
            return (v >= cond.value), ""
        if cond.op == "<":
            return (v < cond.value), ""
        if cond.op == ">":
            return (v > cond.value), ""
    except TypeError:
        return None, f"type mismatch on {cond.name}"
    return None, f"unsupported operator {cond.op}"


@dataclass
class DecisionEngine:
    rules: Sequence[Proposition]
    name: str = "SkyGov-C"

    # ---------------------------------------------------------------- decide
    def fires(self, p: Proposition, facts: Dict[str, Any]) -> Tuple[bool, List[str]]:
        unresolved: List[str] = []
        for c in p.conditions:
            truth, note = _eval_condition(c, facts)
            if truth is None:
                unresolved.append(note)
                return False, unresolved
            if not truth:
                return False, unresolved
        return True, unresolved

    def decide(self, req: FlightRequest, approved: bool = False,
               tier: str = "") -> Verdict:
        facts = req.facts()
        fired: List[Proposition] = []
        chain: List[str] = []
        reasons: List[str] = []
        required_caps: List[str] = []

        blocked_by: List[Proposition] = []
        auth_required = False
        duties: List[Proposition] = []

        for p in self.rules:
            ok, unresolved = self.fires(p, facts)
            if not ok:
                continue
            fired.append(p)
            chain.append(p.cid)
            if p.deontic == "prohibition":
                blocked_by.append(p)
                reasons.append(f"prohibition {p.pid} [{p.cid}]: {p.action}")
            elif p.deontic == "obligation":
                if p.action in AUTHORISATION_ACTIONS:
                    auth_required = True
                    reasons.append(f"authorisation required {p.pid} [{p.cid}]")
                else:
                    duties.append(p)
                    reasons.append(f"duty {p.pid} [{p.cid}]: {p.action}")
            elif p.deontic == "permission":
                reasons.append(f"permission {p.pid} [{p.cid}]")

        allowed = not blocked_by and (not auth_required or approved)

        if blocked_by:
            required_caps.extend(["sensing", "emergency"])
        if auth_required:
            required_caps.extend(["rule", "service"])
        for d in duties:
            if d.action == "share_operational_data":
                required_caps.append("data")
            if d.action == "report_incident":
                required_caps.append("emergency")
            if d.action in ("hold_licence", "register_uas"):
                required_caps.append("rule")

        seen = set()
        caps = [c for c in required_caps if not (c in seen or seen.add(c))]
        return Verdict(
            rid=req.rid,
            allowed=bool(allowed),
            tier=tier,
            triggered=[p.pid for p in fired],
            chain=sorted(set(chain)),
            reasons=reasons,
            required_capabilities=[c for c in caps if c in CAPABILITIES],
        )

    # ------------------------------------------------------- regime induction
    def requires_application(self, req: FlightRequest) -> bool:
        facts = req.facts()
        for p in self.rules:
            if p.deontic != "obligation" or p.action not in AUTHORISATION_ACTIONS:
                continue
            ok, _ = self.fires(p, facts)
            if ok:
                return True
        return False

    def is_prohibited(self, req: FlightRequest) -> bool:
        facts = req.facts()
        for p in self.rules:
            if p.deontic != "prohibition":
                continue
            ok, _ = self.fires(p, facts)
            if ok:
                return True
        return False

    def induce_regime(self, sampler: Callable[[np.random.Generator, int], List[FlightRequest]],
                      n: int = 4000, seed: int = 0) -> Dict[str, Any]:
        """Monte-Carlo estimate of the tier-level parameters implied by the
        rule base.  Returns the probability that a request of each risk class
        needs an authorisation, plus the baseline prohibition rate."""
        rng = np.random.default_rng(seed)
        reqs = sampler(rng, n)
        need = np.array([self.requires_application(r) for r in reqs])
        prohib = np.array([self.is_prohibited(r) for r in reqs])
        risk = np.array([r.risk_class for r in reqs])
        out: Dict[str, Any] = dict(n=n, prohibition_rate=float(prohib.mean()))
        for cls in ("low", "medium", "high"):
            m = risk == cls
            out[f"require_application_{cls}"] = (
                float(need[m].mean()) if m.any() else 0.0)
        return out


CONTROLLED_LETTERS = ("A", "B", "C", "D", "E")


def default_sampler(jurisdiction: str, airspace_class: str = "W",
                    cross_share: float = 0.18) -> Callable:
    """Build a request sampler for one tier.

    ``airspace_class`` may be a concrete class (``"W"``, ``"G"``, ``"A"``...)
    or ``"mixed"``, which draws the share of operations that fall inside
    controlled airspace from that jurisdiction's airspace regime.
    """
    def _s(rng: np.random.Generator, n: int) -> List[FlightRequest]:
        out: List[FlightRequest] = []
        risk = rng.choice(3, size=n, p=(0.6, 0.3, 0.1))
        cats = rng.choice(5, size=n, p=(0.30, 0.40, 0.25, 0.04, 0.01))
        cat_names = ("micro", "light", "small", "medium", "large")
        if airspace_class == "mixed":
            controlled = rng.random(n) < 0.25
            letters = rng.choice(CONTROLLED_LETTERS, size=n)
            acs = [letters[i] if controlled[i]
                   else ("G" if rng.random() < 0.5 else "W") for i in range(n)]
        else:
            acs = [airspace_class] * n
        for i in range(n):
            rc = ("low", "medium", "high")[int(risk[i])]
            ac = acs[i]
            out.append(FlightRequest(
                rid=f"r{i}",
                origin_jurisdiction=jurisdiction,
                dest_jurisdiction=jurisdiction,
                risk_class=rc,
                airspace_class=ac,
                altitude_agl_m=float(rng.uniform(20, 120)),
                uas_category=cat_names[int(cats[i])],
                purpose=("for_hire" if rng.random() < 0.35 else "logistics"),
                cross_border=bool(rng.random() < cross_share * 0.2),
                night_operation=bool(rng.random() < 0.10),
                bvlos=bool(rng.random() < 0.15),
                within_airport_buffer=bool(rng.random() < 0.05),
                over_populated_area=bool(rng.random() < 0.35),
            ))
        return out
    return _s

