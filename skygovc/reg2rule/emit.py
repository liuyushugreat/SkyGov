"""Emission of the executable rule base: RDF/Turtle, SPARQL ASK, SHACL shapes.

Three artefacts are produced from the merged proposition set:

``rulebase.ttl``
    The rule base as RDF, one ``gv:ExecutableRule`` per proposition, each
    carrying its clause-level provenance (``gv:derivedFrom``) and the emitted
    SPARQL text.
``shapes.ttl``
    SHACL shapes for mechanical well-formedness (``ExecutableRuleShape``) and
    for modality coherence (``NoModalityCollisionShape``).
``sparql/*.rq``
    One SPARQL ``ASK`` query per rule, executable against a request graph.

The prose in the paper claims that merging changes whether the rule base
*validates*; ``validate_rulebase`` below is the function that establishes it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from ..core import Proposition
from .merge import object_signature

GV = "urn:skygovc:"
REQ = "urn:skygovc:req:"

DEONTIC_IRI = {
    "obligation": "gv:Obligation",
    "prohibition": "gv:Prohibition",
    "permission": "gv:Permission",
    "procedure": "gv:Procedure",
}

#: condition name -> RDF property local name on a request node
COND_PROP = {
    "airspace_class": "airspaceClass",
    "altitude_agl_m": "altitudeAglM",
    "uas_category": "uasCategory",
    "mass_kg": "massKg",
    "payload_kg": "payloadKg",
    "purpose": "purpose",
    "over_populated_area": "overPopulatedArea",
    "within_airport_buffer": "withinAirportBuffer",
    "cross_border": "crossBorder",
    "night_operation": "nightOperation",
    "bvlos": "bvlos",
    "operator_licence": "operatorLicence",
    "registered": "registered",
    "info_reported": "infoReported",
    "risk_class": "riskClass",
}

_XSD = {
    bool: "xsd:boolean",
    int: "xsd:integer",
    float: "xsd:decimal",
    str: "xsd:string",
}


def rule_iri(p: Proposition) -> str:
    return f"{GV}rule:{p.pid}"


def clause_iri(cid: str) -> str:
    return f"{GV}clause:{cid}"


def _ttl_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    return '"' + str(v) + '"'


def to_sparql_ask(p: Proposition) -> str:
    """Emit an executable ASK query that fires when the rule's antecedent
    matches the request graph bound to ``?r``."""
    triple_lines: List[str] = []
    filters: List[str] = []
    for c in p.conditions:
        prop = COND_PROP.get(c.name)
        if prop is None:
            continue
        var = "?" + prop
        triple_lines.append(f"  ?r g:req/{prop} {var} .")
        if c.op in ("in", "not_in"):
            vals = c.value if isinstance(c.value, (tuple, list)) else (c.value,)
            joined = ", ".join(_ttl_value(v) for v in vals)
            kw = "IN" if c.op == "in" else "NOT IN"
            filters.append(f"  FILTER ({var} {kw} ({joined}))")
        elif c.op in ("==", "!=", "<=", ">=", "<", ">"):
            filters.append(f"  FILTER ({var} {c.op} {_ttl_value(c.value)})")

    head = (
        "PREFIX g: <urn:skygovc:>\n"
        "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n"
        f"# rule {p.pid}  <- clause {p.cid}  ({p.deontic} {p.subject} {p.action})\n"
        "ASK {\n"
        "  ?r a g:FlightRequest .\n"
        + ("\n".join(triple_lines) + "\n" if triple_lines else "")
        + ("\n".join(filters) + "\n" if filters else "")
        + "}\n"
    )
    return head


def to_rule_ttl(p: Proposition) -> str:
    conds = []
    for c in p.conditions:
        conds.append(
            "  gv:condition [ gv:name \"%s\" ; gv:op \"%s\" ; gv:value %s ] ;\n"
            % (c.name, c.op, _ttl_value(list(c.value) if isinstance(c.value, tuple) else c.value))
        )
    sparql = to_sparql_ask(p).replace('"""', '\\"\\"\\"')
    body = (
        f"<{rule_iri(p)}> a gv:ExecutableRule ;\n"
        f"  gv:derivedFrom <{clause_iri(p.cid)}> ;\n"
        f"  gv:deontic {DEONTIC_IRI.get(p.deontic, 'gv:Obligation')} ;\n"
        f"  gv:subject \"{p.subject}\" ;\n"
        f"  gv:action \"{p.action}\" ;\n"
        f"  gv:confidence {p.confidence} ;\n"
        f"  gv:objectTag \"{object_signature(p)}\" ;\n"
        + "".join(conds)
        + f"  gv:sparql \"\"\"{sparql}\"\"\" .\n"
    )
    return body


PREFIXES = """@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix sh:   <http://www.w3.org/ns/shacl#> .
@prefix gv:   <urn:skygovc:v:> .
@prefix g:    <urn:skygovc:> .
"""

VOCAB = """
gv:ExecutableRule a owl:Class .
gv:Clause a owl:Class .
gv:Obligation a owl:Class .
gv:Prohibition a owl:Class .
gv:Permission a owl:Class .
gv:Procedure a owl:Class .
gv:derivedFrom a owl:ObjectProperty .
gv:deontic a owl:ObjectProperty .
gv:subject a owl:DatatypeProperty .
gv:action a owl:DatatypeProperty .
gv:condition a owl:ObjectProperty .
gv:name a owl:DatatypeProperty .
gv:op a owl:DatatypeProperty .
gv:value a owl:DatatypeProperty .
gv:sparql a owl:DatatypeProperty .
gv:confidence a owl:DatatypeProperty .
"""

SHAPES = """
gv:ExecutableRuleShape a sh:NodeShape ;
  sh:targetClass gv:ExecutableRule ;
  sh:property [ sh:path gv:derivedFrom ; sh:minCount 1 ;
                sh:message "every rule must carry clause-level provenance" ] ;
  sh:property [ sh:path gv:deontic ; sh:minCount 1 ] ;
  sh:property [ sh:path gv:subject ; sh:minCount 1 ;
                sh:in ( "operator" "pilot" "manufacturer" "owner"
                        "local_government" "airspace_authority" "public_security"
                        "service_provider" "legislature" "central_government"
                        "service_platform" ) ] ;
  sh:property [ sh:path gv:action ; sh:minCount 1 ] ;
  sh:property [ sh:path gv:sparql ; sh:minCount 1 ] .

gv:NoModalityCollisionShape a sh:NodeShape ;
  sh:targetClass gv:ExecutableRule ;
  sh:sparql [
    sh:message "two rules with the same subject/action carry opposite deontic modalities" ;
    sh:prefixes g: ;
    sh:select \"\"\"
      SELECT $this WHERE {
        $this gv:subject ?s ; gv:action ?a ; gv:deontic ?d .
        ?other gv:subject ?s ; gv:action ?a ; gv:deontic ?d2 .
        ?this gv:objectTag ?tag .
        ?other gv:objectTag ?tag2 .
        FILTER (?other != $this)
        FILTER (?tag = ?tag2)
        FILTER ( ( ?d = gv:Obligation && ?d2 = gv:Prohibition ) ||
                ( ?d = gv:Prohibition && ?d2 = gv:Obligation ) )
      } LIMIT 1
    \"\"\" ;
  ] .
"""


def rulebase_ttl(props: Sequence[Proposition]) -> str:
    parts = [PREFIXES, VOCAB]
    parts.extend(to_rule_ttl(p) for p in props)
    return "\n".join(parts)


def shapes_ttl() -> str:
    return PREFIXES + SHAPES


def sparql_queries(props: Sequence[Proposition]) -> Dict[str, str]:
    return {p.pid: to_sparql_ask(p) for p in props}


# --------------------------------------------------------------------------
# Mechanical validation
# --------------------------------------------------------------------------
@dataclass
class ValidationReport:
    conforms: bool
    violations: int
    messages: List[str]
    n_rules: int

    def to_dict(self) -> Dict[str, Any]:
        return dict(conforms=self.conforms, violations=self.violations,
                    messages=self.messages, n_rules=self.n_rules)


def validate_rulebase(props: Sequence[Proposition]) -> ValidationReport:
    """Run SHACL validation of the rule base with pyshacl."""
    from rdflib import Graph
    from rdflib.namespace import RDF, SH
    from pyshacl import validate

    data = Graph()
    data.parse(data=rulebase_ttl(props), format="turtle")
    shp = Graph()
    shp.parse(data=shapes_ttl(), format="turtle")
    conforms, results_graph, results_text = validate(
        data_graph=data,
        shacl_graph=shp,
        inference=None,
        abort_on_first=False,
        allow_warnings=True,
    )
    msgs: List[str] = []
    violations = list(results_graph.subjects(RDF.type, SH.ValidationResult))
    for v in violations:
        m = results_graph.value(v, SH.resultMessage)
        msgs.append(str(m) if m is not None else "violation")
    return ValidationReport(conforms=bool(conforms), violations=len(violations),
                            messages=msgs[:40], n_rules=len(props))

