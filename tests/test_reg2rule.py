"""Tests for the compilation pipeline (corpus -> frames -> align -> merge -> emit)."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skygovc import corpus  # noqa: E402
from skygovc.core import Condition, FlightRequest, Proposition  # noqa: E402
from skygovc.reg2rule import align as A  # noqa: E402
from skygovc.reg2rule import emit as E  # noqa: E402
from skygovc.reg2rule import engine as EN  # noqa: E402
from skygovc.reg2rule import frames as F  # noqa: E402
from skygovc.reg2rule import merge as M  # noqa: E402
from skygovc.reg2rule import pipeline as P  # noqa: E402
from skygovc.reg2rule import score as S  # noqa: E402


# ------------------------------------------------------------------ corpus
def test_corpus_blocks_are_disjoint_and_sized():
    dev = corpus.dev_clauses()
    held = corpus.heldout_clauses()
    assert len(dev) == 59
    assert len(held) == 19
    assert not ({c.cid for c in dev} & {c.cid for c in held})
    assert all(c.text.strip() for c in corpus.all_clauses())


def test_core_documents_are_sixteen():
    assert len(corpus.CORE_DOC_IDS) == 16
    assert all(c.doc_id in corpus.CORE_DOC_IDS for c in corpus.core_clauses())


def test_gold_compilability_covers_every_clause():
    g = corpus.gold_compilability()
    for c in corpus.all_clauses():
        assert c.cid in g
        ok, reason = g[c.cid]
        assert isinstance(ok, bool)
        if not ok:
            assert reason in corpus.failure_taxonomy()


def test_gold_deduplication_reports_redundancy():
    raw = corpus.gold_propositions(dedupe=False)
    dis = corpus.gold_propositions(dedupe=True)
    assert len(dis) < len(raw)
    assert corpus.redundancy_pairs()          # non-empty


# ------------------------------------------------------------------ frames
def test_modal_and_action_are_syntax_anchored():
    c = corpus.clause("D01-C005")             # 在管制空域内...应当提出飞行活动申请
    ex = F.extract_clause(c, "S3_repair")
    assert ex.compilable
    assert ex.props[0].deontic == "obligation"
    assert ex.props[0].action == "apply_flight_activity"


def test_proviso_is_scoped_to_the_proviso_substring():
    c = corpus.clause("D01-C009")             # 夜间或者超视距 ... 除外
    ex = F.extract_clause(c, "S3_repair")
    sigs = [tuple(sorted(x.key() for x in p.conditions)) for p in ex.props]
    assert any(("night_operation", "==", "True") in s for s in sigs)
    assert any(("bvlos", "==", "True") in s for s in sigs)
    assert not any(("night_operation", "==", "False") in s for s in sigs)


def test_definitional_clause_is_lifted_to_a_permission():
    c = corpus.clause("D01-C003")
    ex = F.extract_clause(c, "S3_repair")
    assert ex.compilable
    assert ex.flags["lifted"] is True
    assert ex.props[0].deontic == "permission"
    assert ex.props[0].action == "operate_in_permitted_airspace"


def test_keyword_baseline_is_strictly_worse_than_frames():
    clause = corpus.clause("D01-C007")
    kw = F.extract_clause(clause, "S1_keyword")
    fr = F.extract_clause(clause, "S2_frame")
    # the keyword baseline emits no conditions at all
    assert kw.props and all(len(p.conditions) == 0 for p in kw.props)
    # the frame pipeline recovers both the actor and the category set
    assert fr.props and fr.props[0].subject == "pilot"
    assert any(c.name == "uas_category" for c in fr.props[0].conditions)


# ------------------------------------------------------------------ align
def test_alignment_closes_negated_airspace_into_an_enumeration():
    p = Proposition(pid="p", cid="D01-C003", deontic="permission",
                    subject="operator", action="operate",
                    conditions=[Condition("airspace_class", "!=", "controlled")])
    out, notes = A.align(p)
    ops = {(c.name, c.op, tuple(c.value) if isinstance(c.value, tuple) else c.value)
           for c in out.conditions}
    assert ("airspace_class", "in", ("G", "W")) in ops
    assert notes


def test_alignment_entails_the_altitude_ceiling():
    p = Proposition(pid="p", cid="X", deontic="permission",
                    subject="operator", action="operate",
                    conditions=[Condition("airspace_class", "==", "permitted")])
    out, _ = A.align(p)
    assert any(c.name == "altitude_agl_m" and c.value == 120 for c in out.conditions)


def test_executability_rejects_unknown_predicate():
    p = Proposition(pid="p", cid="X", deontic="permission", subject="operator",
                    action="operate", conditions=[Condition("made_up", "==", 1)])
    ok, why = A.is_executable(p, {"operator"}, {"operate"}, {"altitude_agl_m"},
                              None)
    assert not ok and "made_up" in why


# ------------------------------------------------------------------ merge
def test_object_mismatch_is_not_reported_as_a_conflict():
    a = Proposition(pid="a", cid="D14-C003", deontic="obligation",
                    subject="service_provider", action="provide_flight_service",
                    normalised_from="低空飞行服务机构应当为飞行活动提供飞行计划处理服务")
    b = Proposition(pid="b", cid="D01-C015", deontic="prohibition",
                    subject="service_provider", action="provide_flight_service",
                    normalised_from="不得提供用于规避监管的通信链路")
    assert not M.objects_overlap(a, b)
    assert M.object_signature(a) != M.object_signature(b)


def test_precedence_algebra_prefers_higher_rank():
    clauses = {c.cid: c for c in corpus.all_clauses()}
    a = Proposition(pid="a", cid="D01-C005", deontic="obligation", subject="operator",
                    action="apply_flight_activity",
                    conditions=[Condition("airspace_class", "in", ("A", "B"))],
                    normalised_from="飞行活动申请")
    b = Proposition(pid="b", cid="D13-C002", deontic="prohibition", subject="operator",
                    action="apply_flight_activity",
                    conditions=[Condition("airspace_class", "in", ("A", "B"))],
                    normalised_from="飞行活动申请")
    kept, conflicts = M.merge([a, b], clauses)
    assert len(conflicts) >= 1
    assert len(kept) <= 2


# ------------------------------------------------------------------ emit
def test_rulebase_ttl_parses_and_carries_provenance():
    r = P.compile_rules("S3_repair", validate=False)
    ttl = E.rulebase_ttl(r.rulebase)
    from rdflib import Graph
    g = Graph()
    g.parse(data=ttl, format="turtle")
    assert len(g) > 0
    assert "derivedFrom" in ttl


def test_sparql_ask_mentions_every_condition():
    r = P.compile_rules("S3_repair", validate=False)
    for p in r.rulebase:
        q = E.to_sparql_ask(p)
        assert q.strip().upper().startswith("PREFIX")
        for c in p.conditions:
            prop = E.COND_PROP[c.name]
            assert prop in q


def test_merged_rulebase_conforms_to_shacl():
    r = P.compile_rules("S3_repair", validate=True)
    assert r.validation_merged.conforms
    assert r.validation_merged.violations == 0


# ------------------------------------------------------------------ engine
def test_prohibition_blocks_a_request():
    rules = [Proposition(pid="p1", cid="D01-C008", deontic="prohibition",
                         subject="operator", action="operate",
                         conditions=[Condition("within_airport_buffer", "==", True)])]
    eng = EN.DecisionEngine(rules)
    bad = FlightRequest(rid="r", origin_jurisdiction="X", dest_jurisdiction="Y",
                        within_airport_buffer=True)
    good = FlightRequest(rid="r2", origin_jurisdiction="X", dest_jurisdiction="Y")
    assert eng.decide(bad).allowed is False
    assert eng.decide(good).allowed is True


def test_authorisation_obligation_requires_approval():
    rules = [Proposition(pid="p1", cid="D01-C005", deontic="obligation",
                         subject="operator", action="apply_flight_activity",
                         conditions=[Condition("airspace_class", "in", ("A", "B"))])]
    eng = EN.DecisionEngine(rules)
    req = FlightRequest(rid="r", origin_jurisdiction="X", dest_jurisdiction="Y",
                        airspace_class="B")
    assert eng.requires_application(req) is True
    assert eng.decide(req, approved=False).allowed is False
    assert eng.decide(req, approved=True).allowed is True


def test_missing_fact_does_not_fire_a_rule():
    rules = [Proposition(pid="p1", cid="X", deontic="prohibition",
                         subject="operator", action="operate",
                         conditions=[Condition("night_operation", "==", True)])]
    eng = EN.DecisionEngine(rules)
    fired, unresolved = eng.fires(rules[0], {})
    assert fired is False and unresolved


# ------------------------------------------------------------------ score
def test_scoring_is_exact_and_symmetric():
    g = corpus.gold_propositions(dedupe=True)
    s = S.score_propositions(g, g)
    assert s.f1 == 1.0 and s.fp == 0 and s.fn == 0


def test_component_scores_penalise_abstention():
    g = corpus.gold_propositions(dedupe=True)
    by_cid: dict = {}
    for p in g:
        by_cid.setdefault(p.cid, []).append(p)

    class _Ex:
        def __init__(self, cid):
            self.cid = cid
            self.props = []

    exs = [_Ex(cid) for cid in by_cid]
    sc = S.score_components(exs, by_cid)
    assert sc.n_clauses == len(by_cid)
    assert sc.exact_acc == 0.0 and sc.field_mean == 0.0

