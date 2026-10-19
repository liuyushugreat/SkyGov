"""Expert gold standard — held out from the compiler.

Each record states, for one clause, whether a competent analyst can turn it
into an executable deontic rule, and if so which normalised proposition(s).

Conventions
-----------
* Airspace conditions are written in their **ontology-grounded** form.  A clause
  that says ``管制空域`` is recorded as ``airspace_class in (A,B,C,D,E)`` because
  that is what the executable rule must contain; the surface form is left to
  the alignment stage.  This is deliberate: it is what makes the alignment
  ablation measurable.
* ``reason`` names the failure category when ``compilable`` is ``False``; the
  ten categories form the compilation-failure taxonomy of Section 6.2.
* ``schema_gap`` marks clauses that are normative but cannot be expressed in
  the current proposition schema (an activity-based antecedent).  Reporting
  them as a *representation* limit rather than as compiler errors is a finding,
  not a concession.

The compiler never reads this module.  Scoring lives in
:mod:`skygovc.reg2rule.score`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

CONTROLLED = ("A", "B", "C", "D", "E")
NON_CONTROLLED = ("G", "W")

#: Compilation-failure taxonomy (ten categories).
FAILURE_MODES: Dict[str, str] = {
    "definitional": "定义性或空间事实；被本体吸收，不发射规则",
    "policy_non_normative": "政策/任务/号召性表述；无确定的义务主体",
    "discretionary": "含“必要时/稳妥有序/会同/临时调整”等裁量或推动性表述",
    "cross_reference": "以“依照国家有关规定办理”等前向引用为主，无自足规范",
    "authorisation": "授权条款；主体为立法机关且需另行“决定”",
    "meta_validity": "位阶层面的效力约束；作为冲突消解公理而非对象级规则",
    "sanction": "罚则条款；作为既有规则的后果元数据而非独立规则",
    "constitutive_schema": "对申请材料/台账的构成性要求；编码为数据模式",
    "methodological": "方法性指引；无规范对象",
    "schema_gap": "规范明确，但其“以行为为前件”的结构无法由当前命题模式表达",
}


def _c(name: str, op: str, value: Any) -> tuple:
    return (name, op, value)


_GOLD_CTRL = _c("airspace_class", "in", CONTROLLED)
_GOLD_PERM = _c("airspace_class", "in", NON_CONTROLLED)

GOLD: List[Dict[str, Any]] = [
    # ------------------------------------------------------------- D01 条例
    dict(cid="D01-C001", compilable=True, props=[
        ("obligation", "owner", "register_uas", []),
        ("prohibition", "operator", "operate", [_c("registered", "==", False)]),
    ]),
    dict(cid="D01-C002", compilable=True, props=[
        ("obligation", "owner", "register_uas", []),
    ]),
    dict(cid="D01-C003", compilable=True, props=[
        ("permission", "operator", "operate_in_permitted_airspace", [
            _c("altitude_agl_m", "<=", 120),
            _c("uas_category", "in", ("micro", "light", "small")),
            _GOLD_PERM,
        ]),
    ]),
    dict(cid="D01-C004", compilable=True, props=[
        ("permission", "operator", "operate_in_permitted_airspace", [
            _c("altitude_agl_m", "<=", 120),
            _c("uas_category", "in", ("micro", "light", "small")),
            _GOLD_PERM,
            _c("purpose", "!=", "for_hire"),
        ]),
    ]),
    dict(cid="D01-C005", compilable=True, props=[
        ("obligation", "operator", "apply_flight_activity", [_GOLD_CTRL]),
    ]),
    dict(cid="D01-C006", compilable=False, reason="constitutive_schema"),
    dict(cid="D01-C007", compilable=True, props=[
        ("obligation", "pilot", "hold_licence", [
            _c("uas_category", "in", ("small", "medium", "large")),
        ]),
        ("permission", "pilot", "hold_licence", [
            _c("uas_category", "in", ("micro", "light")),
        ]),
    ]),
    dict(cid="D01-C008", compilable=True, props=[
        ("prohibition", "operator", "operate", [_c("within_airport_buffer", "==", True)]),
    ]),
    dict(cid="D01-C009", compilable=True, props=[
        ("prohibition", "operator", "operate", [_c("night_operation", "==", True)]),
        ("prohibition", "operator", "operate", [_c("bvlos", "==", True)]),
    ]),
    dict(cid="D01-C010", compilable=True, props=[
        ("obligation", "operator", "carry_payload_for_hire", [_c("purpose", "==", "for_hire")]),
    ]),
    dict(cid="D01-C011", compilable=True, props=[
        ("obligation", "manufacturer", "share_operational_data", []),
    ]),
    dict(cid="D01-C012", compilable=True, props=[
        ("obligation", "operator", "report_incident", []),
    ]),
    dict(cid="D01-C013", compilable=True, props=[
        ("permission", "public_security", "counter_uas", []),
    ]),
    dict(cid="D01-C014", compilable=True, props=[
        ("obligation", "airspace_authority", "provide_flight_service", []),
    ]),
    dict(cid="D01-C015", compilable=True, props=[
        ("prohibition", "operator", "provide_flight_service", []),
    ]),

    # ------------------------------------------------------------- D02 民航法
    dict(cid="D02-C001", compilable=False, reason="authorisation"),
    dict(cid="D02-C002", compilable=True, props=[
        ("obligation", "operator", "carry_payload_for_hire", [_c("purpose", "==", "for_hire")]),
    ]),
    dict(cid="D02-C003", compilable=True, props=[
        ("obligation", "operator", "report_incident", []),
    ]),
    dict(cid="D02-C004", compilable=False, reason="sanction"),

    # ------------------------------------------------------------- D03 立法法
    dict(cid="D03-C001", compilable=False, reason="authorisation"),
    dict(cid="D03-C002", compilable=False, reason="meta_validity"),

    # ------------------------------------------------------------- D04 空域分类
    dict(cid="D04-C001", compilable=False, reason="definitional"),
    dict(cid="D04-C002", compilable=False, reason="definitional"),
    dict(cid="D04-C003", compilable=True, props=[
        ("prohibition", "operator", "operate", [_GOLD_CTRL]),
    ]),
    dict(cid="D04-C004", compilable=True, props=[
        ("prohibition", "local_government", "publish_airspace_notice", []),
    ]),

    # ------------------------------------------------------------- D05-D11 政策
    dict(cid="D05-C001", compilable=False, reason="policy_non_normative"),
    dict(cid="D06-C001", compilable=False, reason="policy_non_normative"),
    dict(cid="D07-C001", compilable=False, reason="discretionary"),
    dict(cid="D07-C002", compilable=False, reason="policy_non_normative"),
    dict(cid="D08-C001", compilable=False, reason="policy_non_normative"),
    dict(cid="D09-C001", compilable=False, reason="policy_non_normative"),
    dict(cid="D10-C001", compilable=False, reason="policy_non_normative"),
    dict(cid="D11-C001", compilable=False, reason="policy_non_normative"),

    # ------------------------------------------------------------- D12 吉林
    dict(cid="D12-C001", compilable=True, props=[
        ("permission", "operator", "operate_in_permitted_airspace", [
            _c("altitude_agl_m", "<=", 120),
            _c("uas_category", "in", ("micro", "light", "small")),
            _GOLD_PERM,
        ]),
    ]),
    dict(cid="D12-C002", compilable=True, props=[
        ("prohibition", "operator", "operate", [_c("within_airport_buffer", "==", True)]),
    ]),
    dict(cid="D12-C003", compilable=False, reason="cross_reference"),

    # ------------------------------------------------------------- D13 天津
    dict(cid="D13-C001", compilable=False, reason="definitional"),
    dict(cid="D13-C002", compilable=True, props=[
        ("obligation", "operator", "apply_flight_activity", [_GOLD_CTRL]),
    ]),
    dict(cid="D13-C003", compilable=True, props=[
        ("obligation", "operator", "share_operational_data", [
            _c("uas_category", "in", ("micro",)), _GOLD_CTRL,
        ]),
        ("prohibition", "operator", "operate", [_c("info_reported", "==", False)]),
    ]),
    dict(cid="D13-C004", compilable=True, props=[
        ("permission", "public_security", "counter_uas", []),
    ]),

    # ------------------------------------------------------------- D14 深圳
    dict(cid="D14-C001", compilable=True, props=[
        ("obligation", "local_government", "fund_industry", []),
    ]),
    dict(cid="D14-C002", compilable=True, props=[
        ("obligation", "local_government", "build_vertiport", []),
    ]),
    dict(cid="D14-C003", compilable=True, props=[
        ("obligation", "service_provider", "provide_flight_service", []),
    ]),
    dict(cid="D14-C004", compilable=True, props=[
        ("obligation", "service_provider", "share_operational_data", []),
    ]),
    dict(cid="D14-C005", compilable=True, props=[
        ("prohibition", "operator", "share_operational_data", []),
    ]),
    dict(cid="D14-C006", compilable=False, reason="discretionary"),

    # ------------------------------------------------------------- D15 北京 3 月
    dict(cid="D15-C001", compilable=False, reason="definitional"),
    dict(cid="D15-C002", compilable=True, props=[
        ("obligation", "operator", "apply_flight_activity", [_GOLD_CTRL]),
    ]),
    dict(cid="D15-C003", compilable=False, reason="schema_gap"),
    dict(cid="D15-C004", compilable=False, reason="discretionary"),
    dict(cid="D15-C005", compilable=False, reason="sanction"),

    # ------------------------------------------------------------- D16 北京 9 月
    dict(cid="D16-C001", compilable=False, reason="definitional"),
    dict(cid="D16-C002", compilable=False, reason="schema_gap"),
    dict(cid="D16-C003", compilable=True, props=[
        ("obligation", "operator", "share_operational_data", [
            _c("uas_category", "in", ("micro", "light")),
        ]),
    ]),
    dict(cid="D16-C004", compilable=False, reason="discretionary"),

    # ------------------------------------------------------------- E 境外对照
    dict(cid="E01-C001", compilable=True, props=[
        ("obligation", "service_provider", "share_operational_data", []),
    ]),
    dict(cid="E01-C002", compilable=True, props=[
        ("obligation", "service_provider", "provide_flight_service", []),
    ]),
    dict(cid="E02-C001", compilable=True, props=[
        ("obligation", "service_provider", "share_operational_data",
         [_c("cross_border", "==", True)]),
    ]),
    dict(cid="E03-C001", compilable=False, reason="methodological"),
]


def gold_by_cid() -> Dict[str, Dict[str, Any]]:
    return {g["cid"]: g for g in GOLD}


def summary() -> Dict[str, Any]:
    n = len(GOLD)
    nc = sum(1 for g in GOLD if g["compilable"])
    nprops = sum(len(g.get("props", [])) for g in GOLD)
    modes: Dict[str, int] = {}
    for g in GOLD:
        if not g["compilable"]:
            modes[g["reason"]] = modes.get(g["reason"], 0) + 1
    return dict(records=n, compilable=nc, non_compilable=n - nc,
                gold_propositions=nprops, failure_modes=modes)


if __name__ == "__main__":  # pragma: no cover
    import json
    print(json.dumps(summary(), ensure_ascii=False, indent=2))

