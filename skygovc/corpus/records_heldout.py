"""Held-out clause block — frozen-grammar, single-run evaluation.

Protocol
--------
The frame grammar and the failure-mode decision procedure were developed on the
``dev`` records (``records_national.py`` + ``records_local.py``).  The records in
this module were authored **after** the grammar and the decision procedure were
frozen, and the pipeline was executed on them **once**; the figures reported in
Section 6.4 are that single run, with no subsequent grammar edits.

The block deliberately covers governance objects that the development set does
not: insurance, vertiport siting, emergency drills, training, cross-border data,
geo-fencing, remote identification, flight-plan amendment, incident
investigation, periodic inspection, night cargo, agricultural operations, test
flights, counter-UAS equipment, and a designated safety officer.  This is a
*domain-extrapolation* test rather than a random split, which is the harder and
more useful question for an information system that must absorb new instruments.

``GOLD_HELDOUT`` lives here (not in ``gold.py``) so that the development gold
and the held-out gold can never be silently pooled.
"""
from __future__ import annotations

from typing import Any, Dict, List

DOCS: Dict[str, Dict[str, Any]] = {
    "H01": {
        "title": "无人驾驶航空器飞行管理暂行条例（保险与应急条款）",
        "title_en": "Interim Regulation — insurance and emergency provisions",
        "issuer": "国务院、中央军事委员会",
        "level": "admin_regulation",
        "instrument_date": "2023-05-31",
        "effective_date": "2024-01-01",
        "jurisdiction": "CN-NATIONAL",
        "source_note": "留出块：保险、应急、演练、安全管理人员相关条款的规范化记录。",
    },
    "H02": {
        "title": "深圳经济特区低空经济产业促进条例（设施与培训条款）",
        "title_en": "Shenzhen Regulation — infrastructure and training provisions",
        "issuer": "深圳市人民代表大会常务委员会",
        "level": "local_regulation",
        "instrument_date": "2023-12-29",
        "effective_date": "2024-02-01",
        "jurisdiction": "CN-SZ",
        "source_note": "留出块：起降设施选址、培训、跨境数据相关条款。",
    },
    "H03": {
        "title": "北京市无人驾驶航空器管理规定（运行管理条款）",
        "title_en": "Beijing Regulation — operational management provisions",
        "issuer": "北京市人民代表大会常务委员会",
        "level": "local_regulation",
        "instrument_date": "2026-09-11",
        "effective_date": "2026-11-15",
        "jurisdiction": "CN-BJ",
        "source_note": "留出块：电子围栏、远程识别、飞行计划变更、夜间货运相关条款。",
    },
    "H04": {
        "title": "天津市无人驾驶航空器管制空域管理实施细则",
        "title_en": "Tianjin implementation rules for controlled airspace",
        "issuer": "天津市人民政府",
        "level": "local_notice",
        "instrument_date": "2026-09-30",
        "effective_date": "2026-10-15",
        "jurisdiction": "CN-TJ",
        "source_note": "留出块：检验、事故调查、试飞相关条款。",
    },
}

CLAUSES: List[Dict[str, Any]] = [
    # ------------------------------------------------- H01 保险与应急
    dict(cid="H01-C001", doc_id="H01", provision="第二十六条",
         text="使用除微型以外的无人驾驶航空器实施飞行活动的，运营人应当投保第三人责任保险。",
         note="留出块。"),
    dict(cid="H01-C002", doc_id="H01", provision="第二十七条",
         text="运营人应当制定飞行事故应急处置预案，并每年至少组织一次应急演练。",
         note="留出块。"),
    dict(cid="H01-C003", doc_id="H01", provision="第二十八条",
         text="运营人应当指定安全管理人员，负责飞行前检查、隐患排查和事件报告。",
         note="留出块。"),
    dict(cid="H01-C004", doc_id="H01", provision="第二十九条",
         text="操控员应当接受安全生产培训并考核合格后方可上岗。",
         note="留出块。"),
    dict(cid="H01-C005", doc_id="H01", provision="第三十条",
         text="发生飞行事故的，运营人不得隐瞒、谎报或者迟报事故信息。",
         note="留出块。"),
    dict(cid="H01-C006", doc_id="H01", provision="第三十一条",
         text="鼓励运营人采用高于国家标准的风险防控措施。",
         note="留出块；鼓励性表述。"),

    # ------------------------------------------------- H02 设施与培训
    dict(cid="H02-C001", doc_id="H02", provision="第十三条",
         text="建设低空飞行起降设施的，应当符合国土空间规划，并避开人口密集区域和机场净空保护区。",
         note="留出块。"),
    dict(cid="H02-C002", doc_id="H02", provision="第十六条",
         text="市交通运输部门应当会同有关部门编制低空飞行基础设施建设专项规划。",
         note="留出块；含“会同”。"),
    dict(cid="H02-C003", doc_id="H02", provision="第十九条",
         text="低空飞行服务机构应当为操控员提供免费的航空信息和安全培训服务。",
         note="留出块。"),
    dict(cid="H02-C004", doc_id="H02", provision="第二十五条",
         text="向境外提供低空飞行相关数据的，应当依法履行数据出境安全评估程序。",
         note="留出块。"),

    # ------------------------------------------------- H03 运行管理
    dict(cid="H03-C001", doc_id="H03", provision="第十条",
         text="无人驾驶航空器系统应当具备电子围栏功能，并在进入管制空域前自动告警。",
         note="留出块。"),
    dict(cid="H03-C002", doc_id="H03", provision="第十一条",
         text="无人驾驶航空器应当具备远程识别功能，飞行全过程持续广播识别信息。",
         note="留出块。"),
    dict(cid="H03-C003", doc_id="H03", provision="第十四条",
         text="飞行计划变更的，运营人应当在起飞前重新提出飞行活动申请。",
         note="留出块。"),
    dict(cid="H03-C004", doc_id="H03", provision="第十七条",
         text="在本市实施夜间载货飞行的，应当经批准并配备地面观察人员。",
         note="留出块。"),
    dict(cid="H03-C005", doc_id="H03", provision="第二十条",
         text="任何单位和个人不得擅自改装无人驾驶航空器以规避电子围栏限制。",
         note="留出块。"),

    # ------------------------------------------------- H04 检验与调查
    dict(cid="H04-C001", doc_id="H04", provision="第六条",
         text="无人驾驶航空器应当按照规定接受年度检验；未通过检验的不得实施飞行活动。",
         note="留出块。"),
    dict(cid="H04-C002", doc_id="H04", provision="第九条",
         text="试飞活动应当在划定的试飞区域内实施，并提前报告空中交通管理机构。",
         note="留出块。"),
    dict(cid="H04-C003", doc_id="H04", provision="第十二条",
         text="事故调查由民用航空管理部门组织，任何单位和个人应当配合调查。",
         note="留出块。"),
    dict(cid="H04-C004", doc_id="H04", provision="第十五条",
         text="因应急处置需要临时使用管制空域的，由市人民政府协调空中交通管理机构办理。",
         note="留出块；含协调性表述。"),
]


def _c(name: str, op: str, value: Any) -> tuple:
    return (name, op, value)


#: Held-out gold.  Written before the single evaluation run.
GOLD_HELDOUT: List[Dict[str, Any]] = [
    dict(cid="H01-C001", compilable=True, props=[
        ("obligation", "operator", "carry_payload_for_hire",
         [_c("uas_category", "!=", "micro")]),
    ]),
    dict(cid="H01-C002", compilable=True, props=[
        ("obligation", "operator", "report_incident", []),
    ]),
    dict(cid="H01-C003", compilable=True, props=[
        ("obligation", "operator", "report_incident", []),
    ]),
    dict(cid="H01-C004", compilable=True, props=[
        ("obligation", "pilot", "train_operator", []),
    ]),
    dict(cid="H01-C005", compilable=True, props=[
        ("prohibition", "operator", "report_incident", []),
    ]),
    dict(cid="H01-C006", compilable=False, reason="policy_non_normative"),
    dict(cid="H02-C001", compilable=True, props=[
        ("obligation", "local_government", "build_vertiport",
         [_c("over_populated_area", "==", False),
          _c("within_airport_buffer", "==", False)]),
    ]),
    dict(cid="H02-C002", compilable=False, reason="discretionary"),
    dict(cid="H02-C003", compilable=True, props=[
        ("obligation", "service_provider", "train_operator", []),
    ]),
    dict(cid="H02-C004", compilable=True, props=[
        ("obligation", "operator", "share_operational_data",
         [_c("cross_border", "==", True)]),
    ]),
    dict(cid="H03-C001", compilable=True, props=[
        ("obligation", "manufacturer", "operate", []),
    ]),
    dict(cid="H03-C002", compilable=True, props=[
        ("obligation", "manufacturer", "share_operational_data", []),
    ]),
    dict(cid="H03-C003", compilable=True, props=[
        ("obligation", "operator", "apply_flight_activity", []),
    ]),
    dict(cid="H03-C004", compilable=True, props=[
        ("obligation", "operator", "operate",
         [_c("night_operation", "==", True), _c("purpose", "==", "logistics")]),
    ]),
    dict(cid="H03-C005", compilable=True, props=[
        ("prohibition", "operator", "operate", []),
    ]),
    dict(cid="H04-C001", compilable=True, props=[
        ("prohibition", "operator", "operate", []),
    ]),
    dict(cid="H04-C002", compilable=True, props=[
        ("obligation", "operator", "apply_flight_activity", []),
    ]),
    dict(cid="H04-C003", compilable=True, props=[
        ("obligation", "operator", "report_incident", []),
    ]),
    dict(cid="H04-C004", compilable=False, reason="discretionary"),
]
