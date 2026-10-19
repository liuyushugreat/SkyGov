"""Chinese legal lexicon used by the frame extractor.

The vocabulary is deliberately *closed*: a term absent here cannot be mapped to
the proposition schema, which is what makes extraction non-trivial and every
vocabulary edit a reviewable act.  Three groups of patterns are separated on
purpose, because they carry different evidential weight:

``HEDGE``
    Discretionary hedges (``必要时``, ``稳妥有序``, ``会同``, ``临时调整``).
    Their presence means the clause delegates judgment, so no determinate rule
    can be emitted.
``POLICY_VERBS``
    Incentive / task verbs (``鼓励``, ``打造``, ``推进``).  A clause whose only
    normative content is an incentive verb has no addressee obligation.
``LIFT``
    Constitutive patterns that *do* licence a rule even without a modal
    particle, because the ontology supplies the missing deontic axiom
    (``X 为适飞空域`` ⇒ permission; ``由 Y 依法处置`` ⇒ competence).
"""
from __future__ import annotations

import re
from typing import Dict, List, Pattern, Tuple

# --------------------------------------------------------------------------
# Deontic modals
# --------------------------------------------------------------------------
MODALS: List[Tuple[str, Pattern]] = [
    ("prohibition", re.compile(r"不得|禁止|严禁|不准|不应")),
    ("obligation", re.compile(r"应当|必须|须")),
    ("permission", re.compile(r"无需|不需|免于|免除|可以|可|有权")),
]

WAIVER_MARKERS = re.compile(r"无需|不需|免于|免除")

# --------------------------------------------------------------------------
# Subjects
# --------------------------------------------------------------------------
SUBJECT_TERMS: Dict[str, List[str]] = {
    "operator": ["运行人", "运营人", "运行组织者", "从事经营性通用航空活动的单位",
                 "使用无人驾驶航空器从事", "实施飞行活动的单位", "任何单位或者个人",
                 "单位或者个人", "有关人员"],
    "pilot": ["操控员", "操控人员", "驾驶员"],
    "manufacturer": ["生产者", "制造者", "无人驾驶航空器系统", "航空器系统", "制造人"],
    "owner": ["所有者", "持有人", "持有者", "购买人"],
    "local_government": ["市、区人民政府", "市人民政府", "各级人民政府", "地方人民政府",
                         "市交通运输部门", "市区人民政府", "任何地方"],
    "airspace_authority": ["空中交通管理机构", "民用航空管理部门", "民航管理部门",
                           "国务院、中央军事委员会"],
    "public_security": ["公安机关", "公安部门", "有关部门", "有关主管部门"],
    "service_provider": ["低空飞行服务机构", "U-space 服务提供者", "服务提供者", "经营者", "企业"],
    "legislature": ["全国人民代表大会及其常务委员会", "全国人民代表大会", "全国人大常委会",
                    "人民代表大会"],
    "central_government": ["国务院办公厅", "国务院"],
    "service_platform": ["监管平台", "平台"],
}

#: "……的，应当……" — the canonical Chinese relative-clause subject.  When the
#: clause uses this construction the actor is implicit and must be inferred.
RELATIVE_CLAUSE = re.compile(r"的[，,]\s*(?:应当|必须|不得|可以|无需|不需|禁止)")
PILOT_RE = re.compile(r"操控[^。；]*?人员|操控员|驾驶人员|操控[^。；]*?执照")
AIRCRAFT_RE = re.compile(r"无人驾驶航空器|航空器")
PERSON_RE = re.compile(r"人员")
ORG_RE = re.compile(r"单位")


def find_subject(text: str) -> Tuple[str, int, int]:
    """Earliest longest alias match, or ``("", -1, -1)``."""
    best: Tuple[str, int, int] = ("", -1, -1)
    for subj, aliases in SUBJECT_TERMS.items():
        for a in sorted(aliases, key=len, reverse=True):
            i = text.find(a)
            if i < 0:
                continue
            if best[1] < 0 or i < best[1]:
                best = (subj, i, i + len(a))
    return best


def infer_subject(text: str) -> Tuple[str, bool]:
    """Return (subject, inferred).  ``inferred`` is True when the actor was not
    named but determined by a documented syntactic rule."""
    subj, _, _ = find_subject(text)
    if subj:
        return subj, False
    if PILOT_RE.search(text):
        return "pilot", True
    if RELATIVE_CLAUSE.search(text):
        return "operator", True
    if AIRCRAFT_RE.search(text):
        return "operator", True
    if PERSON_RE.search(text) or ORG_RE.search(text):
        return "operator", True
    return "", False


# --------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------
ACTION_TERMS: List[Tuple[str, Pattern]] = [
    ("register_uas", re.compile(r"实名登记|变更登记|办理登记|登记")),
    ("hold_licence", re.compile(r"操控员执照|执照|取得培训|考核合格")),
    ("apply_flight_activity", re.compile(r"提出飞行活动申请|重新提出飞行活动申请|飞行活动申请|提出申请|试飞|提前报告|申请")),
    ("operate_in_controlled_airspace", re.compile(r"在管制空域内实施")),
    ("operate_in_permitted_airspace", re.compile(r"在适飞空域内实施|在适飞空域内")),
    ("carry_payload_for_hire", re.compile(r"经营性通用航空活动|经营许可|经营性|从事经营|通用航空经营活动")),
    ("sell_uas", re.compile(r"销售")),
    ("transport_uas", re.compile(r"运输")),
    ("store_uas", re.compile(r"存储|存放|持有")),
    ("counter_uas", re.compile(r"拦截|迫降|捕获|反制|处置|查处")),
    ("publish_airspace_notice", re.compile(r"公布|发布|划设|变更空域类别|作出调整|作出限制")),
    ("build_vertiport", re.compile(r"起降设施|起降场|通导监设施|建设")),
    ("provide_flight_service", re.compile(r"飞行计划处理|提供空域信息|告警等服务|提供飞行服务|飞行服务|提供服务|提供用于")),
    ("report_incident", re.compile(r"立即报告|事故报告|报告当地|报告")),
    ("share_operational_data", re.compile(r"报送|信息报送|数据交换|交换|数据安全|信息能力|身份、位置|查验|采集|获取|传播|电子围栏|远程识别|数据出境|出境")),
    ("train_operator", re.compile(r"安全操作知识|培训")),
    ("fund_industry", re.compile(r"财政资金|安排资金|资金支持")),
    ("insure", re.compile(r"投保|责任保险|保险")),
    ("conduct_drill", re.compile(r"应急演练|组织演练|演练")),
    ("appoint_safety_officer", re.compile(r"指定安全管理人员|安全管理人员|安全负责人")),
    ("undergo_inspection", re.compile(r"年度检验|定期检验|接受检验|通过检验")),
    ("operate", re.compile(r"实施飞行活动|飞行活动|起飞|飞行|进入|作为")),
]

# --------------------------------------------------------------------------
# Conditions
# --------------------------------------------------------------------------
CATEGORY_MAP: Dict[str, str] = {
    "微型": "micro", "轻型": "light", "小型": "small", "中型": "medium", "大型": "large",
}

CONDITION_PATTERNS: List[Tuple[str, Pattern, str, object]] = [
    ("altitude_agl_m", re.compile(r"真高\s*(\d+)\s*米以下"), "<=", "num"),
    ("airspace_class", re.compile(r"不属于管制空域"), "!=", "controlled"),
    ("airspace_class", re.compile(r"管制空域"), "==", "controlled"),
    ("airspace_class", re.compile(r"适飞空域"), "==", "permitted"),
    ("within_airport_buffer", re.compile(r"机场净空|机场飞行区|净空保护区"), "==", True),
    ("night_operation", re.compile(r"夜间"), "==", True),
    ("bvlos", re.compile(r"超视距|视距外"), "==", True),
    ("cross_border", re.compile(r"跨境|出入境|跨越边界|互认"), "==", True),
    ("over_populated_area", re.compile(r"人口密集|人员密集|人群密集"), "==", True),
    ("registered", re.compile(r"未完成实名登记|未实名登记"), "==", False),
    ("info_reported", re.compile(r"未完成报送|未报送"), "==", False),
    ("purpose", re.compile(r"经营性|营运|经营活动"), "==", "for_hire"),
]

CATEGORY_PATTERN = re.compile(r"微型|轻型|小型|中型|大型")

# --------------------------------------------------------------------------
# Failure-mode evidence
# --------------------------------------------------------------------------
AUTHORISATION = re.compile(r"另行规定|决定就|授权|由国务院、中央军事委员会确定")
META_VALIDITY = re.compile(r"相抵触|不得与[^。；]*?抵触|效力")
SANCTION = re.compile(r"责令|处罚|罚款|没收|追究|刑事责任|依法查处")
CONSTITUTIVE_SCHEMA = re.compile(r"应当载明|申请材料|载明|要件|台账|查验")
METHODOLOGICAL = re.compile(r"应当分别评估|评估方法|风险评估应当")
#: Delegation to another instrument: the clause itself carries no self-contained
#: norm, it points elsewhere.
DELEGATION = re.compile(r"依照国家有关规定|按照国家有关规定|另行规定|按有关规定办理|依国家有关规定")
HEDGE = re.compile(r"必要时|稳妥有序|适度|有序推进|逐步|结合实际|会同|"
                   r"作出限制|作出调整|临时调整|特殊保障需要")
POLICY_VERBS = re.compile(r"打造|培育|推进|加快|推动|支持|鼓励|完善|加强|拓展|发展|总结推广|积极")
DEFINITIONAL = re.compile(r"为[^。；]*?空域|划分为|是指|所称")
CROSS_REF = re.compile(r"依照|参照|按照|前款|本条|另有规定|有关规定|另行规定|相关规定|国家有关")
PROVISO = re.compile(r"但[^。；]*?(除外|另有规定)|除外|除[^。；]*?外")

#: Constitutive patterns that licence a rule without a modal particle.
LIFT_PERMITTED_AIRSPACE = re.compile(r"为[^。；]*?适飞空域")
LIFT_COMPETENCE = re.compile(r"由[^，。；]*?(?:依法)?(?:处置|查处|处理|采取)")

ENUM_MARKER = re.compile(r"（[一二三四五六七八九十]+）|\([1-9]\)|^\s*[一二三四五六七八九十]+、")


def find_modal(text: str) -> Tuple[str, int]:
    hits: List[Tuple[int, str]] = []
    for deontic, pat in MODALS:
        m = pat.search(text)
        if m:
            hits.append((m.start(), deontic))
    if not hits:
        return ("", -1)
    hits.sort()
    return (hits[0][1], hits[0][0])


def find_action(text: str) -> Tuple[str, int]:
    """Longest match; ties broken by earliest position."""
    cands: List[Tuple[int, int, str]] = []
    for act, pat in ACTION_TERMS:
        for m in pat.finditer(text):
            cands.append((m.start(), m.end(), act))
    if not cands:
        return ("", -1)
    cands.sort(key=lambda t: (-(t[1] - t[0]), t[0]))
    return (cands[0][2], cands[0][0])


def extract_conditions(text: str) -> List[Tuple[str, str, object]]:
    out: List[Tuple[str, str, object]] = []
    for name, pat, op, val in CONDITION_PATTERNS:
        for m in pat.finditer(text):
            # "不得作为适飞空域": the term is the *object* of a prohibition, not
            # a condition on the activity, so it must not become a condition.
            if name == "airspace_class" and val == "permitted":
                if "作为" in text[max(0, m.start() - 6):m.start()]:
                    continue
            # "不属于管制空域" is the *complement* of 管制空域; do not read both.
            if name == "airspace_class" and val == "controlled":
                if "不属于" in text[max(0, m.start() - 5):m.start()]:
                    continue
            if val == "num":
                try:
                    cand = (name, "<=", int(m.group(1)))
                except (IndexError, ValueError):
                    continue
            else:
                cand = (name, op, val)
            if cand not in out:
                out.append(cand)
    cats: List[str] = []
    for m in CATEGORY_PATTERN.finditer(text):
        c = CATEGORY_MAP.get(m.group(0))
        if c and c not in cats:
            cats.append(c)
    if cats:
        cand = ("uas_category", "in", tuple(cats))
        if cand not in out:
            out.append(cand)
    return out




