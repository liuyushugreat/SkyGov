"""Institutional clause corpus — national-level instruments (D01--D11).

Design notes
------------
Two distinct artefacts live in this package and must not be confused:

``CLAUSES``
    The *input* to :mod:`skygovc.reg2rule`.  Each record holds clause-level
    legal prose, deliberately kept close to the register of the published
    instrument: multi-condition sentences, ``但``/``除外`` provisos,
    cross-references (``依照前款``) and discretionary phrasing are preserved.
    This is what the compiler must read.

``GOLD``  (in ``gold.py``)
    The *held-out* expert gold standard: normalised propositions plus a
    ``compilable`` verdict.  The compiler never sees these records; they are
    used only for scoring.

Normalisation honesty
---------------------
``verbatim`` is ``False`` for every record here.  The clause texts are
normalised renderings consolidated from the published instruments: the
operative verbs, modal particles, numeric thresholds and provisos are
retained, while editorial framing and definitional asides are condensed.
Every record carries a ``source_note`` describing what was condensed, and the
full corpus ships with the artefact so that a reader can audit any record
against the gazette.  Section 7.2 of the paper states this explicitly.
"""
from __future__ import annotations

from typing import Any, Dict, List

DOCS: Dict[str, Dict[str, Any]] = {
    "D01": {
        "title": "无人驾驶航空器飞行管理暂行条例",
        "title_en": "Interim Regulation on the Flight Management of Unmanned Aircraft",
        "issuer": "国务院、中央军事委员会",
        "level": "admin_regulation",
        "instrument_date": "2023-05-31",
        "effective_date": "2024-01-01",
        "jurisdiction": "CN-NATIONAL",
        "source_note": "国务院令第761号；条文按主题归并为条款级记录。",
    },
    "D02": {
        "title": "中华人民共和国民用航空法",
        "title_en": "Civil Aviation Law of the People's Republic of China",
        "issuer": "全国人民代表大会常务委员会",
        "level": "law",
        "instrument_date": "2025-12-27",
        "effective_date": "2026-07-01",
        "jurisdiction": "CN-NATIONAL",
        "source_note": "2025 年修订；仅收录与无人驾驶航空器运行、许可、事故报告相关条款。",
    },
    "D03": {
        "title": "中华人民共和国立法法",
        "title_en": "Legislation Law of the People's Republic of China",
        "issuer": "全国人民代表大会",
        "level": "law",
        "instrument_date": "2023-03-13",
        "effective_date": "2023-03-15",
        "jurisdiction": "CN-NATIONAL",
        "source_note": "2023 年修正；收录授权试点相关条款。",
    },
    "D04": {
        "title": "国家空域基础分类方法",
        "title_en": "National Basic Airspace Classification Method",
        "issuer": "中国民用航空局",
        "level": "dept_rule",
        "instrument_date": "2023-12-21",
        "effective_date": "2024-01-01",
        "jurisdiction": "CN-NATIONAL",
        "source_note": "行业规范；空域类别代号与真高条件的表述已归并。",
    },
    "D05": {
        "title": "国务院2026年度立法工作计划",
        "title_en": "State Council 2026 Legislative Work Plan",
        "issuer": "国务院办公厅",
        "level": "policy",
        "instrument_date": "2026-05",
        "effective_date": None,
        "jurisdiction": "CN-NATIONAL",
        "source_note": "国办发〔2026〕14号；属工作安排，非规范性法律文件。",
    },
    "D06": {
        "title": "通用航空装备创新应用实施方案（2024—2030年）",
        "title_en": "Implementation Plan for Innovative Application of General Aviation Equipment (2024-2030)",
        "issuer": "工业和信息化部、科学技术部、财政部等",
        "level": "policy",
        "instrument_date": "2024-03",
        "effective_date": None,
        "jurisdiction": "CN-NATIONAL",
        "source_note": "产业政策；不含可直接执行的义务性规范。",
    },
    "D07": {
        "title": "国务院办公厅关于加快场景培育和开放推动新场景大规模应用的实施意见",
        "title_en": "Implementation Opinions on Accelerating Scenario Cultivation and Openness",
        "issuer": "国务院办公厅",
        "level": "policy",
        "instrument_date": "2025-11-02",
        "effective_date": None,
        "jurisdiction": "CN-NATIONAL",
        "source_note": "国办发〔2025〕37号。",
    },
    "D08": {
        "title": "中共中央关于制定国民经济和社会发展第十五个五年规划的建议",
        "title_en": "Recommendations for the 15th Five-Year Plan",
        "issuer": "中共中央",
        "level": "policy",
        "instrument_date": "2025-10",
        "effective_date": None,
        "jurisdiction": "CN-NATIONAL",
        "source_note": "规划建议；战略性表述。",
    },
    "D09": {
        "title": "政府工作报告（2024年）",
        "title_en": "Government Work Report 2024",
        "issuer": "国务院",
        "level": "policy",
        "instrument_date": "2024-03",
        "effective_date": None,
        "jurisdiction": "CN-NATIONAL",
        "source_note": "年度政府工作报告。",
    },
    "D10": {
        "title": "政府工作报告（2025年）",
        "title_en": "Government Work Report 2025",
        "issuer": "国务院",
        "level": "policy",
        "instrument_date": "2025-03",
        "effective_date": None,
        "jurisdiction": "CN-NATIONAL",
        "source_note": "年度政府工作报告。",
    },
    "D11": {
        "title": "政府工作报告（2026年）",
        "title_en": "Government Work Report 2026",
        "issuer": "国务院",
        "level": "policy",
        "instrument_date": "2026-03",
        "effective_date": None,
        "jurisdiction": "CN-NATIONAL",
        "source_note": "年度政府工作报告。",
    },
}


# Each clause: (cid, doc_id, provision, text, note)
CLAUSES: List[Dict[str, Any]] = [
    # ---------------------------------------------------------------- D01
    dict(cid="D01-C001", doc_id="D01", provision="第七条",
         text="民用无人驾驶航空器所有者应当依法进行实名登记；未完成实名登记的，不得实施飞行活动。",
         note="合并了登记义务与未登记后果两句。"),
    dict(cid="D01-C002", doc_id="D01", provision="第八条",
         text="无人驾驶航空器所有者、操控员应当如实提供登记信息，登记信息发生变更的，应当及时办理变更登记。",
         note=""),
    dict(cid="D01-C003", doc_id="D01", provision="第十三条第一款",
         text="真高120米以下、不属于管制空域的空域，为微型、轻型、小型无人驾驶航空器的适飞空域。",
         note="定义适飞空域；三个无人机类别的并列关系保留。"),
    dict(cid="D01-C004", doc_id="D01", provision="第十三条第二款",
         text="在适飞空域内实施微型、轻型、小型无人驾驶航空器飞行活动的，无需提出飞行活动申请；但依法需要取得经营许可或者涉及重要目标安全的除外。",
         note="含“无需申请”与但书例外。"),
    dict(cid="D01-C005", doc_id="D01", provision="第十四条",
         text="在管制空域内实施无人驾驶航空器飞行活动的，应当依法提出飞行活动申请，经批准后方可实施。",
         note=""),
    dict(cid="D01-C006", doc_id="D01", provision="第十五条",
         text="飞行活动申请应当载明飞行任务、航空器型号与识别码、操控员、起止时间、飞行空域及高度等事项。",
         note="申请材料要件。"),
    dict(cid="D01-C007", doc_id="D01", provision="第十六条",
         text="操控小型、中型、大型无人驾驶航空器的人员，应当取得相应等级的操控员执照；操控微型、轻型民用无人驾驶航空器的人员无需取得操控员执照，但应当掌握必要的安全操作知识。",
         note="含但书；执照义务仅及小/中/大型。"),
    dict(cid="D01-C008", doc_id="D01", provision="第十七条",
         text="无人驾驶航空器不得在下列区域飞行：（一）机场净空保护区；（二）军事禁区、军事管理区；（三）重要目标周边；（四）法律法规规定的其他区域。",
         note="四项列举按原结构保留。"),
    dict(cid="D01-C009", doc_id="D01", provision="第十八条",
         text="无人驾驶航空器不得在夜间或者超视距条件下实施飞行活动，但经批准并具备相应安全保障能力的除外。",
         note="含但书例外。"),
    dict(cid="D01-C010", doc_id="D01", provision="第十九条",
         text="使用无人驾驶航空器从事经营性通用航空活动的，应当取得相应的经营许可，并持续符合许可条件。",
         note=""),
    dict(cid="D01-C011", doc_id="D01", provision="第二十条",
         text="无人驾驶航空器系统应当具备向监管平台实时报送身份、位置、高度等信息的能力，并在飞行全过程保持有效连接。",
         note="对系统本身的技术性义务。"),
    dict(cid="D01-C012", doc_id="D01", provision="第二十一条",
         text="无人驾驶航空器发生事故的，运营人应当立即报告当地民用航空管理部门、公安机关，并保护现场。",
         note=""),
    dict(cid="D01-C013", doc_id="D01", provision="第二十二条",
         text="对违法飞行的无人驾驶航空器，公安机关等有关部门可以依法采取拦截、迫降、捕获等必要措施；采取措施应当避免或者减少对地面第三人合法权益的损害。",
         note="含“必要时”式裁量与比例原则。"),
    dict(cid="D01-C014", doc_id="D01", provision="第二十三条",
         text="空中交通管理机构应当为无人驾驶航空器飞行活动提供空域信息、飞行计划处理等服务，并公布相关限制信息。",
         note="服务义务主体为空中交通管理机构。"),
    dict(cid="D01-C015", doc_id="D01", provision="第二十四条",
         text="任何单位或者个人不得向无人驾驶航空器提供用于规避监管的通信链路、导航信号或者身份伪造能力。",
         note="供应链侧禁止性规范。"),

    # ---------------------------------------------------------------- D02
    dict(cid="D02-C001", doc_id="D02", provision="第七十条",
         text="空域管理应当兼顾民用航空发展、国防安全和社会公共利益的需要，由国务院、中央军事委员会另行规定。",
         note="原则性条款；主体授权表述。"),
    dict(cid="D02-C002", doc_id="D02", provision="第一百零三条",
         text="从事通用航空经营活动，应当具备与其经营活动相适应的航空器、人员、设施和安全管理条件，并依法取得经营许可。",
         note=""),
    dict(cid="D02-C003", doc_id="D02", provision="第一百三十一条",
         text="民用航空器发生事故，事故现场有关人员应当立即报告，并接受调查。",
         note=""),
    dict(cid="D02-C004", doc_id="D02", provision="第一百四十条",
         text="未经许可擅自从事通用航空经营活动的，由民用航空管理部门责令停止并处以罚款；有违法所得的，没收违法所得。",
         note="罚则；可作为规则的 sanction 来源。"),

    # ---------------------------------------------------------------- D03
    dict(cid="D03-C001", doc_id="D03", provision="第十六条",
         text="全国人民代表大会及其常务委员会可以根据改革发展的需要，决定就特定事项授权在一定期限和范围内暂时调整或者暂时停止适用法律的部分规定。",
         note="授权试点的法律基础。"),
    dict(cid="D03-C002", doc_id="D03", provision="第七十三条第三款",
         text="地方性法规不得与宪法、法律、行政法规相抵触。",
         note="层级冲突的判断依据。"),

    # ---------------------------------------------------------------- D04
    dict(cid="D04-C001", doc_id="D04", provision="二、空域类别",
         text="我国空域划分为A、B、C、D、E、G、W七类。",
         note="空域类别的封闭枚举。"),
    dict(cid="D04-C002", doc_id="D04", provision="三、非管制空域",
         text="G类、W类空域为非管制空域，真高120米以下W类空域适用于微型、轻型、小型无人驾驶航空器飞行活动。",
         note=""),
    dict(cid="D04-C003", doc_id="D04", provision="四、管制空域",
         text="A、B、C、D、E类空域为管制空域，未经批准，航空器不得进入。",
         note=""),
    dict(cid="D04-C004", doc_id="D04", provision="六、空域调整",
         text="空域类别调整由国务院、中央军事委员会确定，任何地方不得自行变更空域类别。",
         note="权限的排他性表述。"),

    # ---------------------------------------------------------------- D05--D11 policy
    dict(cid="D05-C001", doc_id="D05", provision="附件二、预备制定项目",
         text="《空域管理条例》列为预备制定项目。",
         note="立法工作安排，非现行规范。"),
    dict(cid="D06-C001", doc_id="D06", provision="三、重点任务",
         text="加快通用航空装备创新应用，推动无人机物流配送、城市空中交通等场景规模化发展。",
         note="任务性表述。"),
    dict(cid="D07-C001", doc_id="D07", provision="二、重点领域",
         text="稳妥有序拓展低空经济等领域应用场景，完善配套规则和安全保障。",
         note="含“稳妥有序”裁量表述。"),
    dict(cid="D07-C002", doc_id="D07", provision="三、保障措施",
         text="鼓励地方结合实际开展场景创新，及时总结推广可复制的经验做法。",
         note="鼓励性、非义务性。"),
    dict(cid="D08-C001", doc_id="D08", provision="第四部分",
         text="加快培育战略性新兴产业集群，推进低空经济等新增长引擎发展。",
         note="规划建议。"),
    dict(cid="D09-C001", doc_id="D09", provision="二、重点工作",
         text="积极打造生物制造、商业航天、低空经济等新增长引擎。",
         note=""),
    dict(cid="D10-C001", doc_id="D10", provision="二、重点工作",
         text="推动商业航天、低空经济等新兴产业安全健康发展。",
         note=""),
    dict(cid="D11-C001", doc_id="D11", provision="二、重点工作",
         text="打造低空经济等新兴支柱产业，加强安全保障能力建设。",
         note=""),
]
