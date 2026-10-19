"""Institutional clause corpus — local instruments (D12--D16) and external
comparison standards (E01--E05).

Local instruments are the source of the *institutional heterogeneity* that the
paper's requirements R1--R5 are derived from: they regulate different objects
(airspace publication, industry promotion, equipment handling) at different
levels of the hierarchy, and they conflict with the national frame in ways that
a compiler must resolve rather than concatenate.
"""
from __future__ import annotations

from typing import Any, Dict, List

DOCS: Dict[str, Dict[str, Any]] = {
    "D12": {
        "title": "吉林省无人驾驶航空器适飞空域范围公告",
        "title_en": "Jilin Province Announcement of Permitted Airspace for Unmanned Aircraft",
        "issuer": "吉林省交通运输厅",
        "level": "local_notice",
        "instrument_date": "2026-05-12",
        "effective_date": "2026-05-12",
        "jurisdiction": "CN-JL",
        "source_note": "省级适飞空域公告；区域以地理边界描述。",
    },
    "D13": {
        "title": "天津市人民政府关于公布本市无人驾驶航空器管制空域范围的通告",
        "title_en": "Tianjin Municipal Notice on the Controlled Airspace for Unmanned Aircraft",
        "issuer": "天津市人民政府",
        "level": "local_notice",
        "instrument_date": "2026-09-30",
        "effective_date": "2026-09-30",
        "jurisdiction": "CN-TJ",
        "source_note": "全市域管制空域通告。",
    },
    "D14": {
        "title": "深圳经济特区低空经济产业促进条例",
        "title_en": "Shenzhen Special Economic Zone Regulation on the Promotion of the Low-Altitude Economy",
        "issuer": "深圳市人民代表大会常务委员会",
        "level": "local_regulation",
        "instrument_date": "2023-12-29",
        "effective_date": "2024-02-01",
        "jurisdiction": "CN-SZ",
        "source_note": "产业促进型地方性法规。",
    },
    "D15": {
        "title": "北京市无人驾驶航空器管理规定",
        "title_en": "Beijing Municipal Regulation on the Management of Unmanned Aircraft (March text)",
        "issuer": "北京市人民代表大会常务委员会",
        "level": "local_regulation",
        "instrument_date": "2026-03-27",
        "effective_date": "2026-05-01",
        "jurisdiction": "CN-BJ",
        "source_note": "2026 年 3 月通过文本。",
    },
    "D16": {
        "title": "北京市无人驾驶航空器管理规定（2026年9月修订）",
        "title_en": "Beijing Municipal Regulation on the Management of Unmanned Aircraft (September revision)",
        "issuer": "北京市人民代表大会常务委员会",
        "level": "local_regulation",
        "instrument_date": "2026-09-11",
        "effective_date": "2026-11-15",
        "jurisdiction": "CN-BJ",
        "source_note": "2026 年 9 月修订文本；观察期末尚未施行。",
    },
    "E01": {
        "title": "Commission Implementing Regulation (EU) 2021/664 (U-space)",
        "title_en": "Commission Implementing Regulation (EU) 2021/664 (U-space)",
        "issuer": "European Commission",
        "level": "external_standard",
        "instrument_date": "2021-04-22",
        "effective_date": "2023-01-26",
        "jurisdiction": "EU",
        "source_note": "仅用于跨法域比较，不参与国内优先级合并。",
    },
    "E02": {
        "title": "UAS Traffic Management (UTM) Concept of Operations v2.0",
        "title_en": "UAS Traffic Management (UTM) Concept of Operations v2.0",
        "issuer": "Federal Aviation Administration",
        "level": "external_standard",
        "instrument_date": "2020-03",
        "effective_date": None,
        "jurisdiction": "US",
        "source_note": "概念性文件。",
    },
    "E03": {
        "title": "JARUS Guidelines on Specific Operations Risk Assessment (SORA) v2.5",
        "title_en": "JARUS Guidelines on Specific Operations Risk Assessment (SORA) v2.5",
        "issuer": "JARUS",
        "level": "external_standard",
        "instrument_date": "2024-05-13",
        "effective_date": None,
        "jurisdiction": "INTL",
        "source_note": "风险评估方法参照。",
    },
}


CLAUSES: List[Dict[str, Any]] = [
    # ---------------------------------------------------------------- D12 吉林
    dict(cid="D12-C001", doc_id="D12", provision="一、适飞范围",
         text="自2026年5月12日起，本省行政区域内真高120米以下、不属于管制空域的空域，为微型、轻型、小型无人驾驶航空器适飞空域。",
         note="省级适飞公告；与国家规定同构。"),
    dict(cid="D12-C002", doc_id="D12", provision="二、禁止飞行区域",
         text="机场净空保护区、军事管理区、重要目标周边以及依法划定的其他区域，不得作为适飞空域。",
         note=""),
    dict(cid="D12-C003", doc_id="D12", provision="三、申请要求",
         text="在适飞空域内实施飞行活动无需提出飞行活动申请；在管制空域内实施飞行活动的，依照国家有关规定办理。",
         note="含前向引用。"),

    # ---------------------------------------------------------------- D13 天津
    dict(cid="D13-C001", doc_id="D13", provision="一、管制范围",
         text="本市行政区域全部划为无人驾驶航空器管制空域。",
         note="全市域管制。"),
    dict(cid="D13-C002", doc_id="D13", provision="二、飞行申请",
         text="在本市管制空域内实施无人驾驶航空器飞行活动的，应当依法提出飞行活动申请，经批准后方可实施。",
         note="与 D01-C005 同构，位阶不同。"),
    dict(cid="D13-C003", doc_id="D13", provision="三、微型航空器",
         text="微型无人驾驶航空器在本市管制空域内飞行，应当在民航管理部门规定的平台完成信息报送；未完成报送的不得起飞。",
         note="地方对国家规则的加严。"),
    dict(cid="D13-C004", doc_id="D13", provision="四、违规处置",
         text="违反本通告规定飞行的，由公安机关依法处置。",
         note=""),

    # ---------------------------------------------------------------- D14 深圳
    dict(cid="D14-C001", doc_id="D14", provision="第九条",
         text="市、区人民政府应当安排财政资金支持低空经济产业发展，并纳入本级国民经济和社会发展规划。",
         note=""),
    dict(cid="D14-C002", doc_id="D14", provision="第十二条",
         text="市交通运输部门应当统筹推进低空飞行起降设施、通信导航监视设施建设。",
         note=""),
    dict(cid="D14-C003", doc_id="D14", provision="第十五条",
         text="市低空飞行服务机构应当为飞行活动提供飞行计划处理、情报、告警等服务，并保障服务连续性。",
         note=""),
    dict(cid="D14-C004", doc_id="D14", provision="第十八条",
         text="从事低空飞行服务的企业应当建立数据安全管理制度，对位置、影像等数据实行分类分级保护。",
         note=""),
    dict(cid="D14-C005", doc_id="D14", provision="第二十条",
         text="任何单位和个人不得利用低空飞行活动非法采集、获取、传播国家秘密、商业秘密和个人信息。",
         note="禁止性规范。"),
    dict(cid="D14-C006", doc_id="D14", provision="第二十三条",
         text="市人民政府可以会同相邻地区建立低空飞行协同机制，推动飞行计划、限制信息的互认。",
         note="含“可以”，且互认为推动性质。"),

    # ---------------------------------------------------------------- D15 北京 3 月
    dict(cid="D15-C001", doc_id="D15", provision="第四条",
         text="本市行政区域全部划为无人驾驶航空器管制空域。",
         note=""),
    dict(cid="D15-C002", doc_id="D15", provision="第五条",
         text="在本市管制空域内实施飞行活动的，应当依法提出飞行活动申请。",
         note=""),
    dict(cid="D15-C003", doc_id="D15", provision="第八条",
         text="销售、运输、存储无人驾驶航空器的，应当核验购买人身份信息并登记台账。",
         note="调整对象为设备流通，而非飞行活动。"),
    dict(cid="D15-C004", doc_id="D15", provision="第十一条",
         text="因重要活动、应急处置等特殊保障需要，经批准可以对本规定作出临时调整。",
         note="含裁量性例外。"),
    dict(cid="D15-C005", doc_id="D15", provision="第十三条",
         text="对违反本规定飞行的，由公安机关依法查处；构成犯罪的，依法追究刑事责任。",
         note=""),

    # ---------------------------------------------------------------- D16 北京 9 月修订
    dict(cid="D16-C001", doc_id="D16", provision="第四条",
         text="本市行政区域全部划为无人驾驶航空器管制空域。",
         note="与 D15-C001 同文，构成版本比对样本。"),
    dict(cid="D16-C002", doc_id="D16", provision="第八条",
         text="持有、存放无人驾驶航空器的，应当登记并符合存放场所安全要求；运输无人驾驶航空器进入本市的，应当接受查验。",
         note="与 D15-C003 相比，调整对象由“销售”扩至“持有、存放、运输”。"),
    dict(cid="D16-C003", doc_id="D16", provision="第九条",
         text="微型、轻型无人驾驶航空器在本市飞行的，应当在平台完成信息报送。",
         note="新增微型/轻型的报送义务。"),
    dict(cid="D16-C004", doc_id="D16", provision="第十二条",
         text="因特殊保障需要，经依法批准，可以对特定区域、特定时段的飞行活动作出限制或者调整。",
         note=""),

    # ---------------------------------------------------------------- E01 U-space
    dict(cid="E01-C001", doc_id="E01", provision="Art. 5",
         text="U-space 服务应当包括地理感知（geo-awareness）服务，向无人驾驶航空器运行人提供动态空域限制信息。",
         note="境外规范，用于跨法域比较。"),
    dict(cid="E01-C002", doc_id="E01", provision="Art. 6",
         text="U-space 服务提供者应当保证服务连续性与信息可用性，并记录服务事件。",
         note=""),
    dict(cid="E02-C001", doc_id="E02", provision="§3.2",
         text="USS 之间应当交换运行意图与约束信息，以支持跨服务区的运行协调。",
         note="概念性文件，不具约束力。"),
    dict(cid="E03-C001", doc_id="E03", provision="§2.4",
         text="特定运行的风险评估应当分别评估地面风险与空中风险，并据此确定安全目标与缓解措施。",
         note="方法性参照。"),
]
