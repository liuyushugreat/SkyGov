"""Shared data model for SkyGov-C.

This module is deliberately dependency-free (standard library only) so that
every sub-package can import it without creating cycles.  All identifiers are
plain strings; all records are dataclasses so that they serialise trivially to
JSON (``dataclasses.asdict``).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
import dataclasses
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


# --------------------------------------------------------------------------
# Institutional layer
# --------------------------------------------------------------------------
#: Legal hierarchy used for conflict-aware merging.  Lower value == higher rank.
HIERARCHY_RANK: Dict[str, int] = {
    "law": 1,               # 全国人大及其常委会制定的法律
    "admin_regulation": 2,  # 国务院（含与中央军委联合）行政法规
    "dept_rule": 3,         # 部门规章 / 行业规范
    "local_regulation": 4,  # 地方性法规
    "local_notice": 5,      # 地方政府通告 / 公告
    "policy": 6,            # 政策文件（非规范性法律文件）
    "external_standard": 9,  # 境外或国际规范，仅作比较基准，不参与国内优先级
}


@dataclass(frozen=True)
class Clause:
    """One clause-level record of an institutional instrument.

    ``text`` is a *normalised* rendering of the provision at proposition
    granularity.  It keeps the operative verb, the modal particle and every
    condition that the official text attaches to the obligation, but strips
    editorial framing.  ``verbatim`` records whether ``text`` is a literal
    quotation of the published article.
    """

    cid: str
    doc_id: str
    doc_title: str
    issuer: str
    level: str
    instrument_date: str
    effective_date: Optional[str]
    provision: str            # e.g. "第十二条" or "三、(二)"
    text: str
    verbatim: bool
    source_url: Optional[str] = None
    source_note: str = ""
    jurisdiction: str = "CN-NATIONAL"   # CN-NATIONAL, CN-BJ, CN-TJ, ...
    split: str = "dev"                  # "dev" | "heldout"

    @property
    def rank(self) -> int:
        return HIERARCHY_RANK.get(self.level, 8)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Proposition layer (output of extraction)
# --------------------------------------------------------------------------
class Deontic(str, Enum):
    OBLIGATION = "obligation"
    PROHIBITION = "prohibition"
    PERMISSION = "permission"
    PROCEDURE = "procedure"


#: Controlled vocabulary for the regulated action.  Kept small on purpose: a
#: large vocabulary makes matching degenerate, a tiny one makes it trivial.
ACTIONS: Tuple[str, ...] = (
    "register_uas",
    "hold_licence",
    "apply_flight_activity",
    "operate",                      # 泛指的“实施飞行活动”
    "operate_in_permitted_airspace",
    "operate_in_controlled_airspace",
    "carry_payload_for_hire",
    "sell_uas",
    "transport_uas",
    "store_uas",
    "counter_uas",
    "publish_airspace_notice",
    "provide_flight_service",
    "build_vertiport",
    "report_incident",
    "share_operational_data",
    "train_operator",
    "fund_industry",
    "insure",
    "conduct_drill",
    "appoint_safety_officer",
    "undergo_inspection",
)

SUBJECTS: Tuple[str, ...] = (
    "operator",        # 运行组织者 / 运营人
    "pilot",           # 操控员
    "manufacturer",    # 生产者
    "owner",           # 持有者
    "local_government",
    "airspace_authority",
    "public_security",
    "service_provider",
    "legislature",
    "central_government",
    "service_platform",
)

#: Predicate vocabulary for conditions.  A condition is ``name op value``.
CONDITION_NAMES: Tuple[str, ...] = (
    "altitude_agl_m",
    "uas_category",          # micro | light | small | medium | large
    "airspace_class",        # A..E | G | W | permitted | controlled
    "over_populated_area",
    "within_airport_buffer",
    "cross_border",
    "night_operation",
    "bvlos",                 # beyond visual line of sight
    "payload_kg",
    "mass_kg",
    "risk_class",            # low | medium | high
    "operator_licence",
    "registered",
    "info_reported",
    "purpose",               # logistics | survey | emergency | recreation
)

OPERATORS: Tuple[str, ...] = ("<=", ">=", "==", "!=", "in", "not_in", "exists")


@dataclass
class Condition:
    name: str
    op: str
    value: Any

    def key(self) -> Tuple[str, str, str]:
        return (self.name, self.op, str(self.value))

    def __str__(self) -> str:  # for SHACL/SPARQL rendering and explanation
        return f"{self.name} {self.op} {self.value}"


@dataclass
class Proposition:
    """Normalised institutional proposition."""

    pid: str
    cid: str
    deontic: str
    subject: str
    action: str
    conditions: List[Condition] = field(default_factory=list)
    spatial: Optional[str] = None
    temporal: Optional[str] = None
    sanction: Optional[str] = None
    # provenance / confidence
    extractor: str = "symbolic"
    confidence: float = 1.0
    compilable: bool = True
    normalised_from: str = ""     # raw sentence the proposition came from
    unresolved_refs: List[str] = field(default_factory=list)

    def signature(self) -> Tuple[Any, ...]:
        """Equality key used for matching against the expert gold standard."""
        return (
            self.deontic,
            self.subject,
            self.action,
            tuple(sorted(c.key() for c in self.conditions)),
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


# --------------------------------------------------------------------------
# Decision layer
# --------------------------------------------------------------------------
@dataclass
class FlightRequest:
    """A task-level request presented to the governance system."""

    rid: str
    origin_jurisdiction: str
    dest_jurisdiction: str
    risk_class: str = "low"            # low | medium | high
    airspace_class: str = "W"          # A..E | G | W
    altitude_agl_m: float = 80.0
    uas_category: str = "light"
    mass_kg: float = 4.0
    payload_kg: float = 1.0
    purpose: str = "logistics"
    over_populated_area: bool = False
    within_airport_buffer: bool = False
    cross_border: bool = False
    night_operation: bool = False
    bvlos: bool = False
    operator_licence: bool = True
    registered: bool = True

    def facts(self) -> Dict[str, Any]:
        return {
            "airspace_class": self.airspace_class,
            "altitude_agl_m": self.altitude_agl_m,
            "uas_category": self.uas_category,
            "mass_kg": self.mass_kg,
            "payload_kg": self.payload_kg,
            "purpose": self.purpose,
            "over_populated_area": self.over_populated_area,
            "within_airport_buffer": self.within_airport_buffer,
            "cross_border": self.cross_border,
            "night_operation": self.night_operation,
            "bvlos": self.bvlos,
            "operator_licence": self.operator_licence,
            "registered": self.registered,
            "risk_class": self.risk_class,
        }


@dataclass
class Verdict:
    """Outcome of symbolic evaluation, with an auditable provenance chain."""

    rid: str
    allowed: bool
    tier: str
    triggered: List[str] = field(default_factory=list)   # rule ids
    chain: List[str] = field(default_factory=list)       # cid lineage
    reasons: List[str] = field(default_factory=list)
    required_capabilities: List[str] = field(default_factory=list)


#: Six governance tiers used throughout the paper.
TIERS: Tuple[str, ...] = (
    "capital_core",     # 首都政治核心
    "dense_hub",        # 高密度枢纽
    "market_driven",    # 市场化运行
    "mountainous",      # 山地立体
    "border_coastal",   # 边海跨境
    "sparse_resource",  # 低密度资源
)

#: Six capability families.
CAPABILITIES: Tuple[str, ...] = (
    "rule",       # 规则
    "spatial",    # 空间
    "service",    # 运行服务
    "sensing",    # 感知防护
    "emergency",  # 风险应急
    "data",       # 产业数据
)


@dataclass
class ContextFingerprint:
    """Machine-readable description of a local governance context."""

    jurisdiction: str
    law_constraints: List[str] = field(default_factory=list)
    population_exposure: float = 0.0        # 0..1
    critical_facility: float = 0.0          # 0..1
    demand_density: float = 0.0             # 0..1
    terrain_complexity: float = 0.0         # 0..1
    border_coastal: bool = False
    seasonal: bool = False
    airspace_control_share: float = 0.0     # fraction of city under control

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Execution layer results
# --------------------------------------------------------------------------
@dataclass
class RegimeResult:
    name: str
    incident_per_10k: float
    completion: float
    unauth_rate: float
    cancel_rate: float
    duplicates_per_cross: float
    interrupt_rate: float
    mean_delay_h: float
    p95_delay_h: float
    cost_per_1k: float
    flights: int
    extra: Dict[str, Any] = field(default_factory=dict)


def to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses/enums/numpy scalars to JSON-safe values.

    Dataclasses are traversed field-by-field.  The ``to_dict`` protocol is
    deliberately *not* consulted here, because several ``to_dict``
    implementations delegate to this function and would recurse.
    """
    if isinstance(obj, Enum):
        return obj.value
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name))
                for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, "item"):          # numpy scalar
        return obj.item()
    if hasattr(obj, "tolist"):        # numpy array
        return obj.tolist()
    return obj


