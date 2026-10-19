"""``skygovc.protocol`` —— 跨辖区接续协议的建模与显式状态模型检验。

公开 API
--------
状态机（:mod:`skygovc.protocol.handover`）

* :class:`State` —— frozen dataclass（可哈希）的完整协议状态
* :data:`INITIAL` —— 初始状态
* :func:`successors` —— 给定通道模型下的后继 ``(状态, 动作标签)`` 列表
* :data:`CHANNELS` / :data:`MESSAGES` / :data:`PHASES` —— 词汇表常量

性质（:mod:`skygovc.protocol.properties`）

* :func:`safety_p1` —— P1 不重复申报（不变式 ``b_applications <= 1``）
* :func:`is_deadlock_alert` —— P2 的坏状态谓词（ALERT 永久滞留）
* :func:`bounded_liveness_ok` —— P3 有界接续
* :data:`PROPERTIES` —— ``name -> (kind, 描述)`` 注册表

模型检验（:mod:`skygovc.protocol.modelcheck`）

* :func:`check` —— 对 P1/P2/P3 做 BFS 全可达图检验
* :class:`CheckResult` —— 检验结果（含反例路径）
* :func:`reachable_graph` —— 可达状态列表 + 邻接表
* :func:`classify_channel` —— ``pass`` / ``P1_violated`` / ``P2_violated`` / ``P3_violated``
* :func:`retry_boundary` —— 重传预算扫描表
* :func:`counterexample_text` —— 反例路径的可读文本渲染

典型用法::

    from skygovc.protocol import check, classify_channel, counterexample_text

    results = check("lossy")
    print(results["P2"].satisfied)                 # False
    print(counterexample_text(results["P2"].counterexample))
    print(classify_channel("lossy_retry", 2))      # "pass"
"""
from __future__ import annotations

from .handover import (
    CHANNELS,
    DEFAULT_K,
    INITIAL,
    MAX_B_APPLICATIONS,
    MESSAGES,
    PHASES,
    TERMINAL_PHASES,
    State,
    successors,
)
from .modelcheck import (
    CheckResult,
    StateSpaceTooLarge,
    check,
    classify_channel,
    counterexample_text,
    reachable_graph,
    retry_boundary,
)
from .properties import (
    KINDS,
    K_BOUND,
    PROPERTIES,
    PROPERTY_ORDER,
    bounded_liveness_ok,
    is_deadlock_alert,
    property_kind,
    safety_p1,
)

__version__ = "1.0.0"

__all__ = [
    # handover
    "State",
    "INITIAL",
    "successors",
    "PHASES",
    "MESSAGES",
    "CHANNELS",
    "TERMINAL_PHASES",
    "MAX_B_APPLICATIONS",
    "DEFAULT_K",
    # properties
    "safety_p1",
    "is_deadlock_alert",
    "bounded_liveness_ok",
    "PROPERTIES",
    "PROPERTY_ORDER",
    "KINDS",
    "K_BOUND",
    "property_kind",
    # model checking
    "CheckResult",
    "StateSpaceTooLarge",
    "check",
    "reachable_graph",
    "classify_channel",
    "retry_boundary",
    "counterexample_text",
    "__version__",
]
