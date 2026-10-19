"""三条待检验性质的形式化定义（P1 / P2 / P3）。

本模块把论文中的三条性质写成**纯谓词**，不含任何搜索逻辑；搜索与反例生成
放在 :mod:`skygovc.protocol.modelcheck`。谓词与搜索严格分离，是为了让
"性质是什么"与"怎么检验"在论文与代码中都保持可独立审阅。

形式化记号（LTL / CTL 风格）
---------------------------
记 ``Π`` 为可达状态集合，``S0 = INITIAL``，``⟨S, →⟩`` 为 ``successors``
诱导的标记转移系统，``A = {closed}``。

* **P1**::

      ∀ s ∈ Π .  s.b_applications <= 1

  即不变式 ``□(b_applications <= 1)``。

* **P2**::

      ¬∃ s ∈ Π .  deadlock_alert(s) ∧ ¬∃π: s →* closed

  其中 ``deadlock_alert(s) ≡ s.phase = alert_pending ∧ s.alert_issued
  ∧ ¬s.alert_acked ∧ s.in_flight = None``。亦即不存在"ALERT 已发出、未被
  确认、信道已空、且之后再也到不了 CLOSE"的可达状态——论文中"没有未被确认
  的 ALERT 永久滞留"的精确化。标准的"不存在无限循环中 ALERT 不再被改变"
  表述与此等价：在有限状态系统里，"永远无法到达 CLOSE"就等于"被困在一个
  不包含 CLOSE 的强连通/死锁汇中"。

* **P3**::

      ∀ s ∈ Π .  s.phase = closed  ⇒  s.steps <= K

  即完成路径的有界活性 ``□(closed ⇒ steps <= K)``，默认 ``K = 12``。

三条性质都只使用 :class:`skygovc.protocol.handover.State`，因此可以被
单元测试单独驱动（例如手工构造一个 ``b_applications = 2`` 的状态来验证
P1 谓词确实会拒绝它）。
"""
from __future__ import annotations

from typing import Dict, Tuple

from .handover import DEFAULT_K, State

#: P3 的默认界：从 ``REGISTER`` 到 ``CLOSE`` 允许的最大协议推进步数。
K_BOUND: int = DEFAULT_K

#: 性质种类：不变量 / 有界活性 / 无死告警活性。
KINDS: Tuple[str, ...] = (
    "invariant",
    "liveness_bounded",
    "liveness_no_dead_alert",
)


def safety_p1(s: State) -> bool:
    """P1（安全·不重复申报）：任务在 B 辖区最多一次有效申请。

    直接对应论文的不变式 ``b_applications <= 1``。该式在状态机层面由
    ``b_applied`` 守卫结构性保证，模型检验的作用是穷举确认它无法被绕过。
    """
    return s.b_applications <= 1


def is_deadlock_alert(s: State) -> bool:
    """P2（安全·不丢告警）的坏状态谓词：ALERT 永久卡住。

    条件为四个合取项：

    * ``phase == "alert_pending"``：处于等待告警闭环的阶段；
    * ``alert_issued``：ALERT 确实已经发出过；
    * ``not alert_acked``：始终没有被 ACK；
    * ``in_flight is None``：信道上已无 ALERT——也就是说没有"还在路上、
      仍可能被确认"的希望。

    在 ``lossy`` 通道下，该谓词为真即意味着永久卡死（无重传动作可用）；
    在 ``lossy_retry`` 通道下，该谓词还可能是"预算耗尽但可重传"的**暂态**，
    因此 P2 的判定必须再叠加"从该状态无法到达 CLOSE"（见
    :func:`skygovc.protocol.modelcheck.check`）。
    """
    return (
        s.phase == "alert_pending"
        and s.alert_issued
        and not s.alert_acked
        and s.in_flight is None
    )


def bounded_liveness_ok(s: State, K: int) -> bool:
    """P3（活性·有界接续）：``closed`` 状态必须满足 ``steps <= K``。

    非 ``closed`` 状态不承担界（它们还没有"完成"），因此一律返回 ``True``；
    这样该谓词可以直接用于逐状态过滤，而完整判定"所有可达 CLOSE 都在界内"
    由 :mod:`skygovc.protocol.modelcheck` 在可达集上做全称量化。
    """
    if s.phase != "closed":
        return True
    return s.steps <= K


#: 性质注册表：``name -> (kind, 人类可读描述)``。
#: ``kind`` 取值见 :data:`KINDS`。
PROPERTIES: Dict[str, Tuple[str, str]] = {
    "P1": (
        "invariant",
        "P1 不重复申报（安全/不变式）：任意可达状态下 b_applications <= 1，"
        "即任务在 B 辖区最多产生一次有效申请。形式化：□(b_applications <= 1)。",
    ),
    "P2": (
        "liveness_no_dead_alert",
        "P2 不丢告警（安全/无死告警活性）：不存在可达的 alert_pending 状态"
        "同时满足「ALERT 已发、未被 ACK、无在途消息」且无法再到达 CLOSE；"
        "即任何 ALERT 最终都会被 ACK 或升级处置，不会永久滞留。",
    ),
    "P3": (
        "liveness_bounded",
        f"P3 有界接续（活性/有界响应）：任意可达 closed 状态都满足 "
        f"steps <= K（K={K_BOUND}），即从 REGISTER 到 CLOSE 的协议推进"
        f"路径有界。",
    ),
}

#: 报告与结果字典中的固定顺序。
PROPERTY_ORDER: Tuple[str, ...] = ("P1", "P2", "P3")


def property_kind(name: str) -> str:
    """返回性质 ``name`` 的种类；未注册的名字抛 ``KeyError``。"""
    return PROPERTIES[name][0]
