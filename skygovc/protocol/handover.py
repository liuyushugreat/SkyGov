"""Cross-jurisdiction low-altitude mission handover protocol (A -> B).

跨辖区（jurisdiction A 出发地 / jurisdiction B 目的地）低空任务接续协议的
显式状态机与通道模型。

本模块只依赖 Python 标准库，把论文所述的接续协议实现为一个有限、离散、
非确定性的转移系统（Kripke 结构 / 标记转移系统），由
:mod:`skygovc.protocol.modelcheck` 用显式状态 BFS 穷举。五类消息构成的
主干序列为::

    REGISTER -> VERIFY -> DELEGATE -> {b_apply, ALERT -> ACK} -> CLOSE

与 TLA+ / NuSMV / SPIN 等外部模型检验器不同，这里的状态、转移与通道语义
都是普通 Python 数据对象，因此反例路径可以被直接回放、断言与序列化。

三个关键设计点
--------------
* **P1 由结构保证**：``b_apply`` 只在 ``delegated`` 阶段、且 ``b_applied``
  为假时可用；触发后 ``b_applied=True``、``b_applications`` 加一，且永不
  复位。于是任意可达状态的 ``b_applications <= 1``，与通道模型无关。
* **通道是一等公民**：三种通道模型只在两条通道动作上有所不同——``drop:<MSG>``
  与 ``resend:<MSG>``。阶段推进动作与通道无关，因此三组实验是干净对照。
* **ALERT 送达被显式建模**：``b_ack`` 要求 ``in_flight == "ALERT"``，即只有
  仍在信道上的 ALERT 才能被确认。``lossy`` 通道下 ALERT 被丢弃且无重传时，
  状态机永久停在 ``alert_pending``，这正是 P2 的反例。

步数口径（steps）
-----------------
``steps`` 只累计**协议推进**动作（``reg_send`` / ``verify_ok`` /
``verify_fail`` / ``delegate_send`` / ``b_apply`` / ``b_alert`` / ``b_ack`` /
``b_close``），通道动作 ``drop:*`` / ``resend:*`` 不推进协议，其代价由
``retries_used`` 记账。原因见 README「P3 的步数口径」一节：若把 drop/resend
也计入 ``steps``，``lossy_retry``（``max_retries=2``）的最坏完成路径为
7 + 2*2*5 = 27 步，P3 的界 K=12 必然被违反，与"``lossy_retry`` 满足 P3"
的结论直接矛盾。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Tuple

#: 协议阶段（含两个终止阶段 ``closed`` / ``rejected``）。
PHASES: Tuple[str, ...] = (
    "idle",           # 尚无任务
    "registered",     # A 已收下运营人的 REGISTER 申报
    "verified",       # A 核验合法性 / 空域条件通过（VERIFY）
    "delegated",      # A 已向 B 移交状态与责任（DELEGATE）
    "alert_pending",  # B 需要告警并转交第三方处置主体（ALERT 已发出）
    "acked",          # B 已确认收到 ALERT（ACK）
    "closed",         # 任务结束、留痕（CLOSE）
    "rejected",       # A 核验未通过
)

#: 五类协议消息。
MESSAGES: Tuple[str, ...] = ("REGISTER", "VERIFY", "DELEGATE", "ALERT", "ACK")

#: 三种信道模型。
CHANNELS: Tuple[str, ...] = ("reliable", "lossy", "lossy_retry")

#: 终止阶段：不再有阶段推进动作。
TERMINAL_PHASES: Tuple[str, ...] = ("closed", "rejected")

#: ``b_applications`` 的声明上界（可达状态中实际最多为 1，见 ``b_apply``）。
MAX_B_APPLICATIONS: int = 2

#: P3 的默认界（协议推进步数）。
DEFAULT_K: int = 12


@dataclass(frozen=True)
class State:
    """接续协议的完整（全息）状态。

    所有字段都是不可变类型，因此 ``State`` 天然可哈希，可以直接放进
    ``set`` / ``dict`` 作为显式状态搜索的 visited 集合键。

    Attributes
    ----------
    phase:
        当前协议阶段，取值见 :data:`PHASES`。
    b_applications:
        B 辖区内的"有效申请"计数。P1 要求其恒 ``<= 1``。
    b_applied:
        B 是否已经建立过本辖区申请记录（``b_apply`` 的守卫，保证只加一次）。
    alert_issued:
        是否已经发出过 ALERT。
    alert_acked:
        是否已经收到 ALERT 的 ACK。
    retries_used:
        **当前在途消息**已消耗的重传次数（发送新消息时归零）。
    in_flight:
        仍在信道上的消息类型；``None`` 表示当前没有在途消息（已送达或已丢弃）。
    last_msg:
        最近一次尝试发送的消息类型，供 ``resend`` 恢复。
    steps:
        已发生的**协议推进**步数（通道动作不计，见模块 docstring）。
    """

    phase: str
    b_applications: int
    b_applied: bool
    alert_issued: bool
    alert_acked: bool
    retries_used: int
    in_flight: str | None
    last_msg: str | None
    steps: int

    # ------------------------------------------------------------------
    # 便利方法（不参与哈希/相等性）
    # ------------------------------------------------------------------
    @property
    def is_terminal(self) -> bool:
        """是否为终止阶段（``closed`` / ``rejected``）。"""
        return self.phase in TERMINAL_PHASES

    def summary(self) -> str:
        """紧凑的单行摘要，用于反例渲染与日志。"""
        return (
            f"phase={self.phase} b_app={self.b_applications} "
            f"b_applied={int(self.b_applied)} "
            f"alert={'issued' if self.alert_issued else '-'}"
            f"/{'acked' if self.alert_acked else '-'} "
            f"retries={self.retries_used} "
            f"in_flight={self.in_flight or '-'} "
            f"last={self.last_msg or '-'} steps={self.steps}"
        )


#: 初始状态：任务尚未提交，B 辖区无申请记录，无在途消息。
INITIAL: State = State(
    phase="idle",
    b_applications=0,
    b_applied=False,
    alert_issued=False,
    alert_acked=False,
    retries_used=0,
    in_flight=None,
    last_msg=None,
    steps=0,
)


def successors(
    s: State,
    channel: str,
    max_retries: int = 2,
) -> List[Tuple[State, str]]:
    """返回 ``s`` 在给定通道模型下的全部后继 ``(后继状态, 动作标签)``。

    动作标签的取值：

    * 阶段推进：``reg_send`` / ``verify_ok`` / ``verify_fail`` /
      ``delegate_send`` / ``b_apply`` / ``b_alert`` / ``b_ack`` / ``b_close``
    * 通道动作：``drop:<MSG>`` / ``resend:<MSG>``

    返回顺序是确定的（先阶段推进，再 ``drop``，最后 ``resend``），以便 BFS
    产生的反例在多次运行间完全可复现。

    Parameters
    ----------
    s:
        当前状态。
    channel:
        ``"reliable"`` / ``"lossy"`` / ``"lossy_retry"``。
    max_retries:
        ``lossy_retry`` 通道下允许丢弃同一在途消息的最大次数；丢满之后
        必须送达（"强制送达"）。``reliable`` / ``lossy`` 下被忽略。

    Raises
    ------
    ValueError
        通道名未知，或 ``max_retries`` 为负。
    """
    if channel not in CHANNELS:
        raise ValueError(
            f"unknown channel {channel!r}; expected one of {CHANNELS}"
        )
    if not isinstance(max_retries, int) or isinstance(max_retries, bool):
        raise TypeError(f"max_retries must be int, got {type(max_retries).__name__}")
    if max_retries < 0:
        raise ValueError(f"max_retries must be >= 0, got {max_retries}")

    out: List[Tuple[State, str]] = []
    nxt = s.steps + 1
    phase = s.phase

    # ------------------------------------------------------------------
    # 1) 阶段推进（与通道无关）
    # ------------------------------------------------------------------
    if phase == "idle":
        # A 收下运营人的任务申报，并把 REGISTER 发给 A 的受理侧。
        out.append(
            (
                replace(
                    s,
                    phase="registered",
                    in_flight="REGISTER",
                    last_msg="REGISTER",
                    retries_used=0,
                    steps=nxt,
                ),
                "reg_send",
            )
        )

    elif phase == "registered":
        # A 核验通过：产生 VERIFY 记录并进入 verified。
        out.append(
            (
                replace(
                    s,
                    phase="verified",
                    in_flight="VERIFY",
                    last_msg="VERIFY",
                    retries_used=0,
                    steps=nxt,
                ),
                "verify_ok",
            )
        )
        # A 核验不通过：任务被拒，不产生 B 辖区申请。
        out.append(
            (
                replace(
                    s,
                    phase="rejected",
                    in_flight=None,
                    retries_used=0,
                    steps=nxt,
                ),
                "verify_fail",
            )
        )

    elif phase == "verified":
        # A 向 B 移交状态与责任；此时**不**增加 b_applications。
        out.append(
            (
                replace(
                    s,
                    phase="delegated",
                    in_flight="DELEGATE",
                    last_msg="DELEGATE",
                    retries_used=0,
                    steps=nxt,
                ),
                "delegate_send",
            )
        )

    elif phase == "delegated":
        # B 建立本辖区有效申请记录：受 b_applied 守卫，全局至多一次（P1）。
        if not s.b_applied:
            out.append(
                (
                    replace(
                        s,
                        b_applications=min(MAX_B_APPLICATIONS, s.b_applications + 1),
                        b_applied=True,
                        steps=nxt,
                    ),
                    "b_apply",
                )
            )
        # B 判定需要告警并转交第三方处置主体。
        out.append(
            (
                replace(
                    s,
                    phase="alert_pending",
                    alert_issued=True,
                    in_flight="ALERT",
                    last_msg="ALERT",
                    retries_used=0,
                    steps=nxt,
                ),
                "b_alert",
            )
        )
        # 无告警，直接结束并留痕。
        out.append(
            (
                replace(
                    s,
                    phase="closed",
                    in_flight=None,
                    retries_used=0,
                    steps=nxt,
                ),
                "b_close",
            )
        )

    elif phase == "alert_pending":
        # B 确认收到 ALERT——前提是 ALERT 仍在信道上（送达才是可确认的）。
        if s.in_flight == "ALERT" and not s.alert_acked:
            out.append(
                (
                    replace(
                        s,
                        phase="acked",
                        alert_acked=True,
                        in_flight="ACK",
                        last_msg="ACK",
                        retries_used=0,
                        steps=nxt,
                    ),
                    "b_ack",
                )
            )

    elif phase == "acked":
        # 告警闭环后结束并留痕。
        out.append(
            (
                replace(
                    s,
                    phase="closed",
                    in_flight=None,
                    retries_used=0,
                    steps=nxt,
                ),
                "b_close",
            )
        )

    # phase in {"closed", "rejected"}: 无阶段推进动作。

    # ------------------------------------------------------------------
    # 2) 通道动作 drop:<MSG>
    # ------------------------------------------------------------------
    if channel != "reliable" and s.in_flight is not None:
        if channel == "lossy":
            # 无重传：消息永久丢失，相位不变，也没有重试记账。
            out.append(
                (
                    replace(s, in_flight=None),
                    f"drop:{s.in_flight}",
                )
            )
        else:  # lossy_retry
            # 有限重传：丢一次消耗一次预算；预算耗尽后禁止再丢（强制送达）。
            if s.retries_used < max_retries:
                out.append(
                    (
                        replace(
                            s,
                            in_flight=None,
                            retries_used=s.retries_used + 1,
                        ),
                        f"drop:{s.in_flight}",
                    )
                )

    # ------------------------------------------------------------------
    # 3) 通道动作 resend:<MSG>
    # ------------------------------------------------------------------
    if (
        channel == "lossy_retry"
        and s.in_flight is None
        and s.last_msg is not None
        and not s.is_terminal
    ):
        # 发送方重新发出最近一次尝试的消息。注意：此动作不受 retries_used
        # 约束——这正是"预算耗尽后强制送达"得以成立的机制，也是 lossy_retry
        # 恢复 P2 的充分条件。
        out.append(
            (
                replace(s, in_flight=s.last_msg),
                f"resend:{s.last_msg}",
            )
        )

    return out
