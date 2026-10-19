"""显式状态模型检验（explicit-state model checking），纯 Python 实现。

本模块在 :mod:`skygovc.protocol.handover` 给出的有限标记转移系统上，
用**广度优先搜索**穷举全部可达状态，并对三条性质分别给出
``satisfied`` / ``explored`` / ``counterexample`` / ``detail`` 四元组结果，
不依赖 TLA+ / NuSMV / SPIN 等任何外部模型检验器。

算法
----
1. **可达图构造**（``_explore``）：从 ``INITIAL`` 出发做 BFS，记录

   * ``order``：按发现顺序（即最短路径长度非降）排列的可达状态列表；
   * ``parent``：``state -> (前驱, 到达动作)``，用于 O(路径长) 反例重建；
   * ``graph``：邻接表 ``state -> [(后继, 动作)]``。

   状态是 frozen dataclass，天然可哈希，因此 ``visited`` 就是一个 ``dict``。
   若可达状态数超过 ``max_states``，抛 :class:`StateSpaceTooLarge`，避免
   在参数写错时把内存打爆。

2. **P1（不变式）**：在 ``order`` 上过滤 ``not safety_p1(s)``。不变式的
   反例是"到达违规状态的路径"。

3. **P2（无死告警活性）**：先在**反向图**上从全部 ``closed`` 状态做一次
   反向 BFS，得到集合 ``CanReachClosed``；再找出满足 ``is_deadlock_alert``
   却不在该集合中的可达状态。这在有限状态系统里等价于"存在一个 ALERT
   不再被改变、且永远到不了 CLOSE 的无限循环/死锁汇"——比单纯的
   "存在无出边的状态"更精确，因为它同时覆盖了"有出边但只能空转"的情形
   （例如 lossy_retry 中预算耗尽但仍可 resend 的暂态必须**不**被判违规）。

4. **P3（有界活性）**：取所有可达 ``closed`` 状态，检查 ``steps <= K``。
   若根本没有可达的 ``closed`` 状态，则判定活性被平凡违反（"任务永远无法
   完成"），并在 ``detail`` 中说明。

反例格式
--------
``CheckResult.counterexample`` 是 ``list[tuple[State, str]]``，从
``INITIAL`` 开始，形如::

    [(INITIAL, "init"), (s1, "reg_send"), (s2, "verify_ok"), ...]

元组第二个元素是**进入该状态**的动作标签（首项固定为 ``"init"``）；末项
即坏状态本身。用 :func:`counterexample_text` 可渲染成论文附录可直接使用的
文本。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .handover import CHANNELS, INITIAL, State, successors
from .properties import (
    K_BOUND,
    PROPERTY_ORDER,
    bounded_liveness_ok,
    is_deadlock_alert,
    safety_p1,
)

#: 反例路径类型别名：``[(状态, 进入该状态的动作), ...]``。
Path = List[Tuple[State, str]]

#: 邻接表类型别名。
Graph = Dict[State, List[Tuple[State, str]]]


class StateSpaceTooLarge(RuntimeError):
    """可达状态数超过 ``max_states`` 时抛出。"""


@dataclass
class CheckResult:
    """单条性质的检验结果。

    Attributes
    ----------
    satisfied:
        性质是否成立。
    explored:
        本次检验中穷举到的**可达状态数**（即 ``|Π|``）。三个性质的
        ``explored`` 相同，因为它们共享同一张可达图。
    counterexample:
        反例路径，``None`` 表示无违反。格式见模块 docstring。
    detail:
        人类可读的中文说明，含关键计数（便于直接写进论文表格/脚注）。
    """

    satisfied: bool
    explored: int
    counterexample: Path | None
    detail: str


# --------------------------------------------------------------------------
# 可达图构造
# --------------------------------------------------------------------------
def _explore(
    channel: str,
    max_retries: int,
    max_states: int,
) -> Tuple[List[State], Dict[State, Optional[Tuple[State, str]]], Graph]:
    """BFS 构造可达状态列表、父指针表与邻接表。"""
    if channel not in CHANNELS:
        raise ValueError(
            f"unknown channel {channel!r}; expected one of {CHANNELS}"
        )
    if max_states <= 0:
        raise ValueError(f"max_states must be positive, got {max_states}")

    order: List[State] = [INITIAL]
    parent: Dict[State, Optional[Tuple[State, str]]] = {INITIAL: None}
    graph: Graph = {}
    queue: deque = deque([INITIAL])

    while queue:
        state = queue.popleft()
        edges = successors(state, channel, max_retries)
        graph[state] = edges
        for nxt, action in edges:
            if nxt in parent:
                continue
            if len(parent) >= max_states:
                raise StateSpaceTooLarge(
                    f"state space exceeded max_states={max_states} "
                    f"(channel={channel!r}, max_retries={max_retries})"
                )
            parent[nxt] = (state, action)
            order.append(nxt)
            queue.append(nxt)

    return order, parent, graph


def _path_to(
    target: State,
    parent: Dict[State, Optional[Tuple[State, str]]],
) -> Path:
    """回溯父指针，返回从 ``INITIAL`` 到 ``target`` 的最短反例路径。"""
    chain: List[Tuple[State, str]] = []
    cursor: Optional[State] = target
    while cursor is not None:
        step = parent[cursor]
        if step is None:
            chain.append((cursor, "init"))
            cursor = None
        else:
            prev, action = step
            chain.append((cursor, action))
            cursor = prev
    chain.reverse()
    return chain


def reachable_graph(
    channel: str,
    max_retries: int = 2,
    max_states: int = 200000,
) -> Tuple[List[State], Graph]:
    """返回 ``(可达状态列表, 邻接表)``。

    状态列表按 BFS 发现顺序排列（最短路径长度非降），可用于做统计或二次
    分析（例如手工检查某个阶段的状态数）。邻接表把**每一个**可达状态映射
    到它的 ``successors`` 结果（含动作标签），终止状态映射到空列表。

    Raises
    ------
    ValueError
        通道名未知或 ``max_states <= 0``。
    StateSpaceTooLarge
        可达状态数超过 ``max_states``。
    """
    order, _parent, graph = _explore(channel, max_retries, max_states)
    return order, graph


# --------------------------------------------------------------------------
# 三条性质
# --------------------------------------------------------------------------
def _check_p1(order: List[State], parent, channel: str) -> CheckResult:
    bad = next((s for s in order if not safety_p1(s)), None)
    max_b = max(s.b_applications for s in order)
    if bad is not None:
        return CheckResult(
            satisfied=False,
            explored=len(order),
            counterexample=_path_to(bad, parent),
            detail=(
                f"P1 被违反：通道 '{channel}' 下存在 b_applications="
                f"{bad.b_applications} > 1 的可达状态（共 {len(order)} 个"
                f"可达状态）。"
            ),
        )
    return CheckResult(
        satisfied=True,
        explored=len(order),
        counterexample=None,
        detail=(
            f"P1 成立：{len(order)} 个可达状态中 b_applications 的最大值为 "
            f"{max_b}（<= 1）。该上界由 b_applied 守卫结构性保证：b_apply "
            f"只能从 delegated 触发且只触发一次。"
        ),
    )


def _can_reach_closed(order: List[State], graph: Graph) -> set:
    """反向 BFS：返回所有能到达某个 ``closed`` 状态的可达状态集合。"""
    reverse: Dict[State, List[State]] = {s: [] for s in order}
    for src, edges in graph.items():
        for dst, _action in edges:
            reverse[dst].append(src)

    seeds = [s for s in order if s.phase == "closed"]
    can_reach = set(seeds)
    queue: deque = deque(seeds)
    while queue:
        state = queue.popleft()
        for prev in reverse[state]:
            if prev not in can_reach:
                can_reach.add(prev)
                queue.append(prev)
    return can_reach


def _check_p2(order: List[State], graph: Graph, parent, channel: str) -> CheckResult:
    closed_count = sum(1 for s in order if s.phase == "closed")
    can_reach = _can_reach_closed(order, graph)
    stuck = [
        s for s in order if is_deadlock_alert(s) and s not in can_reach
    ]
    alert_wait = sum(1 for s in order if is_deadlock_alert(s))

    if stuck:
        bad = stuck[0]  # order 是 BFS 序，天然是最短反例
        return CheckResult(
            satisfied=False,
            explored=len(order),
            counterexample=_path_to(bad, parent),
            detail=(
                f"P2 被违反：{len(order)} 个可达状态中有 {len(stuck)} 个 "
                f"ALERT 永久滞留状态（alert_pending、ALERT 已发、未 ACK、"
                f"无在途消息，且无法到达 CLOSE）；通道 '{channel}' 不能保证 "
                f"ALERT 的最终送达。"
            ),
        )
    return CheckResult(
        satisfied=True,
        explored=len(order),
        counterexample=None,
        detail=(
            f"P2 成立：{len(order)} 个可达状态中未出现无法到达 CLOSE 的 "
            f"ALERT 滞留状态"
            + (
                f"；另有 {alert_wait} 个「ALERT 已发、未 ACK、信道暂空」"
                f"状态均为可重传恢复的暂态"
                if alert_wait
                else ""
            )
            + f"（{closed_count} 个 CLOSE 状态可达），不存在永久滞留的"
            f"未确认 ALERT。"
        ),
    )


def _check_p3(
    order: List[State],
    parent,
    channel: str,
    K: int,
) -> CheckResult:
    closed = [s for s in order if s.phase == "closed"]
    if not closed:
        return CheckResult(
            satisfied=False,
            explored=len(order),
            counterexample=None,
            detail=(
                f"P3 被违反：通道 '{channel}' 下无法到达 CLOSE，"
                f"有界接续活性被平凡违反。"
            ),
        )

    violating = [s for s in closed if not bounded_liveness_ok(s, K)]
    worst = max(closed, key=lambda s: s.steps)
    if violating:
        bad = min(violating, key=lambda s: s.steps)
        return CheckResult(
            satisfied=False,
            explored=len(order),
            counterexample=_path_to(bad, parent),
            detail=(
                f"P3 被违反：存在 steps={bad.steps} > K={K} 的可达 CLOSE "
                f"状态（共 {len(violating)}/{len(closed)} 个 CLOSE 状态越界）。"
            ),
        )
    return CheckResult(
        satisfied=True,
        explored=len(order),
        counterexample=None,
        detail=(
            f"P3 成立：{len(closed)} 个可达 CLOSE 状态的最大 steps="
            f"{worst.steps}（界 K={K}，余量 {K - worst.steps} 步）。"
        ),
    )


# --------------------------------------------------------------------------
# 对外入口
# --------------------------------------------------------------------------
def check(
    channel: str,
    max_retries: int = 2,
    max_states: int = 200000,
) -> Dict[str, CheckResult]:
    """在给定通道模型下检验 P1 / P2 / P3。

    Parameters
    ----------
    channel:
        ``"reliable"`` / ``"lossy"`` / ``"lossy_retry"``。
    max_retries:
        仅 ``lossy_retry`` 有意义：同一条在途消息允许被丢弃的最大次数。
    max_states:
        可达状态数上限（安全阀）。超限时三条性质都返回
        ``satisfied=False`` 并在 ``detail`` 中说明，不抛异常，便于批量实验
        在参数网格上继续跑。

    Returns
    -------
    dict[str, CheckResult]
        固定键 ``"P1"`` / ``"P2"`` / ``"P3"``，顺序同
        :data:`skygovc.protocol.properties.PROPERTY_ORDER`。三条性质共享同一
        张可达图，因此 ``explored`` 相同。
    """
    try:
        order, parent, graph = _explore(channel, max_retries, max_states)
    except StateSpaceTooLarge as exc:
        detail = f"显式状态空间超出上限，检验中止：{exc}"
        return {
            name: CheckResult(
                satisfied=False,
                explored=max_states,
                counterexample=None,
                detail=detail,
            )
            for name in PROPERTY_ORDER
        }

    results = {
        "P1": _check_p1(order, parent, channel),
        "P2": _check_p2(order, graph, parent, channel),
        "P3": _check_p3(order, parent, channel, K_BOUND),
    }
    # 保证键顺序与性质注册表一致（dict 保序）。
    return {name: results[name] for name in PROPERTY_ORDER}


def classify_channel(channel: str, max_retries: int = 2) -> str:
    """把通道分类为 ``"pass"`` 或首个被违反的性质名。

    优先级为 P1 > P2 > P3（安全性优先于活性，与论文的判定顺序一致）。
    """
    results = check(channel, max_retries=max_retries)
    for name in PROPERTY_ORDER:
        if not results[name].satisfied:
            return f"{name}_violated"
    return "pass"


def retry_boundary(
    channels: Sequence[str] = ("lossy", "lossy_retry"),
    max_retries_range: Iterable[int] = (1, 2, 3, 4, 5),
) -> List[Dict[str, object]]:
    """扫描重传预算，量化"多重传次数 → 性质恢复"的边界。

    对 ``channels × max_retries_range`` 的每个组合跑一次 :func:`check`，
    输出扁平记录列表（可直接 ``pandas.DataFrame(rows).to_markdown()``）::

        {"channel": str, "max_retries": int,
         "P1": bool, "P2": bool, "P3": bool, "explored": int}

    说明：``lossy`` 通道没有重传动作，``max_retries`` 对它无效，因此该通道
    的每一行结果完全相同——这本身就是"参数无法补偿机制缺失"的直接证据。
    """
    rows: List[Dict[str, object]] = []
    for channel in channels:
        for max_retries in max_retries_range:
            results = check(channel, max_retries=int(max_retries))
            rows.append(
                {
                    "channel": channel,
                    "max_retries": int(max_retries),
                    "P1": results["P1"].satisfied,
                    "P2": results["P2"].satisfied,
                    "P3": results["P3"].satisfied,
                    "explored": results["P1"].explored,
                }
            )
    return rows


# --------------------------------------------------------------------------
# 反例渲染
# --------------------------------------------------------------------------
def _as_pairs(path: Sequence) -> List[Tuple[State, str]]:
    """把 ``counterexample``（或裸状态序列）统一成 ``(State, action)`` 列表。"""
    pairs: List[Tuple[State, str]] = []
    for item in path:
        if isinstance(item, State):
            pairs.append((item, "init" if not pairs else "?"))
        elif isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[0], State):
            pairs.append((item[0], str(item[1])))
        else:
            raise TypeError(
                "path items must be State or (State, action) pairs, "
                f"got {type(item).__name__}"
            )
    return pairs


def counterexample_text(path: Sequence) -> str:
    """把反例路径渲染成可直接放进论文附录的中文文本。

    Parameters
    ----------
    path:
        ``CheckResult.counterexample``，或任意 ``[(State, action), ...]`` /
        ``[State, ...]`` 序列。

    Returns
    -------
    str
        多行文本：每行形如 ``[03] --b_alert--> phase=... ``，末尾附一句
        针对末状态的诊断（ALERT 死锁 / 正常 CLOSE / 其它终止）。
    """
    pairs = _as_pairs(path)
    if not pairs:
        return "(空反例路径)"

    lines: List[str] = [
        f"反例路径：{len(pairs)} 个状态，{max(0, len(pairs) - 1)} 个动作",
        "",
    ]
    for index, (state, action) in enumerate(pairs):
        if index == 0:
            head = "[00] <init>"
        else:
            head = f"[{index:02d}] --{action}-->"
        lines.append(f"{head} {state.summary()}")

    last_state = pairs[-1][0]
    lines.append("")
    if is_deadlock_alert(last_state):
        lines.append(
            "诊断：末状态是 ALERT 永久滞留（P2 反例）。ALERT 已发出、从未被 "
            "ACK，且信道上已无该消息（in_flight=None）；当前通道没有可用的"
            "重传动作，状态机因此无法到达 CLOSE，告警被永久丢失。"
        )
    elif last_state.phase == "closed":
        lines.append(
            f"诊断：末状态正常到达 CLOSE（协议推进 steps={last_state.steps}）。"
        )
    elif last_state.phase == "rejected":
        lines.append("诊断：末状态被 A 核验拒绝（rejected），任务未进入 B 辖区。")
    elif last_state.is_terminal:
        lines.append("诊断：末状态为终止状态。")
    else:
        lines.append(
            f"诊断：末状态不可达 CLOSE（phase={last_state.phase}）。"
        )
    return "\n".join(lines)
