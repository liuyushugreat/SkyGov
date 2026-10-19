# `skygovc.protocol` —— 跨辖区接续协议的显式状态建模与模型检验

本子包把论文中"跨辖区（jurisdiction A 出发地 → jurisdiction B 目的地）低空任务
接续协议"实现为一个**有限、离散、非确定性**的标记转移系统，并用**显式状态
广度优先搜索**从零实现了模型检验：

* 安全性不变式 **P1**（不重复申报）；
* 安全性/活性 **P2**（不丢告警，以"无永久滞留的未确认 ALERT"表述）；
* 有界活性 **P3**（有界接续，`K = 12`）；
* 反例路径生成与可读渲染（可直接放进论文附录）。

不依赖 TLA+ / NuSMV / SPIN，也不依赖任何第三方 Python 包：**只用标准库**
（`dataclasses` / `collections` / `typing`）。状态是 `frozen dataclass`，
因此可达集就是一个 `dict`，反例就是普通 Python 列表，可以被单元测试直接
断言与回放。

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | 公开 API 汇总（`__all__`） |
| `handover.py` | 状态 `State`、初始状态 `INITIAL`、转移函数 `successors`、通道模型 |
| `properties.py` | P1/P2/P3 的纯谓词与性质注册表 `PROPERTIES` |
| `modelcheck.py` | BFS 可达图、三条性质的检验、反例重建、重传边界扫描、文本渲染 |
| `../../tests/test_protocol.py` | 21 项单元/回归测试 |

---

## 1. 协议状态机

### 1.1 状态变量

`State`（`handover.py`，`@dataclass(frozen=True)`，全部字段为不可变类型，
因此天然可哈希、可放进 `set`）共 9 个字段，构成完整（全息）状态：

| 字段 | 类型 | 取值 | 含义 |
| --- | --- | --- | --- |
| `phase` | `str` | `idle` / `registered` / `verified` / `delegated` / `alert_pending` / `acked` / `closed` / `rejected` | 协议阶段 |
| `b_applications` | `int` | 声明域 `0..2`（可达集实测最大 **1**） | B 辖区有效申请计数 |
| `b_applied` | `bool` | — | B 是否已建立过本辖区申请记录（`b_apply` 的守卫） |
| `alert_issued` | `bool` | — | ALERT 是否已发出 |
| `alert_acked` | `bool` | — | ALERT 是否已被 ACK |
| `retries_used` | `int` | `0..max_retries` | **当前在途消息**已消耗的重传次数 |
| `in_flight` | `str \| None` | `"REGISTER"` / `"VERIFY"` / `"DELEGATE"` / `"ALERT"` / `"ACK"` / `None` | 仍在信道上的消息；`None` = 无在途 |
| `last_msg` | `str \| None` | 同上 | 最近一次尝试发送的消息，供 `resend` 恢复 |
| `steps` | `int` | `0..7`（三种通道实测一致） | 已发生的**协议推进**步数（口径见 §7.2） |

初始状态：

```python
INITIAL = State(phase="idle", b_applications=0, b_applied=False,
                alert_issued=False, alert_acked=False, retries_used=0,
                in_flight=None, last_msg=None, steps=0)
```

### 1.2 转移表

`successors(s, channel, max_retries=2) -> list[(State, str)]` 返回
`(后继状态, 动作标签)`。标签取值：`reg_send` / `verify_ok` / `verify_fail` /
`delegate_send` / `b_apply` / `b_alert` / `b_ack` / `b_close` /
`drop:<MSG>` / `resend:<MSG>`。返回顺序确定（先推进、再 `drop`、最后
`resend`），保证 BFS 反例在多次运行间完全可复现。

| # | 动作 | 前置条件 | 效果（除 `steps`/`in_flight` 外） | 标签 |
| --- | --- | --- | --- | --- |
| 1 | `reg_send` | `phase = idle` | `phase←registered`，`in_flight←REGISTER`，`last_msg←REGISTER` | `reg_send` |
| 2 | `verify_ok` | `phase = registered` | `phase←verified`，`in_flight←VERIFY` | `verify_ok` |
| 3 | `verify_fail` | `phase = registered` | `phase←rejected`，`in_flight←None` | `verify_fail` |
| 4 | `delegate_send` | `phase = verified` | `phase←delegated`，`in_flight←DELEGATE`；**不加** `b_applications` | `delegate_send` |
| 5 | `b_apply` | `phase = delegated` ∧ `¬b_applied` | `b_applications += 1`，`b_applied←True` | `b_apply` |
| 6 | `b_alert` | `phase = delegated` | `phase←alert_pending`，`alert_issued←True`，`in_flight←ALERT` | `b_alert` |
| 7 | `b_ack` | `phase = alert_pending` ∧ `in_flight = ALERT` ∧ `¬alert_acked` | `phase←acked`，`alert_acked←True`，`in_flight←ACK` | `b_ack` |
| 8 | `b_close` | `phase = delegated` | `phase←closed`，`in_flight←None` | `b_close` |
| 9 | `b_close` | `phase = acked` | `phase←closed`，`in_flight←None` | `b_close` |
| 10 | `drop(m)` | `in_flight ≠ None` ∧ 通道允许（见 §1.3） | `in_flight←None`（`lossy_retry` 另加 `retries_used += 1`） | `drop:ALERT` 等 |
| 11 | `resend` | `lossy_retry` ∧ `in_flight = None` ∧ `last_msg ≠ None` ∧ `¬terminal` | `in_flight←last_msg` | `resend:ALERT` 等 |

所有推进动作 `steps += 1`；`drop` / `resend` **不**推进协议，不改变 `steps`。

> **P1 的结构性保证**：`b_apply` 同时受 `phase = delegated` 与 `¬b_applied`
> 双重守卫，触发后 `b_applied` 永不复位。因此任意可达状态的
> `b_applications ≤ 1`，与通道模型无关。模型检验在这里不是"发现"不变式，
> 而是穷举确认它无法被绕过。

### 1.3 三种通道模型

| 通道 | `drop` | `resend` | 语义 |
| --- | --- | --- | --- |
| `reliable` | **不允许** | 不需要 | 消息必达；`successors` 不产生任何 `drop:*` / `resend:*` |
| `lossy` | 允许（`in_flight ≠ None` 即可），`retries_used` 不变 | **不允许** | 每条消息可被丢弃一次，**无重传**；丢失即永久丢失 |
| `lossy_retry` | 允许，但仅当 `retries_used < max_retries`，且 `retries_used += 1` | 允许（`in_flight = None` 时） | 有限重传；预算耗尽后**强制送达**（禁止再 `drop`），`resend` 本身不受预算约束 |

`max_retries` 只对 `lossy_retry` 有意义；`reliable` / `lossy` 忽略该参数
（`retry_boundary()` 中对 `lossy` 扫 `max_retries ∈ 1..5` 得到的 5 行完全相同，
正是"参数无法补偿机制缺失"的直接证据）。

---

## 2. 三条性质

记号：`Π` 为可达状态集，`S₀ = INITIAL`，`A = {closed}`。

### P1（安全·不重复申报）— `safety_p1`

```
∀ s ∈ Π .  s.b_applications ≤ 1            □(b_applications ≤ 1)
```

判定：在 BFS 可达表上过滤 `not safety_p1(s)`；不变式的反例是"到达违规状态的
最短路径"。上限由 `b_applied` 守卫结构性保证，三种通道都成立。

### P2（安全·不丢告警）— `is_deadlock_alert` + 反向可达

坏状态谓词：

```
deadlock_alert(s) ≡ s.phase = alert_pending ∧ s.alert_issued
                    ∧ ¬s.alert_acked ∧ s.in_flight = None
```

性质（论文"不存在可达的无限循环状态，其中存在一个未被确认的 ALERT 且该
ALERT 之后不会改变状态"的精确化）：

```
¬∃ s ∈ Π .  deadlock_alert(s) ∧ ¬∃π: s →* closed
```

判定：先在**反向图**上从所有 `closed` 状态做一次反向 BFS，得到
`CanReachClosed`；再找出满足 `deadlock_alert` 却不在该集合中的可达状态。
**为什么不能只查"无出边的死状态"**：在 `lossy_retry` 中，`alert_pending` +
`in_flight=None` 的状态同样满足 `deadlock_alert`，但它还能 `resend` 并最终
到达 `closed`——它是**暂态**而非永久滞留。反向可达判定同时覆盖"无出边"与
"有出边但永远回不到 CLOSE"两种情形，是更一般、更安全的表述。

**无环性**：本模型的可达图是 DAG（每次转移都严格增大势函数：推进步数 /
`retries_used` / 清空 `in_flight`），所以"无限循环中 ALERT 不再改变"
在此等价于"落入不含 CLOSE 的死锁汇"。判定实现不依赖无环性，因此将来若
引入 `reset` 等可回退动作仍然正确。

### P3（活性·有界接续）— `bounded_liveness_ok`

```
∀ s ∈ Π .  s.phase = closed  ⇒  s.steps ≤ K       (K = 12)
```

判定：取所有可达 `closed` 状态，检查 `steps ≤ K`；若根本没有可达的
`closed`，判活性的有效性被平凡破坏（"任务永远完不成"），返回不成立。

---

## 3. 模型检验算法

```
_explore(channel, max_retries, max_states):
    order  = [INITIAL]              # BFS 发现序 ⇒ 最短路径长度非降
    parent = {INITIAL: None}        # state -> (前驱, 到达动作)
    graph  = {}                     # 邻接表 state -> [(后继, 动作)]
    BFS(INITIAL)，对每个状态调用 successors()
    若 |parent| 超过 max_states ⇒ raise StateSpaceTooLarge
```

* **P1**：`order` 上线性过滤。
* **P2**：反向图反向 BFS（O(V+E)）得到 `CanReachClosed`，再线性过滤。
* **P3**：`closed` 子集上取 `max steps`。
* **反例**：沿 `parent` 回溯，得到 `[(INITIAL, "init"), (s₁, a₁), …]`，
  末项即坏状态；`counterexample_text()` 渲染成中文多行文本。

复杂度：时间 `O(V + E)`（V = 可达状态数，E = 可达转移数），空间 `O(V + E)`。
本模型 V ≤ 142、E ≤ 317（`max_retries ≤ 8`），全部判定在毫秒级完成。
`max_states` 是安全阀：`check()` 超限时三条性质都返回
`satisfied=False` 并在 `detail` 中说明（便于参数网格继续跑）；
`reachable_graph()` 超限时抛 `StateSpaceTooLarge`。

---

## 4. 实验结果（可复现）

### 4.1 三种通道（`max_retries = 2`）

| 通道 | P1 | P2 | P3 | 可达状态数 `explored` | 可达转移数 | `classify_channel` |
| --- | --- | --- | --- | --- | --- | --- |
| `reliable` | ✓ | ✓ | ✓ | **14** | 13 | `pass` |
| `lossy` | ✓ | **✗** | ✓ | **22** | 31 | `P2_violated` |
| `lossy_retry` | ✓ | ✓ | ✓ | **46** | 89 | `pass` |

* 三条通道的 `max b_applications = 1`、`max closed steps = 7`（界 K=12，余量 5 步）。
* `lossy` 的 P2 反例来自 **2** 个 ALERT 永久滞留状态（`b_applied` 分别取
  0/1 的两条路径各一个），二者均无任何出边。
* `lossy_retry` 有 4 个"ALERT 已发、未 ACK、信道暂空"的状态，**全部**可经
  `resend` 到达 `closed`，因此不构成 P2 违反。

### 4.2 重传边界表（`retry_boundary()` 完整输出）

| channel | max_retries | P1 | P2 | P3 | explored |
| --- | --- | --- | --- | --- | --- |
| `lossy` | 1 | True | **False** | True | 22 |
| `lossy` | 2 | True | **False** | True | 22 |
| `lossy` | 3 | True | **False** | True | 22 |
| `lossy` | 4 | True | **False** | True | 22 |
| `lossy` | 5 | True | **False** | True | 22 |
| `lossy_retry` | 1 | True | True | True | 30 |
| `lossy_retry` | 2 | True | True | True | 46 |
| `lossy_retry` | 3 | True | True | True | 62 |
| `lossy_retry` | 4 | True | True | True | 78 |
| `lossy_retry` | 5 | True | True | True | 94 |

### 4.3 状态空间随重传预算线性增长

`lossy_retry` 的可达状态数满足精确线性律 **`|Π|(m) = 16m + 14`**：

| `max_retries` m | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `explored` | 14 | 30 | 46 | 62 | 78 | 94 | 110 | 126 | 142 |

即每增加 1 次重传预算只多 16 个可达状态：`m = 0` 的 14 个状态之外，
`retries_used` 的每个预算刻度恰好贡献 16 个状态。这解释了为什么边界扫描可以在
毫秒级做满网格，也说明"重传预算"对状态空间是**线性**而非指数级代价
（转移数同理：E(0)=13，E(1)=51，E(2)=89，E(5)=203，E(8)=317）。

### 4.4 `lossy` 通道 P2 反例（论文附录可直接用）

```
反例路径：6 个状态，5 个动作

[00] <init> phase=idle b_app=0 b_applied=0 alert=-/- retries=0 in_flight=- last=- steps=0
[01] --reg_send--> phase=registered b_app=0 b_applied=0 alert=-/- retries=0 in_flight=REGISTER last=REGISTER steps=1
[02] --verify_ok--> phase=verified b_app=0 b_applied=0 alert=-/- retries=0 in_flight=VERIFY last=VERIFY steps=2
[03] --delegate_send--> phase=delegated b_app=0 b_applied=0 alert=-/- retries=0 in_flight=DELEGATE last=DELEGATE steps=3
[04] --b_alert--> phase=alert_pending b_app=0 b_applied=0 alert=issued/- retries=0 in_flight=ALERT last=ALERT steps=4
[05] --drop:ALERT--> phase=alert_pending b_app=0 b_applied=0 alert=issued/- retries=0 in_flight=- last=ALERT steps=4

诊断：末状态是 ALERT 永久滞留（P2 反例）。ALERT 已发出、从未被 ACK，且信道上已无该消息（in_flight=None）；当前通道没有可用的重传动作，状态机因此无法到达 CLOSE，告警被永久丢失。
```

动作序列（计 5 个动作）：
`reg_send → verify_ok → delegate_send → b_alert → drop:ALERT`。

说明：这是 BFS 给出的**最短**反例，因此不包含可选的 `b_apply`
（B 建立申请记录）步骤；包含 `b_apply` 的同类反例同样存在（`b_applied=1`
的那两个死亡状态），只是长 1 步。

### 4.5 严格步数口径的敏感性分析

若把通道动作也计入 `steps`（即"已发生的全部转移数"），在同一张可达图上用
DAG 最长路 DP 可得最坏完成步数：

| 通道 | 最坏完成步数 |
| --- | --- |
| `reliable` | 7 |
| `lossy` | 11 |
| `lossy_retry`, m=1 | 17 |
| `lossy_retry`, m=2 | 27 |
| `lossy_retry`, m=3 | 37 |
| `lossy_retry`, m=4 | 47 |
| `lossy_retry`, m=5 | 57 |

即"`10m + 7`"（每条消息每轮丢+重传各 1 步，5 条消息 × 2m 步）。结论见 §7.2。

---

## 5. API 速查

```python
from skygovc.protocol import (
    State, INITIAL, successors,                 # 状态机
    safety_p1, is_deadlock_alert, bounded_liveness_ok, PROPERTIES,  # 性质
    check, reachable_graph, classify_channel,   # 模型检验
    retry_boundary, counterexample_text, CheckResult, StateSpaceTooLarge,
)

successors(INITIAL, "lossy_retry", max_retries=2)
# [(State(phase='registered', ...), 'reg_send')]

results = check("lossy")                 # dict: P1/P2/P3 -> CheckResult
results["P2"].satisfied                  # False
results["P1"].explored                   # 22
counterexample_text(results["P2"].counterexample)   # 中文反例文本

classify_channel("lossy")                # "P2_violated"
classify_channel("lossy_retry", 2)       # "pass"

order, graph = reachable_graph("lossy_retry", max_retries=2)
rows = retry_boundary()                  # list[dict]，可直接 DataFrame(...).to_markdown()
```

* `CheckResult(satisfied, explored, counterexample, detail)`
* `counterexample`：`[(State, action), ...]`，从 `INITIAL` 起，`action` 是
  **进入该状态**的动作（首项固定 `"init"`），末项为坏状态；`None` = 无违反。
* `PROPERTIES`：`{"P1": ("invariant", 描述), "P2": ("liveness_no_dead_alert", 描述),
  "P3": ("liveness_bounded", 描述)}`。

---

## 6. 复现

```powershell
# 测试（21 项）
& "C:\Python314\python.exe" -m pytest `
  "tests/test_protocol.py" -q

# 冒烟：三种通道的 satisfied / explored + retry_boundary() 全表
$env:PYTHONPATH = "."
& "C:\Python314\python.exe" -c "from skygovc.protocol import *; print(retry_boundary())"
```

测试文件内置 `sys.path` 引导（向 `parents[1]` 注入 `modules/SkyGovC`），
因此从任意工作目录直接 `pytest <该文件>` 均可运行，无需安装包、也无需
`conftest.py`。

---

## 7. 关键发现

### 7.1 P2 的恢复边界在 `max_retries = 1`，不是 2

题面预期"`max_retries=1` 时 `lossy_retry` 的 P2 为 False，`>=2` 时为 True"。
**实测（并已由单调性证明为必然）**：`max_retries = 1` 时 P2 即已恢复，
边界落在"是否存在重传动作"上，而不是某个具体预算值。

证明：`drop` 的守卫是 `retries_used < max_retries`，关于 `max_retries` 单调，
故转移关系满足 `→(m) ⊆ →(m+1)`，从而可达集 `Π(m) ⊆ Π(m+1)`。P2 是"不存在
坏状态"的**全称**性质，于是 `P2(m+1) ⟹ P2(m)`。既然题面同时要求
`check("lossy_retry", max_retries=2)` 的 P2 为 True（本项目
`tests/test_protocol.py` 第 5 项），那么 `m=1` 必然也为 True——**任何单调模型
都不可能产生"m=1 违反、m=2 成立"的组合**。若要真正得到"m=1 失败、m=2 成功"
的分界，只能让模型关于 `max_retries` 非单调（例如让 `resend` 也受预算约束），
但那会同时使 `m=2` 失败，与题面结论冲突。

因此论文里应表述为：**"P2 的恢复不取决于重传次数，而取决于是否存在重传机制：
`lossy` 因缺失重传而丢失告警，`lossy_retry` 只要 `max_retries ≥ 1` 即恢复。"**
这其实是一个更强、更干净的结论——它把失败归因于**机制缺失**而非**参数不足**，
也解释了 §4.2 中 `lossy` 的 5 行完全相同。

### 7.2 P3 的步数口径（`K = 12` 只能按"协议推进步"解读）

若把 `drop`/`resend` 也计入 `steps`，则：

* `lossy` 的最坏完成路径已是 **11 步**（7 个推进步 + 4 次丢弃），距 `K=12`
  只剩 1 步余量；
* 任何一次"重传周期"都会越过上界：`lossy_retry` 在 `m=1` 时最坏 **17 步**、
  `m=2` 时 **27 步**。

也就是说，`K = 12` 这个界恰好是按**无重传通道**标定的；一旦允许重传，"总转移
数"口径下 P3 必然被违反，"`lossy_retry` 满足 P3"就无法成立。因此本模块采用
**协议推进步**口径：`steps` 只累计改变协议阶段的动作（`b_apply` 亦计入），
通道动作的代价由 `retries_used` 单独记账。这一口径下：

* 三通道的最长完成路径都是 **7 步**（`reg_send → verify_ok → delegate_send →
  [b_apply] → b_alert → b_ack → b_close`），距 `K=12` 有 5 步余量；
* P3 刻画的是"协议推进有界"，重传带来的物理时延由 `retries_used` 与
  §4.5 的最坏路径长度体现（论文中建议把两张表并列，说明"协议步有界、
  物理时延随预算线性增长"）。

### 7.3 为什么 `lossy` 只违反 P2

题面转移表没有给 `verify_ok` / `b_apply` / `b_alert` / `b_close` 加
"消息已达"前置条件（只有 `b_ack` 显式依赖 `in_flight = ALERT`）。因此
REGISTER / VERIFY / DELEGATE / ACK 的丢失只影响通道记账，不阻断阶段推进；
`lossy` 的可达集虽然更大（22 个状态），但 P1（结构保证）与 P3（推进步有界）
仍然成立，唯一被打破的是"告警闭环"——这正是论文要讲的故事。

---

## 8. 与规范的偏差（逐条说明）

1. **未实现 `rejected --reset--> idle`**（题面第 12 条，标注为可选）。
   原因有两条：(a) 第二轮任务会对**同一任务**再次 `b_apply`，
   `b_applications` 将到达 2，P1 在**所有通道**下失败，与"三通道 P1 均成立"
   的结论冲突；(b) 每次 `rejected → reset` 循环至少增加 3 个推进步，
   `closed` 会出现在步数 10、13、16…，P3 在**所有通道**下失败。为保住
   P1/P3，本实现省略该动作（题面允许）。
2. **`steps` 只计协议推进步**，不计 `drop`/`resend`。理由与定量证据见 §7.2；
   这是为让 `K=12` 与"`lossy_retry` 满足 P3"同时成立而必须采用的口径。
3. **重传边界的实测值（`m ≥ 1` 即恢复）与题面预期（`m = 2`）不同**。
   原因与单调性证明见 §7.1；`retry_boundary()` 如实输出实测结果。
4. **`b_ack` 增加了 `in_flight = "ALERT"` 前置条件**。题面转移表未显式写出，
   但"`lossy` 下 ALERT 丢失进入死状态"必须以"消息未达则不能确认"为前提；
   否则丢包对告警闭环没有任何影响，P2 反例不可能出现。
5. **`b_applications` 的声明域保留 `0..2`**，但可达集实测最大为 1
   （`b_apply` 的守卫所致）；`b_apply` 用 `min(MAX_B_APPLICATIONS, …)` 截断，
   防止未来扩展越界。P2/P1 的判定也不依赖该域上界。
6. **`check()` 超限时不抛异常**：返回 `satisfied=False` + `detail`，便于在
   参数网格上批量跑；`reachable_graph()` 超限仍抛 `StateSpaceTooLarge`。
7. **测试未新增 `tests/conftest.py`**：仓库中 `SkyGrid_spark` / `SkyRwa` 采用
   `conftest.py` 注入 `sys.path`，但题面只列了 `tests/test_protocol.py`，
   故把等价的路径引导内联在测试文件首部，保证 `pytest <路径>` 在任意 cwd 下可用。

---

## 9. 论文引用草稿

> **表 X**　跨辖区接续协议在三种信道模型下的模型检验结果
> （`K = 12`；`explored` 为显式状态检验的可达状态数）。
> `reliable` 与 `lossy_retry` 满足 P1–P3；`lossy` 因缺少重传机制违反 P2，
> 反例见附录 A。

> **表 Y**　重传预算扫描：P2 的恢复边界。
> `lossy` 通道在任意 `max_retries` 下均违反 P2（参数无法补偿机制缺失）；
> `lossy_retry` 在 `max_retries ≥ 1` 时即恢复 P2，且可达状态数随预算线性增长
> （`|Π| = 16m + 14`）。

> **附录 A**　`lossy` 通道 P2 反例（最短路径，5 个动作）：
> `reg_send → verify_ok → delegate_send → b_alert → drop:ALERT`。
> ALERT 在信道中被丢弃且无重传，状态机永久停在 `alert_pending`，
> 告警既未被确认也未被升级处置。
