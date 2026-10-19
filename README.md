# SkyGov-C

**Compiling institutional rules into an executable, auditable, tiered governance
rule base for urban low-altitude operations.**

[![python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![tests](https://img.shields.io/badge/tests-87%20passed-brightgreen)](#quickstart)
[![license](https://img.shields.io/badge/license-MIT%20%2F%20CC--BY--4.0-lightgrey)](#license)
[![no LLM required](https://img.shields.io/badge/LLM%20required-no-informational)](#use-of-generative-ai)

> **Review status.** This repository accompanies a manuscript currently under
> **double-blind peer review**. To preserve anonymity the venue, authors and the
> formal citation are intentionally omitted here and will be added after the
> notification of acceptance. If you are a reviewer, please do not search for the
> manuscript's authorship; the anonymised mirror submitted with the paper is the
> authoritative artefact.

---

## The problem

An operator flying a drone over a Chinese city in 2026 must act on sixteen
instruments that do not agree on much except that they all apply: a national
regulation that defines permitted airspace below 120 m AGL, a provincial notice
that enlarges the permitted footprint, and two municipal instruments that declare
their entire territory controlled airspace. None of them is wrong — they regulate
different objects at different levels of the legal hierarchy — but the decision the
operator has to make has **no machine-readable representation**.

SkyGov-C treats *compilation* as the central artefact:

| | naïve pipeline | SkyGov-C |
|---|---|---|
| unit of reasoning | one flight | **one legal clause** |
| rule source | hand-authored | **compiled, with provenance** |
| tier assignment | a constant in the model | **derived by rules, and auditable** |
| cross-jurisdiction handover | assumed | **model-checked** |
| calibration | absent | **closed loop with a rollback test** |
| transfer to a new jurisdiction | unknown | **measured on a held-out block** |

## Architecture

```
  16 instruments ─► 59 clause records (dev) + 19 (held-out) + 3 external standards
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                          ▼
      reg2rule/  frames ─► align ─► merge ─► emit    audit/
              │                                   │  RAR / LEC / UCR / TCC
              │                                   ▼
              │                          SHACL shapes + SPARQL ASK + RDF
              │                                   │
              ▼                                   ▼
      classification/  context fingerprint ─► tier rules ─► tier
              │                                   │
              ▼                                   ▼
      allocation/  NSGA-II, budget sweep  ─►  engine/  verdict + provenance chain
              │                                   │
              ▼                                   ▼
      execution/  tier-aware ABM ── protocol/  handover, explicit-state model checking
              │
              ▼
      calibration/  Poisson–Gamma ─► conformal ─► hedged-capital CS ─► rollback
```

Every layer obeys one design rule: **no layer may extend the semantics of the one
below it.** The decision layer cannot invent a condition the rule base does not
contain, and the execution layer cannot invent a regime the decision layer did not
induce.

## Quickstart

```bash
git clone https://github.com/liuyushugreat/SkyGov.git
cd SkyGov
pip install -r requirements.txt

# 87 tests, ~40 s
python -m pytest tests -q

# regenerate every table and figure in the paper, ~40 s
python -m skygovc.experiments.run_all
```

No API keys, no network access, no language model. See
[Use of generative AI](#use-of-generative-ai).

Outputs land in `results/` (15 JSON files; `all_results.json` is the single source
for every number in the paper) and `figures/` (vector PDF, fonts embedded).

### Try it in one screen

```python
from skygovc.reg2rule import pipeline as P
from skygovc.reg2rule.engine import DecisionEngine
from skygovc.core import FlightRequest

res = P.compile_rules("S3_repair")                        # 26 rules, SHACL-clean
print(len(res.rulebase), res.validation_merged.conforms)   # 26 True

eng = DecisionEngine(res.rulebase)
req = FlightRequest(
    rid="demo", origin_jurisdiction="CN-SZ", dest_jurisdiction="CN-TJ",
    risk_class="high", airspace_class="B", uas_category="small",
    purpose="for_hire", night_operation=True, within_airport_buffer=False)
v = eng.decide(req, approved=False, tier="market_driven")
print(v.allowed)          # False
print(v.reasons)          # which rules fired, and on which clause each rests
print(v.chain)            # clause ids — the audit trail
```

## Repository layout

```
skygovc/
  core.py             shared data model (Clause, Proposition, FlightRequest, Verdict)
  corpus/             clause-level corpus + held-out block + expert gold standards
    records_national.py   D01–D11  national instruments (development block)
    records_local.py      D12–D16  local instruments + E01–E03 external standards
    records_heldout.py    H01–H04  held-out block, authored after the grammar freeze
    gold.py               development gold + the ten-category failure taxonomy
    loader.py             assembly, deduplication, institutional-redundancy report
  reg2rule/           text ─► executable rule base
    lexicon.py            closed vocabulary: modals, actors, actions, conditions, hedges
    frames.py             frame extraction, proviso scoping, disjunction splitting,
                          two lifting frames (definitional, competence)
    align.py              ontology-grounded repair of surface conditions
    merge.py              conflict detection, object-overlap test, precedence algebra
    emit.py               RDF/Turtle, SPARQL ASK, SHACL shapes, mechanical validation
    engine.py             forward-chaining decision engine + regime induction
    router.py             learning-to-route (features, grouped CV, cost curve)
    score.py              exact-signature and component-level scoring
    pipeline.py           single entry point: compile_rules(strategy)
  classification/     context fingerprint ─► governance tier (rules + ML baseline)
  allocation/         budget-constrained capability allocation (NSGA-II, hypervolume)
  protocol/           handover state machine + explicit-state model checker
  calibration/        Poisson–Gamma, split conformal, hedged-capital confidence sequence
  audit/              explanation generation + RAR/LEC/UCR/TCC detector validation
  execution/          tier-aware agent-based simulation over induced regimes
  experiments/        run_all.py — every table and figure
tests/                87 tests: reg2rule, allocation, protocol, calibration
results/              one JSON per experiment + all_results.json
figures/              fig1 … fig5 (vector PDF)
```

Each of the three algorithmic subpackages (`allocation/`, `protocol/`,
`calibration/`) ships its own `README.md` with the model, its parameters and a worked
example.

## What the numbers say

Reported honestly, including the ones that are unflattering.

| Question | Result |
|---|---|
| Rule compilation, development block | F1 **0.241** (keyword) → **0.780** (frames) → **1.000** (full pipeline). The last figure is **in-sample**: grammar and gold were authored together. |
| Rule compilation, **held-out** block (grammar frozen, single run) | exact-match F1 **0.231**, field mean **0.711**, compilability accuracy **0.947**, action normalisation **0.500** |
| Rule-base coherence | keyword pipeline leaves **3** object-aware modality collisions; merged rule base **0** (SHACL) |
| Learning to route expert effort | AUC **0.691** (grouped CV by instrument); 80 % of the attainable gain in **31** adjudications vs **40** for confidence-only, unreachable for random |
| Tier classification | **6/6** reference jurisdictions; **0.86 / 0.78 / 0.67** on noisy synthetic contexts vs **0.99 / 0.98 / 0.91** for a random forest — the accuracy we pay for auditability |
| Capability allocation | differentiation buys **−3.2 % → +6.4 %** on incident multiplier as budget headroom grows; **not** a free lunch |
| Handover protocol | `lossy` violates alert-loss safety (5-action counterexample); safety is recovered by the **existence** of retransmission, not its count; reachable states `|Π(m)| = 16m + 14` |
| Calibration | open loop is **exactly stationary** (L0 ≡ L1 element-wise over 180 days); closed loop cuts MAE **0.428 → 0.025**; rollback detects a 35 % drift **200/200**, **0** false alarms, 22.8-day delay |
| System level, 200 repetitions | tiering vs uniform strict: incidents **−0.491** (p = 1.1×10⁻¹⁹), completion **+0.108**; coordination and capacity expansion improve completion and cost but leave the incident rate **statistically unchanged** |

### The most useful result is a negative one

On the held-out block, compilability classification and deontic recognition transfer
almost intact (0.947 / 0.938) while **action normalisation collapses to chance
(0.500)**. A vocabulary pass that added four plausible action terms *lowered*
exact-match F1 from 0.333 to 0.231.

The bottleneck is therefore the **antecedent schema**, not the lexicon. A schema that
cannot express *"the airframe is not equipped with X"* will mis-compile every clause
that turns on equipment, however good the extractor is. Adding words does not help;
adding predicate kinds would.

## Corpus and provenance

* Sixteen Chinese instruments (two laws, one administrative regulation, one
  departmental rule, seven policy documents, three local regulations, two municipal
  notices) decomposed into **59 clause-level records**, plus **19 held-out records**
  from four further instruments, plus three external comparison standards
  (EU U-space, FAA UTM ConOps, JARUS SORA).
* `verbatim` is `False` for every record. Clause texts are **normalised renderings at
  proposition granularity**: operative verb, modal particle, numeric thresholds and
  provisos are retained; editorial framing is condensed. Each record carries a
  `source_note` stating what was condensed, so every record can be audited against the
  gazette. **This is a normalised corpus, not a verbatim one.**
* The gold standard is held out from the compiler: `reg2rule` never reads
  `corpus/gold.py`; scoring lives in `reg2rule/score.py`.
* Institutional redundancy is reported rather than hidden: 37 gold propositions
  collapse to **27 distinct** ones after cross-instrument deduplication (10 redundant
  clause records in 9 groups).

## Reproducibility protocol

| Block | Protocol |
|---|---|
| **Development** (55 core clauses) | grammar and failure-mode procedure authored against these records → reported figures are **in-sample**, and are labelled as such |
| **Held-out** (19 records) | authored **after** the grammar was frozen, evaluated in a **single run**, no grammar edit afterwards → the honest capability estimate |
| **Routing** | `GradientBoostingClassifier`, 8 features, `GroupKFold` **by instrument** so no clause of a test document is seen in training |
| **Allocation** | uniform and differentiated policies spend from the same total budget; the uniform policy must satisfy every tier's floor simultaneously, which is why its minimum feasible cost exceeds the floor sum |
| **Simulation** | authorisation probabilities **induced** by running the decision engine over sampled requests; capability vectors from the allocator; the tier policy overlay expressed as rules in the same language and executed by the same engine; 200 repetitions, Wilcoxon signed-rank on paired runs |

## Limitations encoded in the code

These are not caveats bolted onto a README — they are visible in the artefacts:

* `records_heldout.py` contains one record whose gold label is ambiguous. It was
  **not** revised after the held-out run was seen; the resulting mismatch is left in
  the reported figures.
* `results/heldout_eval_coldstart.json` archives the cold-start held-out run
  (exact-match F1 0.333); `results/heldout.json` is the post-vocabulary run (0.231).
  Both are shipped so the drop can be verified.
* The allocator's cost ladder and the tier floors are **assumptions**. The sensitivity
  of the differentiation result to that ladder is the first thing to attack.
* The model checker covers two jurisdictions and one task. Multi-task concurrency is
  out of scope and would require an interleaving model.
* Nothing here is a field evaluation. The corpus is real; the execution layer is
  simulated, and the behavioural parameters that dominate the system-level table have
  no published source.

## Use of generative AI

`reg2rule/frames.py` implements a **deterministic symbolic extractor**. The module
exposes an extension point for a language-model backend, but `run_all` never calls it:
**no reported number depends on any model version.** Generative tools were used for
language editing of the manuscript and for drafting figure-generation code, which was
reviewed, tested and is included here.

## License

* **Code** — MIT (see [`LICENSE`](LICENSE)).
* **Corpus, gold standards and result files** — CC-BY-4.0.

## Citation

The citation will be added when the double-blind review completes. Until then, if you
need to reference this artefact, please cite the repository URL:

```
SkyGov-C: compiling institutional rules into auditable tiered governance.
https://github.com/liuyushugreat/SkyGov (2026)
```

## 中文速览

本仓库是论文 **《From Regulation to Execution: Compiling Institutional Rules into
Auditable Tiered Governance for Urban Low-Altitude Safety》** 的完整支撑代码，目前该文
处于**双盲评审**阶段，故 README 中隐去会议与作者信息（获奖/录用后再补全引用）。

- **做什么**：把 16 份低空监管文件编译成带法源溯源的**可执行规则库**（SHACL + SPARQL），
  再据此做分级判定、预算约束的能力配置、经模型检验的跨域接续，以及闭环校准与回退。
- **一键复现**：`pip install -r requirements.txt` → `python -m pytest tests -q`（87 项）
  → `python -m skygovc.experiments.run_all`（约 40 秒，生成论文全部表格与图）。
- **不依赖大模型**：抽取器是确定性的符号实现，论文报告的数字与任何模型版本无关。
- **诚实的负结果**：留出块上精确匹配 F1 仅 0.231、动作归一化 0.500——瓶颈在**模式表达力**
  而非词表。这一点写在论文与代码里，没有藏起来。
