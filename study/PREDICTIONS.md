# Paper 8 pilot: where can the course of a computation be pushed?

**Status.** Written on 2026-10-19, before any paper-8 run. No paper-8 data exist when this file is hashed. Its SHA-256 is recorded in `PREDICTIONS.sha256`, together with the hashes of the code that will run (`code/`). Any later change to the design or the analysis is logged in `DEVIATIONS.md`, with its date and reason, and this file is not edited.

The author delegated the design choices to the assistant (Claude). Those choices are recorded here.

## 1. Question

Paper 6 (Ramdan 2026, doi:10.5281/zenodo.23450400) built an oracle direction from each process target. In 17 of 19 runs that direction **reads** the course of the computation at the published layer: settling depth, chain-of-thought necessity and backtracking. Pushing along it never **moves** that target. Answer-choice targets behave differently: D7 hallucination, D8 flips and D9 following a perturbed step become pushable at 2 to 8 gaps.

Every paper-6 push started at the reading position (the last prompt token) and continued to every later position. No push touched earlier prompt tokens. This pilot asks whether token position is the missing factor.

- **H-rederive.** The state that governs the course of the computation is held at source positions (the operands, the question tokens). Attention re-derives it at each layer. A write at the answer position is overwritten or ignored, but a write at the source positions moves the target.
- **H-nonlinear.** The course of the computation cannot be controlled linearly at any of the tested positions or layers.

Either outcome matters. The first would mean the field's probes measure readouts rather than mechanisms. The second would put a hard limit on steering and representation engineering for process variables.

**Relation to prior work.**

- Causal tracing and ROME (Meng et al. 2022) locate factual associations at the subject tokens' mid-layer MLPs. They restore clean states rather than add directions, and their target is a recalled fact, not the course of a computation.
- Geva et al. (2023) describe attribute extraction: the subject representation is enriched first, and attention at the last position then extracts the attribute. That is the read-at-source pattern that H-rederive generalises to arithmetic.
- Billa (2026, arXiv:2604.15557) predicts where steering succeeds. We test position against a process target, with oracle directions and KL-matched controls.

These distinctions must be re-checked against the papers before any write-up.

## 2. Design (fixed)

**Model.** Qwen3-1.7B, thinking off, bf16, greedy decoding. The pilot runs on DQ (RTX 5090). Qwen2.5-7B-Instruct is added only under §6.

**Targets and items.** Both targets reuse paper 6's items and outcome code unchanged (`code/specs.py`, `spec_d8.py`, `cr4_parsers.py` and `p6lib.py` are copied byte for byte).

- **settle** (course of computation). The target is the logit-lens settling depth of the answer digits (paper 6's D3/D4 target, `D` = mean settling layer / L). The items are paper 4's format-matched arithmetic items (paper 6 `items.json` "d3"), in direct mode with the "a * b = " prefill. The Check A split has 240 items, the oracle split 120 and the Check D (push) split 120.
- **flip** (answer choice; positive control). The target is whether the model switches to the user's stated wrong option on ARC, scored only on items it answers correctly without the belief (paper 6's D8 target, with the "The answer is (" prefill). The splits are A 1,828, oracle 220 and D 200.

**Reading position.** This is paper 6's. For flip, it is the last token of the chat prompt before the prefill. For settle, it is the last token of the CR detector's chat template, which is the "\n" after "assistant"; Qwen3's empty `<think>\n\n</think>\n\n` block and the prefill follow it.

**Layers.** Each target uses three layers:

- **Published layer ℓ.** Paper 6's layer for this model. For settle it is block 21, the CR-axis layer of the D3 pilot. Qwen3-1.7B has no LiReFs (D4) run, so D3's layer stands in for both settle targets. For flip it is block 19, the persona-vector layer of the D8 pilot.
- **One earlier layer:** ℓ − round(L/4).
- **One later layer:** ℓ + ceil((L − 1 − ℓ)/2).

With L = 28, the layers are 14, 21 and 24 for settle, and 12, 19 and 23 for flip.

**Oracles.** These are fresh for this model, built on its oracle split with paper 6's `oracle()`. For settle the oracle is the top-tercile mean minus the bottom-tercile mean; for flip it is the class mean of flips minus non-flips. Two oracles are built per layer:

- **o:** from the hidden state at the reading position, as in paper 6;
- **os:** from the mean hidden state over the source tokens.

`os` is added because a state held at the source positions need not use the reading-position basis. Pushing only `o` there could produce a false null.

**Strength unit.** The unit is paper 6's calibration gap G at ℓ: 221.75 for settle and 161.65 for flip, both from the Qwen3-1.7B pilot summaries. At the other two layers, G is scaled by the ratio of median residual norms at the reading position over the oracle-split items. A push of k gaps is therefore the same fraction of the residual stream at every layer.

Pushes use strengths k = 1, 2 and 4 gaps, in both signs.

**Position conditions.** The push vector is added to the output of one block at:

- **(a) Reading position onward.** The reading position and every later prompt, prefill and generated position. This is exactly paper 6's protocol. For settle, this region contains the prefill's copy of the operands ("a * b = "). The source positions in (c) are their first occurrence, in the question.
- **(b) All positions.** Every prompt position from index 1 onward, plus every generated position. Position 0 is excluded because it is the attention-sink token, `<|im_start|>`, which has a very large norm.
- **(c) Source tokens only.** The tokens of the user message content: for settle, "What is a * b?"; for flip, the question, the options, the belief sentence and the instruction. There is no push at the reading position or at any later position.

`o` is pushed under (a), (b) and (c). `os` is pushed under (c) only.

**References.** Twelve unit Gaussian directions are used, from paper 6's generator with seed 1234. Each is pushed at the same positions and layer as the cell it serves, at c ∈ {0.25, 0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24} gaps.

- **Matched-KL effect.** For each item, the change in T from the oracle push minus the change in T from each Gaussian direction, interpolated in log-KL to the oracle push's KL. This is paper 6's `check_D` "rand" family, reimplemented in `code/analyze8.py`. Its CI is a two-level bootstrap over items and reference directions, with 2,000 resamples. Items whose KL is not bracketed are excluded and counted.
- **Matched-norm effect.** The same comparison against the Gaussian pushes at c = k. It is reported alongside the matched-KL effect.

**KL window.** KL(clean ‖ pushed) is teacher-forced on the clean continuation and summed from the reading position to the end. This is the same window for every condition and is paper 6's `kl_all`. Changes to predictions inside the prompt before the reading position are not counted, so matching is on what the model outputs.

**Persistence.** In the same teacher-forced pass, the hidden states are recorded at every layer l′ ≥ l at two answer positions:

- **P1:** the reading position;
- **P2:** the last prompt token, which predicts the first answer token.

The survival of a write is s(l′) = ((h_push − h_clean) · u) / (signed write size), where u is the written unit direction. Under (a) and (b), s(l) = 1 at the push layer. Under (c), s measures how much of a source write arrives at the answer positions.

The oracle's survival is compared with that of norm-matched Gaussian writes at the same positions and layer. The value is reported at l + 3 (primary) and at the last layer, for both targets.

**Readability.** This uses the Check A form, on the A split: the projection on `o` and on `os` at each layer against T. For settle the statistic is Spearman ρ; for flip it is AUC. Partial ρ controlling for prompt length is used if |ρ(T, prompt length)| > 0.1, as in paper 6. The pass rule is paper 6's (SESOI ρ 0.20 or AUC 0.60). The pilot assumes the target is readable at a layer. If it is not, cells at that layer are reported but flagged.

**SESOI.** For settle, 1/L of depth (0.0357). For flip, 10 points of flip rate. Both are paper 6's.

## 3. Outcome measures

1. **Cell effect.** One cell is a (target, condition, kind, layer, k, sign) combination. Its effect is the matched-KL effect with a 95% CI, plus the matched-norm effect and the median KL. Each target has 3 conditions × 3 layers × 3 strengths × 2 signs = 54 cells for `o`, plus 18 for `os`.
2. **Controllable cell.** A cell is controllable when |effect| ≥ SESOI and its CI excludes 0, in either direction. A wrong-sign effect also shows linear control. Paper 6's claimed-sign verdict (pass / reversal / fail / inconclusive) is reported as well.
3. **"Moves."** A condition moves a target if, at some layer and sign, two adjacent strengths (1 and 2, or 2 and 4) are both controllable with the same direction of effect. This dose-consistency rule keeps the chance rate across the 18 cells of a condition near or below 0.1, even though cells are correlated.
4. **Persistence contrast.** The primary contrast is under (a), at ℓ, k = 2, position P2 and layer ℓ + 3. It is the oracle's survival minus the Gaussian survival, with an item-bootstrap CI, for settle and for flip, and the settle-minus-flip difference.

## 4. Predictions

| | settle under (a) | settle under (b) | settle under (c), o or os | flip under (a) | persistence of the settle oracle under (a), vs Gaussian |
|---|---|---|---|---|---|
| **H-rederive** | does not move | moves, or (c) moves | moves, or (b) moves | moves | overwritten (≤ −0.25, CI < 0); more negative than flip |
| **H-nonlinear** | does not move | does not move | does not move | moves | no prediction (kept, overwritten or amplified) |

**Persistence classes.** If the difference is ≤ −0.25 with CI < 0, the write is **overwritten**. If it is ≥ +0.25 with CI > 0, it is **amplified**. Otherwise it is **kept**.

Persistence is auxiliary. A write can be "kept" at the answer position and still go unused if the computation reads its inputs from the source positions directly. That case is compatible with H-rederive, and it is the "readout, not mechanism" case. So persistence decides between "overwritten" and "ignored", not between the two hypotheses.

**Assistant's prior, recorded for calibration** (not a decision rule): H-nonlinear 0.55, H-rederive 0.20, neither or mixed 0.25. Positive control passes: 0.65. In Qwen3-1.7B, paper 6's D8 pilot found no move at one gap, and flips at 1.7B are common (about 20% of correctly answered items).

## 5. Decision rules (implemented in `code/analyze8.py`)

1. **Positive control.** Flips must move under (a), with `o`. If they do not, the verdict is **UNINFORMATIVE**: the pushes are too weak, or the method fails on this model, and nothing is said about the settle hypotheses.
2. **H-REDERIVE.** The positive control passes, settle moves under (b) or under (c) (with `o` or `os`), and settle does not move under (a).
3. **H-NONLINEAR.** The positive control passes, and settle moves under none of (a), (b) or (c).
4. **NEITHER.** Settle moves under (a). The target is pushable from the reading position at some layer or strength that paper 6 did not test, so position is not the factor.
5. Any other pattern is **MIXED**.

The persistence class of the settle oracle is reported with the verdict in every case.

## 6. Go / no-go

**Qwen2.5-7B-Instruct.** It is run with the same code and rules, at paper 6's layers for it: settle ℓ = 14, flip ℓ = 19, with its gaps. The condition is that the 1.7B verdict is H-REDERIVE or H-NONLINEAR.

**The 3-model study.** The study would use paper 6's panel: Qwen2.5-7B-Instruct, Qwen3-8B and OLMo-2-1124-7B-Instruct, with a full preregistration, D4G and D9 added, and a source-patching (activation-restoration) arm. It is justified if:

- the 1.7B verdict is H-REDERIVE or H-NONLINEAR;
- and, when 7B is run, it does not contradict 1.7B. Contradicting means the 7B verdict is the other hypothesis, NEITHER or MIXED.

**No-go outcomes.**

- UNINFORMATIVE: first redesign the strength unit, for example oracle-own gaps or larger k.
- NEITHER: paper 6's conclusion is then about layer, not position. The fundamental claim fails and the result is reported as a correction to paper 6.
- MIXED: report it and do not scale up.

## 7. Known limits, stated in advance

- One small model.
- The source oracle `os` is a mean over source tokens.
- The KL window excludes prompt positions.
- For settle, (a) contains the prefill copy of the operands.
- The positive control is a different task with a different prompt length.
- Settling depth is a logit-lens measure, so late-layer pushes can shift it mechanically; the Gaussian references at the same layer control for this.
