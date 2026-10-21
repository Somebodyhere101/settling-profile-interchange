# Paper 8 self-report arm 2: does the model's per-digit report access its manipulated computation?

**Status.** Written on 2026-10-20, before any arm-2 data on arm-S2 items existed. Its SHA-256 is recorded in `PREDICTIONS.sha256`, with the code, tail and pair hashes. Changes go in `DEVIATIONS.md`; this file is not edited.

**Why this arm exists.** It implements the corrections of an external review that the author wants in (relayed 2026-10-20). Those corrections arrived after arm 1 (`../selfreport/`) was frozen and running, so arm 1 stays as frozen and these corrections form a separately preregistered arm. The assistant made the design choices recorded here.

**What is already known when this is written:**

- **Arm 1's result** (A/B question; `../selfreport/res/sr_summary.json`): RECONSTRUCTION FROM INPUT.
  - Swapping the answer-position states at block 21 transfers depth (τ_gen 0.90) but leaves r unchanged: τ_r −0.00 [−0.06, 0.04].
  - Swapping only the question tokens at block 7 moves r (τ_r 0.70 [0.41, 1.31]) with depth τ 0.10.
  - Clean Spearman(r, D) is −0.01 [−0.19, 0.17].
- **Arm 1's kill check:** PASS by rule. The tuned-lens depth gap is small (0.17 layers), and its copy-free transfer is τ 1.06 [0.36, 2.44].
- **This arm's format check and selftest,** on items outside the run only:
  - the letter mass is 1.000 for every report turn;
  - the clean digit report mostly names the third digit in 3 of 4 rotations, so there is a strong position prior that rotation averaging must remove;
  - temporal isolation is verified (§2).

No digit-question data on arm-S2 items exist.

## 1. Question and positioning

Can an unmodified model report on a computation that was changed by a hidden counterfactual intervention, when the input and the answer are held fixed? A positive result needs:

- the report to follow the manipulated state;
- the report not to follow input content;
- the manipulation not to touch the report turn directly.

**Narrow novelty claim.** Evidence, or its absence, of causal self-access in an unmodified model, under hidden counterfactual interventions that hold input and answer fixed. The question is about per-digit processing, with a target measured independently of the report.

**Related work to position against.** The list was supplied by the author's reviewer as verified on arXiv/PMLR. The assistant has not re-verified it, and it must be checked before any write-up.

- Singh, Linzen & Ravfogel, "Can LLMs Introspect? A Reality Check" (arXiv:2605.26242): the critique this design answers.
- Li, Guo, Huang, Steinhardt & Andreas (arXiv:2511.08579) and Guo, Ruis, Andreas & Li (arXiv:2606.32038): self-explanation *training*. This arm tests an unmodified model.
- Macar et al., "Mechanisms of Introspective Awareness" (ICML 2026, PMLR 306).
- Lederman & Mahowald (arXiv:2603.05414): content-agnostic detection.
- `../DEEPER.md` §6 lists more (Binder et al.; Lindsey 2025; Hatua 2026; Lanham et al. 2023; Geiger et al.; Makelov et al.; Stolfo et al. 2023).

## 2. Design (fixed)

**Model, items and pairs.**

- Qwen3-1.7B, thinking off, bf16, on the laptop RTX 5070, in the `jspace` venv.
- **All** 127 arm-S2 recipients and their far and null donors, 381 items.
- **Selection.** The pairs share the answer *by construction*: same-product pairs were built in the patching arm and kept in S2 when the clean answer tokens matched. The answer was not filtered after any intervention. All preregistered pairs are analysed.

**Engine and selftest.**

- The engine is `../selfreport/srlib.SRModel` (frozen in arm 1, unchanged): one teacher-forced pass per batch, with in-batch interchange at fixed absolute positions of one block.
- **Teacher-forced-answer control.** Every row is forced to the shared answer, which is the recipient's own, so answer identity cannot differ between rows.
- **Answer-change rate.** The rate at which a cell's argmax at the digit positions differs from the forced answer is reported per cell. In S2's free generation, (iv)@21 changed the answer in 2% of rows; in arm 1's teacher-forced pass, 2.8%.

**Report turns.** Each is appended after the answer exactly as the model's generation left it (`run_sr2.py tail_text`), ending with the prefill "The answer is (".

1. **Primary.** "Which digit of your answer were you least sure of: the first, second, third or fourth? Answer A, B, C or D." with four option lines. In **rotation k** (k = 0 … 3), letter j names digit (j + k) mod 4. All 4 rotations are run for every row.
2. **Secondary.** Arm 1's A/B question ("step by step" or "just knew it"), both label orders. Because arm 1's A/B data have been seen, this is a replication with a directional prediction (§4), not a fresh test.

**Temporal isolation.**

- Patches are at fixed absolute positions of one block, in a single teacher-forced pass.
- Under **(iv)@21**, only the answer positions 30 … 30 + K − 2 are overwritten at block 21 (as S2). No report-turn position is patched. The last answer token's position is not patched either.
- The selftest verifies two things:
  - every report position's block outputs up to and including block 21 are **bit-identical** to the clean recipient's;
  - they differ only from block 22 on, through attention to the patched answer states (the KV-cache route).

**Cells, per donor:**

| Cell | Positions | Role |
|---|---|---|
| **(iv)@21** | answer positions | process manipulation, input and answer fixed (S2: depth τ_gen 0.91) |
| **(iv+rep)@21** | answer positions **and** the last answer token's position and every report-turn position | **direct-steering control**: the donor's states written into the report turn itself |
| (i)@7 | question tokens 3–12 | input content, not depth |
| (ii-r)@21 | reading position 17 | inert control |
| (ii)@7 | 17–29 | depth confounded with the operands' prefill copy |

**Gaussian control.** At the (iv) positions at block 21:

- an additive c·σ·g_j, with 4 fixed unit Gaussian directions (seed 1234) and c ∈ {0.125, 0.25, 0.5, 1, 2, 4, 8};
- σ is the recipient's median ‖h_far donor − h_recipient‖ at those positions.

**KL.** KL(clean ‖ row) is summed over positions n … T − 1 and averaged over the 4 rotations.

## 3. Measures

**Report distribution.** For a row, each rotation's 4 letter logits are centred and mapped to digits, then averaged over the 4 rotations. That removes position and letter bias. p is the softmax over digits 1–4.

**Settle profiles.** These are logit-lens settle layers per digit, from this pass:

- s_R: the clean recipient;
- s_D: the clean donor;
- s′: the patched row (reported).

**Primary score.** For a recipient–donor pair:

- w = s_D − s_R per digit, with digit 1 set to 0, because (iv) never touches digit 1.
- y(p) = Σ_k p_k · w_k, in layers.
- Pairs with w = 0 are uninformative and are excluded and counted.
- Row effect: y(p_row) − y(p_R). Pair gap: y(p_D) − y(p_R).

If the report tracks settling ("least sure" falling on later-settling digits), the gap is positive. A report that accesses the manipulated state moves toward the donor's report under (iv)@21.

**Cell effect and τ.** The cell effect is the mean over recipients of the mean over its donors (far and null pooled, because w carries the direction), and τ_rep = effect / gap.

**CIs.** Two-way item bootstrap: a Poisson(1) weight per item, used both as recipient and as donor. 10,000 resamples, seed 80. Ratios use the same resamples.

**Depth transfer τ_D(gen).** Arm S2's s-signed far − null contrast on D_gen, from this pass.

**Matched-KL correction.** For each (iv)@21 pair, the Gaussian y-shift is interpolated in log KL (using the same w) for each direction, then averaged. The corrected effect is the row effect minus that. Pairs that are not bracketed are excluded and counted.

**Input-only and answer-only predictors.**

- **Input-only.** Ridge regression (λ = 1, 5-fold cross-fitted) predicts the 4 centred digit logits of each clean item from one-hot operand digits and one-hot answer digits.
  - The **predicted gap** is y(p̂_D) − y(p̂_R). It reports how much of the clean donor–recipient report difference the input explains.
  - The **predicted cell effect** follows from the input each cell effectively carries. For (iv)@21, (iv+rep)@21 and (ii-r)@21 that is the recipient's own input, so the prediction is 0. For (i)@7 and (ii)@7 it is the donor's operands.
  - The reported statistic is effect − predicted, with a CI.
- **Answer-only.** Donor and recipient share the answer and all rows are forced to it, so the prediction is 0 for every cell, by construction.

**A/B (secondary).** r = logit(step) − logit(just knew), the mean of the two orders. It is analysed with arm 1's s-signed far − null contrast: gap, per-cell effect and τ.

**Clean tracking (descriptive).**

- y_self = Σ p_k (s_k − mean s) over the 381 items.
- How often the report's argmax is a latest-settling digit, compared with chance.

## 4. Decision rules (implemented in `analyze_sr2.py`)

Let τ_D be the τ_D(gen) of (iv)@21 in this pass. A cell **moves** the report if its effect CI excludes 0 and |τ_rep| ≥ 0.2.

1. **NOT EVALUABLE.** The gap's CI does not lie above 0: clean reports do not follow the donor–recipient settling difference.
2. **CAUSAL SELF-ACCESS** requires all of the following:
   - τ_rep of (iv)@21 ≥ 0.5 · τ_D;
   - the (iv)@21 effect CI lies above 0;
   - the KL-corrected (iv)@21 effect CI lies above 0;
   - (iv)@21's effect minus its input-predicted effect has a CI above 0;
   - (i)@7 does not move the report.
3. **RECONSTRUCTION FROM INPUT.** (i)@7 or (ii)@7 moves the report, and (iv)@21 does not meet "τ_rep ≥ 0.5 · τ_D with CI above 0".
4. **DIRECT STEERING ONLY.** (iv+rep)@21 moves the report and (iv)@21 does not meet that criterion.
5. **INCONCLUSIVE.** Anything else.

**Always reported:**

- (iv+rep)@21 next to (iv)@21. Their difference is how much of any effect needs the report turn to be written directly.
- (ii-r)@21 as the inertness check.

**A/B replication prediction** (from arm 1): (iv)@21 τ_r inside ±0.2, and (i)@7 moves r in the same direction as arm 1.

**Second model.** Justified only if the primary verdict is CAUSAL SELF-ACCESS (replicate it) or RECONSTRUCTION FROM INPUT (establish generality).

## 5. Prior (assistant; not a decision rule)

Informed by arm 1's A/B null under (iv)@21, and by the strong third-digit position prior.

| Verdict | Prior |
|---|---|
| NOT EVALUABLE | 0.45 |
| RECONSTRUCTION FROM INPUT | 0.25 |
| INCONCLUSIVE | 0.15 |
| DIRECT STEERING ONLY | 0.10 |
| CAUSAL SELF-ACCESS | 0.05 |

## 6. Limits

- One model, one task and one wording per question.
- The report is read as one forced letter, not free text.
- The latest-settling profile is a logit-lens quantity, and the tuned lens shrinks the early–late depth difference about 6-fold (arm 1's kill check).
- Below block 21, the answer positions keep the recipient's states.
- Nothing is published.
