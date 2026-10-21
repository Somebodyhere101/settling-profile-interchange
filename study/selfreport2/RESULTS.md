# Paper 8 self-report arm 2: results (Qwen3-1.7B, laptop RTX 5070, 2026-10-20)

**Freeze and run.** `PREDICTIONS.md` was frozen at 12:23:11 (+04:00), SHA-256 `b22c4016…`. The run went from 12:23 to 12:47. `DEVIATIONS.md` is empty.

**Sources.** `res/sr2_summary.json` and `logs/analyze_sr2.log`.

**Sample.** All 127 arm-S2 recipients and 354 recipient–donor pairs. 231 pairs are informative, meaning their donor and recipient settle profiles differ on digits 2–4. Every row is teacher-forced to the shared answer.

**Selftest checks** (items outside the run):
- Temporal isolation holds exactly. Under (iv)@21, every report-turn position is bit-identical to the clean recipient's through block 21. The positions diverge only from block 22 on, through attention.
- (iv+rep)@21 writes the donor's states into the report turn exactly.

## Preregistered verdict: **RECONSTRUCTION FROM INPUT**

**Primary score.** y = Σ_k p_k (s_D − s_R)_k, in layers. It is positive if the "least sure" report shifts toward the digits that settle later in the donor than in the recipient. τ_rep = effect / clean gap.

| Cell | Effect y (layers) [95% CI] | τ_rep [CI] | τ_D(gen) | Answer unchanged | Median KL |
|---|---|---|---|---|---|
| **(iv)@21**: answer positions only, temporally isolated | +0.001 [−0.005, 0.007] | **0.00** [−0.03, 0.04] | **0.90** | 0.958 | 0.1 |
| (iv)@21, matched-KL corrected | +0.001 [−0.003, 0.006] | 0.01 [−0.02, 0.04] | | | 4 pairs not bracketed |
| **(iv+rep)@21**: direct-steering control | +0.176 [0.106, 0.260] | **0.99** [0.95, 1.03] | 0.90 | 0.958 | 0.6 |
| (i)@7: question tokens | +0.035 [0.000, 0.076] | 0.20 [0.00, 0.37] | 0.10 | 0.955 | 0.2 |
| **(ii)@7**: prefill operand region | +0.118 [0.068, 0.176] | **0.66** [0.48, 0.88] | 0.57 | 0.938 | 0.3 |
| (ii-r)@21: inert | −0.002 [−0.008, 0.004] | −0.01 | 0.00 | 0.992 | 0.0 |

**Clean gap.** +0.177 layers [0.107, 0.263]. The verdict is therefore evaluable.

**Input-only predictor.** Ridge regression on one-hot operand and answer digits, cross-fitted, predicts the clean digit-report logits with cv R² 0.58. Its predicted gap is 0.65 [0.48, 0.91] of the actual clean gap. Most of what distinguishes one item's report from another's is predictable from the operands.

**Answer-only predictor.** It predicts 0 for every cell by construction, because the answer is shared and forced.

**Verdict by the rules.** (ii)@7 moves the report (τ 0.66). (i)@7 does not meet the preregistered "moves" threshold (τ 0.1998 < 0.2, lower bound 0.0004). (iv)@21 moves depth (τ_D 0.90) but not the report: its upper bound is 0.04 of the gap, against the required 0.45.

**The report is not degenerate** (post hoc descriptive, as requested):
- After rotation correction, the clean report's entropy averages 1.22 bits (median 1.25, range 0.41 to 1.82; the maximum is 2).
- The argmax is "the third digit" for 122 of 127 recipients and "the fourth" for 5. The report's mass varies across items: SD of p(third) 0.12, SD of p(fourth) 0.11.
- The raw letter choice has strong option-position bias, which the 4 rotations remove.
- The result is a null on a graded readout with a positive, input-predictable clean gap. It is not a constant answer.

**The report does not track the item's own settling either:**
- y_self = −0.33 layers [−0.37, −0.29].
- The argmax is a latest-settling digit in 13% of items, against a chance rate of 39%.

**A/B secondary (replication of arm 1).**

| Cell | Effect on r (nats) [95% CI] |
|---|---|
| (iv)@21 | +0.003 [−0.022, 0.035] |
| (i)@7 | −0.450 [−0.669, −0.236] |
| (iv+rep)@21 | −0.631 [−1.016, −0.263] |

The clean gap is −0.646. The first two rows are identical to arm 1, as predicted. The direct-steering row is the new information: it carries 0.98 of the gap.

## Reading

Writing a donor's states into the report turn (iv+rep) transfers the donor's report completely, in both questions. The report readout is fully steerable, and the donor's states do contain the donor's report.

Writing the donor's states only into the answer positions, which moves the logit-lens settling profile (τ_D 0.90), transfers none of it, even though the report turn attends to those patched states from block 22 on. What does move the report is the swap of the reading-position-to-prefill region at block 7 (which also moves depth, τ_D 0.57); the question-token swap falls just below the threshold.

The report about how the answer was reached is reconstructed from the input. It does not read the manipulated computation, either as a global "step by step" judgement or as a per-digit confidence.

**Limits:**
- One model, one task.
- The report is one forced letter.
- Settle profiles are logit-lens quantities (see `../lens/`).
- The related-work list in `PREDICTIONS.md` §1 must be verified before any write-up.
