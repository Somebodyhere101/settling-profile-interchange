# Paper 8 lens-robustness arm: results (Qwen3-1.7B, laptop RTX 5070, 2026-10-20)

**Freeze.** `PREDICTIONS.md` was frozen at 12:55:44 (+04:00), SHA-256 `37dda767…`. The run went from 12:55 to 13:44. `DEVIATIONS.md` is empty.

**Sources.** `res/lens_summary.json`, `res/lens.jsonl` and `logs/analyze_lens.log`.

**Sample.** 840 items (381 S2, 180 fresh M, 279 fresh G), each in 2 prompt forms.

**Checks.** The clean argmax equals the forced digit in 99.3% of items. The corrupt partner's argmax is its own digit in 99.5%.

## Preregistered verdict: **MIXED**

**Primary: item-level Spearman correlations** (mean over digits; 95% item-bootstrap CI)

| | ρ | 95% CI |
|---|---|---|
| (a) logit-lens settling vs (c) causal arrival | **0.26** | [0.20, 0.32] |
| (b) tuned-lens settling vs (c) causal arrival | **0.32** | [0.26, 0.38] |
| (a) vs (b) | 0.23 | [0.16, 0.30] |

ρ(a, c) is below the 0.5 needed for ROBUST and above the 0.2 needed for LENS ARTEFACT, so the verdict is MIXED.

**All three measures are reliable.** Each repeats almost exactly across the two prompt wordings (orig against para):

| Measure | ρ |
|---|---|
| Logit lens | 0.88 |
| Tuned lens | 0.97 |
| Causal arrival | 0.89 |
| 50% arrival | 0.90 |

They measure different things, each very consistently.

**Range.** Means (SDs) in layers:

| Measure | Mean | SD |
|---|---|---|
| Logit lens | 24.8 | 0.56 |
| Tuned lens | 18.2 | 4.2 |
| Causal arrival | 21.1 | 1.9 |
| 50% arrival | 17.0 | 3.7 |

Logit-lens settling varies very little between items. Causal arrival varies more than three times as much.

**Secondary correlations.**

| Comparison | ρ | 95% CI |
|---|---|---|
| a vs c, S2 items only | 0.19 | [0.09, 0.29] |
| a vs c, fresh items | 0.33 | [0.25, 0.42] |
| a vs c, fresh M items | 0.30 | [0.16, 0.44] |
| b vs c, S2 items only | 0.03 | [−0.07, 0.13] |
| b vs c, fresh items | 0.33 | |
| a vs c, D_gen analogues | 0.26 | |
| b vs c, D_gen analogues | 0.40 | |
| a vs c, digit level | 0.32 | |
| b vs c, digit level | 0.56 | |
| a vs 50% arrival | 0.13 | |
| b vs 50% arrival | 0.02 | |

The tuned lens tracks causal arrival digit by digit (0.56), but less well item by item.

**S2 grouping.** The arm-S2 far-versus-null split, s-signed, in layers (two-way bootstrap):

| Measure | Gap (layers) | 95% CI |
|---|---|---|
| Logit lens, D | **1.12** | [1.05, 1.19] |
| Logit lens, D_gen | 1.40 | [1.29, 1.53] |
| Tuned lens, D | 0.29 | [−0.14, 0.73] |
| Tuned lens, D_gen | 0.28 | [0.07, 0.53] |
| **Causal arrival** | **0.18** | [**−0.60**, 0.83] |
| Causal arrival, D_gen | 0.23 | [−0.80, 1.10] |
| 50% arrival | 0.79 | [−0.86, 2.38] |

## Reading

- **Settling depth tracks causal commitment, but only loosely.** Logit-lens settling depth is a highly reliable measurement: the same items settle at the same layers across prompt wordings. It correlates only weakly with when the digit becomes causally available in the residual stream (ρ 0.26). The tuned lens does slightly better (0.32 at item level, 0.56 at digit level).
- **The S2 contrast is a logit-lens contrast.** The early-versus-late split that arm S2 and the self-report arms manipulate (1.1 to 1.4 layers under the logit lens) is not detectably different from zero under causal arrival: the gap is 0.18 layers [−0.60, 0.83], so up to about 0.8 layers (roughly 74% of the logit-lens gap) remain compatible with the data. Under the tuned lens it is 0.29 [−0.14, 0.73], as in the kill check.
- **What S2 moves.** S2's answer-position patch moves whatever the donor's answer-position states carry. What that moves is a logit-lens settling profile, and that profile is only weakly tied to when the computation commits to the digit.
- **Consequences:**
  - "Course of computation" should be stated as "logit-lens settling profile" in papers 4, 6 and 8, unless it is confirmed with a causal measure.
  - The self-report null and the location results stand as results about that profile.
  - Whether a *causally* defined process variable is held at the answer positions, or reportable, is not yet tested.

**Limits.** One model and one task. Causal arrival depends on the corrupt context, which is one partner per item. The paraphrase changes only the user-turn wording.
