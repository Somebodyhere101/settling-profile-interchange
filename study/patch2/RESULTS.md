# Paper 8 patching arm 2: results (Qwen3-1.7B, laptop RTX 5070, 2026-10-20)

**Freeze.** `PREDICTIONS.md` was frozen at 08:40:08 (+04:00), SHA-256 `aa7fd732c5eda288e09d5bdce26c11bb0827763566af52a0a30b9d7e17b6593d`, recorded in `PREDICTIONS.sha256` together with the code and pair hashes. The run went from 08:40 to 08:45 (`patchS2` 284 s, analysis 30 s) with no deviations; `DEVIATIONS.md` is empty.

**Sources.** The full output is in:

- `res/patch2_summary.json` (sha256 `3a5dd4de…`);
- `res/patchS2.jsonl` (sha256 `9d649166…`);
- `logs/analyze_patch2.log`.

**Units.** Effects are toward the donor, far-donor rows minus null-donor rows, in units of depth. SESOI δ = 1/L = 0.0357. CIs are two-way item-bootstrap 95% intervals.

**Sample.** Arm S, token-aligned pairs: 127 recipients (47 early, 80 late) on 381 items. The in-batch gap is 1.147 layers for D and 1.440 for D_gen (pool: 1.145 for D).

## Preregistered verdict

**answer positions @14: NOT HELD (small, nonzero); answer positions @21: HELD.**

### PC1 (corrected ceiling): passed

| Layer | (iii+) τ(D) | τ_exact | E(D) [95% CI] | Same answer |
|---|---|---|---|---|
| 7 | 1.00 | 1.00 | +0.041 [0.039, 0.043] | 1.00 |
| 14 | 1.00 | 1.00 | +0.041 [0.039, 0.043] | 1.00 |
| 21 | 1.00 | 1.00 | +0.041 [0.039, 0.043] | 1.00 |
| 24 (PC1-24) | 0.81 | 0.81 | +0.033 [0.031, 0.036] | 1.00 |

Once the answer positions are included, the whole depth gap transfers at every layer up to 21. At 24 it reaches exactly the lens-limited ceiling.

So the patching arm's failure is now accounted for: the missing part of the gap was at the generated positions. In arm S, the old prompt-only ceiling (iii) gives τ 0.88 at layer 7, 0.61 at 14, 0.10 at 21 and 0.02 at 24. That is the same pattern as arm P's 0.95 / 0.66 / 0.16.

### Primary rule: (iv), answer positions only, on D_gen

| Layer | D_gen effect [95% CI] | τ_gen [CI] | Copy-free D_gen^{>l} | Share of ceiling [CI] | Label |
|---|---|---|---|---|---|
| 7 | −0.001 [−0.004, +0.002] | −0.02 [−0.08, 0.04] | same | −0.02 [−0.08, 0.04] | NOT HELD |
| **14** | +0.009 [+0.003, +0.014] | **0.17** [0.05, 0.28] | same | 0.17 [0.05, 0.28] | **NOT HELD (small, nonzero)** |
| **21** | +0.047 [+0.042, +0.053] | **0.91** [0.81, 1.04] | +0.046 [0.041, 0.052] | 0.91 [0.83, 1.01] | **HELD** |
| 24 | +0.041 [+0.037, +0.048] | 0.80 [0.72, 0.93] | +0.021 [0.017, 0.025] (τ 0.96 of the clipped gap) | 0.98 [0.91, 1.07] | HELD (secondary) |

Recipient-only CIs agree: [0.005, 0.012] at 14 and [0.044, 0.049] at 21. Early and late recipients give the same D effect: +0.007 / +0.006 at 14 and +0.034 / +0.035 at 21.

**Sanity checks.**

- (iv) left digit 1's settle layer unchanged in 100% of rows (causality).
- (iv) changed the answer in only 2% of rows, for far and null donors alike.
- On D (all digits), (iv)@21 is "inconclusive" (τ 0.85; E = 0.0348, just under δ). That is expected: (iv) cannot touch digit 1. The preregistered measure is D_gen.

## Secondary analyses (preregistered)

**Decomposition of the ceiling, on D_gen (τ units of the 1.44-layer gap).**

| Layer | (iii) prompt | (iv) answer | Interaction (iii+) − (iii) − (iv) [CI] | (iii+) |
|---|---|---|---|---|
| 7 | 0.88 | −0.02 | +0.14 [+0.04, +0.25] (small, nonzero) | 1.00 |
| 14 | 0.59 | 0.17 | **+0.24 [+0.09, +0.39]** (small, nonzero) | 1.00 |
| 21 | 0.05 | 0.91 | +0.04 [−0.08, +0.15] (no meaningful move) | 1.00 |
| 24 | −0.02 | 0.80 | +0.04 [−0.04, +0.10] (no meaningful move) | 0.81 |

**S2, rest of the prompt.** (iii+) − (ii+) on D falls inside ±δ at every layer. At 7 and 14 it is small but nonzero: +0.016 [0.010, 0.020] and +0.013 [0.008, 0.018], that is 0.38 and 0.32 of the gap. At 21 and 24 it is about 0. Before layer 21, positions 0–16 still add something on top of the prefill and answer positions, although on their own they move little ((i)@7 τ 0.10, (i)@14 τ 0.06).

**S3 and S4.**

- (ii+) − (iv) (prefill given answer) is 0.63 / 0.51 / 0.07 / 0.01 of the gap at 7 / 14 / 21 / 24.
- (iii+) − (iii) (answer given prompt) is 0.12 / 0.41 / 0.95 / 0.83.

**S5, per-digit effects (layers; per-digit gaps: digit 1 0.27, digit 2 0.84, digit 3 0.47, digit 4 3.01).**

| Cell | Digit 1 | Digit 2 | Digit 3 | Digit 4 |
|---|---|---|---|---|
| (iii)@14 | +0.27 | +0.79 | +0.37 | +1.38 [0.95, 2.05] |
| (iv)@14 | 0 | −0.02 | +0.00 | +0.70 [0.27, 1.15] |
| (iii)@21 | +0.27 | −0.01 | +0.02 | +0.19 [0.00, 0.42] |
| (iv)@21 | 0 | +0.55 [0.21, 0.88] | +0.46 [0.22, 0.69] | +2.82 [2.44, 3.40] |
| (iii+)@21 | +0.27 | +0.84 | +0.47 | +2.99 |

At layer 14, digits 2 and 3 still take their depth entirely from the prompt. Only the last digit (digit 4, which carries most of the gap) already holds part of it at the answer positions. By layer 21, every later digit's depth sits at the answer positions.

**S6, continuity with the patching arm (arm S, all 150 recipients).**

| Cell | Here τ(D) | Patching arm τ(D) |
|---|---|---|
| (ii)@7 | 0.57 | 0.55 |
| (ii)@14 | 0.41 | 0.39 |
| (ii)@21 | 0.07 | 0.06 |
| (i), all layers | −0.00 to 0.10 | 0.00 to 0.11 |

They match within noise.

**Checks.**

| Check | Result |
|---|---|
| In-batch clean D equals pool D | 0.98 |
| Donor D equals pool D | 1.00 |
| Donor argmax under forcing is the shared answer | 0.997 |
| (iii+) rows equal to the exact splice | 0.989 |
| Recipients with chunk mismatches | 25 |

The 1.1% of (iii+) rows that differ from the splice are all second-chunk rows of recipients whose clean or donor lens tables differ between the two batch shapes (bf16 numerics). The splice reference is taken from the first chunk. These rows count equally toward far and null. A few patched rows produced fewer than 4 digits (44 of 8,496); they are left out of the per-digit cells (the "mean of empty slice" warning in the log).

## Calibration against the prior (§6)

| Prior | Outcome |
|---|---|
| PC1 passes (0.97) | passed |
| (iv)@7, τ_gen < 0.15 | −0.02 |
| (iv)@14, NOT HELD or PARTIAL (0.75), τ_gen 0.1 to 0.4 | NOT HELD, 0.17 |
| (iv)@21, HELD (0.45), τ_gen 0.4 to 0.85 | HELD; τ_gen 0.91, above the predicted range |
| (iv)@24, share ≥ 0.7 (0.7) | 0.98 |
| S1, positive interaction at 14 (0.5) | yes, +0.24 |
| S2, inside ±δ at every layer (0.8) | yes, though small but nonzero at 7 and 14 |

## Reading

Settling depth moves over time:

- **Up to layer 14**, depth is held in the prompt: mostly in the prefill operand region, partly separately from the answer value (the patching arm).
- **By layer 21**, it has been written into the residual stream at the generated answer positions. There it is held almost completely: τ_gen 0.91, with no detectable interaction with the prompt (+0.04, CI −0.08 to 0.15), and the copy-free version is unchanged. From then on, what the answer positions read from the prompt above layer 21 does not reset it.
- **At layer 14 the hand-over is under way but incomplete.** The answer positions alone carry 0.17 of the gap, all of it in the last digit. The prompt alone carries 0.59. The remaining 0.24 needs both together: the answer positions' layer-14 state and what they read from the prompt's states above 14.

This is the preregistered reading for "NOT HELD@14, HELD@21": prefill operands at mid layers, then the generated positions at late layers.

**What this means for the series.** It explains the pilot's and paper 6's result: the oracle direction reads D at the reading position, yet pushes along it fail to move D. At the pilot's settle layers (21, 24), D is held neither at the reading position (causally inert) nor anywhere in the prompt. It is held at answer positions that do not yet exist when the reading position is read. A read at the prompt sees a correlate of a quantity that is later computed and held elsewhere. "Readout, not mechanism" here has a concrete location: the mechanism is at the answer positions.

## What this does not show

- **Whether a linear direction at the answer positions would move D.** Whole-state patching shows that the state there holds depth at layer 21. It does not show that a fixed direction can push it. The natural next test is the pilot's `o` / `os` pushes restricted to the answer positions at 21.
- **What inside the answer-position state carries depth.** It could be per-digit progress, a carried value, or the digit itself; the identity of the digit is shared here, so it is not the digit identity alone.
- **Arm P.** Its donors have different answers, so there is no matched answer-position interchange.
- **Scope.** One model, one format (2-digit × 2-digit products with four-digit answers), and the 127 token-aligned recipients of arm S.
- **Prior work** listed in PREDICTIONS.md §8 must be re-checked before any write-up. Nothing has been published.

## Correction (2026-10-20, pre-release audit)

Two statements in "Reading" above need qualifying. No verdict or value changes.

1. **Circularity.** D_gen is read at the answer positions that (iv) overwrites. In the 381 S2 items, 99% of the settle layers of digits 2–4 lie above block 21 (none is below 20). So (iv)@21 shows that blocks 22–28 at those positions carry the block-21 residual forward and are not reset by attention to the recipient's prompt. It does not show that depth is computed there. The copy-free D_gen^{>l} is uninformative at block 21 (0.905 against 0.905).
2. **"Readout, not mechanism … the mechanism is at the answer positions."** The pilot's pushes from the reading position onward also wrote at the answer positions at blocks 21 and 24. The oracle component persisted there (survival 0.996) without moving depth. The gap is therefore between a direction and a whole state at the carrying site, not only between read and hold positions.
3. **The manipulated quantity** is the logit-lens settling profile. The lens arm (`../lens/RESULTS.md`) finds it only weakly related to a lens-free causal arrival layer (ρ 0.26). Under arrival, the S2 grouping is not detectably different from zero.
