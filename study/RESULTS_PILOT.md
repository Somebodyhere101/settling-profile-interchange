# Paper 8 pilot: results (Qwen3-1.7B, laptop RTX 5070, 2026-10-19)

The full output is `code/res/q3-1.7b_summary.json` and `code/logs/analyze_q3-1.7b.log`. Deviations are in `DEVIATIONS.md`: a selftest fix, the move to laptop hardware, and two interim looks.

Effects are matched-KL effects against Gaussian pushes at the same positions and layer, with two-level 95% CIs. Settle effects are in units of depth (SESOI 0.0357). Flip effects are flip-rate points (SESOI 0.10).

## Preregistered verdict: UNINFORMATIVE

The positive control failed rule 1. Flips did not move under (a) at two adjacent strengths.

At the published layer 19, +o under (a) raised flips by:

| Strength | Effect | 95% CI |
|---|---|---|
| 1 gap | +0.067 | [0.026, 0.112] |
| 2 gaps | +0.097 | [0.034, 0.158] |
| 4 gaps | +0.213 | [0.122, 0.308] |

The dose response is monotone and in the claimed sign. It misses the rule only because the 2-gap effect is 0.003 below the SESOI. −o lowered flips by 0.127 [0.070, 0.189] at 2 gaps.

## Settle (course of computation), 120 items

- **Readable.** `o` and `os` read settling depth at layers 14, 21 and 24, with ρ from 0.47 to 0.54, all pass.
- **Source-only (c).** It moved settling depth in 0 of 36 cells. Every |effect| is ≤ 0.006, and the CIs are about ±0.002 to 0.004.
- **Reading position onward (a) and all positions (b).** The only controllable cells are at layer 14, all for −o and all in the wrong direction (settling gets later):

  | Condition | Strength | Effect | 95% CI |
  |---|---|---|---|
  | (a) | 4 gaps | +0.049 | [0.026, 0.074] |
  | (b) | 2 gaps | +0.040 | [0.016, 0.065] |
  | (b) | 4 gaps | +0.045 | [0.033, 0.058] |

  Nothing reaches the SESOI at the published layer 21.
- **Persistence.** The oracle write at the answer position is kept: survival 0.996 against 1.009 for Gaussian writes, a difference of −0.012 [−0.015, −0.009]. It is not overwritten.
- **Rule-based "moves".** a: no, b: yes, c: no. The preregistered H-REDERIVE pattern would have fired from (b) alone. The secondary reading declared before the final data requires (c) to move, so it reads this as **not a source effect**.

## Flip (answer choice), 159 items (exploratory beyond rule 1)

Readability: `o` passes at 19 (AUC 0.66) and 23 (0.72) and fails at 12 (0.56). `os` is inconclusive at every layer (0.57 to 0.59).

**Source-only (c), `os`, layer 12.** Flips move in the claimed sign in both directions:

| Strength | +os | −os |
|---|---|---|
| 1 gap | +0.099 | −0.079 |
| 2 gaps | +0.295 [0.216, 0.379] | −0.130 [−0.193, −0.076] |
| 4 gaps | +0.166 [0.053, 0.278] | −0.079 |

The KLs are 2 to 27.

Under (c), `o` also moves flips at layer 12, but `o` does not read flips there, so its sign means nothing.

## Reading (exploratory, post hoc for the control)

The same source-only write moves the answer choice by up to 30 points, yet leaves settling depth flat to within ±0.004. The settle write at the answer position also survives intact.

In this model, then:

- The answer choice is partly held at the source tokens early on (layer 12).
- The course of the computation is not linearly controllable at any position tested, although it is linearly readable at every layer tested.

This is closer to H-nonlinear (with a layer-14 disruption caveat) than to H-rederive. By the preregistered rules, the pilot does not justify the 3-model study.

The next step is the non-linear patching arm (whole-state interchange at the source and answer positions), with a positive control that fits the target.
