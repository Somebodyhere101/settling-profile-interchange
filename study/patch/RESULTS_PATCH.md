# Paper 8 patching arm: results (Qwen3-1.7B, laptop RTX 5070, 2026-10-19)

**Freeze.** `PREDICTIONS.md` was frozen at 22:37 (+04:00), SHA-256 `2c541eb995f3e1b509c791d6acd9330d957dd402385b6ccbb914ecca4d23c1be`, recorded in `PREDICTIONS.sha256`. The run went from 22:37 to 22:45 with no deviations.

**Sources.** The full output is in `res/patch_summary.json` and `logs/analyze_patch.log`.

**Units.** Effects are toward the donor, far-donor rows minus null-donor rows, in units of depth. SESOI = 1/L = 0.0357. CIs are two-way item-bootstrap 95% intervals. τ is the effect as a fraction of the full clean gap: 1.28 layers in arm P, 1.16 in arm S.

## Preregistered verdict: INVALID

**PC1 failed.** The all-prompt-position patch (iii) did not reach the ceiling at layers 14 and 21:

| Layer | (iii) D effect | τ | Class | Rows giving the donor's answer |
|---|---|---|---|---|
| 7 | +0.044 [0.040, 0.048] | 0.95 | moves | 0.99 |
| 14 | +0.030 [0.024, 0.036] | 0.66 | inconclusive | 0.95 |
| 21 | +0.007 [0.001, 0.014] | 0.16 | small, nonzero | **0.03** |

The rule says nothing is concluded about A against B. **PC2 passed.** At layer 7, the question-token patch raised the log-odds of the donor's answer by +5.3 nats [4.7, 6.0].

## Why PC1 failed (post hoc; does not change the verdict)

The preregistration assumed that by layer l the prompt holds everything that sets D. It does not, for two reasons.

**1. The early–late depth difference sits almost entirely in answer digits 2 to 4.** Mean settle layer by digit (pool baseline, 4-digit items):

| | Digit 1 | Digit 2 | Digit 3 | Digit 4 |
|---|---|---|---|---|
| Early recipients | 24.5 | 25.1 | 24.4 | 22.5 |
| Late recipients | 24.9 | 26.4 | 25.4 | 24.8 |
| Gap | 0.4 | 1.3 | 1.1 | 2.4 |

Digits 2 to 4 are read at generated positions, and no patch touches those positions.

**2. Below the patch layer, those generated positions still attend to the recipient's own prompt states.** Under (iii) at layer 21, the answer changes in 93% of rows but matches the donor's in only 3%. That is a mixture: the later digits are computed afresh at each generated position, from lower-layer reads of the operands.

So at layers ≥ 21, what sets D is not held anywhere in the prompt. It is recomputed at the answer positions themselves, which is why no prompt-only patch can reach the ceiling there.

## Exploratory cell results (arm P, 76 recipients; arm S, 150 same-product recipients)

**Primary cells (pilot layers 14 / 21 / 24).**

| Cell | Arm P D effect | τ | Arm S D effect (same answer) | τ |
|---|---|---|---|---|
| (i) @14 | +0.002 [−0.002, 0.007] | 0.05 | +0.004 [−0.001, 0.007] | 0.08 |
| (i) @21 | −0.000 [−0.001, 0.001] | 0.00 | +0.000 | 0.01 |
| (i) @24 | −0.000 [−0.001, 0.001] | −0.01 | −0.000 | 0.00 |
| (ii) @14 | +0.021 [0.015, 0.027] | 0.46 | +0.016 [0.011, 0.021] | 0.39 |
| (ii) @21 | +0.007 [0.001, 0.013] | 0.15 | +0.003 [−0.001, 0.006] | 0.06 |
| (ii) @24 | +0.005 [−0.002, 0.012] | 0.11 | +0.001 [−0.002, 0.003] | 0.02 |

**Reference layer 7.**

| Cell | Arm P D effect | τ | Arm S D effect | τ |
|---|---|---|---|---|
| (i) @7 | −0.001 [−0.005, 0.003] | −0.02 | +0.005 [0.001, 0.009] | 0.11 |
| (ii) @7 | +0.030 [0.024, 0.036] | 0.65 | +0.023 [0.017, 0.028] | 0.55 |

**Reading position only (ii-r), all layers.** Arm P: |effect| ≤ 0.0006, and the answer log-odds change by 0.00 nats. Arm S: |effect| ≤ 0.0003.

D_gen (digits 2 to n) agrees in sign and size with D in every cell.

**1. The question tokens carry content but not depth.** (i) at layer 7 shifts the answer log-odds by +5.3 nats, and at layer 14 by +4.5 nats. Yet D does not move (τ ≈ 0, CI inside ±0.006), and no row produces the donor's answer. In this format the model reads its operands from the prefill copy. The question tokens' content reaches the answer distribution, but does not set the course of the computation. This agrees with the pilot's (c) null and extends it to whole-state, non-linear patches.

**2. The oracle's reading position is causally inert.** Swapping the entire state at position 17, where `o` reads D with ρ ≈ 0.5, changes neither D nor the answer at any layer, in either arm. The oracle reads a correlate there, not a cause. This is the cleanest "readout, not mechanism" result in the series so far.

**3. Depth goes with the prefill operands, and partly apart from the answer.** (ii) carries 65% of the gap at layer 7 and 46% at layer 14. Arm S, where donor and recipient share the answer, keeps most of it: 55% and 39%. So most of what (ii) transfers is not "a different answer". It is the processing route set by the operands' prefill tokens. By layers 21 and 24 this falls to 0.02 to 0.15, matching the (iii) ceiling.

**4. Answer content and depth come apart at late layers.** Under (ii) at layer 21, the donor's answer log-odds rise by +34 nats and the answer changes in 93% of rows. D moves by only 0.15 of the gap.

**Reading.** In this model and format, settling depth is neither held at the question tokens nor at the reading position. Up to about layer 14, it is carried by the prefill operand region, partly separately from the answer value. From layer 21 on, it is not held in the prompt at all: it is rebuilt at each generated answer position from lower-layer reads. That is closest to outcome B, spread out, but spread over **time** (the answer positions), not only across prompt positions. The pilot's linear null at the reading position is then expected: that position holds nothing causal for D.

**Alternative reading (added 2026-10-20).** "Rebuilt from lower-layer reads" and "not held in the prompt" are not separated by this design.

- **What the patch leaves in place.** Every patch here replaces the prompt states at one layer L. The recipient's prompt states at layers 0 to L−1 stay as they were. Generated answer tokens still attend to those earlier layers' keys and values.
- **What this explains.** The ceiling failure at layers 21 and 24 is then just as consistent with "the state that sets D lives in the early-layer prompt representation, which attention reads directly" as with "D is recomputed from nothing at the answer positions". It also explains the 3% donor-answer rate under (iii) at layer 21, the mixture of donor and recipient content.
- **Discriminating control.** Patch the prompt positions at every layer from 0 to L (equivalent to swapping the prompt's key-value cache up to L). Compare that with patching only the generated answer positions (arm S2).
  - If the all-layers-up-to-L prompt patch moves D while the single-layer patch does not, the state is held in the early-layer prompt representation, not re-derived.
  - If only the answer-position patch moves D, recomputation at the answer positions stands.

Until that control is run, the "spread over time" conclusion above is one of two readings.

## What this does not show

- **Whether the generated positions hold D.** The fix is arm S with generated positions patched too. There, donor and recipient produce identical answer tokens, so donor states at the answer positions are well defined. The conditions would be:
  - (iv) answer positions only;
  - (iii+) all prompt and answer positions, which should reach τ ≈ 1 at every layer and is the corrected PC1.
- **"Non-linear."** That label still rests on the pilot's `o` and `os` pushes.
- **Scope.** One model, one format.
