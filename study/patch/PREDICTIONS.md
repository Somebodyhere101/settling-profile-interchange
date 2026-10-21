# Paper 8 patching arm: where does settling depth live?

**Status.** Written on 2026-10-19, before any patch outcome exists. Its SHA-256 is recorded in `PREDICTIONS.sha256`, together with the hashes of the code that will run. Any later change to the design or the analysis is logged in `DEVIATIONS.md` (in this folder), with its date and reason, and this file is not edited. §7 lists exactly what had been run and seen before the freeze.

The author delegated the design choices to the assistant (Claude). Those choices are recorded here. Nothing in `paper8/PREDICTIONS.md` or the frozen pilot code is changed; this arm's code is new and lives in `paper8/patch/`.

## 1. Question

In the paper 8 pilot (`paper8/RESULTS_PILOT.md`), the oracle direction **reads** settling depth D at layers 14, 21 and 24 (ρ 0.47 to 0.54). Linear pushes along it do not **move** D:

- at the question tokens (pilot condition (c)), 0 of 36 cells, every |effect| ≤ 0.006;
- from the reading position onward ((a)), only a wrong-sign effect at layer 14.

The same source-only pushes do move ARC flips. A linear null leaves two readings open:

- **A, held non-linearly at a position.** The state that sets D sits at some token positions, but not along the oracle direction, or not in a form a fixed vector can shift.
- **B, spread across positions.** No single region of the prompt holds it.

A whole-state interchange (activation patching) separates the two without assuming linearity. It replaces the full block output at a set of positions with the output from a donor prompt.

## 2. Design (fixed)

**Model and engine.** Qwen3-1.7B, thinking off, bf16, greedy decoding, run on the laptop RTX 5070 (8 GB) in the `jspace` venv.

- The engine is paper 8's `P8Model` (`code/p8lib.py`, unchanged), extended by `patchlib.PatchModel`. Its hook replaces the block output and then hands over to the paper 8 hook.
- `run_patch.py selftest` checks the engine against `gen8`. A self-patch must be the identity. A patch must leave every earlier position unchanged and must write exactly the donor's state.

**Target.** D is paper 6's D3 settling depth, computed with `specs.D3.outcome` and `cr4_parsers.py` (both unchanged):

- direct mode, with the "a * b = " prefill, 8 new tokens;
- for each answer digit, the logit-lens settling layer;
- D is the mean settling layer divided by L (L = 28).

**Prompt layout.** Every item is a 2-digit × 2-digit product, so every prompt has exactly 30 tokens, and token positions align between any donor and recipient:

| Positions | Tokens |
|---|---|
| 0–2 | `<|im_start|>user\n` |
| 3–12 | "What is a * b?"; the operand digits are 6, 7, 10 and 11 |
| 13–16 | `<|im_end|>\n<|im_start|>assistant` |
| 17 | "\n", the reading position (paper 6 and the pilot) |
| 18–21 | the empty `<think>` block |
| 22–29 | the prefill "a * b = "; its operand digits are 22, 23, 26 and 27, and position 29 predicts the first answer digit |

**Patch.** At one block l, the output at the patched positions is replaced by the donor's output at the same positions. The donor's states come from one forward pass of the donors.

Only prompt positions are patched. Donor states at generated positions exist only for the donor's own tokens. Generated positions are computed from the patched prompt through the KV cache.

**Position conditions.**

- **(i) Question tokens:** positions 3–12, the pilot's (c) positions.
- **(ii) Reading position onward:** positions 17–29. These are the pilot's (a) positions, limited to the prompt. They include the prefill copy of the operands and position 29.
- **(ii-r) Reading position only:** position 17, where `o` reads D. This condition is secondary.
- **(iii) All prompt positions:** 0–29, the sanity check and ceiling.
  - (iii) is not an exact identity. Below l, generated positions still see the recipient's prompt states.
  - (i) and (ii) do not partition the prompt: positions 13–16 belong to neither.

**Layers.** Block outputs 7, 14, 21 and 24.

- 14, 21 and 24 are the pilot's settle layers, where the linear pushes failed.
- 7 (a quarter of the depth) is a reference layer, before operand information is expected to have left the operand tokens.

### Arm P (primary): early against late settling, paper 4's M items

**Items.** The 240 M items of `code/data/items.json` "d3" (the A and oracle splits), with clean outcomes from the pool stage. Only correctly answered items with 3- or 4-digit answers are used.

**Tails.** Within each answer-digit stratum, with m the stratum's median of D·L:

- the early tail is D·L ≤ m − 0.5;
- the late tail is D·L ≥ m + 0.5.

Every early–late pair therefore differs by at least one layer, which is one SESOI. Every tail item is a recipient.

**Donors**, drawn at random with seed 8:

- **Far donors:** 3 items from the opposite tail of the same stratum, each with a different answer.
- **Null donors:** 3 items from the same stratum, each with a different answer, none of them a far donor, and each with the same depth (|ΔD·L| ≤ 0.125). Only when there are fewer than 3 is the set filled from one grid step away (|ΔD·L| ≤ 0.34).

The null donor shares everything a far donor changes (other operands, another answer, the same disruption) except the depth.

**Realised.**

- 76 recipients (33 early, 43 late) on 78 distinct items: 66 with 4-digit answers (28 early, 38 late; median 24.75 layers) and 10 with 3-digit answers (5 early, 5 late; median 24.33).
- Every recipient has 3 far and 3 null donors.
- The mean far-minus-null clean gap is 1.28 layers (minimum 1.00).
- The mean null |ΔD·L| is 0.02.

The pool stage runs 64 different prompts per batch. Its D equals the pilot's batch-1 D for 225 of the 240 M items, a bf16 batch-numerics difference. Pairing uses the pool D. Each recipient's baseline is the clean row 0 of its own batch.

### Arm S (key secondary): same product, different depth

**Pool.** Every a * b with 11 ≤ a, b ≤ 99 and a four-digit product: 6,610 generated prompts in the same format (89.7% answered correctly). a * b and b * a are distinct prompts.

**Recipients.** A recipient is a correct item whose same-product group (correct members only) has:

- at least one far member, |ΔD·L| ≥ 1.0;
- at least one null member with the same depth, |ΔD·L| ≤ 0.125.

**Donors.** The far side is the side with more far members (on a tie, the one with the larger maximum gap). Each recipient gets up to 2 far and up to 2 null donors, at random. 150 recipients were sampled at random (seed 8), with at most 2 per product.

**Realised.**

- 508 items were eligible and 150 are run (55 early, 95 late), on 445 distinct items.
- The mean gap is 1.16 layers.
- Null |ΔD| is exactly 0.

Donor and recipient share the answer, so a patch cannot move D by changing which answer is computed. If D moves here, the course of computation (depth) travels separately from the answer value. It is still tied to the operands the patched states encode.

**Rows.** For each recipient, every combination of donor, condition and layer is run:

- arm P: 6 × 4 × 4 = 96 rows;
- arm S: up to 4 × 4 × 4 rows.

Batches hold at most 64 rows, and row 0 of each is clean.

## 3. Outcome measures

**Per row.**

- **D** (primary).
- **D_gen:** D over digits 2 to n. These are read at generated positions, which are never patched, so D_gen cannot copy a lens readout from a patched position.
- The produced answer, and whether it is correct for the recipient.
- **Arm P only, Δlm:** the change in log p(donor's answer) − log p(recipient's answer), in nats. Both are teacher-forced and summed over the answer tokens, patched minus clean.

**Cell effect.** For a cell (arm, condition, layer):

E = mean over recipients r of s_r · (mean D over r's far-donor rows − mean D over r's null-donor rows).

s_r is +1 for an early recipient (late donors) and −1 for a late one, so a positive E means "toward the donor". The same contrast is computed for D_gen.

**Ceiling and τ.** The full-transfer reference ("gap") is the same contrast computed on the donors' clean D. τ = E / gap.

**CIs.** The CI is a two-way item bootstrap with 10,000 resamples and percentile 95% limits.

- In each resample, every item gets one Poisson(1) weight. The weight applies both when the item is a recipient and when it is a donor, which accounts for donors being reused.
- A recipient-only bootstrap is reported as well.

**Content and disruption.** For each cell, averaged within each recipient and then over recipients, with a recipient-bootstrap CI:

- Δlm (arm P);
- the rate of producing the donor's answer;
- the rate of changing the answer, separately for far and null donors.

**SESOI.** δ = 1/L = 0.0357 of depth (one layer), as in paper 6 and the pilot.

**Cell classes.**

| Class | Rule |
|---|---|
| **moves** | E ≥ δ and CI lower bound > 0 |
| **moves (wrong sign)** | E ≤ −δ and CI upper bound < 0 |
| **no meaningful move** | the CI lies inside (−δ, δ); flagged "small, nonzero" if it also excludes 0 |
| **inconclusive** | anything else |

**The ceiling matters.** Full transfer is about 1.28 δ in arm P and 1.16 δ in arm S. "Moves" therefore needs τ of about 0.78 (P) or 0.86 (S), which is near-complete transfer. Partial transfer shows up as "small, nonzero" or as inconclusive; the τ readout (§5.2) reports it.

## 4. Positive controls, fitted to the target

The pilot's control needed two adjacent strengths each to clear the SESOI, and it missed by 0.003. Here each control is a single pre-specified cell whose expected effect lies far above its threshold.

- **PC1 (engine and ceiling; tests D itself).** At layers 7, 14 and 21, (iii) must "move" D, and the donor-answer rate under (iii) must be ≥ 0.90. The expected values are τ ≈ 0.9 to 1.0 and a donor-answer rate of ≈ 1.0.
  - Layer 24's (iii) is reported but not required. With the patch at 24, the lens at layer 23 of the last prompt token is still the recipient's, so depths near 24 are read partly from the recipient.
  - **If PC1 fails, the verdict is INVALID** and nothing is said.
- **PC2 (content at the question tokens; fits condition (i)).** At layer 7, (i) must give a mean Δlm ≥ 0.5 nats, with CI lower bound > 0, over all six donors. Passing shows that the question-token states carry operand content that reaches the answer. 0.5 nats makes the donor's answer 1.65 times as likely relative to the recipient's.
  - **If PC2 fails,** (i) is labelled uninformative (the model reads the prefill copy instead) and B is qualified. The verdict is still computed.

## 5. Decision rules (implemented in `analyze_patch.py`)

### 5.1 Primary verdict

The verdict uses arm P only, over (i) and (ii) at the pilot layers 14, 21 and 24: six cells.

1. **INVALID** if PC1 fails.
2. **A (held at a position; non-linearly)** if at least one of the six cells moves.
   - The label is **A-source** if an (i) cell moves and **A-reading** if an (ii) cell moves.
   - "Non-linearly" rests on the pilot. Linear pushes at the same positions and layers did not move D in the claimed direction: under (c), 0 of 36 cells; under (a), no move by the pilot's rule, and only a wrong-sign −o effect at layer 14.
   - **Lens-copy caveat.** An (ii) cell that moves while its D_gen effect has CI lower bound ≤ 0 is flagged. Its move may be the patched readout at position 29 itself.
3. **OTHER** if no cell moves toward the donor but at least one moves with the wrong sign.
4. **B (spread across positions)** if none of the six cells moves in either sign and PC1 passes.
   - **B-strong** if all six are "no meaningful move"; otherwise **B-weak**.
   - If PC2 failed, "(i) uninformative" is appended.

### 5.2 Secondary analyses (preregistered; not part of the verdict)

- **Partial transfer.** Lists any of the six cells that does not move but has CI lower bound > 0 and τ ≥ 0.5.
- **S1, the reading position alone.** The classes of (ii-r) at 14, 21 and 24. This asks whether the state at the oracle's own reading position causes D.
- **S2, the arm-S qualifier.** For each A cell, look at the same cell in arm S:
  - CI lower bound > 0: **answer-independent**. Depth is held there separately from the answer.
  - "No meaningful move" with a CI that includes 0: **answer-bound**. D travels only with the problem that the position encodes.
  - Otherwise: inconclusive.

  The full arm-S table is reported in either case.
- **Also reported:**
  - layer-7 cells;
  - D_gen effects;
  - recipient-only CIs;
  - effects for early and late recipients separately;
  - per-cell content measures.

**Reading map.**

| Result | Reading |
|---|---|
| A-reading, answer-bound | The positions from the reading position onward hold the problem, and depth follows the problem. No separate "course of computation" variable is needed there. The oracle reads a correlate of the problem's difficulty, which a fixed vector cannot rewrite. |
| A-reading, answer-independent | Those positions hold a depth or route state beyond the answer, in a non-linear form. |
| B | No single region holds what sets D at the pilot layers. That would account for why position-limited linear pushes fail. |

## 6. Assistant's prior (for calibration; not a decision rule)

- **Controls.** PC1 passes: 0.95. PC2 passes: 0.6, because the prefill copy of the operands sits right before the answer and may dominate.
- **Verdict.** A: 0.65 (A-reading 0.60, A-source 0.15). B: 0.25. OTHER or INVALID: 0.10.
- **Cell expectations.**
  - (ii) at 14: τ between 0.6 and 1.0, mostly with the donor's answer.
  - (i) at 14, 21 and 24: τ < 0.2.
  - (ii-r) moves at no layer: 0.8.
- **S2.** For the (ii) A cells, "answer-independent": 0.45.

## 7. What was run and seen before the freeze

- **`selftest`.** It passed:
  - 0 token and 0 lens differences from `gen8`;
  - a self-patch changed 0 tokens, and at most 0.8% of lens top-1 entries across all layers and steps (bf16 batch numerics);
  - causality and the exact write differ by exactly 0;
  - (iii) at layer 7 reproduced the donor's answer 3 of 3 times;
  - the teacher-forced log-prob matched generation to 1e-6.
- **`pool`.** Clean outcomes for 240 M items and 6,610 generated prompts. These are baselines, not patch outcomes. The pilot had already seen the M items' clean D.
- **`pair`.** The counts in §2.
- **Crash test.** `patchP 2`, `patchS 2` and `analyze_patch.py smoke` ran with exit code 0, at about 3.5 s per recipient for arm P and 1.5 s for arm S. Their outputs (`res/*_smoke_*`) were not opened, and they are deleted before the freeze.

## 8. Known limits, stated in advance

- **Scope.** One small model and one format (2-digit × 2-digit products).
- **Narrow range of D.** The SD across items is 0.54 layers, so full transfer is only 1.2 to 1.3 SESOI. A null therefore needs CIs inside ±1 layer, and a "move" needs near-complete transfer.
- **The two operand copies.** (ii) contains the prefill copy of the operands, so (ii) moving D at early layers is expected from content alone. Arm S and D_gen address this.
- **Lens copying.** Under (ii) and (iii), the lens readout at the patched last prompt token is the donor's from layer l upward. D_gen is the control for this.
- **Batch numerics.** Donor states come from a batch of donors, so bf16 batch numerics apply. Their effect is equal for far and null donors.
- **"Non-linear" is borrowed from the pilot.** That label rests on the pilot's `o` and `os` pushes at k = 1 to 4 gaps, and does not exclude every possible linear direction.
- **Prior work to re-check before any write-up.**
  - Interchange interventions (Geiger et al. 2021) and causal tracing (Meng et al. 2022).
  - Activation-patching practice (Zhang & Nanda 2024; Heimersheim & Nanda 2024).
  - Arithmetic mediation: Stolfo et al. 2023, who report that operand information is moved to the last token by middle layers.
