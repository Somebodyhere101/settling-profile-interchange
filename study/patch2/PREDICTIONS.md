# Paper 8 patching arm 2: is settling depth held at the answer positions?

**Status.** Written on 2026-10-20, before any patch2 outcome exists. Its SHA-256 is recorded in `PREDICTIONS.sha256`, together with the hashes of the code and the pairs that will be used. Any later change to the design or the analysis is logged in `DEVIATIONS.md` (in this folder), with its date and reason, and this file is not edited. §7 lists exactly what had been run and seen before the freeze.

The author delegated the design choices to the assistant (Claude); they are recorded here. Nothing in `paper8/PREDICTIONS.md`, `paper8/patch/PREDICTIONS.md` or any frozen code is changed. This arm's code is new and lives in `paper8/patch2/`. It imports `paper8/patch/patchlib.py` and `paper8/code/*` unchanged.

## 1. Question

The patching arm (`paper8/patch/RESULTS_PATCH.md`) came out **INVALID** by its preregistered rule. Its all-prompt-position ceiling (iii) reached only τ 0.66 at layer 14 and 0.16 at layer 21. The post-hoc diagnosis:

- the early–late depth gap sits mostly in answer digits 2 to 4;
- those digits are read at **generated** positions, which (iii) never patched;
- below the patch layer, those positions recompute from the recipient's own prompt.

So at late layers, nothing in the prompt holds what sets D. This arm asks the next question:

> **Is depth held at the answer positions themselves?** If the block output at the generated answer positions alone is swapped for a donor's at layer l, do the later digits settle at the donor's depth?

In arm S, donor and recipient produce identical answer tokens. The donor's state at every answer position is therefore defined at the same position, holding the same token. That makes this interchange possible without confounding by answer content.

## 2. Design (fixed)

**Model and engine.** Qwen3-1.7B, thinking off, bf16, greedy decoding, on the laptop RTX 5070 (8 GB), in the `jspace` venv.

The engine is `patchlib2.PatchModel2`, which subclasses `patch/patchlib.PatchModel` (unchanged), which in turn wraps paper 8's `P8Model`. **The interchange is done inside one batch.** Each batch holds:

1. row 0: the clean recipient;
2. the recipient's donors, each on its own prompt and **teacher-forced on the shared answer tokens**;
3. the patched rows, all on the recipient's prompt.

At block l, a patched row's output at its masked absolute positions is replaced by its donor row's output at the same positions. Masked prompt positions are patched in the prompt pass. Masked answer positions are patched at the generation step that processes them.

Because the donor states come from the same forward pass, the copy is bit-exact and free of batch-shape numerics. Paper 8's lens capture runs after the replacement.

**Target.** D is paper 6's D3 settling depth, computed with `specs.D3.outcome` and `cr4_parsers.py` (unchanged):

- direct mode, with the "a * b = " prefill and 8 new tokens;
- for each answer digit, the logit-lens settling layer;
- D is the mean settling layer divided by L (L = 28).

**Positions.** The prompt layout is unchanged from the patching arm: 30 tokens, question tokens 3–12, reading position 17, prefill 22–29. With K shared answer tokens (K = 4 for every recipient but one, which has K = 5 with a comma), the **answer positions** are 30 to 30 + K − 2 (positions 30–32 for K = 4).

- These positions hold answer tokens 1 to K − 1.
- Their outputs predict answer tokens 2 to K, so they are exactly the positions whose lens readouts give the settle layers of digits 2 to n.
- Digit 1 is read at position 29, a prompt position.

**Conditions.**

| Code | Name | Positions |
|---|---|---|
| (i) | question tokens | 3–12 (continuity with the patching arm) |
| (ii) | reading position onward, prompt only | 17–29 (continuity) |
| (iii) | all prompt positions | 0–29 (continuity: the old, failed ceiling) |
| **(iv)** | **answer positions only** | 30 … 30 + K − 2 |
| (ii+) | reading position onward plus the answer positions | 17 … 30 + K − 2 |
| **(iii+)** | **all prompt and answer positions** | 0 … 30 + K − 2 (the corrected ceiling, PC1) |

The reading-position-only condition (ii-r) was causally inert in both arms of the patching arm and is not repeated.

**Layers.** Block outputs 7, 14, 21 and 24, as in the patching arm.

**Pairs.** These are arm S of the patching arm (`../patch/res/pairs.json`, frozen there): 150 recipients, same product, far donors with |ΔD·L| ≥ 1, and null donors with ΔD = 0. The one change is that a donor is kept only if its clean answer tokens (pool) **equal the recipient's**. Otherwise the answer positions do not align: the same product can be written "5,040" or "5040".

A recipient is kept if at least one far and one null donor remain. The resulting `res/pairs2.json` (stage `pair2`) holds:

- 127 recipients (47 early, 80 late);
- 176 far donors and 178 null donors;
- 381 distinct items.

48 of 425 donor slots were dropped for unequal tokens.

**Rows.** Each recipient runs every combination of donor (2 to 4), condition (6) and layer (4): up to 96 patched rows. Batches hold at most 64 rows, including the clean recipient and the donor rows.

## 3. Outcome measures

**Per patched row.**

- **D** over all digits.
- **D_gen** over digits 2 to n. This is the measure (iv) can move: digit 1 is read before any answer position.
- **D_gen^{>l}**, the lens-copy-free D_gen. Each digit's settle layer s is replaced by max(s, l + 1), so only lens layers above the patch layer can lower it. Under (iv), the lens at layer l at an answer position is the donor's copied state. D_gen^{>l} removes that copied readout and everything below it.
- **Per-digit settle layers** (digits 1 to 4, in layers).
- Whether the produced answer tokens equal the shared answer.

**Cell effect.** For a cell (condition, layer) and a measure X:

E = mean over recipients r of s_r · (mean X over r's far-donor rows − mean X over r's null-donor rows).

s_r is +1 for an early recipient (late donors) and −1 for a late one, so a positive E means "toward the donor".

**Gap and τ.** The gap is the same contrast computed on the **in-batch donor rows** (the actual sources of the patch), on the same measure; τ = E / gap. The pool-based gap of the patching arm is reported alongside. On the pool it was 1.14 layers for D and 1.44 for D_gen on these 127 recipients.

**Exact-interchange ceiling.** For each row and layer, the splice of the recipient's clean lens table below l and the donor row's lens table from l up is computed. Its D contrast over gap is **τ_exact(l)**: the most any patch at layer l can transfer, because lens layers below l always belong to the recipient.

**CIs.** The CI is a two-way item bootstrap with 10,000 resamples, seed 80, and percentile 95% limits.

- In each resample, every item gets one Poisson(1) weight. The weight applies both when the item is a recipient and when it is a donor.
- A recipient-only bootstrap is reported as well.
- Differences between cells, the interaction and the share use the same resamples.

**SESOI.** δ = 1/L = 0.0357 of depth (one layer), as in paper 6, the pilot and the patching arm. For per-digit effects, which are in layers, it is 1 layer.

**Cell classes** (unchanged).

| Class | Rule |
|---|---|
| **moves** | E ≥ δ and CI lower bound > 0 |
| **moves (wrong sign)** | E ≤ −δ and CI upper bound < 0 |
| **no meaningful move** | the CI lies inside (−δ, δ); flagged "small, nonzero" if it also excludes 0 |
| **inconclusive** | anything else |

## 4. Positive control: PC1, the corrected ceiling

**Rule (lenient, single cell).** Each of the cells (iii+)@7, (iii+)@14 and (iii+)@21 must pass on its own:

- τ(D) ≥ 0.8 · τ_exact(l);
- the two-way CI lower bound of E(D) > 0;
- the produced answer equals the shared answer in ≥ 90% of rows (recipient-averaged).

**If any of the three fails, the verdict is INVALID.**

This is lenient compared with the patching arm's PC1. That one required the "moves" class, which with a gap of 1.14 δ needs τ ≥ 0.875. Here the cell is not required to clear the SESOI.

**PC1-24.** The same rule at layer 24. It gates only the layer-24 readouts, which are secondary.

**Expected.** τ_exact ≈ 1.00 at 7 and 14, ≈ 1.00 at 21, and ≈ 0.75 at 24. These are estimated from the clean pool's per-digit settle layers (§7). At 24, many donor digits settle at or below 24, where the recipient's own lens takes over. τ ≈ 1 at **every** layer is therefore not reachable by any interchange at 24, and the rule is relative to τ_exact.

**What PC1 does and does not show.** With the in-batch interchange, (iii+) equals the exact splice by construction (the selftest checks this). PC1 therefore checks:

- the run's bookkeeping (masks, donor rows, chunking, forcing);
- that the D gap is fully transferable once the answer positions are included.

That was the failure of the patching arm. PC1 does not test any hypothesis about where depth lives.

## 5. Decision rules (implemented in `analyze_patch2.py`)

### 5.1 Primary: is depth held at the answer positions?

The rule is applied to (iv)@l on **D_gen**, separately for each layer. The primary layers are **14 and 21**. Layers 7 and 24 get the same rule, as secondary readouts.

| Label | Rule |
|---|---|
| **HELD** | D_gen effect "moves" (E ≥ δ, CI lower bound > 0) **and** the copy-free D_gen^{>l} effect has CI lower bound > 0 |
| **NOT HELD** | D_gen effect "no meaningful move" (CI inside ±δ; "small, nonzero" is appended if the CI excludes 0) |
| **WRONG SIGN** | D_gen effect "moves (wrong sign)" |
| **PARTIAL** | none of the above, and the D_gen CI lower bound > 0 |
| **INCONCLUSIVE** | anything else |

**Verdict.**

1. INVALID if PC1 fails.
2. Otherwise: "answer positions @14: ⟨label⟩; answer positions @21: ⟨label⟩".

With a D_gen gap of about 1.44 δ, HELD needs τ_gen ≥ about 0.70 with a CI excluding 0.

**Always reported with the label:**

- τ_gen with its CI;
- the copy-free effect;
- the **share of the ceiling**, E_gen(iv) / E_gen(iii+), with its bootstrap CI.

### 5.2 Secondary analyses (preregistered; not part of the verdict)

All of the following use D_gen except S2, and each is reported with a two-way CI and a class, at every layer.

- **S1, interaction:** E(iii+) − E(iii) − E(iv). A positive interaction means the prompt and the answer positions are needed together at layer l; zero means their contributions add.
- **S2, rest of the prompt:** E(iii+) − E(ii+), on D. A CI inside ±δ means positions 0–16 add nothing once the prefill region and the answer positions are patched.
- **S3, prefill given answer:** E(ii+) − E(iv). What the prefill region adds on top of the answer positions.
- **S4, answer given prompt:** E(iii+) − E(iii). What the answer positions add on top of the whole prompt.
- **S5, per-digit effects** (layers) for every cell, with the per-digit gaps.
- **S6, continuity.** (i), (ii) and (iii) in this subset, against the patching arm's arm-S values. There is no rule: the subset is 127 of 150 recipients and the engine is the in-batch one.
- **Checks:**
  - (iv) must leave digit 1's settle layer unchanged in every row, because digit 1 is read before any answer position;
  - in-batch clean D against the pool D;
  - whether each donor row's own argmax is the shared answer;
  - chunk mismatches.
- **Also reported:** recipient-only CIs, early and late recipients separately, and the same-answer rates per cell.

**Reading map.**

| Result | Reading |
|---|---|
| HELD@14 and HELD@21 | From layer 14 on, the residual stream at the answer positions itself fixes how deep the later digits settle; attention to the prompt above l does not reset it. |
| NOT HELD or PARTIAL @14, HELD@21 | Depth is written into the answer positions between layers 14 and 21. With the patching arm's (ii)@14 (τ 0.39 in arm S), this is a hand-over over time: prefill operands at mid layers, then the generated positions at late layers. |
| NOT HELD@21 | The answer positions' state at 21 does not fix depth. It is set by what they read from the prompt above 21, or by both together; S1 decides which. |
| WRONG SIGN | Reported as found; no reading is preregistered. |

## 6. Assistant's prior (for calibration; not a decision rule)

- **PC1 passes:** 0.97. The engine is exact; the residual risk is bookkeeping.
- **(iv)@7:** τ_gen < 0.15; NOT HELD 0.8.
- **(iv)@14:** τ_gen 0.1 to 0.4; NOT HELD or PARTIAL 0.75, HELD 0.1.
- **(iv)@21:** τ_gen 0.4 to 0.85; HELD 0.45, PARTIAL 0.4, NOT HELD 0.15.
- **(iv)@24:** share of the ceiling ≥ 0.7, with probability 0.7.
- **S1:** a positive interaction at 14 (CI lower bound > 0), probability 0.5.
- **S2:** (iii+) − (ii+) inside ±δ at every layer, probability 0.8.

## 7. What was run and seen before the freeze

- **Clean-pool baselines.** These are the patching arm's `pool_clean.jsonl`, not patch outcomes. The checks were:
  - the answer-token alignment of arm-S pairs, which led to the alignment rule;
  - the pool gaps for D and D_gen on the kept pairs;
  - per-digit pool gaps (digit 1 0.22, digit 2 0.74, digit 3 0.48, digit 4 3.19 layers, on all 150 arm-S recipients);
  - an approximate τ_exact from pool settle lists (≈ 1.00 at 7 and 14, 0.997 at 21, 0.75 at 24);
  - the copy-free D_gen gap (1.44 layers at 7 and 14, 1.41 at 21, 0.60 at 24).
- **`pair2`.** The counts in §2.
- **`selftest`.** It was run four times while the engine was built. The first run stopped on a code error (an empty slice in the causality check). The first two engine versions took donor states from a separate teacher-forced pass: a full forward, then a KV-cache loop. Under a self-patch, those versions changed 1.1% and 1.6% of lens top-1 entries, and 3 of 48 digit settle layers, through bf16 batch-shape numerics. This led to the in-batch interchange. The final version passed with exact zeros on 3 same-product pairs that are not in the run:
  - 0 token or lens differences from `PatchModel.gen`;
  - forcing a row's own answer changed 0 lens entries;
  - a self-patch changed 0 tokens and 0 lens entries;
  - write error 0, and causality error 0 at every layer;
  - (iii+) reproduced the donor row's lens above l exactly, and its settle layers equalled the splice in 12 of 12 rows.

  No far-null contrast is computed in the selftest.
- **Crash test.** `patchS2 3` and `analyze_patch2.py smoke` ran with exit code 0, at about 2.5 s per recipient with 2 donors. Their outputs (`res/patchS2_smoke.jsonl`, `res/patch2_smoke_summary.json`, `logs/smoke_analyze.log`) were not opened and were deleted before the freeze. `logs/smoke_patchS2.log` holds only progress and timing lines.

## 8. Known limits, stated in advance

- **Scope.** One small model and one format (2-digit × 2-digit products with a four-digit answer), and one arm (same product). Arm P cannot be run this way: its donors give different answers, so there are no matched donor states at the answer positions.
- **Lens readout.** D is read by the logit lens, so lens layers below l always belong to the recipient. That caps any transfer at τ_exact(l) and makes layer 24 weak (copy-free gap 0.60 layers, below δ).
- **(iv) still lets the answer positions attend to the recipient's prompt above l.** That is the point of the test, but HELD means only that the answer-position state at l dominates what those reads add. It does not mean the prompt is unused.
- **Teacher forcing.** The donor rows are forced on the shared answer. Their own argmax under forcing is the shared answer in almost all rows (checked and reported). Where it is not, the donor's state is still the state at the shared token.
- **Narrow range of D.** Full transfer is only 1.1 to 1.4 δ, so HELD needs near-complete transfer, and partial transfer shows up as PARTIAL; the τ and share readouts report it.
- **Selection.** The 127 recipients are the token-aligned subset of the 150. The drop depends on how the model writes the product, not on any patch outcome.
- **Prior work to re-check before any write-up.**
  - Interchange interventions (Geiger et al. 2021) and causal tracing (Meng et al. 2022).
  - Activation-patching practice (Zhang & Nanda 2024; Heimersheim & Nanda 2024).
  - Arithmetic mediation (Stolfo et al. 2023).
  - Work on multi-token answers, where later tokens are computed at generated positions.
