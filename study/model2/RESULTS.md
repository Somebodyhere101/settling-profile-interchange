# Paper 8 second model: SmolLM2-1.7B-Instruct (laptop RTX 5070, 2026-10-20)

**Freezes.**
- Stage A (screen): `PREDICTIONS_A.md`, frozen 13:10, SHA-256 `ed88e318…`.
- Stage B (replication): `PREDICTIONS_B.md`, frozen 13:54:57, SHA-256 `59c4f458…`.
- Runs: stage A 13:44–13:46; stage B 13:55–14:41.
- `DEVIATIONS.md` records the model substitution (the author preferred the cached model to a download; the coordinating session made the substitution before any run) and later text and attribution corrections.

**Code.** `mk_m2.py` copies of the Qwen arms' frozen scripts. The only changes are the model key, the layer map (7 → 6, 14 → 12, 21 → 18, 24 → 21) and the report footer.

**Sources.** `patch2/res/patch2_summary.json`, `selfreport/res/kill_summary.json`, `selfreport2/res/sr2_summary.json`, `lens/res/lens_summary.json`, and `logs/*_analyze.log`.

## Stage A: GO

- **Accuracy:** 0.49 on the M items and 0.43 on the generated pool.
- **Pairs:** 150 arm-S2 recipients (54 early, 96 late), 359 items.
- **Gaps:** the mean far gap is 1.13 layers (D) and 1.35 layers (D_gen), in batch.

## Arm S2: **replicates** (answer positions @12: NOT HELD; @18: HELD)

PC1 passes at 6, 12 and 18 (τ 0.99), and at 21 (τ 0.78, the lens-limited ceiling 0.79).

| Cell | D_gen τ [95% CI] | Qwen3-1.7B (block) |
|---|---|---|
| (iv)@6 | 0.00 | −0.02 (7) |
| (iv)@12 | 0.04 [−0.02, 0.12]: NOT HELD | 0.17: NOT HELD (14) |
| **(iv)@18** | **0.79 [0.70, 0.92]: HELD** (copy-free 0.79) | 0.91: HELD (21) |
| (iv)@21 | 0.72: PARTIAL (copy-free 0.82; share of ceiling 0.94) | 0.80 (24) |
| (ii)@6, @12 | 0.58, 0.50 | 0.57, 0.41 (7, 14) |
| (iii)@6 / @12 / @18 | 0.92 / 0.73 / 0.06 | 0.88 / 0.59 / 0.05 |
| (i), any layer | ≤ 0.10 | ≤ 0.10 |

The location result replicates:

- **Hand-off over time.** Depth is held in the prompt (the prefill operand region) at early and mid layers. At the late layer it is held at the generated answer positions.
- **The answer is unchanged.** (iv)@18 leaves the answer unchanged in 94% of rows. Digit 1 is untouched in 100%.

## Kill check: PASS by rule, and more robust than in Qwen

| Lens | (iv)@18 τ_gen [CI] | Far − null gap |
|---|---|---|
| Logit | 0.82 [0.74, 0.92] | 0.054 (1.31 layers) |
| Tuned | 0.79 [0.25, 2.74] | 0.023 (0.55 layers) |
| Tuned, copy-free | **0.79 [0.45, 1.15]** | |

- The two lenses agree on recipients' D more than in Qwen: ρ 0.44, against 0.20.
- The tuned lens shrinks the gap 2.4-fold here, against 8-fold in Qwen.

## Self-report: **NOT EVALUABLE**

**Why the report is not evaluable.** The clean digit report does not differ between donors and recipients in the direction of their settling difference: the gap is +0.004 layers [−0.005, +0.016]. The A/B gap is −0.000 [−0.020, +0.023].

**What the report does instead.** It is close to item-invariant:

- The argmax is "the fourth digit" for 133 of 150 recipients.
- On items outside the run, the raw letter choice is "B" in most option orders.
- Even the direct-steering control barely moves it: (iv+rep)@18 effect +0.004 [−0.005, 0.017], and A/B −0.010 [−0.027, 0.008].
- In Qwen, by contrast, (iv+rep) transferred the report fully.

**The cell values are null but uninterpretable.** (iv)@18 effect −0.000 [−0.003, 0.003]; (i)@6 +0.001; (ii)@6 +0.007 [0.002, 0.011].

**No input reconstruction to detect.** SmolLM2-1.7B's self-report readout carries almost no item-specific information. Its input-only predictor reaches cv R² 0.56 on the digit logits, but there is no donor–recipient report difference to reconstruct. That is the preregistered NOT EVALUABLE case: the readout, not the model's access, is what fails.

## Lens robustness: **MIXED** (as in Qwen)

**Primary correlations.**

| | ρ | 95% CI |
|---|---|---|
| (a) logit lens vs (c) causal arrival | **0.40** | [0.34, 0.46] |
| (b) tuned lens vs (c) causal arrival | **0.41** | [0.34, 0.46] |
| (a) vs (b) | 0.43 | |

The digit-level ρ is 0.39 (a) and 0.56 (b).

**Reliability across paraphrase:** 0.92 to 0.98.

**SDs:** logit lens 0.54 layers, causal arrival 1.76.

**S2 grouping** (far − null, in layers):

| Measure | Gap | 95% CI |
|---|---|---|
| Logit lens | 1.08 | [1.02, 1.17] |
| Tuned lens | 0.44 | [0.06, 0.88] |
| **Causal arrival** | **0.65** | [**0.13**, 1.28] |
| Causal arrival, D_gen | 0.89 | [0.17, 1.74] |

In SmolLM2 the arm-S2 grouping survives the lens-free measure: it is smaller, but its CI excludes 0. In Qwen3-1.7B it did not (0.18 [−0.60, 0.83]).

## Summary against Qwen3-1.7B

| Arm | Qwen3-1.7B | SmolLM2-1.7B | Replicates? |
|---|---|---|---|
| S2 location | NOT HELD@14, HELD@21 | NOT HELD@12, HELD@18 | **yes** |
| Kill check | PASS (tuned gap 0.17 layers) | PASS (tuned gap 0.55 layers; copy-free τ 0.79) | yes, and more robust |
| Self-report | RECONSTRUCTION FROM INPUT | NOT EVALUABLE (item-invariant report) | **not testable** |
| Lens arm | MIXED, ρ(a, c) 0.26; S2 grouping not detectably different from 0 under (c) | MIXED, ρ(a, c) 0.40; S2 grouping differs under (c), below the SESOI | label yes; grouping differs |
