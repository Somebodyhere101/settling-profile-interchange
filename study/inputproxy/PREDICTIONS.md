# Paper 8 input-proxy arm: is the readable "process" an input correlate?

**Status.** Written on 2026-10-20 before any target or probe value of this arm existed. The only things run before the freeze are:

- `run_ip.py items`: greedy answers for the variants, to keep identical-answer variants, plus the choice of the fixed corrupt items;
- `run_ip.py selftest`: 2 items **outside** the base set.

**Item build results** (answers only):
- 1,015 of the 1,050 variants kept an answer identical to v0's: v0 149, v1 147, v2 149, v3 141, v4 136, v5 146, v6 147.
- 148 base items have at least 4 kept variants.
- The corrupt items are G:48*97, G:41*35, G:37*97 and G:78*81.
- The selftest passed on 2 items outside the base set.

The SHA-256 of this file, the code and the item list is in `PREDICTIONS.sha256`. Changes go in `DEVIATIONS.md`.

The author asked for this arm (relayed by the parent session). The assistant made the design choices recorded here.

## 1. Question

Probes and oracles read "process" variables at the reading position (paper 6: oracles read 17 of 19 process targets). The self-report arms found reports reconstructed from operand content. Both could have one explanation: what is readable at the prompt is a correlate of the **input**, a property of the operands, and not of how a particular run of the computation goes.

This arm separates the two:

- **Between items.** Hold the context fixed and vary the operands.
- **Within items.** Hold the operands and the answer fixed and vary the context.

A probe that reads the process should follow the target within items too. An input proxy should not.

## 2. Design (fixed)

**Model and engine.** Qwen3-1.7B, thinking off, bf16, on the laptop RTX 5070. The engine is `../selfreport/srlib.SRModel`, unchanged: teacher-forced passes, with paper 8's per-row additive push used for the replacement.

**Items.**
- 150 base items: correct 4-digit generated items from the Qwen pool, drawn with seed 11.
- None is in arm S2, in the tuned lens's training set, in the lens arm's measured set, or among the corrupt items.

**Context variants.** Every variant uses the same "a * b = " prefill.

| Variant | Context |
|---|---|
| v0 | "What is a * b?" |
| v1 | Two worked examples first: "What is 12 * 34? 408 / What is 56 * 78? 4368 / What is a * b?" |
| v2 | A filler sentence first: "Please read the following question carefully and answer it." |
| v3 | Format change: "Compute the product of a and b." |
| v4 | An irrelevant preceding sum: "Note that 17 + 25 = 42." |
| v5 | A system prompt: "You are a precise calculator." |
| v6 | Paraphrase: "Calculate a * b." |

A variant is kept only if its greedy answer tokens equal the item's v0 answer. An item enters the analysis if v0 and at least 2 other variants are kept.

**Target: the lens-free commitment layer** (the lens arm's causal arrival, adapted as requested).

- **Corrupt runs.** There are 4 fixed corrupt items whose digits differ pairwise at every position. Each is run in v0 form, teacher-forced on its own answer, and is never intervened on except at the single replaced position.
- **Which corrupt run.** For digit k, corrupt run A is the first fixed item whose digit k differs from the clean digit; corrupt run B is the second.
- **Replacement.** The clean run's block-l output at the position that predicts digit k replaces the corrupt run's block-l output at its digit-k position. The clean output is **rescaled to the corrupt run's own norm there**. The replacement is implemented as an additive push equal to the difference.
- **Arrival layer.** The smallest l such that the corrupt run's output at that position is the clean digit for every l′ ≥ l.
- **T.** The mean over the 4 digits, with corrupt run A (primary). T_B uses corrupt run B (reliability).
- **Secondary target:** logit-lens settling depth.

**Probe sites.** Block outputs at:

- the **reading position** (the "\n" after "assistant", paper 6/8's site), at layer 21 (**primary**) and layer 14;
- the last prompt token, at 21 and 14.

**Oracle.** Fitted on **between-item** variation only, in v0 form: the mean of the top tercile of T minus the bottom tercile, normalised. It uses 5-fold cross-fitting over items with seed 12, so each item's projections come from an oracle that never saw it.

## 3. Outcome measures

- **Between-item AUC.** On held-out v0 items, top against bottom tercile of T, with thresholds from the training folds. The held-out between-item Spearman is also reported.
- **ρ_w (within items).** Each item's projections and T are centred over its kept variants, then Spearman is computed on the pooled centred values. CI: item bootstrap, 10,000 resamples, seed 13.
- **Target reliability.** The Pearson correlation of within-item-centred T_A against T_B.
- **Secondary:**
  - the variance decomposition of T (within against between);
  - each variant's mean shift of T relative to v0;
  - all probe sites;
  - the same analyses with logit-lens settling as the target;
  - ρ(settle, arrival).

## 4. Decision rule (the author's, as relayed)

1. **UNDERPOWERED** if the within-item reliability of T is below 0.5. This is reported as such, not as a null.
2. **INPUT-PROXY** if ρ_w's upper 95% bound is below 0.15 **and** the between-item AUC is ≥ 0.75 (primary site).
3. **PROCESS-READABLE** if ρ_w ≥ 0.3 with a CI excluding 0. That case triggers a separately preregistered stage 2. It tests whether an answer-position push of the oracle moves the commitment layer by more than 2× a push of random directions matched for KL and for covariance. It is not run here.
4. **NOT READABLE BETWEEN ITEMS** if the between-item AUC is below 0.75 and none of the above applies.
5. **INCONCLUSIVE** otherwise.

## 5. Prior (assistant; not a decision rule)

| Verdict | Prior |
|---|---|
| INPUT-PROXY | 0.35 |
| UNDERPOWERED | 0.25 |
| NOT READABLE BETWEEN ITEMS | 0.2 |
| INCONCLUSIVE | 0.15 |
| PROCESS-READABLE | 0.05 |

The lens arm showed causal arrival to be reliable across one paraphrase (ρ 0.89) and to vary between items (SD 1.9 layers). Context variants with identical answers may shift it little.

## 6. Limits

- One model and one task.
- The answer-identity filter selects variants in which the context did not change the answer.
- Arrival depends on the fixed corrupt contexts.
- The probe is one oracle direction per site.
- Nothing is published.
