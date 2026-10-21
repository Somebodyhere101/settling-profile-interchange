# Paper 8 self-report arm 1: results (Qwen3-1.7B, laptop RTX 5070, 2026-10-20)

**Freeze.** `PREDICTIONS.md` was frozen at 09:16:48 (+04:00), SHA-256 `4058dab4…`, recorded in `PREDICTIONS.sha256` with the code, the tuned lens and the pairs.

**Runs.**
- Kill check: 09:17–12:09. It ran slowly because the GPU was shared with other applications.
- Self-report: 12:11–12:21.

**Deviations.** `DEVIATIONS.md` holds one entry. It records the late review corrections, which went into arm 2, and the kill-check caveat below.

**Sources.** `res/kill_summary.json`, `res/sr_summary.json`, `logs/analyze_kill.log`, `logs/analyze_sr.log`.

**Sample.** All 127 arm-S2 recipients (381 items). Every row is teacher-forced to the shared answer.

## Step 1: kill check. Preregistered verdict **PASS**, with a caveat

Effects are on D_gen (digits 2 to K) and use the s-signed far − null contrast against the clean-donor gap. CIs are two-way item bootstrap, 95%.

| Lens | Cell | Effect (depth) | Gap | τ [CI] |
|---|---|---|---|---|
| logit | (iv)@21 | +0.0450 [0.0398, 0.0517] | +0.0492 (1.38 layers) | **0.91** [0.84, 1.01] |
| logit | (iv)@14 | +0.0076 | | 0.16 [0.04, 0.27] |
| logit | (iii+)@21 | +0.0491 | | 1.00 |
| tuned | (iv)@21 | +0.0114 [−0.0059, 0.0258] | +0.0062 (0.17 layers) | **1.83** [−6.8, 11.1] |
| tuned, copy-free (layers > 21) | (iv)@21 | +0.0080 [0.0015, 0.0126] | +0.0075 | 1.06 [0.36, 2.44] |
| tuned | (iii+)@21 | +0.0079 | | 1.26 |

The logit lens reproduces arm S2 exactly in teacher-forced form (0.91; ceiling 1.00).

The tuned lens gives PASS under the frozen rule (τ ≥ 0.2), but that ratio rests on a gap that nearly vanishes:

- Under the tuned lens, the far donors and the recipients differ in later-digit settling by only 0.17 layers, against 1.38 layers under the logit lens.
- Clean settling is far earlier: mean settle layer 16.1 against 24.9.
- The two lenses barely agree on which items settle late: Spearman 0.20 between their D on recipients.

What remains of the difference does transfer: the copy-free tuned-lens effect has a CI above 0, and τ is 1.06 [0.36, 2.44].

**Reading.** The answer-position patch moves whatever per-item depth difference exists under either lens. Most of the early–late depth contrast that S2 manipulates, though, is specific to the logit lens. The "process" handle is real but small once basis drift is removed.

## Step 2: self-report (A/B). Preregistered verdict **RECONSTRUCTION FROM INPUT**

**The readout.** r = logit("step by step") − logit("just knew"), in nats, the mean of the two label orders.

- The letter mass is ≥ 0.99999 in every clean row.
- The model always says "step by step": r is +10.7 (SD 1.1). Order 0 gives +12.4 and order 1 gives +8.9, so there is a letter bias, which the two orders cancel.

**Clean relation.**
- Spearman(r, D) on the 127 recipients is −0.01 [−0.19, 0.17]. It is −0.15 with the tuned lens's D, and −0.17 on all 381 items.
- **Gap in r** (clean far − null donors, s-signed): −0.65 nats [−1.04, −0.28]. Donors on the opposite side of the depth split report slightly *less* "step by step". The gap is nonzero, so the arm is evaluable.

**Per cell** (τ_r = E_r / gap_r; τ_D from the same rows):

| Cell | E_r (nats) [CI] | τ_r [CI] | τ_D(gen) | Median KL |
|---|---|---|---|---|
| **(iv)@21**, process only | +0.003 [−0.022, 0.035] | **−0.00** [−0.06, 0.04] | **0.90** | 0.05 |
| (iv)@21, KL-corrected | +0.006 [−0.019, 0.040] | −0.01 [−0.08, 0.03] | | 6 rows not bracketed |
| **(i)@7**, question tokens | **−0.450** [−0.669, −0.236] | **0.70** [0.41, 1.31] | 0.10 | 0.21 |
| (ii)@7, prefill region | +0.024 [−0.207, 0.285] | −0.04 [−0.75, 0.25] | 0.57 | 0.20 |
| (ii-r)@21, inert | −0.001 [−0.027, 0.028] | 0.00 | 0.005 | 0.03 |
| (ii-r)@7, inert | +0.000 [−0.095, 0.092] | −0.00 | 0.004 | 0.06 |

**Gaussian additive control** at the (iv)@21 positions. The r shift stays within ±0.005 nats at every strength, up to a KL of about 20.

**The verdict follows from the rules:**
- (i)@7 moves r with τ_r 0.70.
- (iv)@21 moves depth with τ_D 0.90 but leaves r untouched, with tight CIs: an upper bound of 0.04 of the gap, against the required 0.45.

## Reading

**The report reads the input, not the process.** The model's answer to "did you work that out step by step, or did you just know it?" carries no trace of a manipulation that changes how the later digits settle, with input and answer fixed. It does follow the question tokens' content, which hardly changes depth.

**Two further dissociations:**
- The prefill region at block 7 carries most of the depth (τ_D 0.57) but does not move the report.
- The donors' self-reports differ from the recipients' in a direction set by their operands.

**This is the reconstruction pattern that introspection critics predict.** Here it is shown with a counterfactual that holds the answer fixed.

**Limits:**
- The A/B readout is near ceiling: the model nearly always says "step by step".
- The question is about global strategy, not about the manipulated per-digit variable.
- Arm 2 (`../selfreport2/`) asks the per-digit question, with temporal isolation, a direct-steering control and input baselines.
- One model, one task.
