# Paper 8, second model, stage B: replication on SmolLM2-1.7B-Instruct

**Status.** Written on 2026-10-20 after stage A's screen and before any replication data. The SHA-256 of this file and of the tree is recorded in `PREDICTIONS_B.sha256`. Changes go in `DEVIATIONS.md`.

**Run before the freeze, all on items outside the measured sets:**
- stage A, the clean pool and pairs (reported in §0);
- the S2 selftest;
- the self-report formatcheck and selftest;
- the tuned-lens fit;
- the kill selftest.

The lens arm's item list was also built before the freeze, but its selftest runs after the freeze on 3 of its measured items (§2.4). That is the same disclosure as on Qwen.

## 0. Stage A result

**Stage A.** Frozen at 13:10. It ran from 13:44 to 13:46.

- **Accuracy:** 0.492 on the 240 M items and 0.427 on the 6,610 generated prompts.
- **Pairing:** arm S found 313 eligible recipients and ran 150 (54 early, 96 late).
- **Arm S2:** all 150 recipients kept (no donor dropped for unequal answer tokens), with 217 far and 202 null donors and 359 distinct items.
- **Gaps:** the mean far gap is 1.13 layers. The SD of clean D·L over correct G items is 0.48 layers.
- **Gate: GO** (≥ 100 recipients).

**Before the freeze, on items outside the measured sets:**

- **S2 selftest** passed. The engine reproduces generation; patches are exact and causal; the all-position patch equals the splice.
- **Self-report formatcheck** gave a letter mass of 0.996–0.998, so the §2.3 bar passes. SmolLM2 has a **strong letter bias**:
  - On the digit question it answers "B" in 3 of 4 rotations (all 10 items), and mostly "C" in the fourth.
  - On the A/B question it answers "B" in both orders.
  - The rotation and order averaging removes this bias by design. The item-specific signal left after correction may be small, which would make NOT EVALUABLE more likely.
- **Self-report selftest** passed. Temporal isolation is exact: the report positions are bit-identical through block 18.
- **Tuned lens** fitted. Held-out KL is 0.25–1.4, against 8–33 for the logit lens, whose early layers are very poor in SmolLM2.
- **Kill selftest** passed.
- **Lens-arm items:** 822 (359 S2, 81 fresh M, 382 fresh G). The fresh M items are fewer because accuracy is lower and lens-training items are excluded.

## 1. What is replicated

Each arm runs the Qwen3-1.7B arm's frozen code, copied by `mk_m2.py` with only the model key, the layers (7 → 6, 14 → 12, 21 → 18, 24 → 21) and the report footer changed.

**Decision rules.** These are the arms' own, unchanged: the same contrasts, SESOI (1/L, with L = 24 here), CIs, seeds and verdict thresholds.

1. **Arm S2** (`patch2/`): whole-state interchange at prompt and answer positions, generated (as `../patch2/PREDICTIONS.md`), on the model-2 S2 pairs.
   - **INVALID** if PC1 fails at 6, 12 or 18.
   - Otherwise, a label (HELD / NOT HELD / PARTIAL) for the answer positions at 12 and 18.
   - Qwen gave NOT HELD@14 and HELD@21.
2. **Kill check** (`selfreport/`): a tuned lens fitted for SmolLM2 by the same procedure, on items outside its S2 set. **KILL** if the tuned-lens τ_gen of (iv)@18 is below 0.2.
   - Because the Qwen gap nearly vanished under the tuned lens, the tuned-lens gap and its CI are reported first, as on Qwen.
3. **Self-report** (`selfreport2/`): arm 2's design and rules, with cells (iv)@18, (iv+rep)@18, (i)@6, (ii-r)@18 and (ii)@6.
   - Verdicts: CAUSAL SELF-ACCESS / RECONSTRUCTION FROM INPUT / DIRECT STEERING ONLY / NOT EVALUABLE / INCONCLUSIVE.
   - It runs whatever the kill check gives, because arm 2's design does not depend on it. The kill result is reported next to it.
4. **Lens robustness** (`lens/`): (a) logit lens, (b) tuned lens and (c) causal arrival on the model-2 S2 items plus up to 500 fresh items. The rule is unchanged: ROBUST if ρ(a, c) ≥ 0.5 with lower bound > 0.3; LENS ARTEFACT if ρ(a, c) < 0.2 and ρ(b, c) > ρ(a, c); otherwise MIXED.

## 2. Notes fixed in advance

1. The S2 rule requires near-complete transfer for HELD, because the depth range is narrow. A shallower model may have a smaller gap in layers. The gap is reported in layers and in SESOI units.
2. "Replicates" means the same verdict label as Qwen in each arm. Arm S2's late-layer label is compared at 18 against Qwen's 21. A different label is reported as non-replication; it is not explained away.
3. The self-report formatcheck must show a letter mass ≥ 0.9 on items outside the run. Otherwise the self-report arm is reported as NOT EVALUABLE (readout fails) and is not run.
4. The lens arm's selftest prints values for 3 measured items, as on Qwen.

## 3. Prior (assistant)

| Arm | Prior |
|---|---|
| S2 | HELD@18: 0.5; PC1 passes: 0.9 |
| Kill check | PASS by rule: 0.6 |
| Self-report | RECONSTRUCTION 0.4; NOT EVALUABLE 0.3; INCONCLUSIVE 0.2; CAUSAL SELF-ACCESS 0.05; DIRECT STEERING ONLY 0.05 |
| Lens arm | the same label as Qwen: 0.6 |
