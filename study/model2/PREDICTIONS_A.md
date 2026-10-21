# Paper 8, second model, stage A: accuracy and pairing screen (SmolLM2-1.7B-Instruct)

**Status.** Written on 2026-10-20, before any SmolLM2 run. Its SHA-256 and the hashes of the model-2 tree are recorded in `PREDICTIONS_A.sha256`. Stage B (the replication) gets its own frozen `PREDICTIONS_B.md` before any replication data exist. Changes go in `DEVIATIONS.md`.

**Model.** HuggingFaceTB/SmolLM2-1.7B-Instruct: 24 blocks, bf16, its own chat template (default system prompt, no thinking block). It runs on the laptop RTX 5070 in the `jspace` venv.

**Why this model.** It is a substitution (see `DEVIATIONS.md`):
- The plan named OLMo-2-0425-1B-Instruct, then Qwen2.5-1.5B-Instruct. Neither is cached, and the author decided against a download.
- SmolLM2's tokenizer splits numbers into single digits, as Qwen3's does, so the per-digit design transfers unchanged.
- OLMo-2's tokenizer groups digits, which would break the design.

**Code.** `mk_m2.py` copies paper 8's frozen scripts into this tree, changing only constants (asserted replacements):
- the model key;
- the layers, mapped by relative depth: 7 → 6, 14 → 12, 21 → 18, 24 → 21;
- the report-turn footer (no thinking block).

## Screen (fixed)

1. **`patch/run_patch.py pool`.** Greedy clean outcomes on paper 4's 240 M items and on every a * b with 11 ≤ a, b ≤ 99 and a four-digit product (6,610 prompts), in the patching arm's format. The outcomes are the answer, the per-digit logit-lens settle layers, D, D_gen and the answer tokens. Batches of 64.
2. **`patch/run_patch.py pair`.** The patching arm's arm-S pairing, unchanged (seed 8):
   - same product;
   - far donors at |ΔD·L| ≥ 1.0;
   - null donors at |ΔD·L| ≤ 0.125;
   - at most 150 recipients and 2 per product.
3. **`patch2/run_patch2.py pair2`.** S2's filter: donors must produce exactly the recipient's answer tokens, and each recipient keeps at least one far and at least one null donor.

**Reported:**
- accuracy on the M items and on the generated pool;
- the number of eligible recipients;
- S2 recipients (early and late) and the mean far gap in layers.

**Gate (the author's).**
- **GO** if arm S2 has **at least 100 recipients**.
- **STOP** otherwise. Stopping means reporting and not switching models.

M-item accuracy is reported but does not gate, because the replication uses the generated pool.

**Prior (assistant).**
- M accuracy 0.4–0.8.
- GO with probability 0.6. Fewer correct items and a shallower model make same-product depth spreads of a full layer less common.
