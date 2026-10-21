# Paper 8 lens-robustness arm: is settling depth a process variable or a lens artefact?

**Status.** Written on 2026-10-20, before any lens-arm measurement. Only two things ran before the freeze. The first is the item list (`res/items.json`, built by `run_lens.py items`). The second is a selftest on the **last 3 measured items** (G:56*77, G:96*28, G:58*83). The selftest printed those items' per-digit (a), (b) and (c) values, so 3 of the 840 items were seen. No correlation was computed. The SHA-256 of this file, the code and the item list is in `PREDICTIONS.sha256`. Changes go in `DEVIATIONS.md`; this file is not edited.

The author asked for this arm (relayed by the parent session, 2026-10-20). The assistant made the design choices recorded here.

## 1. Question

Papers 4 and 6 (D3/D4 targets) and every paper-8 arm measure the course of a computation as **logit-lens settling depth**: the earliest layer from which the logit lens's top-1 at a digit's predicting position stays equal to the final output.

Arm 1's kill check (`../selfreport/RESULTS.md`) found two problems:
- A tuned lens settles about 9 layers earlier.
- It shrinks the far-versus-null depth gap of arm S2 from 1.38 to 0.17 layers, and agrees only weakly with the logit lens on which items settle late (ρ 0.20).

If settling depth mostly reflects the logit lens's basis mismatch rather than when the computation delivers the digit, the earlier "process" results concern a lens property. This arm compares both lenses against a **lens-free causal measure**.

## 2. Design (fixed)

**Model and engine.** Qwen3-1.7B, thinking off, bf16, on the laptop RTX 5070, in the `jspace` venv. The engine is `../selfreport/srlib.SRModel`, unchanged: one teacher-forced pass, in-batch interchange, the frozen tuned lens `../selfreport/res/tuned_lens.pt`.

**Items.** 881 in total:
- all 381 arm-S2 items;
- 500 fresh correct items: every correct pool M item (paper 4's two-digit × two-digit items) and correct pool G items drawn with seed 8.

None of the fresh items is in arm S2 or among the 1,500 arithmetic items the tuned lens was fitted on. Items without a corrupt partner (below) are dropped and counted. The item build kept **840** items: 381 S2, 180 fresh M and 279 fresh G. That is 459 fresh items instead of 500, because 41 had no partner.

**Corrupt partner.** One per item, from correct pool items outside the measured set. The partner has:
- different operands;
- the same answer-token count and digit positions;
- the same prompt token length in both prompt forms;
- a **different digit at every digit position**.

Seed 8.

**Prompt forms.**
- **orig** (primary): "What is a * b?", with the "a * b = " prefill, as in every earlier arm.
- **para** (reliability): "Calculate a * b.", with the same prefill.

**Per item and form:** one teacher-forced pass on the item's own (correct) answer.

The settle layer of each digit, at the position that predicts it, is measured with:
- **(a)** the logit lens;
- **(b)** the frozen tuned lens.

**(c) Causal arrival.** For every layer l and digit k:
- The clean item's block-l output at that digit's predicting position is copied, in the same batch, into the corrupt partner's teacher-forced run at the same position and block.
- The patched run's output at that position is read.

Three versions:
- **Stable arrival (primary):** the smallest l such that the clean digit is top-1 for every l′ ≥ l.
- **First arrival:** the smallest l at which the clean digit is top-1.
- **Stable 50% layer:** the smallest l from which (LD − LD_corrupt) / (LD_clean − LD_corrupt) ≥ 0.5 for every l′ ≥ l, where LD = logit(clean digit) − logit(corrupt digit).

No lens is involved: the measure asks from which block onward the clean residual alone, in a foreign context, delivers the digit.

**Item measures.** The mean over an item's digits, in layers (the analogue of D). The D_gen analogues (digits 2 onward) are secondary.

## 3. Outcome measures

**Primary.** On all items, orig form, with 10,000-resample item-bootstrap CIs (seed 90):
- ρ(a, c), the Spearman correlation of logit-lens settling with causal arrival;
- ρ(b, c), the same for the tuned lens.

**Secondary:**
- ρ(a, b);
- ρ with the first-arrival and 50% variants, and with the D_gen analogues;
- ρ on the S2 items, the fresh items and the fresh M items separately;
- digit-level pooled ρ;
- the means and SDs of each measure;
- **split-half reliability:** ρ between orig and para for (a), (b), (c) and the 50% variant;
- **S2 grouping:** the s-signed far − null gap (donor minus recipient) under each measure, with a two-way item bootstrap (seed 80). Does the arm-S2 grouping still separate items under (c)?

## 4. Decision rule (the author's, as relayed)

- **ROBUST PROCESS VARIABLE** if ρ(a, c) ≥ 0.5 and its CI's lower bound is > 0.3.
- **LENS ARTEFACT** if ρ(a, c) < 0.2 and ρ(b, c) > ρ(a, c).
- **MIXED** otherwise.

## 5. Prior (assistant; not a decision rule)

- ROBUST 0.25, MIXED 0.55, LENS ARTEFACT 0.20.
- Causal arrival should be early for digits retrieved as facts and late for digits that need carries. The logit lens will share some of that ordering, and the tuned lens more of it.
- Expected ρ(b, c) > ρ(a, c), probability 0.65.
- Expected reliability across the two forms: above 0.5 for (c), probability 0.6.

## 6. Limits

- One model and one task.
- The corrupt context differs in operands as well as digits, which is intended: the residual must carry the digit on its own.
- Arrival is read at the patched position only.
- The paraphrase keeps the prefill, so its reliability is across user-turn wording only.
- Nothing is published.
