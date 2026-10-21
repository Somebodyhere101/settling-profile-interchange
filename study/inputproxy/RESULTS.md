# Paper 8 input-proxy arm: results (Qwen3-1.7B, laptop RTX 5070, 2026-10-20)

**Freeze.** `PREDICTIONS.md` was frozen at 14:48:36 (+04:00), SHA-256 `ca361f35…`. The run went from 14:48 to 15:33. `DEVIATIONS.md` is empty.

**Sources.** `res/ip_summary.json`, `res/ip.jsonl` and `logs/analyze.log`.

**Sample.** 149 base items with v0 and at least 2 other identical-answer variants: 1,012 item-variant rows.

## Preregistered verdict: **UNDERPOWERED**

The rule checks the within-item reliability of the target first. The Pearson correlation of within-item-centred arrival layers, between corrupt run A and corrupt run B, is **0.17** (Spearman 0.23). That is well below the 0.5 gate, so the arm cannot say whether the probe follows the target within items. Per the preregistration, this is reported as UNDERPOWERED and not as a null.

**Why the target cannot be tested within items.**

- **Variance decomposition.** The commitment layer averages 21.1 layers. Its total variance is 4.58 layers². The variance between items in v0 is 4.40. The mean variance within an item across its context variants is only 0.36, about 8% of the total.
- **Variant shifts.** Each variant shifts the commitment layer relative to v0 by at most 0.16 layers:

| Variant | Shift (layers) | 95% CI |
|---|---|---|
| v1, few-shot prefix | −0.16 | [−0.30, −0.03] |
| v2, filler sentence | −0.15 | [−0.25, −0.07] |
| v3, format change | +0.08 | [−0.10, 0.26] |
| v4, irrelevant sum | −0.04 | [−0.23, 0.14] |
| v5, system prompt | −0.16 | [−0.24, −0.08] |
| v6, paraphrase | +0.05 | [−0.04, 0.17] |

- **Conclusion.** When the answer stays the same, context barely moves when the digit becomes causally available. The variable is almost entirely a property of the operand pair.

**The target is also only moderately reliable between items.** In v0, arrival with corrupt run A against corrupt run B gives r = 0.49. That is lower than the lens arm's paraphrase reliability (0.89), which used a partner-specific corrupt context and no norm rescaling.

## Secondary results (not part of the verdict)

**The oracle does not read the commitment layer even between items, at the primary site:**

| Site | Between-item AUC | Held-out Spearman | ρ_w [95% CI] |
|---|---|---|---|
| Reading position, block 21 (primary) | **0.51** | 0.07 | −0.04 [−0.10, 0.01] |
| Reading position, block 14 | 0.62 | | |
| Last prompt token, block 21 | 0.64 | | −0.11 [−0.18, −0.03] |
| Last prompt token, block 14 | 0.50 | | |

None reaches the 0.75 needed for an INPUT-PROXY call.

**With logit-lens settling as the target** (the measure probes read in paper 6):
- Between-item AUC is 0.61–0.69 and held-out Spearman 0.20–0.41. Block 14 at the last prompt token is best (0.69, ρ 0.41).
- ρ_w is about 0 at every site (−0.06 to +0.03, all CIs straddling 0).
- ρ(settle, arrival) across all rows is 0.33, as in the lens arm.

## Reading

- **No within-item test is possible.** The preregistered question (does the probe follow the process when the input is held fixed and only the context changes?) cannot be answered here. Contexts with identical answers do not move the causal commitment layer enough, or reliably enough.
- **The variable is fixed by the operands.** That is informative in its own right. In this task the course-of-computation variable, measured causally, is almost wholly determined by the operands: 92% of its variance lies between items, and context shifts it by 0.16 layers or less.
- **What this means for the other arms.** A probe reading such a variable at the prompt and an input proxy are hard to tell apart, because the variable is itself nearly a function of the input. That is consistent with the self-report arms, where the report followed operand content.
- **What a real test would need.** Context manipulations that change the process without changing the answer more strongly than these.
