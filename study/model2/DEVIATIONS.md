# Paper 8 second model: deviations from the plan and from the PREDICTIONS files

**2026-10-20: model substitution, before any run.**

- **What changed.** The second model is HuggingFaceTB/SmolLM2-1.7B-Instruct, which is cached locally. The plan named allenai/OLMo-2-0425-1B-Instruct, then Qwen2.5-1.5B-Instruct.
- **Why.**
  - Neither of the planned models is cached, and the author decided against a download.
  - SmolLM2 splits numbers into single digits, like Qwen3, so the per-digit settling and "least sure digit" design transfers unchanged.
  - OLMo-2's cl100k-style tokenizer groups digits, which would break the design.
- **Decided by:** the author, relayed by the parent session.

## 2026-10-20 (+04:00): pre-release audit corrections to RESULTS.md

**What changed.** SmolLM2 lens values in RESULTS.md were rounded twice; corrected from model2/lens/res/lens_summary.json (ρ(a, c) 0.4049 [0.3449, 0.4639] → 0.40 [0.34, 0.46]; ρ(b, c) upper bound 0.4648 → 0.46). The wording about the S2 grouping is aligned with the CIs.

**Status.** No verdict, decision rule or result value changed; only the text of the results note was brought in line with the released summary files.

## 2026-10-21 (+04:00): attribution correction

**Model substitution.** The 2026-10-20 entry above says "the author decided against a download" and "Decided by: the author". `PREDICTIONS_A.md` (frozen) says the same, and titles its gate "Gate (the author's)".

**What actually happened.** The author asked whether SmolLM would be slower and preferred the cached model to a download. The coordinating Claude session (Claude Opus 5.5) then made the substitution. The ≥ 100-recipient gate was also written by the coordinating session.

The frozen file is not edited; this entry corrects the attribution.

**Status.** No design, decision rule, verdict or result changed.
