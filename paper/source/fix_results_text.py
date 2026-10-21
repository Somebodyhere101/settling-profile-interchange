"""Applies the 2026-10-20 audit corrections to the study's RESULTS.md files (text only; no verdict or result value
changes) and logs each in the arm's DEVIATIONS.md. Every replacement is asserted. Idempotent: an entry already
logged is not logged again. python fix_results_text.py"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
S = os.path.normpath(os.path.join(HERE, "..", "paper8"))
TAG = "## 2026-10-20 (+04:00): pre-release audit corrections to RESULTS.md"

FIXES = {
    "model2": [
        ("| (a) logit lens vs (c) causal arrival | **0.41** | [0.35, 0.46] |", "| (a) logit lens vs (c) causal arrival | **0.40** | [0.34, 0.46] |"),
        ("| (b) tuned lens vs (c) causal arrival | **0.41** | [0.34, 0.47] |", "| (b) tuned lens vs (c) causal arrival | **0.41** | [0.34, 0.46] |"),
        ("MIXED, ρ(a, c) 0.41; S2 grouping survives under (c)", "MIXED, ρ(a, c) 0.40; S2 grouping differs under (c), below the SESOI"),
        ("| Lens arm | MIXED, ρ(a, c) 0.26; S2 grouping gone under (c)", "| Lens arm | MIXED, ρ(a, c) 0.26; S2 grouping not detectably different from 0 under (c)"),
    ],
    "selfreport2": [
        ("(ii)@7 moves the report (τ 0.66), and (i)@7 moves it marginally (τ 0.20, lower bound 0.0004).",
         "(ii)@7 moves the report (τ 0.66). (i)@7 does not meet the preregistered \"moves\" threshold (τ 0.1998 < 0.2, lower bound 0.0004)."),
        ("which moves the course of the computation (τ_D 0.90)", "which moves the logit-lens settling profile (τ_D 0.90)"),
        ("What does move the report is the operand content: the prefill copy at block 7, and marginally the question tokens.",
         "What does move the report is the swap of the reading-position-to-prefill region at block 7 (which also moves depth, τ_D 0.57); the question-token swap falls just below the threshold."),
    ],
    ".": [
        ("  | (a) | 4 gaps | +0.049 | [0.026, 0.075] |", "  | (a) | 4 gaps | +0.049 | [0.026, 0.074] |"),
    ],
    "lens": [
        ("does **not** separate items by causal arrival. The gap is 0.18 layers, with a CI spanning −0.6 to 0.8. It largely disappears under the tuned lens too, as in the kill check.",
         "is not detectably different from zero under causal arrival: the gap is 0.18 layers [−0.60, 0.83], so up to about 0.8 layers (roughly 74% of the logit-lens gap) remain compatible with the data. Under the tuned lens it is 0.29 [−0.14, 0.73], as in the kill check."),
    ],
}
NOTES = {
    "model2": "SmolLM2 lens values in RESULTS.md were rounded twice; corrected from model2/lens/res/lens_summary.json (ρ(a, c) 0.4049 [0.3449, 0.4639] → 0.40 [0.34, 0.46]; ρ(b, c) upper bound 0.4648 → 0.46). The wording about the S2 grouping is aligned with the CIs.",
    "selfreport2": "RESULTS.md called the (i)@7 cell a marginal move; under the frozen rule (CI excludes 0 and |τ| ≥ 0.2) τ = 0.1998 does not move the report. \"Course of the computation\" is replaced by \"logit-lens settling profile\" in line with the lens arm.",
    ".": "RESULTS_PILOT.md: the CI upper bound of the (a) 4-gap cell is 0.0745, which rounds to 0.074, not 0.075.",
    "lens": "RESULTS.md said the S2 grouping \"does not separate\" items under causal arrival; the data say it is not detectably different from zero (0.18 [−0.60, 0.83]).",
}
FILE = {".": "RESULTS_PILOT.md"}

for arm, subs in FIXES.items():
    p = os.path.join(S, arm, FILE.get(arm, "RESULTS.md"))
    s = open(p, encoding="utf-8").read()
    for old, new in subs:
        if new in s and old not in s:
            continue  # already applied
        assert s.count(old) == 1, (p, old[:60])
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8", newline="").write(s)
    dp = os.path.join(S, arm, "DEVIATIONS.md")
    d = open(dp, encoding="utf-8").read()
    if TAG not in d:
        d = d.rstrip("\n").replace("(none yet)", "").rstrip("\n") + f"\n\n{TAG}\n\n**What changed.** {NOTES[arm]}\n\n**Status.** No verdict, decision rule or result value changed; only the text of the results note was brought in line with the released summary files.\n"
        open(dp, "w", encoding="utf-8", newline="").write(d)
    print("fixed", p)
