# Paper 8 self-report arm: deviations from PREDICTIONS.md

Each entry records the date, what changed and why, and whether any run data had been seen when the change was made.


## 2026-10-20 ~12:20 (+04:00): review corrections arrive after the freeze; this arm stays as frozen

**Timing.** Design corrections from an external review (relayed by the parent session; the author wants them in) arrived after this arm's freeze at 09:16 and the kill check. The self-report run had started at 12:11, and no self-report output had been read.

**What this arm keeps.** It runs exactly as frozen.

- Its follow-up positions are never patched. The report turn reads the patched answer states only through attention, which already meets the review's temporal-isolation point.
- All arm-S2 pairs are analysed, and every row is teacher-forced to the shared answer.

**What moves to a new arm.** The remaining corrections are implemented as a separately preregistered arm in `../selfreport2/`, frozen before any of its data exist:

- the direct-steering control that also patches the report positions;
- the per-digit "least sure" question as the primary target;
- the input-only and answer-only baselines;
- the related-work positioning.

That arm repeats the A/B question as a secondary replication only, because the A/B data of this arm will have been seen by then.

**Note on the kill check.** The preregistered rule gave PASS (tuned-lens τ_gen 1.83). However, the tuned-lens far-vs-null depth gap is only +0.006 of depth (about 0.17 layers, against 1.38 for the logit lens), so that ratio is unstable: CI −6.8 to 11.1.

- The copy-free tuned-lens transfer has a CI that excludes 0: effect +0.0080 [0.0015, 0.0126], τ 1.06 [0.36, 2.44].
- The step went ahead as preregistered, and this caveat is reported with the verdict.

## 2026-10-21 (+04:00): attribution correction

**Who designed the arm.** `PREDICTIONS.md` (frozen) says "The author asked for this arm and its two steps". The arm's design, its two steps (kill check, self-report) and their decision rules were specified by the coordinating Claude session (Claude Opus 5.5); the author approved going ahead.

**The review source.** The 2026-10-20 entry above calls the later design corrections "an external review … the author wants them in". They came from an assessment of the design written by ChatGPT (OpenAI), which the author pasted into the coordinating session. That session turned the assessment, together with its own corrections, into the changes.

**Status.** No design, decision rule, verdict or result changed.
