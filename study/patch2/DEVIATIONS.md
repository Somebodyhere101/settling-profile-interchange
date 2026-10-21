# Paper 8 patching arm 2: deviations from patch2/PREDICTIONS.md

Each entry records the date, what changed and why, and whether any patch2 outcome had been seen when the change was made.

## 2026-10-20 08:45 (+04:00): none during the run

The run used the frozen code and pairs unchanged (hashes in `PREDICTIONS.sha256`). The diagnosis of the 1.1% of (iii+) rows that differ from the splice (second-chunk batch numerics) was done after the outcomes had been seen. It changes no rule or result.

## 2026-10-20 (+04:00): pre-release audit correction to RESULTS.md

**What changed.** A dated "Correction" section was appended to RESULTS.md. It covers the circularity of reading D_gen at the patched positions, the fact that the pilot's pushes had reached the answer positions, and the lens caveat. The original text is kept above it.

**Status.** No verdict, decision rule or result value changed.
