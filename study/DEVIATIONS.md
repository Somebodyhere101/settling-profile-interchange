# Paper 8: deviations from PREDICTIONS.md

Each entry records the date, what changed and why, and whether any paper-8 data had been seen when the change was made.


## 2026-10-19 ~16:05 (+04:00): selftest write check restricted to rows pushed at the checked layer

**What changed.** In `code/run8.py`, the selftest's `write_c` check now looks only at rows pushed at layer l. The new run8.py SHA-256 is 996b6f015f979e315b90580c4bc5d990126bc30bef1995a34e96004c051a9d86; the frozen one was 248faa36….

**Why.** The first laptop selftest failed on this check (`write_c` = 37). The check compared the reading position at layer l for every row, including rows pushed at an earlier layer. For those rows, a source-only write has legitimately reached the reading position by layer l through attention, so the check was wrong, not the engine.

The same selftest run passed every engine check exactly:
- 0 token differences;
- lp0 and KL differences of 0;
- the causality check at 0;
- the (a) write reproduced to 3e-4.

**Data seen.** None. The selftest uses three items with random vectors and computes no target outcome.

**Push, clean and analysis code.** Unchanged.

## 2026-10-19 (+04:00): the pilot runs on the laptop, not DQ

**Hardware.** DQ is offline during renovation work at the author's home, so the pilot runs on the laptop's RTX 5070 Laptop GPU (8 GB), in the `jspace` venv (torch 2.14.1+cu130, transformers 5.19). The author approved the switch.

**Batch size.** `MAX_ROWS` is 64 instead of 128. Only the batch composition changes, because rows of a batch share one prompt and each chunk has its own clean row.

**Driver.** `code/run_laptop.py` runs the same chain as `jobs_p8.txt`, in the same order, and stops at the first failure. It also keeps Windows from idle-sleeping while it runs.

**Design and rules.** Unchanged.

**Data seen.** The smoke runs (settle on 2 items, flip on 4, in `res/*_smoke_*`) were used only for timing. They are not analysed and not part of the pilot.

## 2026-10-19 ~18:15 (+04:00): interim look at the settle results; secondary reading declared

**What was looked at.** A snapshot of 92 of the 120 settle push items, analysed with the frozen `analyze8.py` under the key `interim` (`res/settle_interim_*`, `res/interim_summary.json`). No flip push data existed yet. The run continues unchanged: there is no stopping rule, and every item is run whatever the snapshot shows.

**Problem noticed.** The frozen rule 2 (§5) would call the result H-REDERIVE whenever (b) moves and (a) does not, even if (c) is null. Two features of the snapshot make that label misleading:

- (b) contains (a).
- The (b) effect at layer 14 has the same sign as the (a) effect there, and a similar size at 4 gaps.

A source-position claim needs the source-only condition to move.

**Declared before the final data and before any flip data.** The final report gives the preregistered label unchanged. It adds a secondary reading in which:

- "position matters through the source tokens" requires (c) to move (`o` or `os`);
- a (b)-only move with (c) null is read as "not a source effect".

This secondary reading does not replace the preregistered verdict.

## 2026-10-19 ~20:10 (+04:00): second interim look (all settle items, 79 of 159 flip items)

**What was looked at.** The frozen `analyze8.py` was run on snapshots under the key `interim`, at the author's request. The run continues unchanged, with no stopping rule.

**Status of anything said about the positive control from here.** The flip data have now been seen. Any reading of the positive control other than preregistered rule 1 (§5) is therefore post hoc. It will be labelled exploratory in every report.

## 2026-10-20 (+04:00): pre-release audit corrections to RESULTS.md

**What changed.** RESULTS_PILOT.md: the CI upper bound of the (a) 4-gap cell is 0.0745, which rounds to 0.074, not 0.075.

**Status.** No verdict, decision rule or result value changed; only the text of the results note was brought in line with the released summary files.
