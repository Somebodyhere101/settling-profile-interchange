# Paper 8 self-report arm

- **`PREDICTIONS.md`**: the design, the decision rules and the prior, frozen before any kill-check or self-report data existed. The hashes are in `PREDICTIONS.sha256`.
- **`srlib.py`**: the engine. It is S2's `PatchModel2`, with one teacher-forced pass, the tuned lens and the letter readout.
- **`run_sr.py`** stages:
  - `formatcheck` and `fitlens`: run before the freeze, on items outside the run;
  - `selftest`;
  - `kill`;
  - `sr`, which runs only after the kill check passes.
- **`analyze_sr.py`**: `kill` writes `res/kill_summary.json`; `sr` writes `res/sr_summary.json`.

Run everything with a Python 3.12 environment built from `requirements.txt` (CUDA build of torch).
