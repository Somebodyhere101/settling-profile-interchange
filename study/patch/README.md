# Paper 8 patching arm: whole-state interchange test of where settling depth lives

- **`PREDICTIONS.md`**: the design, the controls and the decision rules, frozen before any patch outcome existed. Its hash and the code hashes are in `PREDICTIONS.sha256`. Never edit it; log changes in `DEVIATIONS.md`.
- **`patchlib.py`**: `PatchModel`, which is paper 8's `P8Model` (`../code/p8lib.py`, unchanged) plus a prompt-pass patch. The patch replaces the block outputs at chosen positions with a donor's.
- **`run_patch.py`**: the stages, in order:
  1. `selftest`: the engine checks.
  2. `pool`: clean depth D for the M items and the same-format generated pool.
  3. `pair`: the preregistered pairing, written to `res/pairs.json`.
  4. `patchP` and `patchS`: the two arms.
- **`analyze_patch.py`**: the effects, the two-way item-bootstrap CIs, the controls and the verdict. It writes `res/patch_summary.json`.
- **`run_laptop_patch.py`**: a detached driver that runs `patchP`, then `patchS`, then the analysis. Logs go to `logs/`.

Run it with a Python 3.12 environment built from `requirements.txt` (CUDA build of torch), with `MAX_ROWS=64` and `P6_MEM_FRACTION=0.92`. Arm P takes about 5 minutes and arm S about 4 minutes on the laptop RTX 5070.
