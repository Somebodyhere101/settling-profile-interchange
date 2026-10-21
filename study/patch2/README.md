# Paper 8 patching arm 2: is settling depth held at the answer positions?

This is the follow-up to `../patch/`, whose preregistered verdict was INVALID because no prompt-only patch reaches the depth gap carried by answer digits 2 to 4. This arm patches the generated answer positions too, in arm S (same product, same answer tokens).

- **`PREDICTIONS.md`**: the design, PC1 and the decision rules, frozen before any outcome existed. Its hash and the code hashes are in `PREDICTIONS.sha256`. Never edit it; log changes in `DEVIATIONS.md`.
- **`patchlib2.py`**: `PatchModel2`, which extends `../patch/patchlib.PatchModel` (unchanged) with an **in-batch** interchange at prompt and answer positions. Donor rows are teacher-forced on the shared answer.
- **`run_patch2.py`**: the stages:
  1. `selftest`: the engine checks;
  2. `pair2`: the token-aligned arm-S pairs, written to `res/pairs2.json`;
  3. `patchS2`: the run.
- **`analyze_patch2.py`**: the effects, the two-way item-bootstrap CIs, PC1 and the verdict. It writes `res/patch2_summary.json`.
- **`run_laptop_patch2.py`**: a detached driver that runs `patchS2` and then the analysis. Logs go to `logs/`.
- **`RESULTS.md`**: the outcome.

Run it with a Python 3.12 environment built from `requirements.txt` (CUDA build of torch), with `MAX_ROWS=64` and `P6_MEM_FRACTION=0.92`. It takes about 8 minutes on the laptop RTX 5070.
