# Paper 8 pilot: can an oracle direction push the logit-lens settling profile from other token positions?

- **`PREDICTIONS.md`**: the design, the predictions and the decision rules, written before any run. Its hash and the code hashes are in `PREDICTIONS.sha256`. Never edit it; log changes in `DEVIATIONS.md`.
- **`code/`**: the pilot's code.
  - `p8lib.py`: paper 6's engine, with pushes limited to a set of token positions for each row.
  - `run8.py`: the stages `selftest`, `clean` and `push`, each run for `settle` and for `flip`.
  - `analyze8.py`: paper 6's matched-KL `check_D` logic applied to each condition, layer and strength; it also computes readability, persistence and the verdicts.
  - `config8.json`: the published layers and gaps from paper 6, plus the strength grids.
  - `run_laptop.py`: runs the whole pilot chain in order on one GPU, stopping at the first failure.
  - `jobs_p8.txt`, `jobs_p8_7b.txt`: the queue lines originally planned for a desktop GPU (not used; the pilot ran through `run_laptop.py`).
  - `p6lib.py`, `specs.py`, `spec_d8.py`, `cr4_parsers.py`: copied unchanged from paper 6's code.
- The deployment script for the original desktop GPU (`p8.ps1`) is withheld from the release (see the top-level `DATA.md`); its hash line in `PREDICTIONS.sha256` therefore fails.

## To run

Use a Python 3.12 environment built from `requirements.txt`, with the CUDA build of torch, and one GPU with 8 GB or more:

```
cd code
python run_laptop.py
```

This runs `selftest`, `clean` and `push` for `settle` and then for `flip`, and finally `analyze8.py`, which writes `res/q3-1.7b_summary.json` and prints the cell table and the verdicts. The pilot took about 5.7 hours on an 8 GB laptop GPU.

To smoke-test a stage on a few items first, add an item count, for example `python run8.py settle q3-1.7b push 3`. With a count, results go to `res/*_smoke_*`. Then run `python analyze8.py q3-1.7b smoke`.
