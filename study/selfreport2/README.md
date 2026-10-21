# Paper 8 self-report arm 2: per-digit report under temporally isolated patches

**Files**
- **`PREDICTIONS.md`**: frozen before any data on arm-S2 items. Hashes are in `PREDICTIONS.sha256`.
- **`run_sr2.py`**: stages `formatcheck` and `selftest` (items outside the run), then `sr2`.
- **`analyze_sr2.py`**: writes `res/sr2_summary.json`.
- **Engine:** `../selfreport/srlib.py`, unchanged.

## Planned next steps (not part of this arm)

1. **Out-of-family replication on OLMo-2-7B.** This needs a GPU with more than 8 GB of memory. In the meantime, a 1–2B second model was run on the 8 GB laptop GPU.

**Dropped for paper 8 (2026-10-20):**
- the early-by-late crossed intervention (causal-mediation fingerprints, in the style of Stolfo, Belinkov & Sachan 2023);
- the carry / no-carry addition task.

The report does not follow even the transferred late representation (τ_rep 0.00 under (iv)@21, while (iv+rep)@21 transfers it fully). An early-by-late contrast therefore has nothing in the report left to separate. Both stay possible follow-ups for a study of process representation that does not depend on self-report.
