# Where a Logit-Lens Settling Profile Is Carried, and Whether Qwen3-1.7B Reports It

This repository holds the code, preregistrations and results for **"Where a Logit-Lens Settling Profile Is Carried, and Whether Qwen3-1.7B Reports It: Answer-Fixed Interchange in Two Small Language Models"** (Sam Ramdan, October 2026). It is paper 8 in a series on whether hidden-state directions track the processes they are named after.

- **Paper:** [`paper/settling-profile-interchange.pdf`](paper/settling-profile-interchange.pdf); Zenodo: [doi:10.5281/zenodo.23532061](https://doi.org/10.5281/zenodo.23532061) (this version), [doi:10.5281/zenodo.23532060](https://doi.org/10.5281/zenodo.23532060) (all versions). The source, the numbers check (`check_numbers.py`) and the abstract check are in `paper/source/`.
- **Repository:** <https://github.com/Somebodyhere101/settling-profile-interchange>. **Zenodo:** [doi:10.5281/zenodo.23532061](https://doi.org/10.5281/zenodo.23532061). The Zenodo record holds `results-per-item.zip`, which is kept out of git because of its size.
- **Earlier in the series:** [Does the Direction Track the Process? (paper 6)](https://doi.org/10.5281/zenodo.23450400).
- **Changes from the audited draft:** see `V2_CHANGES.md`.

## What the study does

The setting is Qwen3-1.7B multiplying two two-digit numbers, with its answer prefilled to "a * b = ". The variable is the **logit-lens settling profile**: the layer from which each answer digit's logit-lens prediction stays fixed.

Every arm was preregistered. Each has a frozen `PREDICTIONS*.md` with SHA-256 hashes, written before its data existed, and a `DEVIATIONS.md`. Post hoc analyses are marked as such.

| Arm (folder) | Question | Preregistered verdict |
|---|---|---|
| Pilot (`study/`, `study/code/`) | Can an oracle direction push settling depth from the reading position onward, at all positions, or at the question tokens? | UNINFORMATIVE. The positive control missed its rule by 0.003, so all findings are exploratory. Question-token pushes: 0 of 36 cells. All-position pushes "move" depth by the dose rule, but only at block 14 and in the wrong direction. |
| Patching arm (`study/patch/`) | Where is depth carried in the prompt? | INVALID: the prompt-only ceiling failed. The reading position is causally inert. |
| Patching arm 2 (`study/patch2/`) | Is depth carried at the generated answer positions? | NOT HELD at block 14; HELD at block 21 (τ 0.91). This is partly circular: D_gen is read at the patched positions, and 99% of later-digit settle layers lie above block 21. |
| Self-report 1 (`study/selfreport/`) | Does "step by step, or just knew it?" follow the patched state? | Kill check PASS, but unstable. RECONSTRUCTION FROM INPUT. |
| Self-report 2 (`study/selfreport2/`) | Does "which digit were you least sure of?" follow it? (temporal isolation, direct-steering control, input baselines) | RECONSTRUCTION FROM INPUT. The direct-steering control transfers the report (τ 0.99). |
| Lens robustness (`study/lens/`) | Is settling depth a process variable or a lens artefact? | MIXED. ρ with a lens-free causal arrival layer is 0.26. Under arrival, the S2 grouping is not detectably different from zero. |
| Second model (`study/model2/`) | SmolLM2-1.7B-Instruct replication | Location replicates (HELD at block 18). Self-report does **not** replicate: NOT EVALUABLE, because the clean reports do not follow the settling difference. Lens MIXED. |
| Input proxy (`study/inputproxy/`) | Is the readable variable an input correlate? | UNDERPOWERED. The within-item reliability of the target is 0.17. |

Each arm's folder holds its `PREDICTIONS*.md`, `PREDICTIONS*.sha256`, `DEVIATIONS.md`, `RESULTS.md`, its code, and its summaries in `res/`.

**Wording in the frozen files.**
- "The assistant" is Claude.
- "The parent session" is the coordinating Claude session, which passed on the author's requests to the session running an arm.
- **Two corrections to that wording:**
  - The lens and input-proxy preregistrations call their decision rules "the author's, as relayed". Those rules were written by the coordinating session; the author's only instruction then was to continue the paper.
  - The second-model files say the author decided the substitution. The author preferred the cached model to a download, and the coordinating session made the substitution.
  - Dated corrections are in each arm's `DEVIATIONS.md`.

## Verifying the freezes

Run `sha256sum -c PREDICTIONS.sha256` (or `PREDICTIONS_A.sha256` / `PREDICTIONS_B.sha256`) from an arm's folder. The first line of each file is a comment. These lines fail by design:

- **Withheld files** (see `DATA.md`): `code/data/items.json`, `code/data/calibration.json`, `code/data/items_d8.json`, `p8.ps1`, and the tuned lenses `selfreport/res/tuned_lens.pt` and `model2/selfreport/res/tuned_lens.pt`. Their model-2 copies fail the same way.
- **`code/run8.py` in the pilot's `PREDICTIONS.sha256`.** It was fixed after the freeze, under the logged deviation of 2026-10-19 ~16:05 (a selftest-only fix). The released file matches the new hash recorded there: `996b6f015f979e315b90580c4bc5d990126bc30bef1995a34e96004c051a9d86`.

**Shared code.** The arms import shared code from sibling folders: `code/` (paper 6's `p6lib.py`, `specs.py`, `spec_d8.py`, `cr4_parsers.py`), `patch/patchlib.py`, `patch2/patchlib2.py` and `selfreport/srlib.py`. These are unchanged copies wherever reused, and their hashes appear in the arms that use them.

**`study/model2/`** is generated by `mk_m2.py`. It copies the Qwen arms' frozen scripts, changing only constants (the model key, the layer map 7 → 6, 14 → 12, 21 → 18, 24 → 21, and the report footer), and asserts every replacement.

## Reproducing

**Environment.** Python 3.12 with `requirements.txt` (CUDA build of torch) and one GPU with 8 GB or more. The runs used an 8 GB laptop GPU.

**Setup.**

1. Rebuild the withheld data files as described in `DATA.md`. `specs.py` reads `items.json` and `calibration.json` at import.
2. Clone `https://github.com/Somebodyhere101/compute-retrieve` into `third_party/compute-retrieve`. The D3 spec imports its detector.

**Qwen3-1.7B arms, in order:**

1. `study/code/run_laptop.py`: the pilot.
2. `study/patch/run_patch.py` stages `pool`, `pair`, `selftest`, `patchP`, `patchS`, then `analyze_patch.py`.
3. `study/patch2/run_patch2.py` stages `pair2`, `selftest`, `patchS2`, then `analyze_patch2.py`.
4. `study/selfreport/run_sr.py` stages `formatcheck`, `fitlens`, `selftest`, `kill`, then `analyze_sr.py kill`, `run_sr.py sr`, and `analyze_sr.py sr`.
5. `study/selfreport2/run_sr2.py` stages `formatcheck`, `selftest`, `sr2`, then `analyze_sr2.py`.
6. `study/lens/run_lens.py` stages `items`, `selftest`, `run`, then `analyze_lens.py`.
7. `study/inputproxy/run_ip.py` stages `items` and `selftest` (or `run_pre.py`), then `run` and `analyze_ip.py` (or `run_main.py`).

**SmolLM2-1.7B arms** (`study/model2/`):

1. `python mk_m2.py` rebuilds the tree.
2. `python run_stageA.py`: pool, pair, pair2.
3. `python run_steps.py steps_pre.json pre`: selftests, format check, tuned-lens fit, lens items.
4. `python run_steps.py steps_B.json B`: S2, kill check, self-report, lens arm.

## Licence

Code and text are released under the MIT licence. The data licences are in `DATA.md`.

## AI use and author contribution

This study was carried out by an AI system. Claude (Anthropic), specifically Claude Opus 5.5 (`claude-opus-5-5`) for every session, including the audit and reviewer sessions, ran as Claude Code agent sessions on the author's computer and did the following:

- chose the design of every arm and wrote each preregistration (the frozen `PREDICTIONS.md` files record that the design choices were delegated to it);
- wrote all code;
- ran every experiment on the author's hardware;
- analysed the results;
- searched for and checked the cited literature;
- wrote the manuscript and this documentation.

Arm 2's design corrections came from an assessment of the design by ChatGPT (OpenAI), supplied by the author. The coordinating Claude Opus 5.5 session turned it into concrete design changes, together with its own corrections.

The author set the goal of a fundamental result and directed the study through open-ended questions; the study's question was proposed by the AI system in reply. Its self-report arms build on the author's earlier hypothesis linking effort to a system's ability to report its own states. The author approved each stage of the work, including running it on a laptop when the desktop computer was unavailable, passed on an external assessment of the design written by ChatGPT, preferred an already-installed model to a download, provided the computing hardware, and decided to release the study. All design choices were delegated to the AI system; decision rules that two preregistrations first attributed to the author were written by the AI system, as recorded in the deviation logs. The author did not independently check the code, analyses or technical claims.

**How the work was checked.** Automated adversarial audits, run as Claude sessions acting as independent reviewers, compared the manuscript with the released data and code. Their findings and the corrections made are listed in `V2_CHANGES.md`. Everything needed to check the work independently is in this repository and the Zenodo record.
