# Paper 8: changes after the pre-release audit (2026-10-20)

Three automated reviewers audited the first draft: Claude sessions acting as independent reviewers (presentation, claims, numbers). Their findings are summarised below; all three judged it not ready to release. This file maps every blocker and should-fix item to its fix.

**What did not change:**
- No preregistered verdict, decision rule or result value.
- No frozen file (`PREDICTIONS*.md`, code listed in any `PREDICTIONS*.sha256`).

**What changed:**
- Manuscript text.
- Release packaging.
- Unhashed READMEs.
- Results notes (`RESULTS*.md`), with a logged entry in each affected arm's `DEVIATIONS.md`.

**New title.** *Where a Logit-Lens Settling Profile Is Carried, and Whether Qwen3-1.7B Reports It: Answer-Fixed Interchange in Two Small Language Models*.

It replaces *Held at the Answer, Reconstructed in the Report: Locating a Course-of-Computation Variable and Testing Self-Report Under Answer-Fixed Interchange*.

The release name changes from `held-at-the-answer` to `settling-profile-interchange`.

## Blockers

| # | Audit item | Fix |
|---|---|---|
| 1 | **Release hygiene.** DEEPER.md, p8.ps1, IP, key and user paths in READMEs. | `build_release.py` withholds `DEEPER.md`, `p8.ps1` and `items_d8.json`; DATA.md lists each under "Withheld". The study READMEs are rewritten as generic instructions (pilot README; venv paths removed from the `patch`, `patch2` and `selfreport` READMEs; the selfreport2 README no longer names the desktop). `items_d8.json` was withheld because two ARC questions contain that word as ordinary English and the release grep must return zero hits; it can be rebuilt with paper 6's `make_items_d8.py`. The grep for the seven strings named in the audit (the private network address, the key file and folder names, Windows profile and session-workspace path fragments, the user name, and the name of the paper-7 answer-key folder) over repo/ and both zips returns **0 hits**. One qualification: the first two octets of the address also appear as the leading digits of KL values (for example 100.850…) in two per-item files of `results-per-item.zip`. Those are numbers, not the address, and they cannot be changed without altering results. The full address has 0 hits, and `finalize.py` scans for the full address. repo/, both zips and MD5SUMS were rebuilt. |
| 2 | **Circularity of "held at the answer".** | A new paragraph in §3, "A caveat on circularity", states that 99% of later-digit settle layers lie above block 21 (none below 20), at the positions (iv)@21 overwrites. (iv)@21 therefore shows only that blocks 22–28 carry the block-21 residual forward without being reset by the prompt. Copy-free is defined (§3 Definitions) and stated to be uninformative at 21 (0.905 against 0.905). The abstract says "partly by construction". The finding is stated as NOT HELD at 14 (12) and HELD at 21 (18). The title no longer says "Held at the Answer". The same caveat is appended to `study/patch2/RESULTS.md` (dated Correction section). |
| 3 | **"Course of computation".** | The phrase is removed from the title, abstract, Contributions, Table 2 and Discussion. The variable is called the "logit-lens settling profile" throughout. The lens caveats (MIXED; ρ 0.26 [0.20, 0.32]; S2 grouping under arrival 0.18 [−0.60, 0.83], "not detectably different from zero") are in the abstract, §4.4 and the Discussion. Table 2's "What it moves" column became "What is swapped". |
| 4 | **Pilot.** | The §4.1 heading reads "preregistered verdict UNINFORMATIVE; everything below is exploratory". The conditions are relabelled exactly: (a) from the reading position onward, including generated positions; (b) every position from index 1, including generated positions; (c) question tokens only. The text lists all three controllable cells, states that (b) "moves" depth by the preregistered dose rule (`settle_moves.b = True`, wrong direction, block 14), and marks the secondary reading as declared after the interim look. The flip result is marked "post hoc, after the interim looks". The abstract says "Each arm was preregistered; post hoc analyses are marked" and calls the pilot exploratory and uninformative. |
| 5 | **"Held in the prefill operand tokens".** | Removed everywhere. Region (ii) is described as "the reading position through the prefill (17–29)", which includes the thinking block and the operands. It "carries part of the gap" (0.56 at 7, 0.39 at 14), below the SESOI, with the preregistered class given. Only all prompt positions together reach "moves", at 7. The INVALID patching arm is reported separately, with its arm (P) and measure (D). |
| 6 | **Self-report scope.** | The title, Contributions, the §4.3 heading and the Discussion are scoped to Qwen3-1.7B. The "0.66 to 0.70" range is dropped; each cell is reported separately (Table 2 bold marks "moves" cells; (i)@7 = 0.20 "below rule"). The text now includes the preregistered input-reconstruction test (−0.080 [−0.134, −0.036]), the 0.65 [0.48, 0.91] input share, the arm-1 ceiling (+10.7 nats) and ρ(r, D) = −0.01. The Song et al. "nothing beyond" sentence is replaced by "no report followed the state that the input does not determine; its item-to-item variation was largely, but not entirely, predictable from the input". Entropy and anti-tracking are marked post hoc. The digits 2–4 diagnosis is marked post hoc. |
| 7 | **Discussion.** | "Why pushes fail where interchange succeeds" now says the pilot pushes reached the answer positions at 21 and 24 and persisted there. The gap is therefore between a direction and a whole state, and "concrete location" is gone. The input-proxy arm is no longer used as support ("consistent with the self-report arms" is removed; §4.5 marks its numbers descriptive). |
| 8 | **Kill check.** | The preregistered primary (tuned-lens τ_gen 1.83 [−6.75, 11.11], PASS but unstable, gap CI includes 0) is reported first. Then come the earlier tuned settling (16.1 against 24.9) and the lens agreement ρ 0.20, then the copy-free secondary 1.06 [0.36, 2.44]. |
| 9 | **Misrounded numbers.** | SmolLM2 lens: ρ(a, c) 0.40 [0.34, 0.46] (was 0.41 [0.35, 0.46]); ρ(b, c) 0.41 [0.34, 0.46] (was upper 0.47); arrival reliability 0.93 (was 0.94). Pilot: flip +0.29 [0.22, 0.38], with the oracle named; (a) 4-gap CI upper 0.074. Text and Table 1 both use D_gen (0.56/0.39), and the patching-arm values are labelled arm P, measure D. The reading-position log-odds now read "at most 0.01 nats". The block-24 ceiling (0.81) and the SmolLM2 block-21 ceiling (0.76) are in Table 1. The same corrections are made in `study/model2/RESULTS.md` and `study/RESULTS_PILOT.md` and logged. |
| 10 | **Related work.** | Added, each verified: Lindsey et al. 2025, *On the Biology of a Large Language Model* (transformer-circuits.pub; the addition self-explanation result); Huang & Chang 2025, arXiv:2510.09794; Hase et al. 2023, arXiv:2301.04213; Hatua 2026, NeurIPS 2026 workshop poster (verified on neurips.cc; **not on arXiv**, which the bib entry says). OWMI's breadth is stated (8 models, about 78,000 measurements). Novelty is narrowed to on-distribution same-answer donors with fixed answer tokens. "Reported widely" became "also reported in other recent studies". Macar et al. is now cited from arXiv:2603.21396, whose author list was confirmed; the PMLR record (author-name discrepancy) is no longer cited. |
| 11 | **AI statement.** | Replaced by an unnumbered section, "AI use and author contribution". It says Claude chose the design, wrote the preregistrations and code, ran the experiments and analyses, checked the literature and wrote the paper. It contains a red **AUTHOR-ROLE-TODO**, an AI-MODEL-TODO, and a sentence that the author has not checked the technical content. "How the work was checked" names the automated adversarial audits and points to this file. The README section matches, and the README explains "the assistant" and "the parent session". |
| 12 | **Abstract.** | Rewritten: 1,895 characters, plain ASCII (`abstract_check.py`), consistent with items 2–9. |

## Should-fix items

| Audit item | Fix |
|---|---|
| Uncited "paper 4/6"; "M items" undefined | Papers 4 and 6 are cited as `ramdan2026crv2` (paper 4's test, absorbed into the CR-axis v4) and `ramdan2026battery`. M items are defined. "Paper~6's read site" is used in place of "best read site". |
| Undefined jargon | A §3 "Definitions" list covers oracle direction, calibration gap, survival, arm S vs S2, the cell@block notation, τ and τ_rep, copy-free, kill check, and arms 1 and 2. |
| Different measures for the same cell | Fixed (see blocker 9). The far gap is given with its measure and set (1.15 layers, D, S2 in-batch; 1.38 kill-check logit lens; 1.12 lens arm). |
| Unidentified "external review" | Now "reviewers of the design" with a red REVIEW-SOURCE-TODO for the author to name them; the AI statement says the audits were Claude sessions. |
| Wrong repository paths, no URL | Paths now read `study/RESULTS_PILOT.md` and `study/patch/RESULTS_PATCH.md`. REPO-URL-TODO and ZENODO-DOI-TODO are in the paper, README and CITATION.cff. |
| Leftover `\pending`, float package | Both removed. The new red `\todo` marker is used only for the author placeholders. `finalize.py` refuses while any remains. |
| Bib capitalisation, missing URLs | All titles are brace-protected; arXiv URLs added for Qwen3 and SmolLM2. |
| README reproduction steps, hash-failure notes | Reproduction now covers every arm, including the selfreport formatcheck, model2 (`mk_m2.py`, `run_stageA.py`, `run_steps.py` with `steps_pre.json` / `steps_B.json`) and inputproxy (`run_pre.py`, `run_main.py`). The compute-retrieve URL is given. Expected hash failures are listed: the withheld files, and `run8.py`, with its logged new hash. |
| Text-removal claim | Paper and DATA.md state that the result files hold only ids and numbers, and that the 40-character filter removed nothing. ARC is cited (Clark et al. 2018) with its licence. |
| Overstated lens wording | "Not detectably different from zero (CI up to 0.83 layers)" throughout. S2-only ρ (0.19, 0.03) is added. The logit-lens SD of 0.56 layers is below the SESOI. The paraphrase keeps the prefill. SmolLM2's grouping is below the SESOI, and its 50% variant is 0.05 [−0.93, 1.14]. |
| Second-model statements | The substitution reason now follows `model2/DEVIATIONS.md`: not cached, the author decided against a download, and OLMo-2 groups digits. NOT EVALUABLE gives the preregistered criterion (gap +0.004 [−0.005, 0.016]) and is reported as a non-replication. Block 21 is PARTIAL (0.72). |
| Paper-6 backtracking claim | "At four or eight gaps, settling depth and chain-of-thought necessity moved in 1 of the 8 evaluable runs; backtracking was pushed only to two gaps." The oracle's one-gap push moved none of 19. |
| Direct-steering control overstated | Stated to overwrite the readout's own block-21 state, so it rules out only a readout that cannot be moved. |
| Evaluability framed inconsistently | Both models get the preregistered criterion (Qwen gap +0.177 [0.107, 0.263]; SmolLM2 +0.004 [−0.005, 0.016]). Qwen's anti-tracking (−0.33 [−0.37, −0.29]) is stated. |
| Input-proxy within-item null | It is labelled secondary and uninterpretable (within-item reliability 0.17). Between-item reliability is 0.49, the corrupt-reference caveat is given, the variant shift is "point estimate ≤ 0.16 (CI up to 0.26)", and AUC 0.69 is attributed to the last prompt token at block 14. |
| Reproducibility section | It lists the withheld hashed inputs, the `run8.py` hash note, and the per-item results on Zenodo. |
| Macar author discrepancy | Resolved by citing the arXiv version (see blocker 10). |

## Checks run after the fixes

- **`check_numbers.py`** recomputes 55 numbers used in the abstract, introduction, results and conclusions from the released JSONs (and from `results-per-item.zip` for the per-item S2 settle layers). It checks each against the text: **55/55 pass**, on both `../paper8` and the rebuilt `repo/study`.
- **`abstract_check.py`:** 1,895 characters, ASCII.
- **PDF:** 12 pages, with no errors, undefined references or overfull boxes.
- **Release grep for the forbidden strings:** 0 hits in repo/ and in both zips (full network address; see blocker 1 for the KL-value digits).
- **`finalize.py`** refuses while AUTHOR-ROLE-TODO or any other placeholder remains.

Nothing was published, pushed or uploaded.

## Release preparation (2026-10-21)

| Item | Change |
|---|---|
| AI-MODEL-TODO | Filled in the paper and the README: Claude Opus 5.5 (`claude-opus-5-5`) for every session, including the audit and reviewer sessions. |
| REVIEW-SOURCE-TODO | Filled: arm 2's design corrections came from an assessment of the design by ChatGPT (OpenAI), supplied by the author. The coordinating Claude Opus 5.5 session turned it into concrete design changes, together with its own corrections. |
| Attribution correction | Several frozen files credit the author with decisions made by the coordinating Claude session. The frozen files are not edited. Dated corrections are in `lens/`, `inputproxy/`, `model2/`, `selfreport/` and `selfreport2/` `DEVIATIONS.md`, and the paper's AI-use section and the README state them. The corrected items:<br>• The lens and input-proxy decision rules ("the author's, as relayed") were written by the coordinating session; the author's only instruction then was to continue the paper.<br>• The second-model substitution and its ≥ 100-recipient gate: the author preferred the cached model to a download, and the coordinating session made the substitution.<br>• The self-report arm's design and rules came from the coordinating session; the author approved going ahead.<br>• The source of arm 2's corrections is the ChatGPT assessment above.<br>`model2/RESULTS.md` is corrected in place. |
| ZENODO-DOI-TODO | The paper says the per-item results are "linked from the repository README". The README and CITATION.cff say the DOI will be added on release. |
| REPO-URL-TODO | Filled with https://github.com/Somebodyhere101/settling-profile-interchange (created by the author) in the paper, the README and CITATION.cff (`repository-code`). `repo/` is a git repository with that origin. `build_release.py` keeps `repo/.git` when rebuilding and leaves it out of the source zip. |
| Quotations of the author | At the author's request, verbatim quotations of the author's chat messages were replaced by paraphrase (in `lens/` and `inputproxy/` `DEVIATIONS.md` and in this file); the meaning is unchanged. |
| AUTHOR-ROLE-TODO | Replaced in the paper and README by the author-contribution text the author approved. The earlier sentence describing the author's background was removed at his request; the new text ends with the statement that the author did not independently check the code, analyses or technical claims. |
