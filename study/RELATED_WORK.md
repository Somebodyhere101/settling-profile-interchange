# Paper 8 — related work (verified)

Checked 2026-10-20. Every arXiv entry was checked against the arXiv export API (title, authors, first-version date, primary category, comments), with spot checks on arxiv.org/abs pages. Item 11 was also checked against the PMLR v306 index and its abstract page. The Lindsey (2025) page was checked on transformer-circuits.pub and the logit-lens post on LessWrong. All summaries are paraphrased from the abstracts. BibTeX fields copy the source pages: the arXiv record for `@misc` entries, PMLR's own BibTeX for item 11, and the page's suggested citation for Lindsey.

Paper 8 in brief (for the positioning lines): Qwen3-1.7B, multi-digit multiplication with a prefilled `a * b = `. (i) Logit-lens settling depth of the answer digits is a "course of computation" variable. Patching block-21 states at the generated answer positions transfers it between same-answer items, with input and answer held fixed. (ii) The reading position where linear probes decode it is causally inert. (iii) Self-reports ("step by step or just know it?", "least-sure digit?") follow operand content, not the patched process state. Writing donor states into the report turn transfers the report fully. (iv) Logit-lens settling correlates only weakly (ρ 0.26) with a lens-free causal arrival layer.

---

## Part A — items supplied by another session (1–12)

### 1. Ferrara — OWMI — arXiv 2608.20569
**Status: VERIFIED**, with one small numerical difference.
- Title: *Open-Weight Masked Introspection: Measuring What Language Models Can Report About Their Own Computation*
- Author: Emilio Ferrara (sole author). v1 2026-08-20. cs.AI.
- Claim check: the claim "8 open models" is correct (eight open-weight models from seven families). The claim "untrained reports at chance" is correct (AUROC ≈ 0.5007, with an equivalence bound under 0.15 pp). For the claim "probes 75–96%", the abstract says **75% to 95.8%**, so 96 is a rounding. The abstract adds two points not in the claim: a fine-tuned reporter reaches near-perfect recovery, and in one model the confidence, not the yes/no answer, separates the conditions (AUROC 0.647).

**Summary.** OWMI intervenes on residual-stream sites, attention heads and SAE features, then asks the model whether its computation was altered. The answers are compared against sham runs, impact-matched random perturbations and a text-only observer. Across more than 78k measurements, no model's verbal report beats chance, yet linear probes and fine-tuned reporters recover the intervention from the same activations. The authors place the failure in the path from internal state to verbal report.

```bibtex
@misc{ferrara2026owmi,
  title         = {Open-Weight Masked Introspection: Measuring What Language Models Can Report About Their Own Computation},
  author        = {Ferrara, Emilio},
  year          = {2026},
  eprint        = {2608.20569},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2608.20569}
}
```
**vs Paper 8:** Both papers find the same split: the information is linearly readable but the untrained report does not track it. OWMI uses artificial, off-distribution interventions against sham controls. Paper 8 transfers a natural process variable between same-answer items with input and answer fixed. Paper 8 also shows that the report pathway itself can be written (donor states at the report turn transfer the report). So the failure lies in *what the report reads*, not in an inability to change the report.

### 2. Song, Lederman, Hu & Mahowald — arXiv 2508.14802
**Status: VERIFIED.**
- Title: *Privileged Self-Access Matters for Introspection in AI*
- Authors: Siyuan Song, Harvey Lederman, Jennifer Hu, Kyle Mahowald. v1 2025-08-20. cs.AI.

**Summary.** The authors reject a "lightweight" definition of AI introspection and propose a thicker one. On their definition, introspection is a process that yields information about internal states more reliably than any process of equal or lower cost available to a third party. In experiments where LLMs reason about their own sampling temperature, models look introspective under the lightweight definition but fail the stricter one.

```bibtex
@misc{song2025privileged,
  title         = {Privileged Self-Access Matters for Introspection in AI},
  author        = {Song, Siyuan and Lederman, Harvey and Hu, Jennifer and Mahowald, Kyle},
  year          = {2025},
  eprint        = {2508.14802},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2508.14802}
}
```
**vs Paper 8:** Paper 8 puts their criterion into practice. Holding input fixed while varying the internal process state removes the third-party route. Reports that follow operand content are exactly what a cheap third-party reader of the input could produce.

### 3. Aoki et al. — In-Context Neurofeedback — arXiv 2609.00904
**Status: VERIFIED.** The full title is longer than the one given.
- Title: *In-Context Neurofeedback: Can LLMs Control Their Internal Representations through Privileged Access?*
- Authors: Koshiro Aoki, Ryota Takatsuki, Gouki Minegishi, Yusuke Haruki, Daisuke Kawahara. v1 2026-09-01. cs.AI.

**Summary.** A prior neurofeedback study claimed that LLMs can control their internal representations. Its control targets, however, could be inferred from the prompt. This paper redesigns the paradigm so the target is privileged, closer to human neurofeedback studies. Under that design the models show no reliable control, which suggests the earlier result may rest on superficial mechanisms.

```bibtex
@misc{aoki2026neurofeedback,
  title         = {In-Context Neurofeedback: Can LLMs Control Their Internal Representations through Privileged Access?},
  author        = {Aoki, Koshiro and Takatsuki, Ryota and Minegishi, Gouki and Haruki, Yusuke and Kawahara, Daisuke},
  year          = {2026},
  eprint        = {2609.00904},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2609.00904}
}
```
**vs Paper 8:** Both papers require a target that cannot be inferred from the prompt. Aoki et al. test *control* of such a target and Paper 8 tests *report*. Both find null access once prompt cues are removed.

### 4. Xun — Lagged Coupling — arXiv 2609.01048
**Status: VERIFIED.**
- Title: *Lagged Coupling: Internal Representations Become Readable Before They Become Causal*
- Author: Xining Xun (sole author). v1 2026-09-01. cs.CL. The comments field says it is a pre-registered study on the full Pythia suite with an OLMo-2 replication.

**Summary.** Across Pythia checkpoints (160M–12B) and four task families, a probe reads the target variable almost perfectly from early training onward. Steering along the probe direction, however, is null-equivalent in 43 of 48 model-checkpoint cells. The paper separates internal readability, behavioural readability and causal efficacy, and reports that readability consistently comes before causal use. Both pre-registered onset hypotheses came out INDETERMINATE.

```bibtex
@misc{xun2026lagged,
  title         = {Lagged Coupling: Internal Representations Become Readable Before They Become Causal},
  author        = {Xun, Xining},
  year          = {2026},
  eprint        = {2609.01048},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2609.01048}
}
```
**vs Paper 8:** Xun's "readable but not causal" probe direction parallels Paper 8's finding that the reading position is causally inert. The papers differ in two ways. Paper 8 finds the causal carrier elsewhere (answer positions, block 21) by whole-state patching rather than steering along a direction. Paper 8 also studies one trained model, not a developmental trajectory.

### 5. Galeone et al. — Perfect Detection, Failed Control — arXiv 2606.24952
**Status: VERIFIED.**
- Title: *Perfect Detection, Failed Control: The Geometry of Knowing vs. Steering in Language Models*
- Authors: Cosimo Galeone, Anna Ettorre, Minsu Park, Giuseppe Ettorre, Daniele Ligorio. v1 2026-06-23. cs.CL.

**Summary.** The paper measures the angle between the direction that best detects a behaviour and the direction that best causes it. Hallucination of fake entities is perfectly linearly detectable in Gemma-2-2B-it, yet the detection direction sits at cos ≈ 0.12 to the refusal-producing direction. The gap holds across four models (1B–9B) and is unchanged by instruction tuning. The cosine does not predict steerability, which the authors argue is a functional property rather than one readable from a static angle.

```bibtex
@misc{galeone2026perfect,
  title         = {Perfect Detection, Failed Control: The Geometry of Knowing vs. Steering in Language Models},
  author        = {Galeone, Cosimo and Ettorre, Anna and Park, Minsu and Ettorre, Giuseppe and Ligorio, Daniele},
  year          = {2026},
  eprint        = {2606.24952},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2606.24952}
}
```
**vs Paper 8:** Both papers find a read/push split. Galeone et al. measure it as an angle between directions at a site. Paper 8 measures it as a difference in *position* (the read site is not the causal site) and uses whole-state transfer rather than direction steering.

### 6. Ferreira da Silva & Heimersheim — feature-specific error correction — arXiv 2606.24964
**Status: VERIFIED.**
- Title: *Evidence for feature-specific error correction in LLMs*
- Authors: Francisco Ferreira da Silva, Stefan Heimersheim. v1 2026-06-23. cs.LG.

**Summary.** The authors perturb residual-stream activations and find plateaus of robustness to small perturbations. Activations are less robust along candidate feature directions (contrastive, MELBO, SAE-decoder) than along mixtures of two such directions. Modelling the perturbation response with an Lp-norm gives p > 2 for feature directions and p ≈ 2 for random and PCA controls, which the authors read as feature-specific error correction. The result replicates across six models, **including Qwen3-1.7B**.

```bibtex
@misc{ferreiradasilva2026error,
  title         = {Evidence for feature-specific error correction in LLMs},
  author        = {Ferreira da Silva, Francisco and Heimersheim, Stefan},
  year          = {2026},
  eprint        = {2606.24964},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2606.24964}
}
```
**vs Paper 8:** This uses the same model. Their activation plateaus give an alternative reading of an "inert" site: small or mixed pushes can be absorbed by error correction. Paper 8 should state the push magnitude at the reading position relative to the plateau scale before calling that position causally unused.

### 7. Hahami et al. — arXiv 2512.12411
**Status: VERIFIED.** The description "injected-thought detection; routing and recovery" matches.
- Title: *Detecting the Disturbance: A Nuanced View of Introspective Abilities in LLMs*
- Authors: Ely Hahami, Ishaan Sinha, Lavik Jain, Josh Kaplan, Jon Hahami. v1 2025-12-13 (v2 2026-03-01). cs.AI.

**Summary.** On Llama-3.1-8B-Instruct, apparent binary detection of steering injections is fully explained by global logit shifts that bias the model toward "yes". On differential tasks the model still shows partial introspection: it localizes which of 10 sentences was injected (up to 88%) and compares injection strengths (83%). Both abilities hold only for early-layer injections. The authors explain the layer dependence by attention-based routing and recovery dynamics in the residual stream.

```bibtex
@misc{hahami2025detecting,
  title         = {Detecting the Disturbance: A Nuanced View of Introspective Abilities in LLMs},
  author        = {Hahami, Ely and Sinha, Ishaan and Jain, Lavik and Kaplan, Josh and Hahami, Jon},
  year          = {2025},
  eprint        = {2512.12411},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2512.12411}
}
```
**vs Paper 8:** Both papers find that apparent self-report signal is driven by a cheaper factor (global yes-bias for Hahami et al., operand content for Paper 8). Hahami et al. report positive differential access for early-layer injections. Paper 8 tests a mid-depth (block 21), on-distribution process state and finds no access.

### 8. Singh, Linzen & Ravfogel — Can LLMs Introspect? A Reality Check — arXiv 2605.26242
**Status: VERIFIED.**
- Authors: Shashwat Singh, Tal Linzen, Shauli Ravfogel. v1 2026-05-25 (v2 2026-08-21). cs.AI. The comments field says it was accepted at COLM 2026.

**Summary.** The authors propose two conditions for showing introspection: privileged access (the task is not solvable from input cues) and second-order computation (designs where first-order and meta-representational accounts predict different results). Re-examining two paradigms, they find that input-only classifiers match models' "self-predictions". They also find that models cannot tell internal tampering apart from input manipulation, which fits generic anomaly detection. They conclude that current evidence does not establish metacognitive monitoring.

```bibtex
@misc{singh2026reality,
  title         = {Can LLMs Introspect? A Reality Check},
  author        = {Singh, Shashwat and Linzen, Tal and Ravfogel, Shauli},
  year          = {2026},
  eprint        = {2605.26242},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2605.26242}
}
```
**vs Paper 8:** Paper 8's same-input, patched-state design is a test where the predictions diverge in the sense of Singh et al. A process-reading account predicts that reports follow the donor state, while an input-cue account predicts that they follow operands. The result favours the input-cue account. Paper 8 also uses natural on-distribution states rather than tampering.

### 9. Li, Guo, Huang, Steinhardt & Andreas — arXiv 2511.08579
**Status: VERIFIED.**
- Title: *Training Language Models to Explain Their Own Computations*
- Authors: Belinda Z. Li, Zifan Carl Guo, Vincent Huang, Jacob Steinhardt, Jacob Andreas. v1 2025-11-11 (v3 2026-02-09). cs.CL.

**Summary.** The authors fine-tune LMs to describe (1) what features encode, (2) the causal structure of internal activations and (3) input-token influence, using existing interpretability methods as ground truth. With tens of thousands of examples, explainers generalize to new queries. A model explains its *own* computations better than a different, even stronger, model does, which the authors attribute partly to privileged access.

```bibtex
@misc{li2025training,
  title         = {Training Language Models to Explain Their Own Computations},
  author        = {Li, Belinda Z. and Guo, Zifan Carl and Huang, Vincent and Steinhardt, Jacob and Andreas, Jacob},
  year          = {2025},
  eprint        = {2511.08579},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2511.08579}
}
```
**vs Paper 8:** Paper 8 measures the *untrained* baseline that this line of work starts from. Its null for settling depth is the natural target for a trained-explainer follow-up, and the patching design would supply a ground truth that cannot be read from the input.

### 10. Guo, Ruis, Andreas & Li — arXiv 2606.32038
**Status: VERIFIED.**
- Title: *Introspective Coupling: Self-Explanation Training Tracks Behavioral Change Despite Fixed Supervision*
- Authors: Zifan Carl Guo, Laura Ruis, Jacob Andreas, Belinda Z. Li. v1 2026-06-30. cs.CL.

**Summary.** LMs are trained to explain which input features influenced their behaviour, with counterfactual behaviour as supervision. Explanations trained on *fixed* targets, taken from earlier checkpoints or from similar models, often end up more faithful to the model's *current* behaviour than to the targets. The authors call this "introspective coupling". It persists when behaviour shifts under concurrent post-training (e.g., sycophancy, refusal) and is robust to label noise.

```bibtex
@misc{guo2026introspective,
  title         = {Introspective Coupling: Self-Explanation Training Tracks Behavioral Change Despite Fixed Supervision},
  author        = {Guo, Zifan Carl and Ruis, Laura and Andreas, Jacob and Li, Belinda Z.},
  year          = {2026},
  eprint        = {2606.32038},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2606.32038}
}
```
**vs Paper 8:** Guo et al. study explanations of *input-feature influence*, which is close to what Paper 8's untrained reports already follow (operand content). Paper 8's target is an input-independent process variable, which is exactly the case their setup does not test.

### 11. Macar et al. — Mechanisms of Introspective Awareness — ICML 2026, PMLR 306
**Status: VERIFIED**, with an **author-name mismatch between the venues**.
- PMLR: *Proceedings of the 43rd International Conference on Machine Learning*, PMLR 306:85403–85429, published 2026-09-29. Authors on PMLR: Uzay Macar, Li Yang, **Zifan Wang**, Peter Wallich, Emmanuel Ameisen, Jack Lindsey.
- arXiv preprint 2603.21396 (v1 2026-03-22, v5 2026-06-10, cs.LG) lists **Atticus Wang** as the third author. Everything else matches. Check which name is correct before citing.
- The two abstracts differ slightly: arXiv says ablating refusal improves detection by +53%, while the PMLR abstract says ~50%. Both say +75% for the trained vector.

**Summary.** In open-weight models, detection of injected steering vectors is behaviourally robust: moderate true-positive rates with 0% false positives. The ability emerges from post-training: DPO elicits it but supervised fine-tuning does not. Detection is not a single linear confound with "yes" directions. It runs through a two-stage MLP circuit in which early "evidence carrier" features suppress downstream "gate" features that implement a default "no". Concept identification uses largely separate later-layer mechanisms. The ability is under-elicited: refusal ablation and a trained bias vector raise detection substantially.

```bibtex
@InProceedings{pmlr-v306-macar26a,
  title     = {Mechanisms of Introspective Awareness},
  author    = {Macar, Uzay and Yang, Li and Wang, Zifan and Wallich, Peter and Ameisen, Emmanuel and Lindsey, Jack},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  pages     = {85403--85429},
  year      = {2026},
  editor    = {Zhang, Tong and Dudik, Miroslav and Jaggi, Martin and Agarwal, Alekh and Li, Sharon and Schuurmans, Dale and Zhu, Jerry and Berkenkamp, Felix and Dong, Hanze and Bietti, Alberto},
  volume    = {306},
  series    = {Proceedings of Machine Learning Research},
  month     = {06--11 Jul},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v306/macar26a.html}
}
```
**vs Paper 8:** Macar et al. explain detection of *anomalous* injected directions through an evidence-carrier/gate circuit. Paper 8's donor states are on-distribution states from same-answer items, so there is no anomaly for such a circuit to flag. Paper 8's null is therefore consistent with access that is limited to anomaly detection.

### 12. Lederman & Mahowald — arXiv 2603.05414
**Status: VERIFIED.**
- Title: *Emergent Introspection in AI is Content-Agnostic*
- Authors: Harvey Lederman, Kyle Mahowald. v1 2026-03-05 (v2 2026-04-07). cs.AI. The v2 comments field says it "supersedes the earlier posted preprint". Cite v2.

**Summary.** The authors replicate Lindsey's thought-injection paradigm extensively in large open models. Models can detect *that* something was injected without reliably identifying *what*. They confabulate frequent, concrete concepts (e.g., "apple"), and they need fewer tokens to detect an injection than to name it. The authors argue that a content-agnostic mechanism fits leading philosophical and psychological accounts.

```bibtex
@misc{lederman2026contentagnostic,
  title         = {Emergent Introspection in AI is Content-Agnostic},
  author        = {Lederman, Harvey and Mahowald, Kyle},
  year          = {2026},
  eprint        = {2603.05414},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2603.05414}
}
```
**vs Paper 8:** Both papers find no content-specific access. Paper 8 extends this from injected concepts to a graded process variable, and in Paper 8's design there is no anomaly signal for content-agnostic detection to use.

---

## Part B — standard references

### B1. Meng et al. 2022 — ROME / causal tracing — arXiv 2202.05262
**VERIFIED.** *Locating and Editing Factual Associations in GPT*. Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov. v1 2022-02-10. cs.CL. NeurIPS 2022 (per the comments field).
**Summary.** Causal tracing restores clean hidden states into a noise-corrupted run and finds that mid-layer MLPs at subject tokens mediate factual recall. Rank-One Model Editing (ROME) then edits those weights to change specific facts while keeping specificity and generalization.
```bibtex
@misc{meng2022locating,
  title         = {Locating and Editing Factual Associations in GPT},
  author        = {Meng, Kevin and Bau, David and Andonian, Alex and Belinkov, Yonatan},
  year          = {2022},
  eprint        = {2202.05262},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2202.05262}
}
```
**vs Paper 8:** Paper 8 uses clean-to-clean patching between same-answer items, not corruption and restoration. The patched quantity is a *process* variable (settling depth), with the output held fixed, rather than the output itself.

### B2. Geva et al. 2023 — dissecting factual recall — arXiv 2304.14767
**VERIFIED.** *Dissecting Recall of Factual Associations in Auto-Regressive Language Models*. Mor Geva, Jasmijn Bastings, Katja Filippova, Amir Globerson. v1 2023-04-28. cs.CL. EMNLP 2023.
**Summary.** Attention-edge knockouts identify two critical information flows to the prediction: one from relation positions and one from subject positions. The paper then describes a three-step mechanism: MLP-driven subject enrichment, propagation of relation information, and attribute extraction by attention heads.
```bibtex
@misc{geva2023dissecting,
  title         = {Dissecting Recall of Factual Associations in Auto-Regressive Language Models},
  author        = {Geva, Mor and Bastings, Jasmijn and Filippova, Katja and Globerson, Amir},
  year          = {2023},
  eprint        = {2304.14767},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2304.14767}
}
```
**vs Paper 8:** Geva et al. separate positions by where information flows. Paper 8 makes a similar positional point: the position where a variable is decodable is not the position where it is causally used.

### B3. Stolfo, Belinkov & Sachan 2023 — arithmetic via causal mediation — arXiv 2305.15054
**VERIFIED.** *A Mechanistic Interpretation of Arithmetic Reasoning in Language Models using Causal Mediation Analysis*. Alessandro Stolfo, Yonatan Belinkov, Mrinmaya Sachan. v1 2023-05-24. cs.CL. EMNLP 2023.
**Summary.** Causal mediation on arithmetic queries shows that attention moves operand and operator information from earlier positions to the final token in early layers. A set of MLPs then writes result-related information into the residual stream. The pattern is compared against number-retrieval and factual queries.
```bibtex
@misc{stolfo2023mechanistic,
  title         = {A Mechanistic Interpretation of Arithmetic Reasoning in Language Models using Causal Mediation Analysis},
  author        = {Stolfo, Alessandro and Belinkov, Yonatan and Sachan, Mrinmaya},
  year          = {2023},
  eprint        = {2305.15054},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2305.15054}
}
```
**vs Paper 8:** This is the closest arithmetic-mechanism precedent. Stolfo et al. localize where the *result* is computed. Paper 8 holds the result fixed and asks whether *how* it was reached (settling depth) is a transferable state, then whether the model can report it.

### B4. Belrose et al. 2023 — tuned lens — arXiv 2303.08112
**VERIFIED.** *Eliciting Latent Predictions from Transformers with the Tuned Lens*. Nora Belrose, Igor Ostrovsky, Lev McKinney, Zach Furman, Logan Smith, Danny Halawi, Stella Biderman, Jacob Steinhardt. v1 2023-03-14 (v6 2025-11-11). cs.LG.
**Summary.** A per-layer affine probe ("translator") decodes each hidden state into a vocabulary distribution. The tuned lens is more predictive, reliable and unbiased than the logit lens on models up to 20B parameters. Causal tests suggest it uses features similar to the model's own.
```bibtex
@misc{belrose2023eliciting,
  title         = {Eliciting Latent Predictions from Transformers with the Tuned Lens},
  author        = {Belrose, Nora and Ostrovsky, Igor and McKinney, Lev and Furman, Zach and Smith, Logan and Halawi, Danny and Biderman, Stella and Steinhardt, Jacob},
  year          = {2023},
  eprint        = {2303.08112},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2303.08112}
}
```
**vs Paper 8:** The tuned lens is the obvious robustness check for Paper 8's logit-lens settling measure. The paper's documented logit-lens brittleness fits Paper 8's weak ρ 0.26 against the lens-free causal arrival layer.

### B5. nostalgebraist 2020 — logit lens (LessWrong)
**VERIFIED.** *interpreting GPT: the logit lens*, by nostalgebraist, posted 2020-08-31 on LessWrong (not on arXiv).
**Summary.** The post applies GPT-2's final unembedding to intermediate residual-stream states. The decoded distributions are already sensible in middle layers and converge gradually toward the final prediction. This introduced the "logit lens" view of iterative refinement.
```bibtex
@misc{nostalgebraist2020logitlens,
  title        = {interpreting {GPT}: the logit lens},
  author       = {{nostalgebraist}},
  year         = {2020},
  month        = aug,
  howpublished = {LessWrong},
  url          = {https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens},
  note         = {Blog post, 31 August 2020}
}
```
**vs Paper 8:** This is the source of Paper 8's settling-depth measure. Paper 8's lens-free arrival comparison is a direct check on the lens's validity in this setting.

### B6. Lindsey 2025 — Emergent introspective awareness
**VERIFIED.** *Emergent Introspective Awareness in Large Language Models*. Jack Lindsey (Anthropic). Transformer Circuits Thread, published 2025-10-29. An arXiv copy also exists: **arXiv 2601.01828** (v1 2026-01-05, cs.CL, sole author Jack Lindsey).
**Summary.** Representations of known concepts are injected into a model's activations, and the model's self-reports are checked against the injection. In some settings, models (Claude Opus 4/4.1 strongest) notice and correctly name the injected concepts. Some models also use recall of prior intentions to tell their own outputs from prefills. The author stresses that the ability is unreliable and context-dependent.
```bibtex
@article{lindsey2025emergent,
  author  = {Lindsey, Jack},
  title   = {Emergent Introspective Awareness in Large Language Models},
  journal = {Transformer Circuits Thread},
  year    = {2025},
  url     = {https://transformer-circuits.pub/2025/introspection/index.html}
}

@misc{lindsey2026emergent_arxiv,
  title         = {Emergent Introspective Awareness in Large Language Models},
  author        = {Lindsey, Jack},
  year          = {2026},
  eprint        = {2601.01828},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2601.01828}
}
```
(The `@article` is the page's own suggested citation.)
**vs Paper 8:** Lindsey injects off-distribution concept vectors into large frontier models. Paper 8 patches on-distribution process states in a 1.7B model and finds no report access.

### B7. Binder et al. 2024 — Looking Inward — arXiv 2410.13787
**VERIFIED.** *Looking Inward: Language Models Can Learn About Themselves by Introspection*. Felix J Binder, James Chua, Tomek Korbak, Henry Sleight, John Hughes, Robert Long, Ethan Perez, Miles Turpin, Owain Evans. v1 2024-10-17. cs.CL.
**Summary.** Models are fine-tuned to predict properties of their own behaviour in hypothetical scenarios. A model M1 predicts itself better than a model M2 trained on M1's behaviour, which the authors take as evidence of privileged self-access. M1 keeps tracking its behaviour after that behaviour is deliberately changed. The effect fails on complex or out-of-distribution tasks.
```bibtex
@misc{binder2024looking,
  title         = {Looking Inward: Language Models Can Learn About Themselves by Introspection},
  author        = {Binder, Felix J and Chua, James and Korbak, Tomek and Sleight, Henry and Hughes, John and Long, Robert and Perez, Ethan and Turpin, Miles and Evans, Owain},
  year          = {2024},
  eprint        = {2410.13787},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2410.13787}
}
```
**vs Paper 8:** Binder et al. use fine-tuned self-prediction of *behaviour*. Paper 8 tests untrained report of an *internal process* variable with behaviour (the answer) held fixed.

### B8. Geiger et al. — interchange intervention training — arXiv 2112.00826
**VERIFIED.** *Inducing Causal Structure for Interpretable Neural Networks*. Atticus Geiger, Zhengxuan Wu, Hanson Lu, Josh Rozner, Elisa Kreiss, Thomas Icard, Noah D. Goodman, Christopher Potts. v1 2021-12-01 (v2 2022-07-20). cs.LG. The arXiv record has no venue in its comments field.
**Summary.** Interchange intervention training (IIT) aligns variables of a causal model with neural representations. It trains the network so that swapping aligned representations from a source input yields the counterfactual output of the causal model. At zero loss, this guarantees that the causal model is an abstraction of the network. IIT beats multi-task and augmentation baselines on MNIST-PVR, ReaSCAN and MQNLI.
```bibtex
@misc{geiger2021inducing,
  title         = {Inducing Causal Structure for Interpretable Neural Networks},
  author        = {Geiger, Atticus and Wu, Zhengxuan and Lu, Hanson and Rozner, Josh and Kreiss, Elisa and Icard, Thomas and Goodman, Noah D. and Potts, Christopher},
  year          = {2021},
  eprint        = {2112.00826},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2112.00826}
}
```
**vs Paper 8:** Paper 8's same-answer patching is an interchange intervention, but it is used only diagnostically. Paper 8 does not train an alignment and makes no causal-abstraction claim beyond the transfer of settling depth.

### B9. Makelov, Lange & Nanda 2023 — subspace patching illusion — arXiv 2311.17030
**VERIFIED.** *Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching*. Aleksandar Makelov, Georg Lange, Neel Nanda. v1 2023-11-28. cs.LG. NeurIPS 2023 ATTRIB workshop.
**Summary.** Patching a subspace can change outputs as if a feature had been set when it has not. The effect can come from activating a dormant parallel pathway that is otherwise causally disconnected from the output. The paper shows this in a toy example, in IOI and in factual recall, where it links to rank-1 editing. It also describes the extra evidence needed for a faithful subspace claim.
```bibtex
@misc{makelov2023subspace,
  title         = {Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching},
  author        = {Makelov, Aleksandar and Lange, Georg and Nanda, Neel},
  year          = {2023},
  eprint        = {2311.17030},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2311.17030}
}
```
**vs Paper 8:** Paper 8 patches whole states, which avoids the dormant-subspace illusion but gives up specificity. The donor state carries everything at that site, not only settling depth. That limits what the transfer says about a dedicated "settling" variable.

### B10. Lanham et al. 2023 — CoT faithfulness — arXiv 2307.13702
**VERIFIED.** *Measuring Faithfulness in Chain-of-Thought Reasoning*. Tamera Lanham, Anna Chen, Ansh Radhakrishnan, Benoit Steiner, Carson Denison, Danny Hernandez, Dustin Li, Esin Durmus, Evan Hubinger, Jackson Kernion, Kamilė Lukošiūtė, Karina Nguyen, Newton Cheng, Nicholas Joseph, Nicholas Schiefer, Oliver Rausch, Robin Larson, Sam McCandlish, Sandipan Kundu, Saurav Kadavath, Shannon Yang, Thomas Henighan, Timothy Maxwell, Timothy Telleen-Lawton, Tristan Hume, Zac Hatfield-Dodds, Jared Kaplan, Jan Brauner, Samuel R. Bowman, Ethan Perez. v1 2023-07-17. cs.AI.
**Summary.** The authors intervene on chain-of-thought text (adding mistakes, truncating, paraphrasing) to see how much answers depend on it. Dependence varies widely across tasks. The CoT benefit is not explained by extra compute or phrasing alone, and larger models give less faithful reasoning on most tasks.
```bibtex
@misc{lanham2023measuring,
  title         = {Measuring Faithfulness in Chain-of-Thought Reasoning},
  author        = {Lanham, Tamera and Chen, Anna and Radhakrishnan, Ansh and Steiner, Benoit and Denison, Carson and Hernandez, Danny and Li, Dustin and Durmus, Esin and Hubinger, Evan and Kernion, Jackson and Luko{\v{s}}i{\={u}}t{\.{e}}, Kamil{\.{e}} and Nguyen, Karina and Cheng, Newton and Joseph, Nicholas and Schiefer, Nicholas and Rausch, Oliver and Larson, Robin and McCandlish, Sam and Kundu, Sandipan and Kadavath, Saurav and Yang, Shannon and Henighan, Thomas and Maxwell, Timothy and Telleen-Lawton, Timothy and Hume, Tristan and Hatfield-Dodds, Zac and Kaplan, Jared and Brauner, Jan and Bowman, Samuel R. and Perez, Ethan},
  year          = {2023},
  eprint        = {2307.13702},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2307.13702}
}
```
**vs Paper 8:** Lanham et al. test whether *stated reasoning* drives answers. Paper 8 tests whether a *post-hoc report about reasoning* ("step by step vs just know") tracks the actual internal process, and finds that it tracks the input instead.

### B11. Zhang & Nanda 2023 — activation-patching best practices — arXiv 2309.16042
**VERIFIED.** *Towards Best Practices of Activation Patching in Language Models: Metrics and Methods*. Fred Zhang, Neel Nanda. v1 2023-09-27. cs.LG. ICLR 2024.
**Summary.** The paper systematically varies evaluation metrics (e.g., probability vs logit difference) and corruption methods (Gaussian noise vs symmetric token replacement) in activation patching. These choices can change localization and circuit-discovery conclusions. The authors give conceptual arguments and recommended practices.
```bibtex
@misc{zhang2023best,
  title         = {Towards Best Practices of Activation Patching in Language Models: Metrics and Methods},
  author        = {Zhang, Fred and Nanda, Neel},
  year          = {2023},
  eprint        = {2309.16042},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2309.16042}
}
```
**vs Paper 8:** Paper 8's same-answer donors avoid off-distribution noise corruption, in the spirit of the recommendation for in-distribution corruption. The paper's metric-sensitivity point applies to how Paper 8 scores "transfer" of settling depth.

### B12. Elazar et al. 2021 — amnesic probing — arXiv 2006.00995
**VERIFIED.** *Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals*. Yanai Elazar, Shauli Ravfogel, Alon Jacovi, Yoav Goldberg. v1 2020-06-01 (v3 2021-02-19). cs.CL. TACL (published 2021). Note: the first arXiv version is from 2020, though the TACL paper is commonly cited as 2021.
**Summary.** The authors argue that probe accuracy cannot support behavioural conclusions. They propose removing a property from representations (via iterative nullspace projection) and measuring the effect on the task. On BERT, probing accuracy did not track task importance (e.g., for POS information in masked-word prediction).
```bibtex
@misc{elazar2020amnesic,
  title         = {Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals},
  author        = {Elazar, Yanai and Ravfogel, Shauli and Jacovi, Alon and Goldberg, Yoav},
  year          = {2020},
  eprint        = {2006.00995},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2006.00995}
}
```
**vs Paper 8:** This is the classic statement that decodable information need not be used. Paper 8's causally inert reading position is a positional instance of that point.

---

## Summary table

| # | Ref | Id | Verified | One-line relevance to Paper 8 |
|---|-----|----|----------|-------------------------------|
| 1 | Ferrara (OWMI) | arXiv 2608.20569 | Yes (probe range is 75–95.8%) | Reports at chance while probes read interventions; P8 shows the same split for a natural process variable |
| 2 | Song, Lederman, Hu, Mahowald | arXiv 2508.14802 | Yes | Defines introspection as beating third-party access; P8's fixed-input design puts this into practice |
| 3 | Aoki et al. | arXiv 2609.00904 | Yes (title longer) | Privileged targets abolish apparent self-control; parallel null for report |
| 4 | Xun | arXiv 2609.01048 | Yes | Readable before causal; parallels P8's inert reading position |
| 5 | Galeone et al. | arXiv 2606.24952 | Yes | Detection vs steering direction gap; P8 finds a positional read/push gap |
| 6 | Ferreira da Silva & Heimersheim | arXiv 2606.24964 | Yes | Error-correction plateaus in Qwen3-1.7B; alternative explanation for "inert" pushes |
| 7 | Hahami et al. | arXiv 2512.12411 | Yes | Detection explained by yes-bias; analogous cheap-cue confound |
| 8 | Singh, Linzen, Ravfogel | arXiv 2605.26242 | Yes | Privileged-access and second-order criteria; P8 is a test where predictions diverge |
| 9 | Li, Guo, Huang, Steinhardt, Andreas | arXiv 2511.08579 | Yes | Trained self-explainers; P8 is the untrained baseline and a target for training |
| 10 | Guo, Ruis, Andreas, Li | arXiv 2606.32038 | Yes | Introspective coupling for input-feature explanations; P8 targets an input-independent variable |
| 11 | Macar et al. | PMLR 306:85403–85429 (arXiv 2603.21396) | Yes, but author mismatch: PMLR "Zifan Wang" vs arXiv "Atticus Wang" | Anomaly-detection circuit; P8's on-distribution states give it nothing to detect |
| 12 | Lederman & Mahowald | arXiv 2603.05414 | Yes | Content-agnostic detection; P8 also finds no content-specific access |
| B1 | Meng et al. (ROME) | arXiv 2202.05262 | Yes | Causal tracing precedent; P8 uses clean-to-clean, output-fixed patching |
| B2 | Geva et al. | arXiv 2304.14767 | Yes | Positional information flow; the read site differs from the use site |
| B3 | Stolfo, Belinkov, Sachan | arXiv 2305.15054 | Yes | Arithmetic mechanism precedent; P8 studies process, not result |
| B4 | Belrose et al. (tuned lens) | arXiv 2303.08112 | Yes | Logit-lens brittleness; fits ρ 0.26 |
| B5 | nostalgebraist (logit lens) | LessWrong, 2020-08-31 | Yes (not on arXiv) | Source of the settling-depth measure |
| B6 | Lindsey (introspection) | transformer-circuits 2025; arXiv 2601.01828 | Yes | Concept injection; P8 uses on-distribution states |
| B7 | Binder et al. (Looking Inward) | arXiv 2410.13787 | Yes | Fine-tuned behavioural self-prediction vs untrained process report |
| B8 | Geiger et al. (IIT) | arXiv 2112.00826 | Yes | Interchange interventions; P8 uses them diagnostically only |
| B9 | Makelov, Lange, Nanda | arXiv 2311.17030 | Yes | Subspace illusion; whole-state patch avoids it but loses specificity |
| B10 | Lanham et al. | arXiv 2307.13702 | Yes | Faithfulness of stated reasoning; P8 tests faithfulness of process reports |
| B11 | Zhang & Nanda | arXiv 2309.16042 | Yes | Patching metric and corruption choices |
| B12 | Elazar et al. (amnesic probing) | arXiv 2006.00995 | Yes (v1 2020; TACL 2021) | Decodable does not mean used; matches the inert reading position |
