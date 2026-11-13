# Manifest

Every script in this repository, the file name it had during the experiments, and its purpose.
Paths inside the scripts were rewritten to be relative to the workspace (see README.md).

| Release path | Original file | Purpose |
|---|---|---|
| `notebooks/experiment_notebook.ipynb` | `bitirme_tezi_discharge.ipynb` | Full experiment notebook, outputs cleared (BLOK 0–16) |
| `04_qlora/qwen_qlora_train.py` | notebook section BLOK 15B | Qwen QLoRA training cell, extracted verbatim |
| `01_data/13a_triage_input_v3.py` | `__run_blok13a_standalone.py` | Section parsing and admission-time triage input (V3, precursor of V4) |
| `01_data/13a1_residual_audit.py` | `__run_blok13a1.py` | Residual-term audit of the V3 input |
| `01_data/13a1_hpi_imaging_audit.py` | `__run_blok13a1_hpi_imaging.py` | Audit of imaging mentions in HPI |
| `01_data/13a2_sentence_cleanup.py` | `__run_blok13a2.py` | Conservative sentence-level cleanup |
| `01_data/13a3_vital_preserving_cleanup.py` | `__run_blok13a3.py` | Cleanup that preserves vital signs |
| `01_data/13a4_residual_lab_cleanup.py` | `__run_blok13a4.py` | Residual laboratory-value cleanup |
| `01_data/13c_leakage_audit.py` | `__run_blok13c_leakage_audit.py` | Leakage screen of the triage input |
| `01_data/13c_leakage_severity.py` | `__run_blok13c_severity.py` | Three-tier leakage severity grading |
| `01_data/13d_time_aware_v4_initial.py` | `__run_blok13d_time_aware.py` | Time-aware triage input (V4), first version |
| `01_data/13e_time_aware_refined.py` | `__run_blok13e_refined.py` | V4 refined time-aware detector |
| `01_data/13e_v2_time_aware_scope.py` | `__run_blok13e_v2.py` | V4 audit scope restricted to HPI/PMH |
| `01_data/13f_manual_review_cleanup.py` | `__run_blok13f_manual.py` | Rule-based review of the test records still flagged by the screen, and final cleanup (V4) |
| `01_data/verify_nested_split.py` | `__verify_nested_split.py` | Checks patient-level, nested train subsets |
| `01_data/15a_export_json_splits.py` | `__run_blok15a_v4_qlora_json.py` | Exports train/val/test JSON files used by all methods |
| `02_zero_shot/qwen_zero_shot_val.py` | `__run_blok13b_v4_val.py` | Qwen3.5-4B zero-shot, validation split |
| `02_zero_shot/qwen_zero_shot_test.py` | `__run_blok13b_v4_test.py` | Qwen3.5-4B zero-shot, test split (Table I) |
| `02_zero_shot/parser.py` | `output/zeroshot/zeroshot_parser_FINAL.py` | Output parser shared by all generative runs |
| `02_zero_shot/gemma_setup_colab.py` | `gemma/blokG01_kurulum_ve_model.py` | Colab setup for Gemma-4-E4B |
| `02_zero_shot/gemma_zero_shot_val_bf16.py` | `gemma/blokG02_zeroshot_val.py` | Gemma zero-shot (bf16), validation |
| `02_zero_shot/gemma_zero_shot_val_4bit.py` | `gemma/blokG03_4bit_zeroshot_val.py` | Gemma zero-shot (4-bit), validation |
| `02_zero_shot/gemma_zero_shot_test.py` | `gemma/blokG04_zeroshot_TEST.py` | Gemma zero-shot, test split (Table I) |
| `03_few_shot/select_examples_medoid.py` | `__run_blok14_v4_select.py` | Selects in-context examples (class-wise medoids) from the training pool |
| `03_few_shot/reselect_other_3shot.py` | `__run_blok14_v4_reselect_other_3shot.py` | Re-selects the 33 examples of the compressed prompt: skips candidates whose shortened text reveals the current encounter and, for OTHER, candidates resembling a target class (22 of 33 stay those of the full prompt) |
| `03_few_shot/compress_examples.py` | `__run_blok14v4_3shot_ultracompact.py` | Compresses the 33 examples to fit 8 GB |
| `03_few_shot/qwen_fewshot_compressed_val.py` | `__run_blok14c_v4_3shot_ultra_val.py` | Qwen compressed few-shot, validation (8 GB device) |
| `03_few_shot/freeze_compressed_prompt.py` | `freeze_compressed_prompt.py` | Freezes the exact compressed prompt |
| `03_few_shot/freeze_full_prompt.py` | `freeze_full_prompt.py` | Freezes the exact full prompt |
| `03_few_shot/colab_B1_qwen_fewshot_compressed.py` | `b1_colab_upload/B1_colab_compressed_fewshot.py` | Qwen compressed few-shot on A100 (test + device control) |
| `03_few_shot/colab_B2_qwen_fewshot_full.py` | `b1_colab_upload/B2_colab_full_fewshot.py` | Qwen full few-shot on A100 (test) |
| `03_few_shot/colab_fewshot_setup.py` | `colab_fewshot/blok01_kurulum_ve_model_indir.py` | Colab setup for full few-shot |
| `03_few_shot/colab_fewshot_load_data.py` | `colab_fewshot/blok02_veri_yukle.py` | Colab data loading for full few-shot |
| `03_few_shot/colab_qwen_fewshot_full_test_initial.py` | `colab_fewshot/blok04_full_3shot_TEST.py` | First full few-shot test run (superseded by B2) |
| `03_few_shot/gemma_fewshot_full.py` | `gemma/blokG05_full_3shot.py` | Gemma full few-shot (bf16) |
| `03_few_shot/analyze_compression_and_device.py` | `analyze_b1.py` | Compression effect and device-control analysis |
| `03_few_shot/v3_superseded/select_1shot.py` | `__run_blok14_select_1shot.py` | Earlier 1-shot selection (V3, superseded; called by the notebook) |
| `03_few_shot/v3_superseded/section_aware_compaction.py` | `__run_blok14d_compact.py` | Earlier section-aware compaction (V3, superseded) |
| `03_few_shot/v3_superseded/compact_query.py` | `__run_blok14e_compact_query.py` | Earlier compact query input (V3, superseded) |
| `04_qlora/qwen_qlora_eval_fullcontext.py` | `eval_qlora_fullcontext.py` | Evaluates the Qwen adapter on full-length test inputs (Table I) |
| `04_qlora/gemma_qlora_load_data.py` | `gemma/blokG06a_qlora_veri_yukle.py` | Colab data loading for Gemma QLoRA |
| `04_qlora/gemma_qlora_train_fair.py` | `gemma/blokG06b_v3_qlora_train.py` | Gemma QLoRA, the paper's shared adapter (r=8, q/v; called *fair* in file names) |
| `04_qlora/gemma_qlora_eval.py` | `gemma/blokG06c_qlora_eval.py` | Gemma QLoRA evaluation |
| `04_qlora/gemma_qlora_train_optimized.py` | `gemma/blokG07_qlora_optimize.py` | Gemma QLoRA, the paper's high-capacity adapter (r=16, all linear layers; called *optimized* in file names) |
| `04_qlora/colab_B5_gemma_qlora_eval.py` | `b1_colab_upload/B5_colab_gemma_qlora_eval.py` | Re-run of Gemma fair QLoRA evaluation with per-record output |
| `05_classical/tfidf_baselines.py` | `klasik_ml_v4_baseline.py` | TF-IDF + LR / linear SVM / RF baselines on the same splits |
| `05_classical/tfidf_baselines_rerun.py` | `klasik_ml_v4_kayit_tahminleri.py` | Re-run of the three baselines with the same settings and seed that keeps per-record predictions of all of them; source of the LR and RF rows of T01 and T05 (the paper reports the SVM only) |
| `06_memory/profile_memory.py` | `profile_memory.py` | Measures GPU memory of Qwen on the 8 GB device |
| `06_memory/colab_B3_gemma_memory.py` | `b1_colab_upload/B3_colab_gemma_memory.py` | Measures Gemma memory (bf16 / 4-bit) on A100 |
| `06_memory/merge_b3.py` | `merge_b3.py` | Merges memory measurements |
| `06_memory/make_memory_table.py` | `make_memory_table.py` | Builds the memory table (Table II) |
| `06_memory/make_resource_table.py` | `make_resource_table.py` | Builds the resource/latency table |
| `07_analysis/results_master.py` | `A0_results_master.py` | Recomputes all metrics from per-record predictions |
| `07_analysis/mcnemar.py` | `mcnemar2.py` | Matched McNemar tests |
| `07_analysis/pairwise_tests.py` | `pairwise.py` | Pairwise comparisons |
| `07_analysis/analyze_full_fewshot_and_gemma_qlora.py` | `analyze_b2_b5.py` | Analysis of the B2 and B5 re-runs |
| `07_analysis/audit_numbers.py` | `audit_numbers.py` | Cross-checks reported numbers |
| `07_analysis/impact.py` | `impact.py` | Impact of the macro-F1 denominator correction |
| `07_analysis/build_results_tables.py` | `build_results_tables.py` | Builds the aggregate result tables in `results/` (T01–T14) |
| `07_analysis/build_reviewer_checks.py` | `build_reviewer_checks.py` | Additional checks: class prevalence, unmasked-term and laboratory-mention checks, SVM weights, label gaps, SVM learning curve and truncation, unparseable outputs (T15–T22, T24, T25) |
| `07_analysis/build_leak_screen_table.py` | `build_leak_screen_table.py` | Final state of the leakage screen in each split (T23) |
| `08_paper/make_tables.py` | `paper_v2/make_tables.py` | Generates the paper tables and number macros |
| `08_paper/make_paper_figures.py` | `make_paper_figures.py`, rewritten to read `results/` | Draws the paper figures in `figures/` from the result tables |
| `08_paper/verify_all.py` | `paper_v2/verify_all.py` | Verifies numbers in the paper against per-record results |
| `08_paper/verify_prose.py` | `paper_v2/verify_prose.py` | Verifies ratios written in the text |

## Result tables

`results/` holds the aggregate tables generated by the three `07_analysis/build_*.py` scripts: metrics
with confidence intervals, paired tests, per-class F1, confusion matrices, memory, run time, class
prevalence and leakage checks.
See [results/README.md](results/README.md).

## Figures

`figures/` holds the paper figures (PDF and PNG), drawn from `results/` by
`08_paper/make_paper_figures.py`; see [figures/README.md](figures/README.md).

## Not included

- MIMIC-IV data and anything derived from it: triage inputs, selected and compressed
  in-context examples, frozen prompts with examples, per-record predictions
- Trained LoRA adapters
- Internal notes, poster and presentation material

## Provenance notes

- **Paper scripts.** `08_paper/make_tables.py`, `verify_all.py` and `verify_prose.py` are the scripts
  of the arXiv version, unchanged. They take the parent of their own folder as the workspace (they
  ran from `workdir/paper_v2/`), and the two verify scripts also read the LaTeX source of the paper.
- **Adapter names.** The paper calls the adapters *shared* and *high-capacity*; file names and
  `config` keys keep *fair* and *optimized*.
- **TF-IDF baselines.** The first run (`05_classical/tfidf_baselines.py`, scikit-learn 1.8.0) kept
  per-record predictions for the SVM only. The logistic-regression and random-forest rows of T01 and T05 come from
  `05_classical/tfidf_baselines_rerun.py` (scikit-learn 1.7.2, same settings and seed), which
  reproduces every SVM prediction of the first run. The paper reports the linear SVM only.

- **Gemma QLoRA, optimized.** Train with `04_qlora/gemma_qlora_train_optimized.py`, then run
  `04_qlora/gemma_qlora_eval.py` in the same Colab session: the evaluation script scores the model
  held in memory. Its output, `gemma_qlora_test_preds.csv`, overwrote the earlier predictions of
  the fair adapter, so the fair Gemma row in the paper comes from the re-run
  `04_qlora/colab_B5_gemma_qlora_eval.py` (`b5_gemma_qlora_test.csv`).
- **Colab results.** Per-record outputs of the Colab runs (Gemma zero-shot at bf16 and 4-bit,
  Qwen full few-shot validation) are read from `workdir/output/colab_downloads/`.
- **Qwen QLoRA input length.** The Qwen adapter was trained on inputs truncated to their first
  320 tokens (`MAX_SEQ_LEN=512`) and evaluated on full-length inputs; Gemma adapters were trained
  with `MAX_SEQ_LEN=2048`.
