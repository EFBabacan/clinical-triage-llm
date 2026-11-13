# Results

Aggregate tables behind the numbers in the paper, generated from the per-record predictions
and run logs by [`07_analysis/build_results_tables.py`](../07_analysis/build_results_tables.py) (T01–T14),
[`build_reviewer_checks.py`](../07_analysis/build_reviewer_checks.py) (T15–T22, T24, T25) and
[`build_leak_screen_table.py`](../07_analysis/build_leak_screen_table.py) (T23)
(Python 3.10.12, pandas 2.3.3, numpy 2.2.6, scipy 1.15.3, scikit-learn 1.7.2).
They contain counts, rates and test statistics only: no note text, no record identifiers and
no per-record predictions.
[PROVENANCE.md](PROVENANCE.md) lists the source files behind every table with their SHA-256
checksums, the software and model versions, and the known gaps.

## Conventions

- Splits: test 1,650 records and validation 550, both class-balanced (150 and 50 per class);
  the training pool S3 has 4,400 records (T11).
- Accuracy counts unparseable outputs (`n_invalid`) as errors. Macro-F1 averages the 11 target
  classes; unparseable outputs do not form a twelfth class.
- 95% intervals: Wilson score interval for accuracy; percentile bootstrap for macro-F1
  (2,000 resamples of records, seed 0).
- Paired comparisons: exact McNemar test (two-sided binomial test on the discordant records).
  `p_holm` applies Holm's correction across all 20 comparisons in T03.
- The `config` keys are shared by all tables:

| `config` | Model | Method | Weights | Hardware |
|---|---|---|---|---|
| `qwen_zero` | Qwen3.5-4B | zero-shot | 4-bit NF4 | RTX 4060, 8 GB |
| `qwen_fs_comp` | Qwen3.5-4B | few-shot, 33 compressed examples | 4-bit NF4 | A100, 80 GB |
| `qwen_fs_full` | Qwen3.5-4B | few-shot, 33 full examples | 4-bit NF4 | A100, 80 GB |
| `qwen_qlora` | Qwen3.5-4B | QLoRA, shared adapter (r=8, q/v, 2 epochs) | 4-bit NF4 | RTX 4060, 8 GB |
| `qwen_qlora_trunc` | Qwen3.5-4B | same adapter, evaluation inputs truncated (ablation) | 4-bit NF4 | RTX 4060, 8 GB |
| `gemma_zero_bf16` | Gemma-4-E4B | zero-shot | bf16 | A100, 80 GB |
| `gemma_zero_4bit` | Gemma-4-E4B | zero-shot | 4-bit NF4 | A100, 80 GB |
| `gemma_fs_full` | Gemma-4-E4B | few-shot, 33 full examples | bf16 | A100, 80 GB |
| `gemma_qlora_fair` | Gemma-4-E4B | QLoRA, shared adapter (r=8, q/v, 2 epochs) | 4-bit NF4 | A100, 80 GB |
| `gemma_qlora_opt` | Gemma-4-E4B | QLoRA, high-capacity adapter (r=16, all linear layers, 3 epochs) | 4-bit NF4 | A100, 80 GB |
| `svm`, `lr`, `rf` | TF-IDF features | linear SVM, logistic regression, random forest | – | CPU |

The paper's *shared* and *high-capacity* adapters are called *fair* and *optimized* in the `config`
keys and in some table text. Validation runs carry a `_val` suffix. `qwen_fs_comp_val_4060` and `qwen_fs_comp_val_a100` form
the device control.

## Tables

| File | Contents | Used for |
|---|---|---|
| `T01_main_results_test.csv` | Test accuracy and macro-F1 with 95% intervals, unparseable outputs and fit in 8 GB, for the 9 LLM configurations and the 3 TF-IDF baselines (logistic regression and random forest from the re-run; the paper reports the SVM only) | main results table |
| `T02_validation_results.csv` | The same metrics on the validation split | device control |
| `T03_paired_tests.csv` | 20 paired comparisons: accuracy difference in points, discordant counts, exact and Holm-adjusted p | p-values in the results |
| `T04_per_class_f1_test.csv` | Per-class F1 on the test split, 10 configurations | per-class analysis |
| `T05_classical_per_class_f1.csv` | Per-class F1 of the three TF-IDF baselines, from per-record predictions (logistic regression and random forest: re-run) | baseline comparison |
| `T06_confusion_matrices_long.csv` | 11×11 confusion matrices of the 10 configurations, long format | error analysis |
| `T07_top_confusions.csv` | The eight most frequent confusions of `qwen_zero`, `qwen_qlora`, `gemma_qlora_opt` and `svm` | error analysis |
| `T08_mask_control.csv` | The `[DIAG_MASKED]` placeholder: frequency, accuracy from the placeholder count alone, SVM with and without it | leakage discussion |
| `T09_error_overlap.csv` | Test records missed by SVM, Qwen QLoRA and optimized Gemma QLoRA alike; rank correlation of per-class F1 | error analysis |
| `T10_leakage_audit_test.csv` | Leakage-screen category of the test inputs | data description |
| `T11_dataset_splits.csv` | Split sizes and class balance | data description |
| `T12_memory.csv` | Measured GPU memory: weights, non-quantized part, peak inference, activations, QLoRA training peak | memory table |
| `T13_memory_components_gemma.csv` | Gemma-4-E4B parameters and memory by component, bf16 and 4-bit | memory analysis |
| `T14_resources_latency.csv` | Mean prompt length, latency, inference time and QLoRA training time from the run logs | prompt lengths, run time |
| `T15_pool_class_distribution.csv` | Class distribution of the eligible pool (118,087 admissions) | label table, prevalence |
| `T16_prevalence_reweighted.csv` | Accuracy, macro-F1 and mean target-class precision on the balanced test set and reweighted to the pool distribution | discussion of prevalence |
| `T17_leakage_term_checks.csv` | Unmasked target terms and laboratory mentions in the chief complaint or symptom history; SVM retrained with them deleted | leakage discussion |
| `T18_svm_top_features.csv` | Five largest weights of the linear SVM for each class | appendix table |
| `T19_label_gaps.csv` | Primary codes that the prefix mapping sends to *other* although they describe a target condition | limitations |
| `T20_svm_learning_curve.csv` | SVM accuracy against the number of training labels, including the 33 examples of the compressed few-shot prompt | baseline comparison |
| `T21_svm_truncation.csv` | SVM accuracy on inputs cut to their first words | context ablation |
| `T22_cohort_and_time.csv` | Cohort sizes, mean input length, SVM run time, number of J44.9 admissions in the COPD class | data description, run time |
| `T23_leakage_screen.csv` | Leakage-screen state of each split and the rule-based review of the flagged test records | data description |
| `T24_term_stratified_recall.csv` | For every configuration: recall on target records with an unmasked own-class term against records of the same classes without one; AKI recall with and without a laboratory mention | leakage discussion, per-class analysis |
| `T25_unparseable_outputs.csv` | Unparseable outputs of the Qwen prompting runs: counts, length in tokens, accuracy if all of them were correct | results |

## Notes

- **LR and RF (T01, T05).** The first run of the baselines (scikit-learn 1.8.0) stored per-record
  predictions for the SVM only; it gave 0.5727 for logistic regression and 0.5055 for the random
  forest. The rows reported here come from a re-run with the same settings and seed in the
  environment above (`05_classical/tfidf_baselines_rerun.py`): 0.5721 and 0.5152. The re-run
  reproduces all 1,650 SVM predictions of the first run. The paper reports the linear SVM only.
- **Mask control (T08).** Both SVMs in T08 are retrained in the environment above with the settings
  of the SVM in T01, one with the placeholder token and one without it. The one with the
  placeholder reproduces the accuracy of the SVM in T01 (0.5915).
- **Leakage checks (T17, T24).** The deletion checks in T17 retrain the SVM; the language models
  were not re-run on inputs with the terms removed. T24 is observational: it compares records with
  and without a term or a laboratory mention under the existing predictions.
- **Unparseable outputs (T25).** Token counts use the Qwen tokenizer (`tokenizers` package and the
  tokenizer file stored with the adapter).
- **Memory (T12, T13).** Qwen was profiled on the RTX 4060 in fp16 and in 4-bit. Gemma does not
  fit in 8 GB and was profiled on the A100 in bf16 and in 4-bit. Peak inference values are reserved memory at batch size 1 with 2,048-token prompts. The QLoRA
  training peak in T12 is allocated memory, as reserved memory was not recorded during training.
- **Run time (T14).** From per-record timing in the run logs. Runs without it are marked
  `not recorded` and left blank. Timings from the A100 and the RTX 4060 are not comparable. The
  training time is that of the Qwen adapter on the RTX 4060.
- **Device control.** `qwen_fs_comp_val_4060` and `qwen_fs_comp_val_a100` share the compressed
  examples, the query construction and greedy decoding, but their tokenized prompts differ record
  by record (mean 2,016 versus 2,028 tokens, T14). The control therefore compares the two devices together with their software environments.
- **Few-shot examples.** `qwen_fs_full` and `gemma_fs_full` use the 33 examples selected by
  `03_few_shot/select_examples_medoid.py`. `qwen_fs_comp` uses 33 examples re-selected by
  `03_few_shot/reselect_other_3shot.py`, which skips candidates whose shortened text reveals the
  current encounter and, for *other*, candidates resembling a target class. The two sets share 22
  records. The SVM learning curve (T20) uses the compressed prompt's 33 records.
- **Repeated evaluations.** `qwen_fs_full` and `gemma_qlora_fair` were each evaluated twice. The
  tables use the second run, which kept per-record outputs. The first runs reached 0.5552 and
  0.6285 accuracy, 0.37 and 0.30 points above the second.

## Regenerating

From the workspace root (see the top-level README):

```
python /path/to/repo/05_classical/tfidf_baselines_rerun.py
python /path/to/repo/07_analysis/build_results_tables.py
python /path/to/repo/07_analysis/build_reviewer_checks.py
python /path/to/repo/07_analysis/build_leak_screen_table.py
```

These write `workdir/output/results_tables/`. They need the split files, the cohort tables, the
per-record outputs of every run listed in the scripts and the memory tables, and therefore
credentialed MIMIC-IV access.
