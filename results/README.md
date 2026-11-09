# Results

Aggregate tables behind the numbers in the paper, generated from the per-record predictions
and run logs by [`07_analysis/build_results_tables.py`](../07_analysis/build_results_tables.py)
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
| `qwen_qlora` | Qwen3.5-4B | QLoRA, fair adapter (r=8, q/v, 2 epochs) | 4-bit NF4 | RTX 4060, 8 GB |
| `qwen_qlora_trunc` | Qwen3.5-4B | same adapter, evaluation inputs truncated (ablation) | 4-bit NF4 | RTX 4060, 8 GB |
| `gemma_zero_bf16` | Gemma-4-E4B | zero-shot | bf16 | A100, 80 GB |
| `gemma_zero_4bit` | Gemma-4-E4B | zero-shot | 4-bit NF4 | A100, 80 GB |
| `gemma_fs_full` | Gemma-4-E4B | few-shot, 33 full examples | bf16 | A100, 80 GB |
| `gemma_qlora_fair` | Gemma-4-E4B | QLoRA, fair adapter (r=8, q/v, 2 epochs) | 4-bit NF4 | A100, 80 GB |
| `gemma_qlora_opt` | Gemma-4-E4B | QLoRA, optimized adapter (r=16, all linear layers, 3 epochs) | 4-bit NF4 | A100, 80 GB |
| `svm`, `lr`, `rf` | TF-IDF features | linear SVM, logistic regression, random forest | – | CPU |

Validation runs carry a `_val` suffix. `qwen_fs_comp_val_4060` and `qwen_fs_comp_val_a100` form
the device control.

## Tables

| File | Contents | Used for |
|---|---|---|
| `T01_main_results_test.csv` | Test accuracy and macro-F1 with 95% intervals, unparseable outputs and fit in 8 GB, for the 9 LLM configurations and the 3 TF-IDF baselines | main results table, baseline comparison |
| `T02_validation_results.csv` | The same metrics on the validation split | device control |
| `T03_paired_tests.csv` | 20 paired comparisons: accuracy difference in points, discordant counts, exact and Holm-adjusted p | p-values in the results |
| `T04_per_class_f1_test.csv` | Per-class F1 on the test split, 10 configurations | per-class analysis |
| `T05_classical_per_class_f1.csv` | Per-class F1 of the three TF-IDF baselines, as stored at training time | baseline comparison |
| `T06_confusion_matrices_long.csv` | 11×11 confusion matrices of the 10 configurations, long format | error analysis |
| `T07_top_confusions.csv` | The eight most frequent confusions of `qwen_zero`, `qwen_qlora`, `gemma_qlora_opt` and `svm` | error analysis |
| `T08_mask_control.csv` | The `[DIAG_MASKED]` placeholder: frequency, accuracy from the placeholder count alone, SVM with and without it | leakage discussion |
| `T09_error_overlap.csv` | Test records missed by SVM, Qwen QLoRA and optimized Gemma QLoRA alike; rank correlation of per-class F1 | error analysis |
| `T10_leakage_audit_test.csv` | Leakage-screen category of the test inputs | data description |
| `T11_dataset_splits.csv` | Split sizes and class balance | data description |
| `T12_memory.csv` | Measured GPU memory: weights, non-quantized part, peak inference, activations, QLoRA training peak | memory table |
| `T13_memory_components_gemma.csv` | Gemma-4-E4B parameters and memory by component, bf16 and 4-bit | memory analysis |
| `T14_resources_latency.csv` | Mean prompt length, latency, inference time and QLoRA training time from the run logs | prompt lengths, run time |

## Notes

- **LR and RF (T01).** Only aggregate scores were stored for these two baselines. Their accuracy
  intervals are computed from the stored accuracy; macro-F1 intervals are not available. The SVM
  row is recomputed from its stored per-record predictions and agrees with its stored aggregate
  scores (T05).
- **Mask control (T08).** Both SVMs in T08 are retrained with identical settings in the
  environment above, one with the placeholder token and one without it. Compare them with each
  other, not with the SVM in T01, which was trained in the original environment (accuracy 0.5915
  there, 0.5909 here).
- **Memory (T12, T13).** Qwen was profiled on the RTX 4060 in fp16 and in 4-bit. Gemma does not
  fit in 8 GB and was profiled on the A100 in bf16 and in 4-bit. Peak values are reserved memory
  at batch size 1 with 2,048-token prompts.
- **Run time (T14).** From per-record timing in the run logs. Runs without it are marked
  `not recorded` and left blank. Timings from the A100 and the RTX 4060 are not comparable. The
  training time is that of the Qwen adapter on the RTX 4060.
- **Device control.** `qwen_fs_comp_val_4060` and `qwen_fs_comp_val_a100` share the compressed
  examples, the query construction and greedy decoding, but their tokenized prompts differ record
  by record (mean 2,016 versus 2,028 tokens, T14). The control therefore compares the two devices
  together with their software environments.

## Regenerating

From the workspace root (see the top-level README):

```
python /path/to/repo/07_analysis/build_results_tables.py
```

This writes `workdir/output/results_tables/`. It needs the split files, the per-record outputs of
every run listed in the script and the memory tables, and therefore credentialed MIMIC-IV access.
