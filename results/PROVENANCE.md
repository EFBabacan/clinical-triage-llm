# Provenance

The tables in this folder were computed from per-record model outputs that cannot be released,
because they are derived from MIMIC-IV and covered by the PhysioNet Credentialed Health Data Use
Agreement. To fix exactly which files the reported numbers come from, this list gives every source
file with its row count and SHA-256 checksum. The files are kept by the authors under the data use
agreement; the checksums pin their exact content.

| Group | File | Rows | SHA-256 | Used for |
|---|---|---|---|---|
| predictions, test | `output/zeroshot/zeroshot_v4_test.csv` | 1650 | `6b8cb586e5dc4014ab31e8a9b11d1db1fdf2c23ff0db11a071550315e59f06c9` | Qwen zero-shot; leakage tiers (T10) |
| predictions, test | `output/b1_compressed_a100/b1_compressed_test.csv` | 1650 | `7b6bd715352015a883176cac553674eb711311062dd9ff49ed737336cd5a1e0e` | Qwen few-shot compressed |
| predictions, test | `output/b2_b5_verify/b2_full3shot_test.csv` | 1650 | `2de30f4f5ea8366fc155208f73f80dd5ec084f7b351a5cfdb2a563c7fbb808a0` | Qwen few-shot full |
| predictions, test | `output/qlora_eval_fullctx/test_predictions.csv` | 1650 | `72b46f25651c9b79bb864f92cc780b9165dbeeeb7047c0e39dbaa1d4434b1570` | Qwen QLoRA (fair) |
| predictions, test | `output/qlora_eval/v4_s3_full_2epoch_lr1e4/test_predictions.csv` | 1650 | `1774162bf82b5e60dbbbad9eaf9e103b5ea85441b4b6542b013988304146385a` | Qwen QLoRA, truncated input (ablation) |
| predictions, test | `output/colab_downloads/gemma_bf16_zeroshot_test.csv` | 1650 | `19579b48d711c1f1e66e40f96d515b463169ac5bf86f54f15903acad80be7c16` | Gemma zero-shot bf16 |
| predictions, test | `output/colab_downloads/gemma_4bit_zeroshot_test.csv` | 1650 | `9be5aeaab771099e2c79e25f50896795811d8db2d2c80e04adaf4e95f05f7c2b` | Gemma zero-shot 4-bit |
| predictions, test | `output/gemma_full3shot_test.csv` | 1650 | `2646c34b9a01fbcdb1098e3826b3492301f8ea34b3342a42ba5733b4de4ef583` | Gemma few-shot full |
| predictions, test | `output/b2_b5_verify/b5_gemma_qlora_test.csv` | 1650 | `af665688b0786f062f6d11fdd10d091e571056a74e25496ff90347d5446d34b3` | Gemma QLoRA (fair) |
| predictions, test | `output/gemma_qlora_test_preds.csv` | 1650 | `863f6e033a7fd827894f31176e5fdf1a393b17f8f1cf6f3a977b30c31bb6565e` | Gemma QLoRA (optimized); labels only, order-aligned |
| predictions, test | `output/classical_ml_v4/classical_ml_v4_predictions.csv` | 1650 | `c7d8259c1e5843a510edfbf84392027ce5015b2c80826f8a2265bbb4bd69db99` | TF-IDF linear SVM |
| predictions, test | `output/classical_ml_v4/classical_ml_v4_predictions_all_models.csv` | 1650 | `2f5d6af6d55df8efa1ad17b9501e4bc0f54a1a9a4f5569038c77b9dbdce58537` | TF-IDF baselines, re-run: logistic regression and random forest (T01, T05); labels only, order-aligned |
| predictions, validation | `output/zeroshot/zeroshot_v4_val.csv` | 550 | `7a7dad687aa92e719d6d4229776fe6b636718b4328cd267b7f64fa9d5b998cfb` | Qwen zero-shot (val) |
| predictions, validation | `output/fewshot/fewshot_v4_3shot_ultracompact_val.csv` | 550 | `63b857e754522fdc97db1068694a866184252e1a46d029d0c472c6c07eae3112` | device control, 8 GB device |
| predictions, validation | `output/b1_compressed_a100/b1_compressed_val.csv` | 550 | `e189ae2656658f9bdab7b6e7b6c724bd8c193959da72b1d6a922e3cea6e8809a` | device control, A100 |
| predictions, validation | `output/colab_downloads/full_3shot_val.csv` | 550 | `cd02771273aff8c8a99161ca26da82e4fef5efe50486abedb61428fda70abd59` | Qwen few-shot full (val) |
| predictions, validation | `output/colab_downloads/gemma_zeroshot_val.csv` | 550 | `dd89adfe117bc54dbbc015919dd08144876a9d024d1862cae438ae04ce97bbe3` | Gemma zero-shot bf16 (val) |
| predictions, validation | `output/colab_downloads/gemma_4bit_zeroshot_val.csv` | 550 | `69f03591e3cab55582c46a7f78c1ba59e2c7da8b90ec1c4557cc8cc4847711b1` | Gemma zero-shot 4-bit (val) |
| aggregate only | `output/classical_ml_v4/classical_ml_v4_results.csv` | 5 | `f0244e8b1732fc335d49efa1d2d586343e91fdde839e0e9b8d8262a5ad3d5079` | LR and RF scores of the first run (not reported in the paper) |
| aggregate only | `output/classical_ml_v4/classical_ml_v4_per_class_f1.csv` | 11 | `25f408dc0f9dd571815f59e5845f989f404c5eeb78ac9d23e452153c24aa5a16` | per-class F1 of the first run; its SVM column is checked against T04 |
| aggregate only | `output/gemma_qlora_summary.csv` | 2 | `47d4dbbae642635caeb738445c90e1ee4000f73c51c0705d867181a9cb3b0a8a` | Gemma QLoRA optimized, run summary |
| cohort | `output/discharge_etiketli_full.parquet` | 122288 | `350ef6771e18bf92ec00d1bd7d10630008be50b1ecbfb2da61d40d209b4fd996` | admissions with a discharge summary and a primary ICD-10 code (T22) |
| cohort | `output/discharge_parsed.parquet` | 118087 | `7f35bc8c1fe18061121169075440aeac25ac0d62f4deffcd7fcb114b71dd35f8` | eligible pool: class distribution and primary codes (T15, T19, T22) |
| cohort | `output/discharge_split.parquet` | 6600 | `a23d59a8d381f811c4cec27ad9e8117310b5f073e86200068b3eb2a49e358f9f` | leakage-screen state of the three splits (T23) |
| splits | `output/qlora_json/qlora_v4_train_S1.json` | 1100 | `2b960f56719d8e23503a98a96d3e5d2442a38a84a217785acde23ccb360277e9` | training subset S1 |
| splits | `output/qlora_json/qlora_v4_train_S2.json` | 2200 | `24f818b82f4e0cd4cf487d3b035ec55b700f59934bd3a89397fa5ae0a3d5cfde` | training subset S2 |
| splits | `output/qlora_json/qlora_v4_train_S3.json` | 4400 | `8e6211b9d406dd83d4665e05ed5abd68acb5fee9443f1e2d53711b95f9402574` | training set (4400) |
| splits | `output/qlora_json/qlora_v4_val.json` | 550 | `a6786bea67005ece136ed77166982419e5b904b43de95d57b27d9cb323a1d000` | validation set (550) |
| splits | `output/qlora_json/qlora_v4_test.json` | 1650 | `11df062c362ed4acee53e09d216ec8297bfe6f782a4dfbe2f661555485341b7a` | test set (1650) |
| prompts | `output/zeroshot/zeroshot_prompt_FINAL.txt` |  | `6f3031e72f3770e4fe2e026097e9f837e596bc86be5391c7e1bc6ae6abf2fb8e` | zero-shot prompt |
| prompts | `output/fewshot/compressed_3shot_PROMPT_FROZEN.txt` |  | `0d93f1ac538e03ad612affda7ed793ccc1096f83d8d2f0b5f9bd80a8f1f61361` | compressed few-shot prompt (contains training examples) |
| prompts | `output/fewshot/full_3shot_PROMPT_FROZEN.txt` |  | `6e5eba4e6d0682d8305c49f58a1c633f904f73125dc60db515ac5ad6d810360a` | full few-shot prompt (contains training examples) |
| prompts | `output/fewshot/fewshot_v4_3shot_ultracompact_examples.json` | 33 | `d0e3836c71f226afa5ffaaf9cec9990097aee0cf813638fe809f4de27df765d3` | the 33 examples of the compressed few-shot prompt; the SVM learning curve (T20) uses them |
| prompts | `output/fewshot/fewshot_v4_3shot_examples.json` | 33 | `ff89a87b7ca84b62615f98cd3faf14a7eddd578c42a80a64fa846938e2dcce73` | the 33 examples of the full few-shot prompt; 22 of them are also in the compressed prompt |
| logs and measurements | `output/qlora_adapters/v4_s3_full_2epoch_lr1e4/training_curve.json` |  | `b1006118dfd34e72ff1812f7a21e2554ffb602a594aeb0d8ec9081388dafa1ae` | Qwen QLoRA training time and peak memory |
| logs and measurements | `output/classical_ml_v4/classical_ml_v4_rerun_meta.json` |  | `f4a28f409c8b2eb8d1f8d08d81f143b27a16b5b5963e0b13e5d747382585b35b` | versions, seed and reproduction checks of the TF-IDF re-run |
| logs and measurements | `output/memory_profile/memory_profile_ozet.csv` | 5 | `09a28cf77c68528ee8bae99c9477287700a596f1f60f5e03f88af1c056c10ba1` | Qwen memory profile, summary |
| logs and measurements | `output/memory_profile/profil_4bit_20260806_121415.json` |  | `a95bd7ef023cc74c545a9c935f95f15db2f1b65ad320337f6db48282f5c84152` | Qwen 4-bit memory profile |
| logs and measurements | `output/memory_profile/profil_fp16_20260806_121446.json` |  | `befa812460418bcf7f577007746383d2603f97c294be01de449a032a40b90ce8` | Qwen fp16 profile (out of memory) |
| logs and measurements | `output/b3_gemma_memory/b3_gemma_bf16.json` |  | `d02845a63d67479feeb554b4a92ca86ade7bb46084e4cc2be84800b2cf77ab14` | Gemma bf16 memory (A100) |
| logs and measurements | `output/b3_gemma_memory/b3_gemma_4bit.json` |  | `072144080c8cfba5e1959239d361b84ce99da7084cea5a96b52582ff2ba353b7` | Gemma 4-bit memory (A100) |
| logs and measurements | `__blok14b_format_output.txt` |  | `f4341c766ebe1647677e0e627f4a03bd26e529ce2886d1346de13d43fc648b52` | one-shot prompt token budget (7,656 tokens on average) |
| logs and measurements | `__blok14c_val_output.txt` |  | `cf07ec5462e427f1c95d027cd3d8a3ea172498e06ccf92bfec54d039d049a1ba` | out-of-memory error of the one-shot prompt on the 8 GB GPU |
| model | `output/qlora_adapters/v4_s3_full_2epoch_lr1e4/adapter/adapter_model.safetensors` |  | `9a332538ebbbe2d13048fb773232b481e17ea83f798f060af2bdaca3293ce86f` | Qwen QLoRA adapter (fair) |
| model | `output/qlora_adapters/v4_s3_full_2epoch_lr1e4/adapter/adapter_config.json` |  | `aef9d5f9caa4b98e32a4f684dfe75d292a4a25fd1bb06933d47be925f4d42700` | Qwen QLoRA adapter configuration |
| model | `output/qlora_adapters/v4_s3_full_2epoch_lr1e4/adapter/tokenizer.json` |  | `92b95cc075eb3b4ad8454383694d3e4189ebca27abc944d0a51dfa4e1d57024d` | Qwen tokenizer, token counts of the unparseable outputs (T25) |
| model | `gemma_qlora_adapter/adapter_model.safetensors` |  | `decb4ece2d8c995af98118f2027f71dc499de23447d3ae9695539eaa5f2e2911` | Gemma QLoRA adapter (fair), evaluated by `04_qlora/colab_B5_gemma_qlora_eval.py` |
| model | `gemma_qlora_adapter/adapter_config.json` |  | `20ef1a8ea67d6b4d5d2984d247c438320a5c7f0fca19de3acfd7d91aa8a3b71c` | its configuration: r=8, query and value projections of the language model (66 modules) |
| model | `gemma_tum_modeller/adapter/adapter_model.safetensors` |  | `4d4cbf76d7675a64566939bb4ea6988fad3ce540ba78475c3ef397d04ac8c1c5` | Gemma QLoRA adapter (optimized) |
| model | `gemma_tum_modeller/adapter/adapter_config.json` |  | `eb014ac6c236d5ef1fcd24ee7767156139ae63bc5f94afdf965202a7bfc674ca` | its configuration: r=16, all linear projections of the language model (258 modules) |
| derived tables | `results_master.csv` | 15 | `4421018eb497c77dc49c11c5edcbd523773a6233bd1934ed5938b2baa53d245c` | metrics recomputed from predictions |
| derived tables | `memory_table.csv` | 4 | `ab84cd109173f69ea9afc305b3f5ef5c52423d659d014e662de6928f3ec025b9` | memory table (Table II) |
| derived tables | `memory_components.csv` | 8 | `e26070c128e26505653ecc77fbf04e83e777208dd5be825069a9d9ed528a0dbc` | Gemma memory by component |
| derived tables | `resource_table.csv` | 13 | `17c163d8929c5073d98b278be3768f0dc8be9ab3f138fbe4d7c1ee38200b8756` | read by `08_paper/make_tables.py` but no longer used |
| notebook | `bitirme_tezi_discharge.ipynb` |  | `fca2e097633da05c1c27e151a11eaec8e42bce738018803af215bbab4c9a2562` | full experiment notebook with outputs |

## Software

- Local environment (RTX 4060 Laptop GPU) as installed at release: Python 3.11.9, torch 2.5.1+cu121,
  transformers 5.6.2, peft 0.19.1, bitsandbytes 0.49.2, accelerate 1.13.0, trl 1.2.0, tokenizers 0.22.2;
  scikit-learn 1.8.0 for the TF-IDF baselines. torch, transformers and trl were installed on 11 May 2026 at
  15:30. The evaluation of the Qwen adapter on full-length inputs and the Qwen memory measurements (August 2026)
  ran on these versions. The earlier local runs (zero-shot, 9 May; compressed few-shot validation, 10 May;
  QLoRA training, finished 11 May 07:47; truncated-input evaluation, 11 May 13:22) predate that
  installation, and their library versions were not recorded.
- Runs on Google Colab (A100 80 GB), including the Gemma memory measurements: library versions were
  not recorded.
- Building the tables in this folder: Python 3.10.12, pandas 2.3.3, numpy 2.2.6, scipy 1.15.3,
  scikit-learn 1.7.2, tokenizers 0.23.3. The reported logistic-regression and random-forest rows
  (re-run, `random_state=42`) were produced in this environment.

## Models

- `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` (local cache of the 8 GB device).
  The Hub repository has no later revision (checked 13 November 2026), so the Colab runs loaded the same one.
- `google/gemma-4-E4B-it` (Colab runs); the revision was not recorded. The Hub repository changed its
  chat template in July 2026; the Gemma runs span June to August 2026.

## Known gaps

- The first run of the TF-IDF baselines (scikit-learn 1.8.0) stored per-record predictions for the
  SVM only. Logistic regression and the random forest are reported from a re-run (scikit-learn 1.7.2,
  same settings and seed) that reproduces every SVM prediction; their accuracy differs from the
  first run by 0.06 and 0.97 points.
- The language models were not re-run on inputs with unmasked target terms or laboratory mentions
  removed; T24 compares existing predictions only.
- The first evaluation of the fair Gemma adapter kept no per-record output (the optimized adapter's
  evaluation overwrote it). The adapter was evaluated again (`b5_gemma_qlora_test.csv`): 0.6285 in the
  first run, 0.6255 in the second. The full few-shot prompt of Qwen was also run twice (0.5552, then
  0.5515). The tables and the paper use the second runs.
- The full and the compressed few-shot prompts share 22 of their 33 example records (compare the two
  example files above); the difference between them combines shortening with the replaced examples.
- The QLoRA training peak (7,055 MiB) is allocated memory; reserved memory was not recorded during
  training.
- Predictions of the optimized Gemma adapter hold the gold label and the prediction only, in test-split
  order; the gold labels match the split record for record.
- Run times of the Colab runs were not recorded.
- The 7,656-token mean length of the one-example-per-class prompt was estimated on 200 test inputs;
  the out-of-memory error occurred on the validation run.
