# Clinical triage text classification with open-source LLMs on an 8 GB GPU

Code for the paper *"A Comparison of Text Classification Methods Using Open-Source LLMs on Clinical Triage Data in Resource-Constrained Environments"* (Enes Furkan Babacan and Abdurrahman Gümüş; arXiv preprint,
link to be added).

We compare ways of classifying clinical text with open-weight large language models under a
fixed memory budget of one 8 GB consumer GPU: zero-shot prompting, few-shot (in-context)
prompting and QLoRA fine-tuning, with Qwen3.5-4B and Gemma-4-E4B, alongside TF-IDF baselines.
The task assigns the primary reason for admission to one of 11 classes from a time-aware triage
input derived from MIMIC-IV discharge summaries.

> **No data is included.** MIMIC-IV is credentialed data. This repository contains code and
> aggregate result tables only: no notes, no derived text, no per-record predictions and no
> trained adapters.

## Data access

1. Obtain credentialed access on [PhysioNet](https://physionet.org/) and sign the data use
   agreement for:
   - MIMIC-IV-Note v2.2 (`note/discharge.csv.gz`)
   - MIMIC-IV v3.1, hosp module (`diagnoses_icd`)
2. Keep the data on hardware you control. Do not send it to third-party model APIs or AI
   assistants: every model in this repository runs inside your own environment (a local GPU or
   your own Colab runtime) and none calls a model API. Delete copies from cloud storage after use.

## Workspace layout

Scripts use paths relative to a workspace directory. Run them from the workspace root, for
example `cd workspace && python /path/to/repo/02_zero_shot/qwen_zero_shot_test.py`.

```
workspace/
├── data/        raw PhysioNet files (you provide them)
├── workdir/     created by the pipeline: intermediate files, predictions, adapters, tables
│   └── output/colab_downloads/   per-record results of Colab runs, read by the analysis scripts
└── downloads/   optional landing folder for files downloaded from Colab; three ingestion
                 scripts copy them into workdir/output/ when present
```

## Hardware

| Environment | Used for |
|---|---|
| NVIDIA RTX 4060 Laptop GPU, 8 GB | Every Qwen3.5-4B configuration that fits in 8 GB, TF-IDF baselines (CPU), memory profiling of Qwen |
| Google Colab, A100 80 GB | Gemma-4-E4B (does not fit in 8 GB, even at 4-bit), the uncompressed few-shot prompt, Gemma memory profiling |

Measured on the 8 GB device: zero-shot test run ≈ 40 min, QLoRA training ≈ 9.1 h (peak 7,055 MiB),
QLoRA evaluation ≈ 45 min.

## Pipeline

| Step | Location | What it does |
|---|---|---|
| 0 | `notebooks/experiment_notebook.ipynb`, sections BLOK 0–6 | Cohort, ICD-10 labels, section parsing, term masking, patient-level class-balanced splits |
| 1 | `01_data/` | Time-aware triage input (V4), leakage screen, JSON export of the splits |
| 2 | `02_zero_shot/` | Zero-shot runs (Qwen locally, Gemma on Colab) and the shared output parser |
| 3 | `03_few_shot/` | In-context example selection, compression, few-shot runs |
| 4 | `04_qlora/` | QLoRA training and evaluation |
| 5 | `05_classical/` | TF-IDF + logistic regression / linear SVM / random forest |
| 6 | `06_memory/` | GPU memory measurements and the memory table |
| 7 | `07_analysis/` | Metrics recomputed from per-record predictions, McNemar tests |
| 8 | `08_paper/` | Tables, figures and checks of the numbers quoted in the paper |

[MANIFEST.md](MANIFEST.md) lists every script, its original file name and its purpose.

## Results

Aggregate tables behind the numbers in the paper (accuracy and macro-F1 with confidence
intervals, paired tests, per-class F1, confusion matrices, memory, run time) are in
[results/](results/README.md). `07_analysis/build_results_tables.py` regenerates them. The figures in
[figures/](figures/README.md) are drawn from these tables by `08_paper/make_paper_figures.py`, which
needs no data access. [results/PROVENANCE.md](results/PROVENANCE.md) lists the files behind the
numbers with checksums, software and model versions.

## Notes

- The notebook is the chronological record of the experiments and includes superseded versions
  (V1–V3). Its outputs were cleared. It calls some scripts by their original file names; see
  MANIFEST.md for the mapping.
- Comments and log messages are partly in Turkish.
- Colab scripts expect Google Drive under `/content/drive/MyDrive/`. Shell lines such as
  `!pip install` are kept as comments marked `# [Colab]`.
- Macro-F1 is averaged over the 11 target classes. Unparseable outputs count as errors in
  accuracy but do not form a twelfth class.

## License

Code: MIT (see [LICENSE](LICENSE)). Data: not included; its use is governed by the PhysioNet
Credentialed Health Data Use Agreement.

## Citation

Babacan, E. F., and Gümüş, A. A Comparison of Text Classification Methods Using Open-Source LLMs on Clinical Triage Data in Resource-Constrained Environments. arXiv preprint, 2026 (identifier to be added).
