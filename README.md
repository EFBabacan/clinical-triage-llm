# Classifying clinical admission notes with open-weight LLMs on an 8 GB GPU

Code for the paper *"Classifying Clinical Admission Notes with Open-Weight LLMs on an 8 GB GPU"*
(Enes Furkan Babacan and Abdurrahman Gümüş; arXiv preprint, link to be added).

We compare ways of classifying clinical text with open-weight large language models under a
fixed memory budget of one 8 GB consumer GPU: zero-shot prompting, few-shot (in-context)
prompting and QLoRA fine-tuning, with Qwen3.5-4B and Gemma-4-E4B, alongside a TF-IDF linear SVM
that runs on a CPU. The task assigns the primary reason for admission to one of 11 classes from an
admission note: the sections of a MIMIC-IV discharge summary that describe the patient at
presentation. File names and comments call this input the *time-aware triage input* (V4).

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
| Google Colab, A100 80 GB | Gemma-4-E4B (does not fit in 8 GB, even at 4-bit), the uncompressed few-shot prompt, the test run of the compressed few-shot prompt (so that both prompts share a device), Gemma memory profiling |

Measured on the 8 GB device: zero-shot test run ≈ 40 min, QLoRA training ≈ 9.1 h (peak 7,055 MiB allocated),
QLoRA evaluation ≈ 45 min.

## Pipeline

| Step | Location | What it does |
|---|---|---|
| 0 | `notebooks/experiment_notebook.ipynb`, sections BLOK 0–6 | Cohort, ICD-10 labels, section parsing, term masking, patient-level class-balanced splits |
| 1 | `01_data/` | Admission-note input (V4, *time-aware* in file names), leakage screen, JSON export of the splits |
| 2 | `02_zero_shot/` | Zero-shot runs (Qwen locally, Gemma on Colab) and the shared output parser |
| 3 | `03_few_shot/` | In-context example selection, re-selection and compression for the compressed prompt, few-shot runs |
| 4 | `04_qlora/` | QLoRA training and evaluation |
| 5 | `05_classical/` | TF-IDF + logistic regression / linear SVM / random forest, and the re-run that keeps per-record predictions of all three (the paper reports the linear SVM) |
| 6 | `06_memory/` | GPU memory measurements and the memory table |
| 7 | `07_analysis/` | Metrics recomputed from per-record predictions, McNemar tests, and the additional checks reported in the paper (class prevalence, leakage, learning curve) |
| 8 | `08_paper/` | Tables, figures and checks of the numbers quoted in the paper |

[MANIFEST.md](MANIFEST.md) lists every script, its original file name and its purpose.

## Results

Aggregate tables behind the numbers in the paper (accuracy and macro-F1 with confidence
intervals, paired tests, per-class F1, confusion matrices, memory, run time, class prevalence and
leakage checks) are in [results/](results/README.md). `07_analysis/build_results_tables.py`,
`build_reviewer_checks.py` and `build_leak_screen_table.py` regenerate them. The figures in
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
- The full and the compressed few-shot prompts do not use the same examples. The compressed
  prompt's examples were re-selected with two extra rules (`03_few_shot/reselect_other_3shot.py`),
  and the two prompts share 22 of their 33 example records.
- Two configurations were evaluated twice: Qwen with the full few-shot prompt and the shared Gemma
  adapter. The tables and the paper use the second runs, which kept per-record outputs (see
  [results/PROVENANCE.md](results/PROVENANCE.md)).
- Macro-F1 is averaged over the 11 target classes. Unparseable outputs count as errors in
  accuracy but do not form a twelfth class.
- The paper calls the two adapter configurations *shared* (r=8, query and value projections, both
  models) and *high-capacity* (r=16, all linear projections, Gemma only). File names, `config` keys
  and some table text keep the earlier names *fair* and *optimized*.

## License

Code: MIT (see [LICENSE](LICENSE)). Data: not included; its use is governed by the PhysioNet
Credentialed Health Data Use Agreement.

## Citation

Babacan, E. F., and Gümüş, A. Classifying Clinical Admission Notes with Open-Weight LLMs on an
8 GB GPU. arXiv preprint, 2026 (identifier to be added).
