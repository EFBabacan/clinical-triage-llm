# Figures

The paper's figures, drawn by [`08_paper/make_paper_figures.py`](../08_paper/make_paper_figures.py)
from the aggregate tables in [`results/`](../results/README.md) only, so they can be redrawn
without MIMIC-IV access. Each figure is saved as PDF (vector, for LaTeX) and PNG (300 dpi preview).

| File | Shows | Data | Paper section |
|---|---|---|---|
| `fig1_paradigm_progression` | Accuracy and macro-F1 of zero-shot prompting, few-shot prompting (33 full examples) and QLoRA for both models (Gemma with the shared and the high-capacity adapter), with 95% intervals and the TF-IDF + linear SVM baseline | T01 | Results |
| `fig2_fewshot_by_base_strength` | Zero-shot to few-shot accuracy for both models, with the paired difference and the Holm-adjusted p of the exact McNemar test. Same data as the first two steps of Fig. 1(a); not used in the arXiv version | T01, T03 | Results |
| `fig3_quantization_decomposition` | GPU memory at 16-bit and 4-bit, split into quantizable weights, embedding tables and activations, with the measured peak (Qwen on the RTX 4060, Gemma on the A100) and the 8 GB limit | T12 | Resources |
| `fig4_prompt_budget_vs_accuracy` | Accuracy against prompt length for Qwen3.5-4B: zero-shot, compressed few-shot and full few-shot, with the cost of compression and its Holm-adjusted p; not used in the current arXiv version | T01, T03, T14 | Results |
| `fig5_per_class_f1` | Per-class F1 of all ten configurations, classes ordered by mean F1, macro-F1 in the last row | T04, T01 | Results, per class |
| `fig6_confusion_qwen_qlora` | Confusion matrix of Qwen3.5-4B with QLoRA, the best configuration that fits in 8 GB; classes grouped by organ system | T06 | Error analysis |

In the current arXiv version these files appear as Fig. 1 (`fig1`), Fig. 2 (`fig3`), Fig. 3 (`fig6`)
and Fig. 4 (`fig5`, appendix). `fig2` and `fig4` are not used there.

## Conventions

- Blue is Qwen3.5-4B and orange is Gemma-4-E4B in every figure. Line style and marker shape
  repeat the distinction for grayscale printing.
- Filled markers: the configuration runs on the 8 GB GPU. Open markers: it needed the A100.
- Error bars are 95% intervals: Wilson for accuracy, bootstrap for macro-F1 (see `results/`).
- Gemma's zero-shot and few-shot points in Figs. 1 and 2 are bf16; its QLoRA points are 4-bit.
- Text is set in Arial, or in Liberation Sans (same metrics) where Arial is missing.

## Values not taken from `results/`

Two constants at the top of the script:

- `GPU_MIB = 8188`: total memory of the RTX 4060 Laptop GPU as reported by CUDA.
- `OOM_TOKENS = 7656`: mean length of the uncompressed one-shot prompt that ran out of memory on
  that GPU (token audit of the V4 runs). The shaded region of `fig4` starts there.

## Regenerating

```
python 08_paper/make_paper_figures.py
```

Requires matplotlib, numpy and pandas.
