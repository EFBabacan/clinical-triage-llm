# -*- coding: utf-8 -*-
"""Draws the paper figures from the released result tables.

Reads  : results/T01, T03, T04, T06, T12 and T14 (see results/README.md)
Writes : figures/<name>.pdf (vector, for LaTeX) and figures/<name>.png (300 dpi preview)
Run    : python 08_paper/make_paper_figures.py   (from anywhere; no MIMIC-IV access needed)

Design: one colour pair for the two models (blue = Qwen3.5-4B, orange = Gemma-4-E4B, checked
for colour-vision deficiency), a one-hue blue ramp for magnitudes, and a second channel for every
colour distinction (marker shape, line style, hatching, labels) so the figures survive grayscale
printing. Filled markers: the configuration runs on the 8 GB GPU. Open markers: it needed the A100.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB, FIG = os.path.join(REPO, "results"), os.path.join(REPO, "figures")
os.makedirs(FIG, exist_ok=True)

# The only two numbers not taken from results/ (see figures/README.md)
GPU_MIB = 8188      # total memory of the RTX 4060 Laptop GPU as reported by CUDA (T12, hardware column)
OOM_TOKENS = 7656   # mean length of the uncompressed one-shot prompt that ran out of memory on that GPU

# ---- style ------------------------------------------------------------------------------
QWEN, GEMMA = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS, SHADE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#f0efec"
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
        "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("blue_ramp", RAMP)
CMAP.set_bad("white")
GEMMA_LS, REF_LS = (0, (4, 2)), (0, (5, 3))
_have = {f.name for f in font_manager.fontManager.ttflist}
SANS = next(f for f in ("Arial", "Liberation Sans", "Helvetica", "Nimbus Sans", "DejaVu Sans") if f in _have)
plt.rcParams.update({
    "font.family": SANS, "mathtext.fontset": "custom", "mathtext.rm": SANS,
    "mathtext.it": f"{SANS}:italic", "mathtext.bf": f"{SANS}:bold", "mathtext.cal": SANS, "mathtext.sf": SANS,
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "legend.frameon": False,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "text.color": INK,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def tidy(ax, fmt=None):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.grid(axis="x", visible=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if fmt:
        ax.yaxis.set_major_formatter(FuncFormatter(fmt))


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"))
    plt.close(fig)
    print("  ", name)


def ptxt(p):
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e")
        return rf"$p = {m}\times10^{{{int(e)}}}$"
    return rf"$p = {p:.2f}$"


def pts(v):
    return f"{v:+.1f}".replace("-", "−") + " pts"


def text_on(rgba):  # white or near-black label inside a filled cell, by luminance
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgba[:3]]
    return "white" if 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2] < 0.30 else INK


read = lambda name: pd.read_csv(os.path.join(TAB, name))
t1 = read("T01_main_results_test.csv").set_index("config")
t3 = read("T03_paired_tests.csv").set_index("comparison")
t4 = read("T04_per_class_f1_test.csv").set_index("class")
t6 = read("T06_confusion_matrices_long.csv")
t12 = read("T12_memory.csv")
t14 = read("T14_resources_latency.csv").set_index("config")
CI = {"accuracy": ("acc_ci_low", "acc_ci_high"), "macro_f1": ("mf1_ci_low", "mf1_ci_high")}


def fits(k):
    return str(t1.at[k, "fits_8gb"]).startswith("yes")


def point(ax, x, k, col, color, shape, size=6.0):
    y = t1.at[k, col]
    lo, hi = (t1.at[k, c] for c in CI[col])
    ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="none", ecolor=color, elinewidth=0.8, alpha=0.8, zorder=3)
    ax.plot(x, y, marker=shape, ms=size, color=color, mec=color, mew=1.3, ls="none", zorder=4,
            mfc=color if fits(k) else "white")


FILL_KEY = [Line2D([], [], marker="o", ms=5.5, color=MUTED, ls="none", label="Runs on the 8 GB GPU"),
            Line2D([], [], marker="o", ms=5.5, color=MUTED, mfc="white", mew=1.2, ls="none", label="Needs the A100")]
NAMES = {"SEPSIS": "Sepsis", "STROKE": "Stroke", "ATRIAL_FIBRILLATION": "Atrial fibrillation",
         "AKI": "Acute kidney injury", "PNEUMONIA": "Pneumonia", "UTI": "Urinary tract infection",
         "HEART_FAILURE": "Heart failure", "GI_BLEEDING": "GI bleeding",
         "DIABETES_COMPLICATION": "Diabetes complication", "COPD_EXACERBATION": "COPD exacerbation", "OTHER": "Other"}
print("Figures from", TAB)

# ---- Fig. 1: adaptation methods for both models, against the TF-IDF baseline -------------
SERIES = [("Qwen3.5-4B", QWEN, "o", "-", -0.07, ["qwen_zero", "qwen_fs_full", "qwen_qlora"]),
          ("Gemma-4-E4B, fair adapter", GEMMA, "s", GEMMA_LS, 0.07, ["gemma_zero_bf16", "gemma_fs_full", "gemma_qlora_fair"])]
ALL1 = ["qwen_zero", "qwen_fs_full", "qwen_qlora", "gemma_zero_bf16", "gemma_fs_full", "gemma_qlora_fair", "gemma_qlora_opt"]
fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.55))
for ax, (col, title) in zip(axes, [("accuracy", "(a) Accuracy"), ("macro_f1", "(b) Macro-F1")]):
    svm = t1.at["svm", col]
    ax.axhline(svm, color=INK2, lw=0.9, ls=REF_LS, zorder=2)
    if col == "accuracy":
        ax.annotate("TF-IDF + linear SVM", (-0.36, svm), (0, 2.5), textcoords="offset points",
                    ha="left", va="bottom", fontsize=6.8, color=INK2)
    for name, color, shape, ls, dx, keys in SERIES:
        x = np.arange(3) + dx
        ax.plot(x, t1.loc[keys, col].values, color=color, lw=1.5, ls=ls, zorder=3)
        for xi, k in zip(x, keys):
            point(ax, xi, k, col, color, shape)
    point(ax, 2.22, "gemma_qlora_opt", col, GEMMA, "^", 6.5)
    lo = min(t1.loc[ALL1, CI[col][0]].min(), svm) - 0.012
    hi = max(t1.loc[ALL1, CI[col][1]].max(), svm) + 0.012
    ax.set_ylim(lo, hi)
    ax.set_xlim(-0.4, 2.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Zero-shot", "Few-shot\n(33 examples)", "QLoRA"])
    ax.set_title(title, loc="left", fontweight="bold", color=INK)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    tidy(ax, lambda v, _: f"{v:.2f}")
handles = [Line2D([], [], color=QWEN, marker="o", ms=5.5, lw=1.5, label="Qwen3.5-4B"),
           Line2D([], [], color=GEMMA, marker="s", ms=5, lw=1.5, ls=GEMMA_LS, mfc="white", mew=1.2, label="Gemma-4-E4B, fair adapter"),
           Line2D([], [], color=GEMMA, marker="^", ms=6, mfc="white", mew=1.2, ls="none", label="Gemma-4-E4B, optimized adapter"),
           Line2D([], [], color=INK2, lw=0.9, ls=REF_LS, label="TF-IDF + linear SVM (CPU)")] + FILL_KEY
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, handlelength=2.4, columnspacing=1.6)
fig.subplots_adjust(wspace=0.22)
save(fig, "fig1_paradigm_progression")

# ---- Fig. 2: few-shot benefit by base model (same data as Fig. 1a, first two steps) -------
fig, ax = plt.subplots(figsize=(3.4, 2.5))
for name, color, shape, ls, (z, f), comp in [
        ("Qwen3.5-4B", QWEN, "o", "-", ("qwen_zero", "qwen_fs_full"), "Few-shot benefit, Qwen (full)"),
        ("Gemma-4-E4B", GEMMA, "s", GEMMA_LS, ("gemma_zero_bf16", "gemma_fs_full"), "Few-shot benefit, Gemma (bf16)")]:
    dx = -0.04 if color == QWEN else 0.04
    ax.plot([dx, 1 + dx], t1.loc[[z, f], "accuracy"].values, color=color, lw=1.5, ls=ls, zorder=3)
    for xi, k in zip([dx, 1 + dx], (z, f)):
        point(ax, xi, k, "accuracy", color, shape)
    r = t3.loc[comp]
    ax.annotate(f"{pts(r.diff_points)}, {ptxt(r.p_exact)}", (1.04, t1.at[f, "accuracy"]), (9, 0),
                textcoords="offset points", ha="left", va="center", fontsize=7, color=INK)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Zero-shot", "Few-shot (33 examples)"])
ax.set_xlim(-0.2, 1.95)
ax.set_ylim(0.485, 0.6)
tidy(ax, lambda v, _: f"{v:.2f}")
ax.set_ylabel("Accuracy")
ax.legend(handles=[Line2D([], [], color=QWEN, marker="o", ms=5.5, lw=1.5, label="Qwen3.5-4B"),
                   Line2D([], [], color=GEMMA, marker="s", ms=5, lw=1.5, ls=GEMMA_LS, mfc="white", mew=1.2, label="Gemma-4-E4B")] + FILL_KEY,
          loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, handlelength=2.4, fontsize=7)
save(fig, "fig2_fewshot_by_base_strength")

# ---- Fig. 3: what 4-bit quantization can and cannot shrink -------------------------------
QPART, EPART, APART = "#b4b2a9", "#6250d6", "#e6e4dd"
fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.0), sharey=True)
top = 0
for ax, (model, precs) in zip(axes, [("Qwen3.5-4B", ["fp16", "4-bit NF4"]), ("Gemma-4-E4B", ["bf16", "4-bit NF4"])]):
    rows = [t12[(t12.model == model) & (t12.precision == p)].iloc[0] for p in precs]
    for x, r in enumerate(rows):
        quant, emb = r.weights_MiB - r.non_quantized_MiB, r.non_quantized_MiB
        act = 0.0 if pd.isna(r.activations_MiB) else r.activations_MiB
        kw = dict(width=0.5, edgecolor="white", linewidth=1.2, zorder=3)
        ax.bar(x, quant, color=QPART, **kw)
        ax.bar(x, emb, bottom=quant, color=EPART, hatch="////", **kw)
        if act:
            ax.bar(x, act, bottom=quant + emb, color=APART, hatch="....", **kw)
        bar_top = quant + emb + act
        if pd.isna(r.peak_inference_MiB):
            note, y = f"weights {r.weights_MiB:,.0f}\nout of memory\nwhile loading", bar_top
        else:
            peak = r.peak_inference_MiB
            ax.plot([x - 0.3, x + 0.3], [peak, peak], color=INK, lw=1.3, solid_capstyle="butt", zorder=5)
            note, y = f"peak {peak:,.0f}\n" + ("fits" if peak <= GPU_MIB else "exceeds 8 GB"), max(bar_top, peak)
        ax.annotate(note, (x, y), (0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=6.8, color=INK, linespacing=1.15)
        top = max(top, y)
    ax.axhline(GPU_MIB, color=INK2, lw=0.9, ls=REF_LS, zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(precs)
    ax.set_xlim(-0.6, 1.6)
    save_pct = t12[(t12.model == model) & (t12.precision == "4-bit NF4")].quantization_saving_pct.iloc[0]
    share = t12[(t12.model == model) & (t12.precision == precs[0])].non_quantized_share_pct.iloc[0]
    ax.set_title(model, loc="left", fontweight="bold", color=INK, pad=13)
    ax.text(0, 1.015, f"4-bit saves {save_pct:.1f}% \u00b7 embeddings: {share:.0f}% of 16-bit weights",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7, color=INK2)
    tidy(ax, lambda v, _: f"{v:,.0f}")
axes[0].set_ylabel("GPU memory (MiB)")
axes[0].set_ylim(0, top * 1.17)
axes[0].annotate(f"8 GB GPU ({GPU_MIB:,} MiB)", (1.58, GPU_MIB), (0, 3), textcoords="offset points",
                 ha="right", va="bottom", fontsize=6.8, color=INK2)
fig.legend(handles=[Patch(facecolor=QPART, edgecolor="white", label="Quantizable weights"),
                    Patch(facecolor=EPART, edgecolor="white", hatch="////", label="Embedding tables (stay 16-bit)"),
                    Patch(facecolor=APART, edgecolor="white", hatch="....", label="Activations and KV cache"),
                    Line2D([], [], color=INK, lw=1.3, label="Measured peak (reserved)"),
                    Line2D([], [], color=INK2, lw=0.9, ls=REF_LS, label="8 GB GPU")],
           loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=5, handlelength=1.8, columnspacing=1.2, fontsize=7)
fig.subplots_adjust(wspace=0.12)
save(fig, "fig3_quantization_decomposition")

# ---- Fig. 4: prompt length against accuracy, Qwen3.5-4B at 4-bit -------------------------
fig, ax = plt.subplots(figsize=(3.5, 2.7))
ax.axvspan(OOM_TOKENS, 60000, color=SHADE, lw=0, zorder=0)
ax.text(43000, 0.507, "out of memory\non the 8 GB GPU", ha="right", va="center", fontsize=6.8, color=INK2)
P4 = {k: (t14.at[k, "prompt_tokens_mean"], t1.at[k, "accuracy"]) for k in ("qwen_zero", "qwen_fs_comp", "qwen_fs_full")}
for k, shape, size in [("qwen_zero", "o", 6), ("qwen_fs_comp", "X", 7.5), ("qwen_fs_full", "o", 6)]:
    point(ax, P4[k][0], k, "accuracy", QWEN, shape, size)
lab = dict(fontsize=7, color=INK, textcoords="offset points")
ax.annotate("Zero-shot", (P4["qwen_zero"][0], t1.at["qwen_zero", "acc_ci_high"]), (0, 3), ha="center", va="bottom", **lab)
ratio = P4["qwen_fs_full"][0] / P4["qwen_fs_comp"][0]
ax.annotate(f"Few-shot, compressed\n({ratio:.1f}× shorter)", P4["qwen_fs_comp"], (9, -9), ha="left", va="top", **lab)
ax.annotate("Few-shot, full", P4["qwen_fs_full"], (0, 31), ha="center", va="bottom", **lab)
ax.annotate("", xy=(P4["qwen_fs_comp"][0] * 1.25, P4["qwen_fs_comp"][1] + 0.004),
            xytext=(P4["qwen_fs_full"][0] * 0.8, P4["qwen_fs_full"][1] - 0.003),
            arrowprops=dict(arrowstyle="->", color=INK2, lw=0.9, connectionstyle="arc3,rad=0.2"), zorder=2)
r = t3.loc["Compression cost, Qwen"]
ax.text(3300, 0.545, f"{pts(r.diff_points)}\nmatched McNemar\n{ptxt(r.p_exact)}", ha="center", va="center",
        fontsize=6.8, color=INK, linespacing=1.2)
ax.set_xscale("log")
ax.set_xlim(550, 45000)
ax.set_xticks([1000, 10000])
ax.set_xticklabels(["1k", "10k"])
ax.set_xlabel("Prompt length (tokens, log scale)")
ax.set_ylim(0.47, 0.594)
ax.set_ylabel("Accuracy")
tidy(ax, lambda v, _: f"{v:.2f}")
ax.legend(handles=[Line2D([], [], color=QWEN, marker="o", ms=5.5, ls="none", label="Runs on the 8 GB GPU"),
                   Line2D([], [], color=QWEN, marker="o", ms=5.5, mfc="white", mew=1.2, ls="none", label="Needs the A100"),
                   Line2D([], [], color=QWEN, marker="X", ms=6.5, ls="none", label="Compressed to fit 8 GB")],
          loc="upper left", fontsize=6.8, handletextpad=0.3, borderaxespad=0.2)
save(fig, "fig4_prompt_budget_vs_accuracy")

# ---- Fig. 5: per-class F1 of every configuration -----------------------------------------
COLS = [("svm", "TF-IDF\n+ SVM"),
        ("qwen_zero", "Zero-\nshot"), ("qwen_fs_comp", "Few-shot,\ncompr."), ("qwen_fs_full", "Few-shot,\nfull"), ("qwen_qlora", "QLoRA,\nfair"),
        ("gemma_zero_bf16", "Zero-shot,\nbf16"), ("gemma_zero_4bit", "Zero-shot,\n4-bit"), ("gemma_fs_full", "Few-shot,\nfull"),
        ("gemma_qlora_fair", "QLoRA,\nfair"), ("gemma_qlora_opt", "QLoRA,\noptimized")]
keys = [k for k, _ in COLS]
order = t4[keys].mean(axis=1).sort_values(ascending=False).index
mat = np.vstack([t4.loc[order, keys].values, t1.loc[keys, "macro_f1"].values])
ylabels = [NAMES[c] for c in order] + ["Macro-F1"]
norm = Normalize(0, 1)
fig, ax = plt.subplots(figsize=(6.9, 3.75))
ax.imshow(mat, cmap=CMAP, norm=norm, aspect="auto")
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.6, color=text_on(CMAP(norm(v))),
                fontweight="bold" if i == mat.shape[0] - 1 else "normal")
for x in (0.5, 4.5):
    ax.axvline(x, color="white", lw=3)
ax.axhline(mat.shape[0] - 1.5, color="white", lw=3)
ax.set_xticks(range(len(COLS)))
ax.set_xticklabels([lbl for _, lbl in COLS], fontsize=6.8)
ax.xaxis.tick_top()
ax.set_yticks(range(len(ylabels)))
ax.set_yticklabels(ylabels)
ax.tick_params(length=0)
for side in ax.spines.values():
    side.set_visible(False)
for x0, x1, label in [(-0.45, 0.45, "Classical"), (0.55, 4.45, "Qwen3.5-4B"), (4.55, 9.45, "Gemma-4-E4B")]:
    ax.annotate("", xy=(x0, -1.75), xytext=(x1, -1.75), xycoords="data", annotation_clip=False,
                arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8))
    ax.text((x0 + x1) / 2, -1.85, label, ha="center", va="bottom", fontsize=7.2, fontweight="bold", color=INK,
            clip_on=False)
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), ax=ax, fraction=0.03, pad=0.02)
cb.set_label("F1", color=INK2)
cb.outline.set_visible(False)
cb.ax.tick_params(length=0, labelsize=7)
save(fig, "fig5_per_class_f1")

# ---- Fig. 6: confusion matrix of the best configuration that fits 8 GB --------------------
GROUPED = ["SEPSIS", "PNEUMONIA", "UTI", "COPD_EXACERBATION", "HEART_FAILURE", "ATRIAL_FIBRILLATION",
           "AKI", "DIABETES_COMPLICATION", "STROKE", "GI_BLEEDING", "OTHER"]
SHORT = {"SEPSIS": "Sepsis", "PNEUMONIA": "Pneumonia", "UTI": "UTI", "COPD_EXACERBATION": "COPD", "HEART_FAILURE": "HF",
         "ATRIAL_FIBRILLATION": "AF", "AKI": "AKI", "DIABETES_COMPLICATION": "Diabetes", "STROKE": "Stroke",
         "GI_BLEEDING": "GI bleed", "OTHER": "Other"}
cm = (t6[t6.config == "qwen_qlora"].pivot(index="true_class", columns="predicted_class", values="count")
      .loc[GROUPED, GROUPED].values.astype(float))
cnorm = PowerNorm(0.5, vmin=0, vmax=cm.sum(axis=1).max())
fig, ax = plt.subplots(figsize=(4.7, 4.1))
ax.imshow(np.ma.masked_equal(cm, 0), cmap=CMAP, norm=cnorm)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        if cm[i, j]:
            ax.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center", fontsize=6.4,
                    color=text_on(CMAP(cnorm(cm[i, j]))), fontweight="bold" if i == j else "normal")
ax.set_xticks(range(11))
ax.set_xticklabels([SHORT[c] for c in GROUPED], rotation=45, ha="right", rotation_mode="anchor", fontsize=7)
ax.set_yticks(range(11))
ax.set_yticklabels([NAMES[c] for c in GROUPED], fontsize=7)
ax.set_xlabel("Predicted class")
ax.set_ylabel("True class (150 records each)")
ax.tick_params(length=0)
for side in ax.spines.values():
    side.set_visible(False)
ax.set_xticks(np.arange(-0.5, 11, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 11, 1), minor=True)
ax.grid(which="minor", color=GRID, lw=0.4)
ax.tick_params(which="minor", length=0)
cb = fig.colorbar(plt.cm.ScalarMappable(norm=cnorm, cmap=CMAP), ax=ax, fraction=0.04, pad=0.03)
cb.set_label("Records (square-root scale)", color=INK2)
cb.outline.set_visible(False)
cb.ax.tick_params(length=0, labelsize=7)
save(fig, "fig6_confusion_qwen_qlora")
print("Saved to", FIG)
