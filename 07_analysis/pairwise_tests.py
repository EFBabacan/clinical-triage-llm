# -*- coding: utf-8 -*-
"""TEST (n=1650) uzerindeki kilit ikili karsilastirmalarin anlamliligi."""
import numpy as np

N = 1650
# (ad, acc) -- hepsi test n=1650
A = {
    "Qwen zero-shot":        0.5152,
    "Qwen full 3-shot":      0.5552,   # per-kayit CSV YOK, rapor edilen deger
    "Qwen QLoRA adil":       0.6345,
    "Gemma zero-shot bf16":  0.5648,
    "Gemma zero-shot 4bit":  0.5388,
    "Gemma full 3-shot":     0.5624,
    "Gemma QLoRA adil":      0.6285,   # per-kayit CSV YOK
    "Gemma QLoRA optimize":  0.6467,
}

PAIRS = [
    ("Qwen full 3-shot", "Qwen zero-shot", "few-shot faydasi (Qwen)"),
    ("Gemma full 3-shot", "Gemma zero-shot bf16", "few-shot faydasi (Gemma)"),
    ("Qwen QLoRA adil", "Qwen zero-shot", "QLoRA kazanci (Qwen)"),
    ("Gemma QLoRA optimize", "Gemma zero-shot bf16", "QLoRA kazanci (Gemma)"),
    ("Gemma zero-shot 4bit", "Gemma zero-shot bf16", "4-bit niceleme bedeli (Gemma)"),
    ("Qwen QLoRA adil", "Gemma QLoRA adil", "adil QLoRA'da model farki"),
    ("Gemma QLoRA optimize", "Qwen QLoRA adil", "optimize Gemma vs Qwen"),
    ("Gemma zero-shot bf16", "Qwen zero-shot", "base model secimi"),
]

print("=" * 96)
print(f"TEST SETI IKILI KARSILASTIRMALAR  (n={N}, %95 CI, bagimsiz-oran yaklasimi = muhafazakar)")
print("=" * 96)
print(f"{'karsilastirma':34s} {'fark':>8s} {'%95 CI':>20s}   sonuc")
print("-" * 96)
for a, b, ad in PAIRS:
    p1, p2 = A[a], A[b]
    se = np.sqrt(p1 * (1 - p1) / N + p2 * (1 - p2) / N)
    d = p1 - p2
    lo, hi = d - 1.96 * se, d + 1.96 * se
    sig = "ANLAMLI" if (lo > 0 or hi < 0) else "gurultu icinde"
    print(f"{ad:34s} {100*d:+7.1f}p  [{100*lo:+6.1f}, {100*hi:+6.1f}]   {sig}")

print("\nNot: ayni test seti uzerinde eslesmis (paired) test daha dar CI verir;")
print("     asagidaki McNemar per-kayit CSV'si olan ciftler icin gercek anlamliliktir.")

# --- gercek eslesmis test: per-kayit CSV'si olan cift ---
import os
import pandas as pd
from scipy.stats import binomtest

OUT = "./workdir/output"  # workspace/workdir (see README)
DL = os.path.join(OUT, "colab_downloads")  # Colab sonuc dosyalari (5 Kasim 2026: Indirilenler'den projeye tasindi)

bf16 = pd.read_csv(os.path.join(DL, "gemma_bf16_zeroshot_test.csv"))
q4 = pd.read_csv(os.path.join(DL, "gemma_4bit_zeroshot_test.csv"))
m = bf16[["note_id", "label", "parsed_pred"]].merge(
    q4[["note_id", "parsed_pred"]], on="note_id", suffixes=("_bf16", "_4bit"))
c_bf = (m["parsed_pred_bf16"] == m["label"])
c_q4 = (m["parsed_pred_4bit"] == m["label"])
b = int((c_bf & ~c_q4).sum())   # bf16 dogru, 4bit yanlis
c = int((~c_bf & c_q4).sum())   # 4bit dogru, bf16 yanlis
p = binomtest(b, b + c, 0.5).pvalue
print("\n" + "=" * 96)
print("McNEMAR (eslesmis) — Gemma bf16 vs 4-bit zero-shot, ayni 1650 kayit")
print("=" * 96)
print(f"  bf16 dogru & 4-bit yanlis : {b}")
print(f"  4-bit dogru & bf16 yanlis : {c}")
print(f"  p = {p:.4g}   -> {'ANLAMLI (p<0.05)' if p < 0.05 else 'anlamli DEGIL'}")
