# -*- coding: utf-8 -*-
"""B2 + B5 sonuc analizi. Ozellikle B5'teki Macro-F1 tutarsizligini kovala."""
import os
import shutil

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score, precision_recall_fscore_support

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")
DL = r"./downloads"
DEST = os.path.join(OUT, "b2_b5_verify")
os.makedirs(DEST, exist_ok=True)

CLASSES = ["SEPSIS", "STROKE", "ATRIAL_FIBRILLATION", "AKI", "PNEUMONIA", "UTI",
           "HEART_FAILURE", "GI_BLEEDING", "DIABETES_COMPLICATION",
           "COPD_EXACERBATION", "OTHER"]

for f in ["b2_full3shot_test.csv", "b5_gemma_qlora_test.csv"]:
    s = os.path.join(DL, f)
    if os.path.exists(s):
        shutil.copy(s, os.path.join(DEST, f))

b2 = pd.read_csv(os.path.join(DEST, "b2_full3shot_test.csv"))
b5 = pd.read_csv(os.path.join(DEST, "b5_gemma_qlora_test.csv"))


def met(d):
    yt = d["label"].astype(str)
    yp = d["parsed_pred"].fillna("INVALID").astype(str)
    acc = float((yt == yp).mean())
    mf11 = f1_score(yt, yp, average="macro", labels=CLASSES, zero_division=0)
    mf_union = f1_score(yt, yp, average="macro", zero_division=0)
    wf1 = f1_score(yt, yp, average="weighted", zero_division=0)
    return acc, mf11, mf_union, wf1


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


print("=" * 84)
print("B2 — Qwen tam 3-shot  (raporlanan: acc 0.5552 · MF1 0.5621)")
print("=" * 84)
a, m11, mu, wf = met(b2)
lo, hi = wilson(int(round(a * len(b2))), len(b2))
print(f"  acc      = {a:.4f}  [{lo:.4f}, {hi:.4f}]   fark {100*(a-0.5552):+.2f} puan")
print(f"  MF1(/11) = {m11:.4f}   fark {100*(m11-0.5621):+.2f} puan")
print(f"  MF1(union)={mu:.4f}   wF1 = {wf:.4f}")
print(f"  INVALID  = {int((b2['parsed_pred']=='INVALID').sum())}  "
      f"direct = %{100*(b2['parse_kategori']=='direct').mean():.1f}")
print(f"  -> {'DOGRULANDI' if abs(a-0.5552) < 0.01 else 'FARK BUYUK'}")

print("\n" + "=" * 84)
print("B5 — Gemma adil QLoRA  (raporlanan: acc 0.6285 · MF1 0.5653)")
print("=" * 84)
a5, m5, mu5, wf5 = met(b5)
lo5, hi5 = wilson(int(round(a5 * len(b5))), len(b5))
print(f"  acc      = {a5:.4f}  [{lo5:.4f}, {hi5:.4f}]   fark {100*(a5-0.6285):+.2f} puan  <- TUTUYOR")
print(f"  MF1(/11) = {m5:.4f}   fark {100*(m5-0.5653):+.2f} puan  <- BUYUK FARK")
print(f"  MF1(union)={mu5:.4f}   wF1 = {wf5:.4f}")
print(f"  INVALID  = {int((b5['parsed_pred']=='INVALID').sum())}")

print("\n  --- 0.5653 nereden gelmis olabilir? ---")
print(f"     MF1(/11) x 11/12       = {m5*11/12:.4f}")
print(f"     MF1(/11) x 11/13       = {m5*11/13:.4f}")
print(f"     wF1                    = {wf5:.4f}")
print(f"     raporlanan             = 0.5653")

print("\n  --- SINIF BAZINDA F1 (B5) ---")
yt, yp = b5["label"].astype(str), b5["parsed_pred"].fillna("INVALID").astype(str)
P, R, F, S = precision_recall_fscore_support(yt, yp, labels=CLASSES, zero_division=0)
for c, p_, r_, f_, s_ in zip(CLASSES, P, R, F, S):
    print(f"     {c:<24s} P={p_:.3f} R={r_:.3f} F1={f_:.3f} N={s_}")
print(f"     {'MAKRO':<24s} {'':>13s} F1={F.mean():.4f}")

print("\n  --- Tahmin dagilimi (sinif cokusu var mi?) ---")
vc = yp.value_counts()
for k, v in vc.items():
    print(f"     {k:<24s} {v:>5d}  (%{100*v/len(b5):.1f})   gercek 150")

print("\n" + "=" * 84)
print("ESLESMIS TESTLER (McNemar)")
print("=" * 84)


def mcnemar(da, db, na, nb, ca="parsed_pred", cb="parsed_pred"):
    m = da[["note_id", "label", ca]].merge(db[["note_id", cb]], on="note_id",
                                           suffixes=("_a", "_b"))
    ka = m[f"{ca}_a"] == m["label"]
    kb = m[f"{cb}_b"] == m["label"]
    x, y = int((ka & ~kb).sum()), int((~ka & kb).sum())
    pv = binomtest(x, x + y, 0.5).pvalue if (x + y) else 1.0
    print(f"\n  {na}  vs  {nb}   (n={len(m)})")
    print(f"     yalniz {na}: {x}   yalniz {nb}: {y}")
    print(f"     p = {pv:.4g}  ->  {'ANLAMLI' if pv < 0.05 else 'anlamli DEGIL'}")
    return pv


comp = pd.read_csv(os.path.join(OUT, "b1_compressed_a100", "b1_compressed_test.csv"))
mcnemar(b2, comp, "TAM few-shot", "SIKISTIRILMIS")

qw = pd.read_csv(os.path.join(OUT, "qlora_eval_fullctx", "test_predictions.csv"))
qw = qw.rename(columns={"gold": "label", "pred": "parsed_pred"})
mcnemar(qw, b5, "Qwen QLoRA", "Gemma QLoRA adil")

zero = pd.read_csv(os.path.join(OUT, "zeroshot", "zeroshot_v4_test.csv"))
mcnemar(b2, zero, "TAM few-shot", "zero-shot")
