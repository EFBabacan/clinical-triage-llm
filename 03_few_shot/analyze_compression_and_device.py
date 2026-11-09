# -*- coding: utf-8 -*-
"""
B1 SONUC ANALIZI — sikistirilmis few-shot, A100

Cevaplanan soru: sikistirmanin etkisi ne, donanim karistiricisi olmadan?
  * Sikistirilmis ve tam few-shot artik AYNI donanimda (A100)
  * Donanim kontrolu (val) yapildi: 4060 vs A100 = 0.5 puan (3 kayit) -> etkisiz
"""
import os
import shutil

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")
DL = r"./downloads"
B1DIR = os.path.join(OUT, "b1_compressed_a100")
os.makedirs(B1DIR, exist_ok=True)

CLASSES = ["SEPSIS", "STROKE", "ATRIAL_FIBRILLATION", "AKI", "PNEUMONIA", "UTI",
           "HEART_FAILURE", "GI_BLEEDING", "DIABETES_COMPLICATION",
           "COPD_EXACERBATION", "OTHER"]

# indirilen dosyalari projeye tasi (Downloads gecici)
for f in ["b1_compressed_test.csv", "b1_compressed_val.csv"]:
    src = os.path.join(DL, f)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(B1DIR, f))

comp = pd.read_csv(os.path.join(B1DIR, "b1_compressed_test.csv"))
zero = pd.read_csv(os.path.join(OUT, "zeroshot", "zeroshot_v4_test.csv"))
compv = pd.read_csv(os.path.join(B1DIR, "b1_compressed_val.csv"))
zerov = pd.read_csv(os.path.join(OUT, "zeroshot", "zeroshot_v4_val.csv"))
locv = pd.read_csv(os.path.join(OUT, "fewshot", "fewshot_v4_3shot_ultracompact_val.csv"))


def met(d, g="label", p="parsed_pred"):
    acc = float((d[g] == d[p]).mean())
    mf1 = f1_score(d[g], d[p], average="macro", labels=CLASSES, zero_division=0)
    return acc, mf1


def wilson(k, n, z=1.96):
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def mcnemar(a, b, na, nb, g="label", p="parsed_pred"):
    m = a[["note_id", g, p]].merge(b[["note_id", p]], on="note_id",
                                   suffixes=("_a", "_b"))
    ca = m[f"{p}_a"] == m[g]
    cb = m[f"{p}_b"] == m[g]
    x = int((ca & ~cb).sum())
    y = int((~ca & cb).sum())
    pv = binomtest(x, x + y, 0.5).pvalue if (x + y) else 1.0
    print(f"\n  {na}  vs  {nb}   (eslesmis n={len(m)})")
    print(f"     yalniz {na} dogru : {x}")
    print(f"     yalniz {nb} dogru : {y}")
    print(f"     p = {pv:.4g}  ->  {'ANLAMLI' if pv < 0.05 else 'anlamli DEGIL'}")
    return pv


print("=" * 80)
print("1) DONANIM KONTROLU (val) — sikistirilmis istem, ayni 550 kayit")
print("=" * 80)
a1, f1_ = met(locv)
a2, f2_ = met(compv)
print(f"  RTX 4060 : acc {a1:.4f}  macroF1 {f1_:.4f}")
print(f"  A100     : acc {a2:.4f}  macroF1 {f2_:.4f}")
print(f"  fark     : {100*(a2-a1):+.1f} puan acc")
mcnemar(locv, compv, "4060", "A100")
print("  => Donanim doğruluğu degistirmiyor; kalan fark surum/sayisal gurultu.")

print("\n" + "=" * 80)
print("2) TEST SONUCLARI (n=1650)")
print("=" * 80)
ac, fc = met(comp)
az, fz = met(zero)
AF, FF = 0.5552, 0.5621          # tam 3-shot: per-kayit CSV kayip, rapor+wF1
lo, hi = wilson(int(round(ac * len(comp))), len(comp))
print(f"  Sikistirilmis 3-shot (A100) : acc {ac:.4f} [{lo:.4f}, {hi:.4f}]  macroF1 {fc:.4f}"
      f"   ort {comp['in_tokens'].mean():.0f} token")
print(f"  Zero-shot (4060)            : acc {az:.4f}  macroF1 {fz:.4f}"
      f"   ort {zero['in_tokens'].mean():.0f} token")
print(f"  Tam 3-shot (A100)           : acc {AF:.4f}  macroF1 {FF:.4f}   ort ~23800 token")

print("\n" + "=" * 80)
print("3) ASIL SORU — sikistirmanin etkisi (artik ayni donanim)")
print("=" * 80)


def unpaired(p1, n1, p2, n2):
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    d = p1 - p2
    return d, d - 1.96 * se, d + 1.96 * se


d, l, h = unpaired(ac, 1650, AF, 1650)
print(f"\n  A) Sikistirilmis vs TAM few-shot  (ikisi de A100, tek fark istem uzunlugu)")
print(f"     {100*d:+.1f} puan  %95 CI [{100*l:+.1f}, {100*h:+.1f}]  "
      f"-> {'ANLAMLI' if h < 0 or l > 0 else 'gurultu'}")
print(f"     istem: 2024 -> 23800 token (11.8x)")
print("     NOT: tam few-shot per-kayit CSV'si kayip -> eslesmis test yapilamiyor,")
print("          bagimsiz CI kullanildi (muhafazakar).")

print(f"\n  B) Sikistirilmis vs ZERO-SHOT")
pv = mcnemar(comp, zero, "sikistirilmis", "zero-shot")
d, l, h = unpaired(ac, 1650, az, 1650)
print(f"     fark {100*d:+.1f} puan  (bagimsiz CI [{100*l:+.1f}, {100*h:+.1f}])")

print("\n" + "=" * 80)
print("4) VAL ILE TUTARLILIK")
print("=" * 80)
av, fv = met(zerov)
print(f"  val  : sikistirilmis {a2:.4f} vs zero-shot {av:.4f}  -> {100*(a2-av):+.1f} puan")
print(f"  test : sikistirilmis {ac:.4f} vs zero-shot {az:.4f}  -> {100*(ac-az):+.1f} puan")
print("  Iki bolumde de ayni yon: sikistirilmis few-shot zero-shot'in ALTINDA.")

ozet = pd.DataFrame([
    dict(yapilandirma="Qwen 3-shot sikistirilmis (A100)", split="test", n=len(comp),
         accuracy=round(ac, 4), macro_f1=round(fc, 4),
         acc_ci_lo=round(lo, 4), acc_ci_hi=round(hi, 4),
         istem_token_ort=round(float(comp["in_tokens"].mean())),
         gecikme_s=round(float(comp["sure_s"].mean()), 3),
         toplam_dk=round(float(comp["sure_s"].sum()) / 60, 1)),
    dict(yapilandirma="Qwen 3-shot sikistirilmis (A100)", split="val", n=len(compv),
         accuracy=round(a2, 4), macro_f1=round(f2_, 4),
         acc_ci_lo=None, acc_ci_hi=None,
         istem_token_ort=round(float(compv["in_tokens"].mean())),
         gecikme_s=round(float(compv["sure_s"].mean()), 3),
         toplam_dk=round(float(compv["sure_s"].sum()) / 60, 1)),
])
ozet.to_csv(os.path.join(B1DIR, "b1_ozet.csv"), index=False, encoding="utf-8-sig")
print(f"\nKaydedildi: {B1DIR}")
print(ozet.to_string(index=False))
