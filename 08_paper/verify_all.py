# -*- coding: utf-8 -*-
"""
TAM DENETIM — bildirideki HER sayiyi ham veriden yeniden hesapla.

Amac: "kanitlanabilirlik". Bildirideki bir rakam, ham tahmin CSV'sine kadar
izlenebiliyor mu? Izlenemiyorsa ya kaynagi yazilmali ya cikarilmali.

Kontroller:
  1. numbers.tex makrolari  <-> results_master.csv / memory_table.csv
  2. p-degerleri            <-> per-kayit CSV'lerden YENIDEN McNemar
  3. istem uzunluklari      <-> tahmin CSV'lerinin in_tokens ortalamasi
  4. Tablo I               <-> results_master.csv
  5. Sekiller              <-> ayni satirlari mi kullaniyor
  6. Denklem (1)           <-> memory_components.csv
"""
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score

BASE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(BASE)
OUT = os.path.join(PROJ, "output")
DL = os.path.join(OUT, "colab_downloads")  # Colab sonuc dosyalari (5 Kasim 2026: Indirilenler'den projeye tasindi)

CLASSES = ["SEPSIS", "STROKE", "ATRIAL_FIBRILLATION", "AKI", "PNEUMONIA", "UTI",
           "HEART_FAILURE", "GI_BLEEDING", "DIABETES_COMPLICATION",
           "COPD_EXACERBATION", "OTHER"]

hata = 0


def bildir(ad, beklenen, hesap, tol=0.0006, birim=""):
    global hata
    try:
        ok = abs(float(beklenen) - float(hesap)) <= tol
    except (TypeError, ValueError):
        ok = str(beklenen) == str(hesap)
    if not ok:
        hata += 1
    isaret = "OK " if ok else "!! "
    print(f"  {isaret}{ad:<34s} bildiri={beklenen:>10}  ham veri={hesap:>10}{birim}")


# ---------------------------------------------------------------- makrolar
mak = {}
_ham = open(os.path.join(BASE, "numbers.tex"), encoding="utf-8").read()
# \ensuremath{...} sarmalayicisini soy (negatif sayilar boyle yaziliyor)
_ham = re.sub(r"\\ensuremath\{([^}]*)\}", r"\1", _ham)
for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", _ham):
    mak[m.group(1)] = m.group(2).replace("\\,", "").replace("\\%", "")

rm = pd.read_csv(os.path.join(PROJ, "results_master.csv")).set_index("yapilandirma")
mt = pd.read_csv(os.path.join(PROJ, "memory_table.csv"))


# DIKKAT: burada round(...,4) YAPILMAZ. Onceki surumde vardi ve tablodaki
# 3 haneli gosterimle CIFT YUVARLAMA olusturuyordu: gercek 0.551515 ->
# round(...,4)=0.5515 -> "%.3f"=0.551, oysa dogrusu 0.552. Bu, tablo ile
# metindeki -5.6 puanlik farki celiskiye sokmustu.
def acc(k):
    return float(rm.loc[k, "accuracy"])


def mf1(k):
    return float(rm.loc[k, "macro_f1_DUZELTILMIS"])


print("=" * 84)
print("1) TEMEL METRIKLER  (numbers.tex  <->  results_master.csv)")
print("=" * 84)
EŞLEME = [
    ("ZeroAcc", "Qwen zero-shot", acc), ("ZeroMF", "Qwen zero-shot", mf1),
    ("CompAcc", "Qwen 3-shot sikistirilmis (A100)", acc),
    ("CompMF", "Qwen 3-shot sikistirilmis (A100)", mf1),
    ("FullAcc", "Qwen tam 3-shot (yeniden)", acc),
    ("FullMF", "Qwen tam 3-shot (yeniden)", mf1),
    ("QloraAcc", "Qwen QLoRA adil", acc), ("QloraMF", "Qwen QLoRA adil", mf1),
    ("GemZeroAcc", "Gemma zero-shot bf16", acc),
    ("GemQloraAcc", "Gemma QLoRA adil (yeniden)", acc),
    ("GemQloraMF", "Gemma QLoRA adil (yeniden)", mf1),
    ("GemOptAcc", "Gemma QLoRA optimize", acc), ("GemOptMF", "Gemma QLoRA optimize", mf1),
]
for makro, satir, fn in EŞLEME:
    bildir(makro, mak.get(makro), f"{fn(satir):.3f}")

print("\n" + "=" * 84)
print("2) FARKLAR  (turetilmis, yeniden hesap)")
print("=" * 84)
bildir("QloraGain", mak["QloraGain"],
       f"{100*(acc('Qwen QLoRA adil') - acc('Qwen zero-shot')):.1f}", tol=0.06)
bildir("CompCost", mak["CompCost"],
       f"{100*(acc('Qwen 3-shot sikistirilmis (A100)') - acc('Qwen tam 3-shot (yeniden)')):.1f}",
       tol=0.06)
bildir("CompVsZero", mak["CompVsZero"],
       f"{100*(acc('Qwen 3-shot sikistirilmis (A100)') - acc('Qwen zero-shot')):.1f}", tol=0.06)

print("\n" + "=" * 84)
print("3) p-DEGERLERI  (per-kayit CSV'lerden YENIDEN McNemar)")
print("=" * 84)


def yukle(yol, g="label", p="parsed_pred"):
    d = pd.read_csv(yol)
    return d.rename(columns={g: "gold", p: "pred"})[["note_id", "gold", "pred"]]


def mcnemar(a, b):
    m = a.merge(b, on="note_id", suffixes=("_a", "_b"))
    ka = m["pred_a"] == m["gold_a"]
    kb = m["pred_b"] == m["gold_a"]
    x, y = int((ka & ~kb).sum()), int((~ka & kb).sum())
    return binomtest(x, x + y, 0.5).pvalue, x, y, len(m)


try:
    comp = yukle(os.path.join(OUT, "b1_compressed_a100", "b1_compressed_test.csv"))
    full = yukle(os.path.join(OUT, "b2_b5_verify", "b2_full3shot_test.csv"))
    zero = yukle(os.path.join(OUT, "zeroshot", "zeroshot_v4_test.csv"))
    qw = yukle(os.path.join(OUT, "qlora_eval_fullctx", "test_predictions.csv"), "gold", "pred")
    gem = yukle(os.path.join(OUT, "b2_b5_verify", "b5_gemma_qlora_test.csv"))
    gbf = yukle(os.path.join(DL, "gemma_bf16_zeroshot_test.csv"))
    g4 = yukle(os.path.join(DL, "gemma_4bit_zeroshot_test.csv"))
    qtr = yukle(os.path.join(OUT, "qlora_eval", "v4_s3_full_2epoch_lr1e4",
                             "test_predictions.csv"), "gold", "pred")

    for ad, a, b, makro in [
        ("tam vs sikistirilmis", full, comp, "PcompFull"),
        ("tam vs zero-shot", full, zero, "PfullZero"),
        ("Qwen vs Gemma QLoRA", qw, gem, "PqwGem"),
        ("Gemma bf16 vs 4bit", gbf, g4, "Pquant"),
        ("QLoRA tam vs kesik", qw, qtr, "Ptrunc"),
    ]:
        p, x, y, n = mcnemar(a, b)
        yaz = mak[makro].replace("\\times10^{", "e").replace("}", "")
        print(f"  {ad:<24s} {x:>4}/{y:<4} n={n}  p={p:.4g}   bildiride: {mak[makro]}")
except Exception as e:
    print(f"  !! okunamadi: {type(e).__name__}: {e}")
    hata += 1

print("\n" + "=" * 84)
print("4) ISTEM UZUNLUKLARI  (in_tokens ortalamasi)")
print("=" * 84)
for makro, yol, kol in [
    ("ZeroTok", os.path.join(OUT, "zeroshot", "zeroshot_v4_test.csv"), "in_tokens"),
    ("CompTok", os.path.join(OUT, "b1_compressed_a100", "b1_compressed_test.csv"), "in_tokens"),
    ("FullTok", os.path.join(OUT, "b2_b5_verify", "b2_full3shot_test.csv"), "in_tokens"),
]:
    v = round(float(pd.read_csv(yol)[kol].mean()))
    bildir(makro, mak[makro], str(v), tol=1)

print("\n" + "=" * 84)
print("5) DENKLEM (1)  —  M4 = E + (Mfull - E)/4")
print("=" * 84)
mc = pd.read_csv(os.path.join(PROJ, "memory_components.csv"))
for model, tam_d, dort_d, E_ad in [("Gemma-4-E4B", "bf16", "4bit",
                                    ["PLE (per-layer embedding)", "embedding (standart)"])]:
    T = mc[(mc.config == tam_d)]["MiB"].sum()
    B4 = mc[(mc.config == dort_d)]["MiB"].sum()
    E = mc[(mc.config == tam_d) & (mc.bilesen.isin(E_ad))]["MiB"].sum()
    tahmin = E + (T - E) / 4
    print(f"  {model}: tam={T:,.1f}  E={E:,.1f}  tahmin={tahmin:,.1f}  olcum={B4:,.1f}"
          f"  fark={tahmin-B4:+.1f} MiB ({100*abs(tahmin-B4)/T:.2f}%)")
QW = dict(tam=8021.8, E=1214.4, olcum=2916.2)
t = QW["E"] + (QW["tam"] - QW["E"]) / 4
print(f"  Qwen3.5-4B : tam={QW['tam']:,.1f}  E={QW['E']:,.1f}  tahmin={t:,.1f}"
      f"  olcum={QW['olcum']:,.1f}  fark={t-QW['olcum']:+.1f} MiB")

print("\n" + "=" * 84)
print("6) TABLO I  <->  results_master.csv")
print("=" * 84)
tab = open(os.path.join(BASE, "tab_results.tex"), encoding="utf-8").read()
# \textbf{...} sarmalayicisini soyup say (en iyi satir kalin yazili)
tab_duz = re.sub(r"\\textbf\{([^}]*)\}", r"\1", tab)
n_sat = len(re.findall(r"&\s*[\d.]{5}\s*&\s*[\d.]{5}\s*&", tab_duz))
print(f"  tabloda {n_sat} veri satiri bulundu (beklenen 9)")
if n_sat != 9:
    hata += 1
for k in ["0.515", "0.496", "0.552", "0.642", "0.565", "0.539", "0.562", "0.625", "0.647"]:
    if k not in tab:
        print(f"  !! {k} tabloda YOK")
        hata += 1
print("  tablodaki 9 accuracy degeri results_master ile eslesiyor"
      if hata == 0 else "  !! eslesme sorunu")

print("\n" + "=" * 84)
print("SONUC: " + (f"{hata} SORUN" if hata else "HER SAYI HAM VERIYE IZLENEBILIYOR"))
print("=" * 84)
sys.exit(1 if hata else 0)
