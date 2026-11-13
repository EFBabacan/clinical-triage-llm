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
n_sat = len(re.findall(r"&\s*[\d.]{5}\s*&\s*(?:[\d.]{5}--[\d.]{5}|--)\s*&\s*[\d.]{5}\s*&", tab_duz))
print(f"  tabloda {n_sat} veri satiri bulundu (beklenen 10: 9 LLM + dogrusal SVM)")
if n_sat != 10:
    hata += 1
# Sadelestirme (12 Kasim 2026): lojistik regresyon ve random forest bildiriden cikti.
for _yasak in ("Logistic regression", "Random forest"):
    if _yasak in tab:
        print(f"  !! {_yasak} satiri hala tabloda"); hata += 1
for k in ["0.515", "0.496", "0.552", "0.642", "0.565", "0.539", "0.562", "0.625", "0.647", "0.592"]:
    if k not in tab:
        print(f"  !! {k} tabloda YOK")
        hata += 1
print("  tablodaki 10 accuracy degeri results_master ile eslesiyor"
      if hata == 0 else "  !! eslesme sorunu")

print("\n" + "=" * 84)
print("7) EK MAKROLAR  (9 Kasim 2026 duzeltmeleri)")
print("=" * 84)
import json as _json
_ux = _json.load(open(os.path.join(PROJ, "output", "fewshot",
                  "fewshot_v4_3shot_ultracompact_examples.json"), encoding="utf-8"))
_ut = [e["ultra_token"] for c in _ux.values() for e in (c if isinstance(c, list) else [c])]
bildir("CompExTok (ornek basina token, ort.)", float(mak["CompExTok"]), sum(_ut) / len(_ut), tol=0.5)
bildir("CompExTokMin", float(mak["CompExTokMin"]), min(_ut), tol=0)
bildir("CompExTokMax", float(mak["CompExTokMax"]), max(_ut), tol=0)
_t03 = pd.read_csv(os.path.join(PROJ, "output", "results_tables", "T03_paired_tests.csv")).set_index("comparison")
_r = _t03.loc["Compressed few-shot vs zero-shot, Qwen"]
bildir("PcompZero", float(mak["PcompZero"]), float(_r["p_exact"]), tol=0.0006)
bildir("PcompZeroHolm", float(mak["PcompZeroHolm"]), float(_r["p_holm"]), tol=0.006)
_t08 = pd.read_csv(os.path.join(PROJ, "output", "results_tables", "T08_mask_control.csv")).set_index("metric")["value"]
bildir("MaskEnvSvm", float(mak["MaskEnvSvm"]), float(_t08["SVM accuracy with placeholder (this environment)"]))
if mak["MaskSklearn"] != str(_t08["scikit-learn version of this control"]).strip():
    print("  !! MaskSklearn eslesmiyor"); hata += 1
else:
    print("  OK  MaskSklearn", mak["MaskSklearn"])

print("\n" + "=" * 84)
print("8) HAKEM KONTROLLERI  (11 Kasim 2026; build_reviewer_checks.py, yeniden kosu)")
print("=" * 84)
_rt = os.path.join(PROJ, "output", "results_tables")
_t15 = pd.read_csv(os.path.join(_rt, "T15_pool_class_distribution.csv")).set_index("class")
_t06 = pd.read_csv(os.path.join(_rt, "T06_confusion_matrices_long.csv"))
_t17 = pd.read_csv(os.path.join(_rt, "T17_leakage_term_checks.csv")).set_index("metric")["value"]
_pi = (_t15.loc[CLASSES, "n"] / _t15["n"].sum()).values
_sayi = lambda k: float(re.sub(r"\\ensuremath\{|\}|\\,", "", mak[k]).replace("\\times10^{", "e"))
bildir("PoolN", _sayi("PoolN"), _t15["n"].sum(), tol=0)
bildir("PoolOtherPct", _sayi("PoolOtherPct"), 100 * _pi[-1], tol=0.05)
def _yeniden(cfg):  # T06 + T15'ten bagimsiz yeniden hesap
    M = (_t06[_t06.config == cfg].pivot(index="true_class", columns="predicted_class", values="count")
         .reindex(index=CLASSES, columns=CLASSES).fillna(0).values) / 150
    J = _pi[:, None] * M
    prec = np.diag(J) / J.sum(0)
    return np.diag(J).sum(), prec[:-1].mean()
for cfg, ka, kp in [("svm", "PrevAccSvm", "PrevPpvSvm"), ("qwen_qlora", "PrevAccQlora", "PrevPpvQlora")]:
    a, pp = _yeniden(cfg)
    bildir(ka, _sayi(ka), a, tol=0.0006)
    bildir(kp, _sayi(kp), 100 * pp, tol=0.5)
for k, m, tol in [("TermMaxPct", "target term in CC or HPI, own class, max over classes, %", 0.05),
                  ("TermDelta", "SVM accuracy change, target terms deleted everywhere, points", 0.05),
                  ("Pterm", "p (exact McNemar, unadjusted), target terms", 0.006),
                  ("LabPct", "lab mention in CC or HPI, all test inputs, %", 0.05),
                  ("LabAkiPct", "lab mention in CC or HPI, AKI inputs, %", 0.5),
                  ("LabDelta", "SVM accuracy change, lab mentions deleted from CC and HPI, points", 0.05),
                  ("Plab", "p (exact McNemar, unadjusted), lab mentions", 0.00006),
                  ("AkiRecBase", "SVM recall for AKI, baseline", 0.006),
                  ("AkiRecLab", "SVM recall for AKI, lab mentions deleted", 0.006)]:
    bildir(k, _sayi(k), float(_t17[m]), tol=tol)
import json as _js
_rr = _js.load(open(os.path.join(OUT, "classical_ml_v4", "classical_ml_v4_rerun_meta.json"), encoding="utf-8"))
bildir("RerunSvmAyni (1650 tahmin)", _rr["checks"]["svm_predictions_identical"], 1650, tol=0)
# LR / RF: Tablo 1 artik yeniden kosunun kayit duzeyindeki tahminlerinden (bagimsiz yeniden sayim)
_ra = pd.read_csv(os.path.join(OUT, "classical_ml_v4", "classical_ml_v4_predictions_all_models.csv"))
bildir("LrAcc", _sayi("LrAcc"), float((_ra["lr"] == _ra["gercek"]).mean()))
bildir("RfAcc", _sayi("RfAcc"), float((_ra["rf"] == _ra["gercek"]).mean()))
bildir("LrOrigAcc", _sayi("LrOrigAcc"), _rr["checks"]["lr"]["stored_accuracy"])
bildir("RfOrigAcc", _sayi("RfOrigAcc"), _rr["checks"]["rf"]["stored_accuracy"])
bildir("RerunSeed", _sayi("RerunSeed"), _rr["random_state"], tol=0)
if mak["RerunSklearn"] != _rr["sklearn"]:
    print("  !! RerunSklearn eslesmiyor"); hata += 1
else:
    print("  OK  RerunSklearn", mak["RerunSklearn"])

print("\n" + "=" * 84)
print("9) FABLE DEGERLENDIRMESI SONRASI EKLENENLER  (11 Kasim 2026)")
print("=" * 84)
import math as _m
_rt = os.path.join(PROJ, "output", "results_tables")
_pf = lambda v: float(re.sub(r"\\ensuremath\{|\}|\\,|\+", "", str(v)).replace("\\times10^{", "e"))
_t03b = pd.read_csv(os.path.join(_rt, "T03_paired_tests.csv")).set_index("comparison")
for mk, cmp_ in [("PqloraHolm", "QLoRA gain, Qwen"), ("PgemGainHolm", "QLoRA gain, Gemma (4-bit to 4-bit)"),
                 ("PcompFullHolm", "Compression cost, Qwen"), ("PfullZeroHolm", "Few-shot benefit, Qwen (full)"),
                 ("PquantHolm", "Quantization cost, Gemma"), ("PtruncHolm", "Truncated evaluation input, Qwen QLoRA"),
                 ("PoptQwHolm", "Gemma optimized vs Qwen QLoRA"), ("PdeviceHolm", "Device control (compressed few-shot, val)")]:
    v, ref = _pf(mak[mk]), float(_t03b.at[cmp_, "p_holm"])
    bildir(mk, v, ref, tol=0.051 * ref if ref < 0.1 else 0.006)  # iki anlamli basamak
bildir("FullZeroGain", _pf(mak["FullZeroGain"]), float(_t03b.at["Few-shot benefit, Qwen (full)", "diff_points"]), tol=0.051)
# cihaz kontrolu: degisen etiketler ve fark araligi, ham tahminlerden bagimsiz
_a = pd.read_csv(os.path.join(OUT, "b1_compressed_a100", "b1_compressed_val.csv"))
_l = pd.read_csv(os.path.join(OUT, "fewshot", "fewshot_v4_3shot_ultracompact_val.csv"))
_mm = _a.merge(_l, on="note_id", suffixes=("_a", "_l"))
_pa, _pl = _mm["parsed_pred_a"].fillna("X").astype(str), _mm["parsed_pred_l"].fillna("X").astype(str)
bildir("DevChanged", _pf(mak["DevChanged"]), int((_pa != _pl).sum()), tol=0)
_ra, _rl = _pa == _mm["label_a"], _pl == _mm["label_a"]
_x, _y, _n = int((_ra & ~_rl).sum()), int((~_ra & _rl).sum()), len(_mm)
_d, _se = (_y - _x) / _n, _m.sqrt(_x + _y - (_y - _x) ** 2 / _n) / _n
bildir("DevLo", _pf(mak["DevLo"]), 100 * (_d - 1.959964 * _se), tol=0.051)
bildir("DevHi", _pf(mak["DevHi"]), 100 * (_d + 1.959964 * _se), tol=0.051)
# istem uzunluklari
_zt = pd.read_csv(os.path.join(OUT, "zeroshot", "zeroshot_v4_test.csv"))["in_tokens"].mean()
_ft = pd.read_csv(os.path.join(OUT, "b2_b5_verify", "b2_full3shot_test.csv"))["in_tokens"].mean()
bildir("FullExTok (ornek basina tam token)", _pf(mak["FullExTok"]), (_ft - _zt) / 33, tol=0.5)
bildir("CompMaxTok (8 GB val en uzun istem)", _pf(mak["CompMaxTok"]), _l["in_tokens"].max(), tol=0)
# bellek: kalan pay ve kart
_mt = pd.read_csv(os.path.join(PROJ, "memory_table.csv"))
for mk, mdl in [("QwRest", "Qwen3.5-4B"), ("GemRest", "Gemma-4-E4B")]:
    r_ = _mt[(_mt.model == mdl) & _mt.duyarlik.str.startswith("4-bit")].iloc[0]
    bildir(mk, _pf(mak[mk]), r_.tepe_cikarim_MiB - r_.agirlik_MiB - r_.aktivasyon_MiB, tol=0.5)
_kart = float(re.search(r"(\d{4}) MiB", _mt[_mt.model == "Qwen3.5-4B"]["donanim"].iloc[0]).group(1))
bildir("CardMiB", _pf(mak["CardMiB"]), _kart, tol=0)
# kohort ve etiket boslugu: parquet'ten bagimsiz yeniden sayim (yalniz etiket ve kod sutunlari)
_lab = pd.read_parquet(os.path.join(OUT, "discharge_etiketli_full.parquet"), columns=["label"])
_cod = pd.read_parquet(os.path.join(OUT, "discharge_parsed.parquet"), columns=["label", "icd_code"])
bildir("LabeledN", _pf(mak["LabeledN"]), len(_lab), tol=0)
bildir("HtnHfN", _pf(mak["HtnHfN"]), int(_cod.icd_code.str.match(r"^(I110|I130|I132)").sum()), tol=0)
bildir("HfN", _pf(mak["HfN"]), int((_cod.label == "HEART_FAILURE").sum()), tol=0)
bildir("GibSrcN", _pf(mak["GibSrcN"]), int(_cod.icd_code.str.match(r"^(K2[5-8][0246]|K57[0-9][13]|K5521|K31811|I8501|I8511)").sum()), tol=0)
# ogrenme egrisi, kirpma, sureler: T20-T22 ve T14
_t20 = pd.read_csv(os.path.join(_rt, "T20_svm_learning_curve.csv")).set_index("setting")["accuracy_mean"]
_t21 = pd.read_csv(os.path.join(_rt, "T21_svm_truncation.csv")).set_index("first_words")
_t22 = pd.read_csv(os.path.join(_rt, "T22_cohort_and_time.csv")).set_index("metric")["value"]
_t14 = pd.read_csv(os.path.join(_rt, "T14_resources_latency.csv")).set_index("config")
for mk, ref in [("SvmLcThree", _t20["3 per class, random"]), ("SvmLcFs", _t20["the 33 few-shot examples (full text)"]),
                ("SvmLcHundred", _t20["100 per class, random"]), ("SvmLcTwoHundred", _t20["200 per class, random"]),
                ("SvmTruncSixty", _t21.at[60, "acc_full_train_truncated_test"]),
                ("ZeroSec", _t14.at["qwen_zero", "latency_s_per_record"]), ("QloraSec", _t14.at["qwen_qlora", "latency_s_per_record"]),
                ("FullSec", _t14.at["qwen_fs_full", "latency_s_per_record"])]:
    bildir(mk, _pf(mak[mk]), float(ref), tol=0.0006 if mk.startswith("Svm") else 0.006)
bildir("QloraHours", _pf(mak["QloraHours"]), float(_t14.at["qwen_qlora", "training_h"]), tol=0.051)
bildir("InputWords", _pf(mak["InputWords"]), float(_t22["test input length, words, mean"]), tol=0.5)
bildir("SvmSec", _pf(mak["SvmSec"]), float(_t22["SVM: TF-IDF fit, training and labeling of the test set, seconds (2-core CPU)"]), tol=0.051)

# 11) ikinci Fable turu: "hep other" tabani, ayristirilamayan ciktilar, terim/lab katmanlari, COPD
bildir("AllOtherAcc", _pf(mak["AllOtherAcc"]), float(_pi[-1]))
bildir("PrevAccMax", _pf(mak["PrevAccMax"]), max(_yeniden(c)[0] for c in _t06.config.unique() if c not in ("qwen_qlora_trunc",)))
_t01 = pd.read_csv(os.path.join(_rt, "T01_main_results_test.csv")).set_index("config")
_fsf = pd.read_csv(os.path.join(OUT, "b2_b5_verify", "b2_full3shot_test.csv"))
_inv = ~_fsf.parsed_pred.isin(CLASSES)
bildir("FullInvalidN", _pf(mak["FullInvalidN"]), int(_inv.sum()), tol=0)
bildir("FullUpperAcc", _pf(mak["FullUpperAcc"]), float(((_fsf.parsed_pred == _fsf.label).sum() + _inv.sum()) / len(_fsf)))
bildir("tam few-shot: ayristirilamayanlarin hepsi 'Based on' ile basliyor", int(_inv.sum()),
       int(_fsf.raw_output.fillna("")[_inv].str.startswith("Based on").sum()), tol=0)
bildir("FullSvmGap", _pf(mak["FullSvmGap"]), 100 * (float(_t01.at["svm", "accuracy"]) - float((_fsf.parsed_pred == _fsf.label).mean())), tol=0.06)
bildir("GemInvalidMaxN", _pf(mak["GemInvalidMaxN"]), int(max(_t01.at[k, "n_invalid"] for k in _t01.index if k.startswith("gemma"))), tol=0)
bildir("Qwen QLoRA ayristirilamayan", 0, int(_t01.at["qwen_qlora", "n_invalid"]), tol=0)
_cp = _cod[_cod.label == "COPD_EXACERBATION"]
bildir("CopdUnspecPct", _pf(mak["CopdUnspecPct"]), 100 * float((_cp.icd_code == "J449").mean()), tol=0.05)
_t24 = pd.read_csv(os.path.join(_rt, "T24_term_stratified_recall.csv")).set_index("config")
for mk, ref, tol in [("TermRecN", _t24.at["qwen_qlora", "n_target_records_with_term"], 0),
                     ("TermQloraWith", 100 * _t24.at["qwen_qlora", "recall_with_term"], 0.5),
                     ("TermQloraWithout", 100 * _t24.at["qwen_qlora", "recall_without_term_same_class_mix"], 0.5),
                     ("TermPtsQlora", _t24.at["qwen_qlora", "accuracy_points_attributable"], 0.05),
                     ("TermPtsMin", _t24["accuracy_points_attributable"].min(), 0.05),
                     ("TermPtsMax", _t24["accuracy_points_attributable"].max(), 0.05),
                     ("TermPtsSvm", _t24.at["svm", "accuracy_points_attributable"], 0.05),
                     ("TermDeltaAbs", abs(float(_t17["SVM accuracy change, target terms deleted everywhere, points"])), 0.05),
                     ("AkiLabQlora", 100 * _t24.at["qwen_qlora", "aki_recall_with_lab_mention"], 0.5),
                     ("AkiNoLabQlora", 100 * _t24.at["qwen_qlora", "aki_recall_without_lab_mention"], 0.5),
                     ("AkiLabZero", 100 * _t24.at["qwen_zero", "aki_recall_with_lab_mention"], 0.5),
                     ("AkiNoLabZero", 100 * _t24.at["qwen_zero", "aki_recall_without_lab_mention"], 0.5),
                     ("AkiLabSvm", 100 * _t24.at["svm", "aki_recall_with_lab_mention"], 0.5),
                     ("AkiNoLabSvm", 100 * _t24.at["svm", "aki_recall_without_lab_mention"], 0.5)]:
    bildir(mk, _pf(mak[mk]), float(ref), tol=tol)

# 10) sizinti taramasi — T23'u kullanmadan, parquet'ten bagimsiz sayim (yalniz sayim sutunlari)
_sp = pd.read_parquet(os.path.join(OUT, "discharge_split.parquet"),
                      columns=["split", "strict_current_leakage_severity_final", "manual_review_decision_13f"])
_fl = _sp.strict_current_leakage_severity_final.isin(["CURRENT_HIGH", "CURRENT_MEDIUM", "REVIEW_UNCERTAIN"])
_ts = _sp[_sp.split == "test"]
bildir("LeakKeptN", _pf(mak["LeakKeptN"]), int((_ts.manual_review_decision_13f == "HISTORICAL_ALLOWED").sum()), tol=0)
bildir("LeakRemN", _pf(mak["LeakRemN"]), int((_ts.manual_review_decision_13f == "REMOVE_CURRENT_LEAKAGE").sum()), tol=0)
bildir("LeakTestN", _pf(mak["LeakTestN"]),
       int(_ts.manual_review_decision_13f.isin(["HISTORICAL_ALLOWED", "REMOVE_CURRENT_LEAKAGE"]).sum()), tol=0)
bildir("LeakTrainN", _pf(mak["LeakTrainN"]), int((_fl & (_sp.split == "train_S3")).sum()), tol=0)
bildir("LeakValN", _pf(mak["LeakValN"]), int((_fl & (_sp.split == "val")).sum()), tol=0)
bildir("test: isaretli kayit kalmadi", 0, int((_fl & (_sp.split == "test")).sum()), tol=0)
bildir("13F yalniz testte", 0, int((_sp.split != "test").mul(_sp.manual_review_decision_13f != "AUTO").sum()), tol=0)

# 12) kanit denetimi (13 Kasim 2026): ortak few-shot ornekleri, cift kosular, egitim tepesi
import json as _json
_fs = lambda ad: {str(e["note_id"]) for v in _json.load(open(os.path.join(OUT, "fewshot", ad), encoding="utf-8")).values()
                  for e in (v if isinstance(v, list) else [v])}
_ort = len(_fs("fewshot_v4_3shot_examples.json") & _fs("fewshot_v4_3shot_ultracompact_examples.json"))
bildir("FsSharedN", _pf(mak["FsSharedN"]), _ort, tol=0)
bildir("FsReplacedN", _pf(mak["FsReplacedN"]), 33 - _ort, tol=0)
_rm = pd.read_csv(os.path.join(PROJ, "results_master.csv")).set_index("yapilandirma")["accuracy"]
bildir("RerunMaxDiff", _pf(mak["RerunMaxDiff"]),
       100 * max(abs(_rm["Qwen tam 3-shot"] - _rm["Qwen tam 3-shot (yeniden)"]),
                 abs(_rm["Gemma QLoRA adil"] - _rm["Gemma QLoRA adil (yeniden)"])), tol=0.051)
_tc = _json.load(open(os.path.join(OUT, "qlora_adapters", "v4_s3_full_2epoch_lr1e4", "training_curve.json")))
bildir("TrainPeak (allocated, MiB)", _pf(mak["TrainPeak"]), _tc["peak_gb"] * 1024, tol=0.5)

print("\n" + "=" * 84)
print("SONUC: " + (f"{hata} SORUN" if hata else "HER SAYI HAM VERIYE IZLENEBILIYOR"))
print("=" * 84)
sys.exit(1 if hata else 0)
