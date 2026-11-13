# -*- coding: utf-8 -*-
"""
C1/C4 — BILDIRI TABLOLARINI DOGRULANMIS CSV'LERDEN URET

Kural: bildiriye elle rakam YAZILMAZ. Bu script results_master.csv ve
memory_table.csv'den LaTeX tablolari uretir; bir sayi degisirse script
tekrar kosturulur ve bildiri kendini gunceller.

Cikti:  tab_results.tex · tab_resources.tex  (main.tex bunlari \input eder)
"""
import math
import json
import os
import re

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(BASE)

rm = pd.read_csv(os.path.join(PROJ, "results_master.csv")).set_index("yapilandirma")
mt = pd.read_csv(os.path.join(PROJ, "memory_table.csv"))
rt = pd.read_csv(os.path.join(PROJ, "resource_table.csv"))

# ---------------------------------------------------------------- toplu sonuc tablolari
# build_results_tables.py ciktisi: per-kayit tahminlerden, results_master ile birebir.
RTD = os.path.join(PROJ, "output", "results_tables")
_t = lambda ad: pd.read_csv(os.path.join(RTD, ad))
T01 = _t("T01_main_results_test.csv").set_index("config")
T03 = _t("T03_paired_tests.csv").set_index("comparison")
T04 = _t("T04_per_class_f1_test.csv").set_index("class")
T06 = _t("T06_confusion_matrices_long.csv")
T08 = _t("T08_mask_control.csv").set_index("metric")["value"]
T09 = _t("T09_error_overlap.csv").set_index("metric")["value"]


def accx(k):
    """T01 dogrulugu, dogru sayisindan (cift yuvarlama olmasin)."""
    return round(float(T01.at[k, "accuracy"]) * 1650) / 1650


def tok_mean(rel, col="in_tokens"):
    return float(pd.read_csv(os.path.join(PROJ, "output", rel))[col].mean())


def g(ad, kol):
    return rm.loc[ad, kol]


# ---------------------------------------------------------------- Tablo I
# Kaynak: T01. Iki blok: 8 GB butcesi icinde kosanlar / 80 GB A100 gerektirenler.
# (model, yontem, bit, T01 anahtari). Sikistirilmis istem 8 GB'a sigiyor; test kosusu
# tam istemle ayni cihazi paylassin diye A100'de yapildi (dipnot).
BLOKLAR = [
    (r"Runs within the 8\,GB budget", [
        ("Qwen3.5-4B", "Zero-shot", "4-bit", "qwen_zero"),
        ("Qwen3.5-4B", "Few-shot, compressed", "4-bit", "qwen_fs_comp"),
        ("Qwen3.5-4B", "QLoRA (shared)", "4-bit", "qwen_qlora"),
        ("TF-IDF", "Linear SVM", "--", "svm"),
    ]),
    (r"Needs the 80\,GB A100", [
        ("Qwen3.5-4B", "Few-shot, full", "4-bit", "qwen_fs_full"),
        ("Gemma-4-E4B", "Zero-shot", "bf16", "gemma_zero_bf16"),
        ("Gemma-4-E4B", "Zero-shot", "4-bit", "gemma_zero_4bit"),
        ("Gemma-4-E4B", "Few-shot, full", "bf16", "gemma_fs_full"),
        ("Gemma-4-E4B", "QLoRA (shared)", "4-bit", "gemma_qlora_fair"),
        ("Gemma-4-E4B", "QLoRA (high-capacity)", "4-bit", "gemma_qlora_opt"),
    ]),
]
SATIRLAR = [(m, y, b, k, blok.startswith("Runs")) for blok, satirlar in BLOKLAR for m, y, b, k in satirlar]
# blok tanimi olculen sigma durumuyla ayni olmali (T01, fits_8gb)
for _m, _y, _b, _k, _sigar in SATIRLAR:
    assert str(T01.at[_k, "fits_8gb"]).startswith("yes") == _sigar, f"blok/sigma uyusmuyor: {_k}"


def aralik(lo, hi):
    return "--" if pd.isna(lo) else f"{lo:.3f}--{hi:.3f}"


lines = [
    r"% OTOMATIK URETILDI — make_tables.py. ELLE DUZENLEME.",
    r"\begin{table*}[t]",
    r"\caption{Test set performance ($n=1650$, 150 per class) with 95\% intervals",
    r"(Wilson for accuracy, bootstrap for macro-F1); macro-F1 over the 11 target",
    r"classes. The best accuracy and macro-F1 of each block are in bold. \emph{Shared} is",
    r"the adapter configuration held identical across models; \emph{high-capacity} is a larger",
    r"adapter trained for Gemma only. Run on: the device that produced the predictions.",
    r"Unpars.: outputs the parser could not map to a class, in percent.",
    r"$^{*}$The compressed prompt fits in 8\,GB (its longest instance, \CompMaxTok{} tokens, ran",
    r"on the RTX 4060); its test run used the A100 so that it shares a device with the full prompt.}",
    r"\label{tab:results}",
    r"\centering",
    r"\small",
    r"\setlength{\tabcolsep}{4pt}",
    r"\begin{tabular}{@{}llccrcrcr@{}}",
    r"\toprule",
    r"Model & Method & Bits & Run on & Acc. & 95\% CI & Macro-F1 & 95\% CI & Unpars. \\",
]
_keys = [k for *_, k, _ in SATIRLAR]
en_iyi_acc = max(accx(k) for k in _keys)
en_iyi_mf1 = max(float(T01.at[k, "macro_f1"]) for k in _keys)
for blok, satirlar in BLOKLAR:
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{9}{@{}l}{\emph{" + blok + r"}} \\")
    b_acc = max(accx(k) for *_, k in satirlar)
    b_mf1 = max(float(T01.at[k, "macro_f1"]) for *_, k in satirlar)
    prev = None
    for model, yontem, bit, k in satirlar:
        r = T01.loc[k]
        a_s, m_s = f"{accx(k):.3f}", f"{r.macro_f1:.3f}"
        if abs(accx(k) - b_acc) < 1e-9:
            a_s = f"\\textbf{{{a_s}}}"
        if abs(r.macro_f1 - b_mf1) < 1e-9:
            m_s = f"\\textbf{{{m_s}}}"
        md = model if model != prev else ""
        prev = model
        cihaz = {"RTX 4060 8 GB": "RTX 4060", "A100 80 GB": "A100", "CPU": "CPU"}[r.hardware]
        if k == "qwen_fs_comp":
            assert cihaz == "A100"
            cihaz = "A100$^{*}$"
        unp = "--" if model.startswith("TF-IDF") else f"{100 * r.n_invalid / r.n:.1f}"
        lines.append(f"{md} & {yontem} & {bit} & {cihaz} & {a_s} & {aralik(r.acc_ci_low, r.acc_ci_high)} & {m_s} & "
                     f"{aralik(r.mf1_ci_low, r.mf1_ci_high)} & {unp} " + r"\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
with open(os.path.join(BASE, "tab_results.tex"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

# ---------------------------------------------------------------- Tablo II
def mem(model, duy, kol):
    s = mt[(mt["model"] == model) & (mt["duyarlik"].str.startswith(duy))]
    return None if s.empty or pd.isna(s.iloc[0][kol]) else float(s.iloc[0][kol])


def fmt(v, birim=""):
    return "--" if v is None else f"{v:,.0f}{birim}".replace(",", "\\,")


QW_FULL = mem("Qwen3.5-4B", "fp16", "agirlik_MiB")
QW_4 = mem("Qwen3.5-4B", "4-bit", "agirlik_MiB")
QW_4T = mem("Qwen3.5-4B", "4-bit", "tepe_cikarim_MiB")
GE_FULL = mem("Gemma-4-E4B", "bf16", "agirlik_MiB")
GE_4 = mem("Gemma-4-E4B", "4-bit", "agirlik_MiB")
# 8 GB kartin CUDA'nin bildirdigi toplam bellegi (memory_table donanim sutunu); Sekil 2 de bunu kullaniyor
CARD = int(re.search(r"(\d{4}) MiB", mt[mt.model == "Qwen3.5-4B"]["donanim"].iloc[0]).group(1))
GE_4T = mem("Gemma-4-E4B", "4-bit", "tepe_cikarim_MiB")
QW_PAY = mem("Qwen3.5-4B", "fp16", "nicelenmeyen_pay_%")
GE_PAY = mem("Gemma-4-E4B", "bf16", "nicelenmeyen_pay_%")
QW_KAZ = mem("Qwen3.5-4B", "4-bit", "niceleme_kazanci_%")
GE_KAZ = mem("Gemma-4-E4B", "4-bit", "niceleme_kazanci_%")
QW_E = mem("Qwen3.5-4B", "fp16", "nicelenmeyen_MiB")
GE_E = mem("Gemma-4-E4B", "bf16", "nicelenmeyen_MiB")
# QLoRA egitim tepesi: training_curve.json "peak_gb" = torch.cuda.max_memory_allocated()/1024**3
# (ALLOCATED; egitimde reserved kaydedilmedi). Tablo 2 basligi ve 4.3 bunu "allocated" diye yaziyor.
QLORA_TEPE = float(round(json.load(open(os.path.join(PROJ, "output", "qlora_adapters", "v4_s3_full_2epoch_lr1e4",
                                                   "training_curve.json")))["peak_gb"] * 1024))
assert QLORA_TEPE == 7055.0, QLORA_TEPE

# Denklem (1) artigi: M4 = E + (Mfull - E)/4 tahmini ile OLCUM arasindaki fark.
# BU TABLODAN turetiliyor, cunku hakem elinde yalnizca bu tablo var; ayni
# sayilari kullanmazsak okur hesabi tutturamaz. (memory_components.csv bilesen
# yuvarlamasi yuzunden 0.1 MiB kadar farkli cikar — kaynak karistirmayin.)
_mc = pd.read_csv(os.path.join(PROJ, "memory_components.csv"))
GE_VA = float(_mc[(_mc.config == "4bit") & (_mc.bilesen == "vision/audio")]["MiB"].iloc[0])

QW_FULL_TOK = float(pd.read_csv(os.path.join(
    PROJ, "output", "qlora_eval_fullctx", "test_predictions.csv"))["input_tokens"].mean())
QW_TRUNC_TOK = float(pd.read_csv(os.path.join(
    PROJ, "output", "qlora_eval", "v4_s3_full_2epoch_lr1e4",
    "test_predictions.csv"))["input_tokens"].mean())

# ARTIK, TABLODA BASILAN yuvarlanmis degerlerden hesaplanir. Ham degerlerden
# hesaplayinca 0.05 ve 23.70 cikiyordu; okur tabloyu alip ayni islemi yapinca
# 0.0 ve 23.5 buluyor ve sayilar tutmuyordu. (Ayni kural GemNoVA/QwNonEmb icin
# de gecerli — turev sayilar okurun gorebildigi degerlerden uretilir.)
QW_ARTIK = abs(round(QW_E) + (round(QW_FULL) - round(QW_E)) / 4 - round(QW_4))
GE_ARTIK = abs(round(GE_E) + (round(GE_FULL) - round(GE_E)) / 4 - round(GE_4))

t2 = [
    r"% OTOMATIK URETILDI — make_tables.py",
    r"\begin{table}[t]",
    # Baslik "in MiB" diyordu ama pay/kazanc satirlari YUZDE, son satir evet/hayir.
    # Birim satir etiketlerine tasindi; baslik artik yanlis vaat etmiyor.
    # Baslik "iki blok" diyordu ama tabloda UC blok var, ve "ilk blok
    # parametre boyutu" dedigi blogun iki satiri yuzde. Blok anlatimi yerine
    # satir birimlerine gore tarif ediliyor.
    r"\caption{Measured GPU memory, for Qwen on the RTX 4060 and for Gemma on the A100. Inference rows",
    r"use batch size 1 and 2048-token prompts; the",
    r"training peak comes from the QLoRA training run itself (sequences of at most 512 tokens,",
    r"batch 1 with gradient accumulation over 8 steps).",
    r"Weight rows are parameter sizes, which leave out",
    r"quantization constants; the share row is the fraction of the 16-bit weights",
    r"(fp16 for Qwen, bf16 for Gemma) that quantization cannot reach. Peak inference figures are",
    r"\emph{reserved} memory, which also includes quantization constants, activations, the key-value",
    r"cache, and allocator overhead. The training peak is \emph{allocated} memory, as reserved memory was",
    r"not recorded during training. Gemma exceeds the budget already at",
    r"inference, so no training peak on the target device exists to report (--).}",
    r"\label{tab:memory}",
    r"\centering",
    r"\setlength{\tabcolsep}{4pt}",
    r"\begin{tabular}{lrr}",
    r"\toprule",
    r" & Qwen3.5-4B & Gemma-4-E4B \\",
    r"\midrule",
    # "full precision" YANLISTI: bf16 kullaniyoruz, full precision fp32'dir.
    # Denklemdeki bolen 4 zaten 16-bit tabana dayaniyor.
    f"Weights, 16-bit (MiB) & {fmt(QW_FULL)} & {fmt(GE_FULL)} \\\\",
    # E satiri EKLENDI: metin "Taking E from Table II" diyordu ama 1,214 ve
    # 6,656 tabloda yoktu; okur sayiyi bulamiyordu. Ayrica %15/%44 payinin
    # neyin payi oldugunu da bu satir acikliyor.
    f"\\textbf{{Embedding tables (MiB)}} & \\textbf{{{fmt(QW_E)}}} & \\textbf{{{fmt(GE_E)}}} \\\\",
    f"Weights, 4-bit NF4 (MiB) & {fmt(QW_4)} & {fmt(GE_4)} \\\\",
    f"Embedding share of 16-bit & {QW_PAY:.0f}\\% & {GE_PAY:.0f}\\% \\\\",
    f"\\textbf{{Quantization saving}} & \\textbf{{{QW_KAZ:.1f}\\%}} & \\textbf{{{GE_KAZ:.1f}\\%}} \\\\",
    r"\midrule",
    f"Peak inference, 4-bit (MiB) & {fmt(QW_4T)} & {fmt(GE_4T)} \\\\",
    f"Peak QLoRA training (MiB) & {fmt(QLORA_TEPE)} & -- \\\\",
    r"\midrule",
    r"\textbf{Fits in 8\,GB (4-bit)} & \textbf{yes} & \textbf{no} \\",
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
]
with open(os.path.join(BASE, "tab_resources.tex"), "w", encoding="utf-8") as f:
    f.write("\n".join(t2) + "\n")

# ---------------------------------------------------------------- Tablo III (ek)
AD = {"qwen_zero": "Qwen, zero-shot", "qwen_fs_comp": "Qwen, few-shot compressed",
      "qwen_fs_full": "Qwen, few-shot full", "qwen_qlora": "Qwen, QLoRA (shared)",
      "qwen_qlora_trunc": "Qwen, QLoRA, truncated input", "gemma_zero_bf16": "Gemma, zero-shot bf16",
      "gemma_zero_4bit": "Gemma, zero-shot 4-bit", "gemma_fs_full": "Gemma, few-shot full",
      "gemma_qlora_fair": "Gemma, QLoRA (shared)", "gemma_qlora_opt": "Gemma, QLoRA (high-capacity)",
      "svm": "TF-IDF linear SVM", "qwen_fs_comp_val_4060": "Qwen, compressed, 8\\,GB device$^\\ast$",
      "qwen_fs_comp_val_a100": "same, A100$^\\ast$"}


def p_tex(p, ondalik=None):
    """Tek kural, tablo ve metin icin: p<0.001 bilimsel; p<0.1 iki anlamli basamak; aksi 2 ondalik."""
    p = float(p)
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e")
        return f"{m}\\times10^{{{int(e)}}}"
    if ondalik is None:
        ondalik = 1 - math.floor(math.log10(p)) if p < 0.1 else 2
    return f"{p:.{ondalik}f}"


def isaretli(v):
    return f"$-${abs(v):.1f}" if v < 0 else f"$+${v:.1f}"


t3 = [r"% OTOMATIK URETILDI — make_tables.py",
      r"\begin{table*}[!htb]",
      r"\caption{Every paired comparison between configurations reported in this paper",
      r"(exact McNemar test). $\Delta$: accuracy",
      r"of A minus B in points. A only, B only: records that only A, or only B, classifies",
      r"correctly. $p_{\mathrm{Holm}}$: Holm-adjusted over all rows. $^\ast$Validation",
      r"split (550 records); every other row uses the 1650 test records.}",
      r"\label{tab:tests}",
      r"\centering",
      r"\small",
      r"\setlength{\tabcolsep}{4pt}",
      r"\begin{tabular}{@{}llrrrrr@{}}",
      r"\toprule",
      r"A & B & $\Delta$ & A only & B only & $p$ & $p_{\mathrm{Holm}}$ \\",
      r"\midrule"]
for _, r in T03.reset_index().iterrows():
    t3.append(f"{AD[r.a]} & {AD[r.b]} & {isaretli(r.diff_points)} & {int(r.a_right_b_wrong)} & "
              f"{int(r.a_wrong_b_right)} & ${p_tex(r.p_exact)}$ & ${p_tex(r.p_holm)}$ " + r"\\")
t3 += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
with open(os.path.join(BASE, "tab_tests.tex"), "w", encoding="utf-8") as f:
    f.write("\n".join(t3) + "\n")

# ---------------------------------------------------------------- ornek uzunlugu
# Dondurulmus sikistirilmis istem bu JSON'dan kuruluyor (freeze_compressed_prompt.py).
_ux = json.load(open(os.path.join(PROJ, "output", "fewshot",
                                  "fewshot_v4_3shot_ultracompact_examples.json"), encoding="utf-8"))
_ut = [e["ultra_token"] for c in _ux.values() for e in (c if isinstance(c, list) else [c])]
assert len(_ut) == 33, len(_ut)
ORNEK_TOK_ORT, ORNEK_TOK_MIN, ORNEK_TOK_MAX = sum(_ut) / len(_ut), min(_ut), max(_ut)

# ---------------------------------------------------------------- numbers.tex
# Metinde de elle rakam yazilmasin: her sayi bir makro.
def acc(k):
    return float(g(k, "accuracy"))


def mf1(k):
    return float(g(k, "macro_f1_DUZELTILMIS"))


def _dev():
    """Cihaz kontrolu: ayni istem 8 GB cihazda ve A100'de, 550 dogrulama kaydi.
    Discordant ciftleri ve McNemar p'sini HAM tahminlerden hesaplar."""
    from scipy.stats import binomtest
    a = pd.read_csv(os.path.join(PROJ, "output", "b1_compressed_a100",
                                 "b1_compressed_val.csv"))
    l = pd.read_csv(os.path.join(PROJ, "output", "fewshot",
                                 "fewshot_v4_3shot_ultracompact_val.csv"))
    m = a[["note_id", "label", "parsed_pred"]].merge(
        l[["note_id", "parsed_pred"]], on="note_id", suffixes=("_a", "_l"))
    ka = m["parsed_pred_a"] == m["label"]
    kl = m["parsed_pred_l"] == m["label"]
    x, y = int((ka & ~kl).sum()), int((~ka & kl).sum())
    # degisen etiket sayisi (dogruluk degismeden degisenler dahil) ve cihaz etkisinin %95 araligi
    degisen = int((m["parsed_pred_a"].fillna("INVALID").astype(str)
                   != m["parsed_pred_l"].fillna("INVALID").astype(str)).sum())
    n, d = len(m), (y - x) / len(m)               # 8 GB eksi A100
    se = math.sqrt(x + y - (y - x) ** 2 / n) / n
    sg = lambda v: f"\\ensuremath{{{'+' if v >= 0 else '-'}{abs(v):.1f}}}"
    return {"DevA": str(x), "DevB": str(y), "DevDisc": str(x + y),
            "Pdevice": f"{binomtest(x, x + y, 0.5).pvalue:.2f}", "DevChanged": str(degisen),
            "DevLo": sg(100 * (d - 1.959964 * se)), "DevHi": sg(100 * (d + 1.959964 * se))}


def other_orani(yol, kol="parsed_pred"):
    """'OTHER' tahmin edilen kayitlarin yuzdesi (gercek oran %9.1)."""
    d = pd.read_csv(yol)
    return 100.0 * (d[kol].astype(str) == "OTHER").sum() / len(d)


def invalid_orani(yol, kol="parsed_pred"):
    """Cikti parser'inin cozemedigi kayitlarin yuzdesi."""
    d = pd.read_csv(yol)
    return 100.0 * (d[kol].astype(str).str.upper() == "INVALID").sum() / len(d)


def eksi(v, basamak=1):
    """Negatif sayiyi LaTeX matematik eksisiyle yaz (tire degil, kirilmaz)."""
    s = f"{abs(v):.{basamak}f}"
    return f"\\ensuremath{{-{s}}}" if v < 0 else s


ZERO_TOK = tok_mean("zeroshot/zeroshot_v4_test.csv")
COMP_TOK = tok_mean("b1_compressed_a100/b1_compressed_test.csv")
FULL_TOK = tok_mean("b2_b5_verify/b2_full3shot_test.csv")
DEV_TOK_A = tok_mean("fewshot/fewshot_v4_3shot_ultracompact_val.csv")
DEV_TOK_B = tok_mean("b1_compressed_a100/b1_compressed_val.csv")
EGITIMSIZ = ["qwen_zero", "qwen_fs_comp", "qwen_fs_full", "gemma_zero_bf16", "gemma_zero_4bit", "gemma_fs_full"]


def P(karsilastirma, kol="p_exact", ondalik=None):
    return p_tex(T03.at[karsilastirma, kol], ondalik)


def KM(gercek, tahmin, k="qwen_qlora"):
    x = T06[(T06.config == k) & (T06.true_class == gercek) & (T06.predicted_class == tahmin)]
    return str(int(x["count"].iloc[0]))


N = {
    "ZeroAcc": f"{acc('Qwen zero-shot'):.3f}",
    "ZeroMF": f"{mf1('Qwen zero-shot'):.3f}",
    "CompAcc": f"{acc('Qwen 3-shot sikistirilmis (A100)'):.3f}",
    "CompMF": f"{mf1('Qwen 3-shot sikistirilmis (A100)'):.3f}",
    "FullAcc": f"{acc('Qwen tam 3-shot (yeniden)'):.3f}",
    "FullMF": f"{mf1('Qwen tam 3-shot (yeniden)'):.3f}",
    "QloraAcc": f"{acc('Qwen QLoRA adil'):.3f}",
    "QloraMF": f"{mf1('Qwen QLoRA adil'):.3f}",
    "GemZeroAcc": f"{acc('Gemma zero-shot bf16'):.3f}",
    "GemQloraAcc": f"{acc('Gemma QLoRA adil (yeniden)'):.3f}",
    "GemQloraMF": f"{mf1('Gemma QLoRA adil (yeniden)'):.3f}",
    "GemOptAcc": f"{acc('Gemma QLoRA optimize'):.3f}",
    "GemOptMF": f"{mf1('Gemma QLoRA optimize'):.3f}",
    "GemZeroMF": f"{mf1('Gemma zero-shot bf16'):.3f}",
    "GemFourAcc": f"{acc('Gemma zero-shot 4bit'):.3f}",
    "GemFourMF": f"{mf1('Gemma zero-shot 4bit'):.3f}",
    # nicelemenin DOGRULUK bedeli (Gemma bf16 -> 4bit)
    "QuantAccDrop": eksi(100 * (acc('Gemma zero-shot 4bit') - acc('Gemma zero-shot bf16'))),
    "QuantMFDrop": eksi(100 * (mf1('Gemma zero-shot 4bit') - mf1('Gemma zero-shot bf16'))),
    # Gemma'nin QLoRA kazanci — IKISI AYRI:
    #   GemFairGain : adil tarif (Qwen ile karsilastirilabilir olan BUDUR)
    #   GemGain     : optimize varyant (adil kiyas DEGIL, ayri raporlanmali)
    # DIKKAT — TABAN CIZGISI ESLESMELI.
    # Qwen'in kazanci 4-bit zero -> 4-bit QLoRA olarak olculuyor. Gemma icin
    # bf16 tabanini kullanmak (GemFairGainBf) niceleme kaybini kazanca yikiyor
    # ve "matched" kelimesini bosa cikariyor. Metinde ESLESMIS olan kullanilir.
    "GemFairGain": f"{100*(acc('Gemma QLoRA adil (yeniden)') - acc('Gemma zero-shot 4bit')):.1f}",
    "GemFairGainBf": f"{100*(acc('Gemma QLoRA adil (yeniden)') - acc('Gemma zero-shot bf16')):.1f}",
    "GemGain": f"{100*(acc('Gemma QLoRA optimize') - acc('Gemma zero-shot bf16')):.1f}",
    # Gemma'da few-shot: ORNEKLER FAYDA VERMEDI. Sikistirma bulgusunu
    # daraltiyor — yok edilecek fayda yoksa iddia tek modele ozgu.
    "GemFewAcc": f"{acc('Gemma tam 3-shot'):.3f}",
    "GemFewDelta": eksi(100 * (acc('Gemma tam 3-shot') - acc('Gemma zero-shot bf16'))),
    # farklar (puan)
    # NEGATIF degerler \ensuremath ile: duz "-" TIRE olarak dizilir ve satir
    # sonunda "-\n3.1" gibi kopabilir. Matematik eksi dogru glif + kirilmaz.
    "QloraGain": f"{100*(acc('Qwen QLoRA adil') - acc('Qwen zero-shot')):.1f}",
    "CompCost": eksi(100 * (acc('Qwen 3-shot sikistirilmis (A100)')
                            - acc('Qwen tam 3-shot (yeniden)'))),
    "CompVsZero": eksi(100 * (acc('Qwen 3-shot sikistirilmis (A100)')
                              - acc('Qwen zero-shot'))),
    # bellek
    "QwFull": f"{QW_FULL:,.0f}".replace(",", "\\,"),
    "QwFour": f"{QW_4:,.0f}".replace(",", "\\,"),
    "QwPeak": f"{QW_4T:,.0f}".replace(",", "\\,"),
    "GemFull": f"{GE_FULL:,.0f}".replace(",", "\\,"),
    "GemFour": f"{GE_4:,.0f}".replace(",", "\\,"),
    "GemPeak": f"{GE_4T:,.0f}".replace(",", "\\,"),
    "QwSave": f"{QW_KAZ:.1f}",
    "GemSave": f"{GE_KAZ:.1f}",
    "QwShare": f"{QW_PAY:.0f}",
    "GemShare": f"{GE_PAY:.0f}",
    # Nicelemeden SONRA gomme tablolarinin payi. DIKKAT: ayni tanim
    # (PLE + standart gomme), yalniz PLE degil. Iki tanimi karistirmak
    # bildiride %61 vs %76 celiskisine yol acmisti.
    "QwShareAfter": f"{100*mem('Qwen3.5-4B', '4-bit', 'nicelenmeyen_MiB')/QW_4:.0f}",
    "GemShareAfter": f"{100*mem('Gemma-4-E4B', '4-bit', 'nicelenmeyen_MiB')/GE_4:.0f}",
    # denklem (1) artiklari + egitim tepesi — metinde elle yazilmasin
    # Hakem itirazina hazir cevap: "vision/audio kulelerini yuklemeseydiniz?"
    # Yuklemesek bile sigmiyor — baglayici kisit gomme tablolari. Sayi
    # bildiride olmali, yoksa hakem hesabi kendi yapamaz.
    # DIKKAT — TUREV SAYILAR YUVARLANMIS GIRDILERDEN HESAPLANIR.
    # Okur tabloda 8,802 ve 251 goruyor; 8802-251=8551 yapar. Ham degerlerden
    # (8802.3-250.8=8551.5) hesaplarsak "8,552" yazariz ve okurun aritmetigi
    # tutmaz. Ayni sorun Qwen'in gomme-haric boyutunda da vardi (6,807 vs 6,808).
    "GemVA": f"{round(GE_VA):.0f}",
    "GemNoVA": f"{round(GE_4) - round(GE_VA):,.0f}".replace(",", "\\,"),
    "GemOverVA": f"{round(GE_4) - round(GE_VA) - CARD:.0f}",
    "GemEmbMiB": f"{GE_E:,.0f}".replace(",", "\\,"),
    # Gomme HARIC boyut: "benzer nominal boyut" iddiasini test eder
    "QwEmbMiB": f"{QW_E:,.0f}".replace(",", "\\,"),
    # Ayni kural: okur tablodaki 8,022 - 1,214 = 6,808 yapar.
    "QwNonEmb": f"{round(QW_FULL) - round(QW_E):,.0f}".replace(",", "\\,"),
    "GemNonEmb": f"{round(GE_FULL) - round(GE_E):,.0f}".replace(",", "\\,"),
    "GemEmbBudget": f"{100 * GE_E / CARD:.0f}",
    "QwResid": f"{QW_ARTIK:.1f}",
    "GemResid": f"{GE_ARTIK:.1f}",
    "GemResidPct": f"{100*GE_ARTIK/GE_FULL:.2f}",
    "TrainPeak": fmt(QLORA_TEPE),
    # Parse hata oranlari — Macro-F1 > Accuracy anomalisini aciklar.
    # Per-kayit CSV'lerden sayilir, elle girilmez.
    # "other" tahmin oranlari. Few-shot istemine eklenen iki kural
    # "other"i AZALTMAK icindi; olcum tam tersini gosteriyor, dolayisiyla
    # few-shot kazanci "other duzeltmesi" ile aciklanamaz.
    # CIHAZ KONTROLU — 8 GB RTX 4060 vs A100, ayni sikistirilmis istem, 550 val.
    # ONCEKI METIN "moved accuracy by 3 of those 550 records" diyordu; bu
    # YANILTICIYDI: net dogruluk farki 3 kayit ama DEGISEN tahmin sayisi 33.
    # Hakem McNemar'i kontrol etmek icin discordant ciftleri (15/18) ister.
    **_dev(),
    "Chance": f"{100/11:.1f}",
    # Olculmus OOM: tek ornekli sikistirilmamis istem, ortalama 7,656 token,
    # RTX 4060'ta CUDA OOM (v4_final/docs/README.md, token audit). Tam 3-shot
    # istem bunun ~3 kati; dolayisiyla Tablo I'deki carpi OLCUME dayaniyor.
    "OomTok": "7\\,656",
    "ZeroOther": f"{other_orani(os.path.join(PROJ, 'output', 'zeroshot', 'zeroshot_v4_test.csv')):.1f}",
    "FullOther": f"{other_orani(os.path.join(PROJ, 'output', 'b2_b5_verify', 'b2_full3shot_test.csv')):.1f}",
    "QloraOther": f"{other_orani(os.path.join(PROJ, 'output', 'qlora_eval_fullctx', 'test_predictions.csv'), 'pred'):.1f}",
    "ZeroInvalid": f"{invalid_orani(os.path.join(PROJ, 'output', 'zeroshot', 'zeroshot_v4_test.csv')):.1f}",
    "FullInvalid": f"{invalid_orani(os.path.join(PROJ, 'output', 'b2_b5_verify', 'b2_full3shot_test.csv')):.1f}",
    "QloraInvalid": f"{invalid_orani(os.path.join(PROJ, 'output', 'qlora_eval_fullctx', 'test_predictions.csv'), 'pred'):.1f}",
    "QwAct": f"{mem('Qwen3.5-4B', '4-bit', 'aktivasyon_MiB'):,.0f}".replace(",", "\\,"),
    "GemAct": f"{mem('Gemma-4-E4B', '4-bit', 'aktivasyon_MiB'):,.0f}".replace(",", "\\,"),
    # istem uzunluklari
    "ZeroTok": f"{ZERO_TOK:,.0f}".replace(",", "\\,"),
    "CompTok": f"{COMP_TOK:,.0f}".replace(",", "\\,"),
    "FullTok": f"{FULL_TOK:,.0f}".replace(",", "\\,"),
    # istem uzunluk oranlari ve kesme ablasyonu — hepsi turetilmis
    "CompVsZeroTok": f"{round(COMP_TOK) / round(ZERO_TOK):.1f}",
    "QwFullTok": f"{QW_FULL_TOK:,.0f}".replace(",", "\\,"),
    "QwTruncTok": f"{QW_TRUNC_TOK:,.0f}".replace(",", "\\,"),
    "TruncPct": f"{100*(QW_FULL_TOK-QW_TRUNC_TOK)/QW_FULL_TOK:.0f}",
    "TruncCost": f"{100*(acc('Qwen QLoRA adil') - acc('Qwen QLoRA adil [kesik 409 tok]')):.1f}",
    # p=0.5'te (en genis) Wilson %95 yari-genisligi, n=TestN
    "MarginCI": f"{100*1.959964*math.sqrt(0.25/1650 + 1.959964**2/(4*1650**2))/(1+1.959964**2/1650):.1f}",
    # istatistik (analyze_b1.py / analyze_b2_b5.py ciktilarindan)
    "PcompFull": P("Compression cost, Qwen"),
    "PfullZero": P("Few-shot benefit, Qwen (full)"),
    "PqwGem": P("Model gap, fair QLoRA"),
    "PoptQw": P("Gemma optimized vs Qwen QLoRA"),
    "Ptrunc": P("Truncated evaluation input, Qwen QLoRA"),
    "Pquant": P("Quantization cost, Gemma"),
    "PqwGemHolm": P("Model gap, fair QLoRA", "p_holm"),
    # sikistirilmis few-shot ile zero-shot (Qwen) — "harmful" iddiasi yerine olcum
    "PcompZero": P("Compressed few-shot vs zero-shot, Qwen"),
    "PcompZeroHolm": P("Compressed few-shot vs zero-shot, Qwen", "p_holm"),
    "HolmN": str(len(T03)),
    # klasik taban cizgileri (T01) ve SVM'e karsi esli testler (T03)
    "SvmAcc": f"{accx('svm'):.3f}", "SvmMF": f"{T01.at['svm', 'macro_f1']:.3f}",
    "LrAcc": f"{accx('lr'):.3f}", "RfAcc": f"{accx('rf'):.3f}",
    "SvmMinMargin": f"{100 * (accx('svm') - max(accx(k) for k in EGITIMSIZ)):.1f}",
    "SvmMaxMargin": f"{100 * (accx('svm') - min(accx(k) for k in EGITIMSIZ)):.1f}",
    "PsvmGemZeroHolm": P("gemma_zero_bf16 vs TF-IDF SVM", "p_holm"),
    "PsvmGemFewHolm": P("gemma_fs_full vs TF-IDF SVM", "p_holm"),
    "SvmQloraGain": f"{100 * (accx('qwen_qlora') - accx('svm')):.1f}",
    "PsvmQloraHolm": P("qwen_qlora vs TF-IDF SVM", "p_holm"),
    "SvmGemOptGain": f"{100 * (accx('gemma_qlora_opt') - accx('svm')):.1f}",
    "PsvmGemOptHolm": P("gemma_qlora_opt vs TF-IDF SVM", "p_holm"),
    "SvmGemFairGain": f"{100 * (accx('gemma_qlora_fair') - accx('svm')):.1f}",
    "PsvmGemFairHolm": P("gemma_qlora_fair vs TF-IDF SVM", "p_holm"),
    # maske kontrolu (T08)
    "MaskPct": f"{float(T08['test notes containing the placeholder, %']):.1f}",
    "MaskPerNote": f"{float(T08['placeholders per test note, mean']):.1f}",
    "MaskOnlyAcc": f"{100 * float(T08['accuracy from placeholder count alone']):.1f}",
    "MaskDelta": f"{abs(float(T08['difference, points'])):.1f}",
    "Pmask": f"{float(T08['McNemar exact p']):.3f}",
    # maske kontrolunun ortami (Ek B): ayri scikit-learn surumu ve o ortamdaki SVM
    "MaskEnvSvm": f"{float(T08['SVM accuracy with placeholder (this environment)']):.3f}",
    "MaskSklearn": str(T08['scikit-learn version of this control']).strip(),
    # sikistirilmis orneklerin gercek uzunlugu (betikteki 60-80 HEDEFTI, olcum degil)
    "CompExTok": f"{ORNEK_TOK_ORT:.0f}", "CompExTokMin": f"{ORNEK_TOK_MIN}",
    "CompExTokMax": f"{ORNEK_TOK_MAX}",
    # cihaz kontrolunde istem uzunluklari
    "DevTokA": f"{DEV_TOK_A:,.0f}".replace(",", "\\,"),
    "DevTokB": f"{DEV_TOK_B:,.0f}".replace(",", "\\,"),
    # sinif bazinda F1 (T04, Sekil 5 ile ayni 2 ondalik) ve karisiklik (T06)
    "RhoSvmQlora": f"{float(T09['Spearman rho, per-class F1 of SVM vs Qwen QLoRA']):.2f}",
    "EasyMin": f"{math.floor(100 * T04.loc[['STROKE', 'GI_BLEEDING']].values.min()) / 100:.2f}",
    "SepsisLo": f"{T04.loc['SEPSIS'].min():.2f}", "SepsisHi": f"{T04.loc['SEPSIS'].max():.2f}",
    "OtherLo": f"{T04.loc['OTHER'].min():.2f}", "OtherHi": f"{T04.loc['OTHER'].max():.2f}",
    "AkiZeroF": f"{T04.at['AKI', 'qwen_zero']:.2f}", "AkiQloraF": f"{T04.at['AKI', 'qwen_qlora']:.2f}",
    "UtiZeroF": f"{T04.at['UTI', 'qwen_zero']:.2f}", "UtiQloraF": f"{T04.at['UTI', 'qwen_qlora']:.2f}",
    "SepsisQloraF": f"{T04.at['SEPSIS', 'qwen_qlora']:.2f}",
    "SepCorrect": KM("SEPSIS", "SEPSIS"), "SepPneu": KM("SEPSIS", "PNEUMONIA"),
    "SepUti": KM("SEPSIS", "UTI"), "PneuCopd": KM("PNEUMONIA", "COPD_EXACERBATION"),
    "HfCopd": KM("HEART_FAILURE", "COPD_EXACERBATION"),
    "AllWrong": f"{float(T09['all three wrong (SVM, Qwen QLoRA, Gemma QLoRA optimized), %']):.1f}",
    "Oracle": f"{float(T09['at least one correct (oracle upper bound), %']):.1f}",
    "TestN": "1650",
}
# ---------------------------------------------------------------- hakem kontrolleri (11 Kasim 2026)
# build_reviewer_checks.py ciktisi (T15-T18) ve klasik taban cizgilerinin yeniden kosusu
# (klasik_ml_v4_kayit_tahminleri.py). Elle rakam yok; tablolar da buradan uretiliyor.
T15 = _t("T15_pool_class_distribution.csv").set_index("class")
T16 = _t("T16_prevalence_reweighted.csv").set_index("config")
T17 = _t("T17_leakage_term_checks.csv").set_index("metric")["value"]
T18 = _t("T18_svm_top_features.csv")
RR = json.load(open(os.path.join(PROJ, "output", "classical_ml_v4", "classical_ml_v4_rerun_meta.json"),
                    encoding="utf-8"))
assert RR["checks"]["svm_predictions_identical"] == 1650, "yeniden kosu SVM tahminlerini uretmiyor"
_v = lambda m: float(T17[m])
N.update({
    "PoolN": f"{int(T15['n'].sum()):,}".replace(",", "\\,"),
    "PoolOtherPct": f"{T15.at['OTHER', 'pct']:.1f}",
    "PrevAccSvm": f"{T16.at['svm', 'accuracy_pool']:.3f}",
    "PrevAccQlora": f"{T16.at['qwen_qlora', 'accuracy_pool']:.3f}",
    "PrevPpvSvm": f"{100 * T16.at['svm', 'target_precision_pool']:.0f}",
    "PrevPpvQlora": f"{100 * T16.at['qwen_qlora', 'target_precision_pool']:.0f}",
    "TermMaxPct": f"{_v('target term in CC or HPI, own class, max over classes, %'):.1f}",
    "TermDelta": eksi(_v("SVM accuracy change, target terms deleted everywhere, points")),
    "Pterm": p_tex(_v("p (exact McNemar, unadjusted), target terms")),
    "LabPct": f"{_v('lab mention in CC or HPI, all test inputs, %'):.1f}",
    "LabAkiPct": f"{_v('lab mention in CC or HPI, AKI inputs, %'):.0f}",
    "LabDelta": eksi(_v("SVM accuracy change, lab mentions deleted from CC and HPI, points")),
    "Plab": p_tex(_v("p (exact McNemar, unadjusted), lab mentions")),
    "AkiRecBase": f"{_v('SVM recall for AKI, baseline'):.2f}",
    "AkiRecLab": f"{_v('SVM recall for AKI, lab mentions deleted'):.2f}",
    "LrOrigAcc": f"{RR['checks']['lr']['stored_accuracy']:.3f}",
    "RfOrigAcc": f"{RR['checks']['rf']['stored_accuracy']:.3f}",
    "RerunSklearn": RR["sklearn"],
    "RerunSeed": str(RR["random_state"]),
})

# Ek C tablolari. Kodlar output/... etiketleyicisi icd10_to_label_v3 (__add_blok1d.py) ile ayni.
SINIF = [
    ("SEPSIS", "Sepsis", "A40, A41"),
    ("STROKE", "Stroke", "I60--I63"),
    ("ATRIAL_FIBRILLATION", "Atrial fibrillation", "I48"),
    ("AKI", "Acute kidney injury", "N17"),
    ("PNEUMONIA", "Pneumonia", "J12--J18"),
    ("UTI", "Urinary tract infection", "N39.0"),
    ("HEART_FAILURE", "Heart failure", "I50"),
    ("GI_BLEEDING", "Gastrointestinal bleeding", "K92.0--K92.2"),
    ("DIABETES_COMPLICATION", "Diabetes complication",
     "E10, E11, E13 with subcategory .0 or .1 (hyperosmolarity, ketoacidosis), .64 or .65 (hypo- and hyperglycemia)"),
    ("COPD_EXACERBATION", "COPD exacerbation",
     r"J44 (\CopdUnspecPct\% of the class is J44.9, with no exacerbation coded)"),
    ("OTHER", r"\emph{Other}", "every other code"),
]
tl = [r"% OTOMATIK URETILDI — make_tables.py",
      r"\begin{table}[!htb]",
      r"\caption{Label definition. The primary ICD-10 code of an admission is mapped to a class by",
      r"prefix. The last column is the class distribution of the eligible pool of \PoolN{} admissions",
      r"with a parsable discharge summary, from which the balanced splits were drawn.}",
      r"\label{tab:labels}", r"\centering", r"\small",
      r"\begin{tabular}{@{}lp{0.56\linewidth}r@{}}", r"\toprule",
      r"Class & ICD-10 codes & Pool (\%) \\", r"\midrule"]
for k, ad, kod in SINIF:
    tl.append(f"{ad} & {kod} & {T15.at[k, 'pct']:.1f} " + r"\\")
tl += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
with open(os.path.join(BASE, "tab_labels.tex"), "w", encoding="utf-8") as f:
    f.write("\n".join(tl) + "\n")

SIRA = ["qwen_zero", "qwen_fs_comp", "qwen_fs_full", "qwen_qlora", "gemma_zero_bf16", "gemma_zero_4bit",
        "gemma_fs_full", "gemma_qlora_fair", "gemma_qlora_opt", "svm"]
for k in SIRA:  # dengeli satirlar Tablo 1 ile ayni olmali
    assert abs(T16.at[k, "accuracy_balanced"] - accx(k)) < 6e-4, k
    assert abs(T16.at[k, "macro_f1_balanced"] - float(T01.at[k, "macro_f1"])) < 6e-4, k
tp = [r"% OTOMATIK URETILDI — make_tables.py",
      r"\begin{table}[!htb]",
      r"\caption{Results on the balanced test set and reweighted to the class distribution of the",
      r"eligible pool (Table~\ref{tab:labels}), assuming that the predictions within each class stay",
      r"the same. Target precision is the mean precision over the ten target classes.}",
      r"\label{tab:prev}", r"\centering", r"\small",
      r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
      r" & \multicolumn{2}{c}{Accuracy} & \multicolumn{2}{c}{Macro-F1} & \multicolumn{2}{c}{Target precision} \\",
      r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
      r"Configuration & Balanced & Pool & Balanced & Pool & Balanced & Pool \\", r"\midrule"]
for k in SIRA:
    r = T16.loc[k]
    tp.append(f"{AD[k]} & {accx(k):.3f} & {r.accuracy_pool:.3f} & {float(T01.at[k, 'macro_f1']):.3f} & "
              f"{r.macro_f1_pool:.3f} & {r.target_precision_balanced:.3f} & {r.target_precision_pool:.3f} " + r"\\")
tp += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
with open(os.path.join(BASE, "tab_prev.tex"), "w", encoding="utf-8") as f:
    f.write("\n".join(tp) + "\n")

tf = [r"% OTOMATIK URETILDI — make_tables.py",
      r"\begin{table}[!htb]",
      r"\caption{Largest weights of the TF-IDF linear SVM for each class, among word unigrams and",
      r"bigrams found in at least 20 training inputs, leaving out the masking placeholder and",
      r"de-identification marks. Bigrams such as \emph{fever symptom} join the end of the chief",
      r"complaint to the next section heading.}",
      r"\label{tab:features}", r"\centering", r"\small",
      r"\begin{tabular}{@{}lp{0.62\linewidth}@{}}", r"\toprule",
      r"Class & Five largest weights \\", r"\midrule"]
for k, ad, _ in SINIF:
    ozl = T18[T18["cls"] == k].sort_values("rank")["feature"].tolist()
    assert len(ozl) == 5, k
    tf.append(f"{ad} & " + ", ".join(x.replace("_", r"\_") for x in ozl) + r" \\")
tf += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
with open(os.path.join(BASE, "tab_features.tex"), "w", encoding="utf-8") as f:
    f.write("\n".join(tf) + "\n")


# ---------------------------------------------------------------- hakem degerlendirmesi (11 Kasim 2026)
T14 = _t("T14_resources_latency.csv").set_index("config")
T19 = _t("T19_label_gaps.csv")
T20 = _t("T20_svm_learning_curve.csv").set_index("setting")
T21 = _t("T21_svm_truncation.csv").set_index("first_words")
T22 = _t("T22_cohort_and_time.csv").set_index("metric")["value"]
_sep = lambda v: f"{int(v):,}".replace(",", "\\,")
_kalan = lambda mdl: (mem(mdl, "4-bit", "tepe_cikarim_MiB") - mem(mdl, "4-bit", "agirlik_MiB")
                      - mem(mdl, "4-bit", "aktivasyon_MiB"))
N.update({
    "CardMiB": _sep(CARD),
    "LabeledN": _sep(float(T22["admissions with a discharge summary and a primary ICD-10 code"])),
    "HtnHfN": _sep(T19.loc[T19.group.str.startswith("hypertensive"), "n_in_other"].sum()),
    "HfN": _sep(T19.loc[T19.target_class == "HEART_FAILURE", "n_in_target_class"].iloc[0]),
    "GibSrcN": _sep(T19.loc[T19.target_class == "GI_BLEEDING", "n_in_other"].sum()),
    "SvmLcThree": f"{T20.at['3 per class, random', 'accuracy_mean']:.3f}",
    "SvmLcFs": f"{T20.at['the 33 few-shot examples (full text)', 'accuracy_mean']:.3f}",
    "SvmLcHundred": f"{T20.at['100 per class, random', 'accuracy_mean']:.3f}",
    "SvmLcTwoHundred": f"{T20.at['200 per class, random', 'accuracy_mean']:.3f}",
    "SvmTruncSixty": f"{T21.at[60, 'acc_full_train_truncated_test']:.3f}",
    "InputWords": f"{float(T22['test input length, words, mean']):.0f}",
    "SvmSec": f"{float(T22['SVM: TF-IDF fit, training and labeling of the test set, seconds (2-core CPU)']):.1f}",
    "ZeroSec": f"{T14.at['qwen_zero', 'latency_s_per_record']:.2f}",
    "QloraSec": f"{T14.at['qwen_qlora', 'latency_s_per_record']:.2f}",
    "FullSec": f"{T14.at['qwen_fs_full', 'latency_s_per_record']:.2f}",
    "QloraHours": f"{T14.at['qwen_qlora', 'training_h']:.1f}",
    "FullZeroGain": f"{T03.at['Few-shot benefit, Qwen (full)', 'diff_points']:.1f}",
    "FullExTok": f"{(FULL_TOK - ZERO_TOK) / 33:.0f}",
    "CompMaxTok": _sep(pd.read_csv(os.path.join(PROJ, "output", "fewshot",
                                                "fewshot_v4_3shot_ultracompact_val.csv"))["in_tokens"].max()),
    "QwRest": f"{_kalan('Qwen3.5-4B'):.0f}", "GemRest": f"{_kalan('Gemma-4-E4B'):.0f}",
    "PqloraHolm": P("QLoRA gain, Qwen", "p_holm"),
    "PgemGainHolm": P("QLoRA gain, Gemma (4-bit to 4-bit)", "p_holm"),
    "PcompFullHolm": P("Compression cost, Qwen", "p_holm"),
    "PfullZeroHolm": P("Few-shot benefit, Qwen (full)", "p_holm"),
    "PquantHolm": P("Quantization cost, Gemma", "p_holm"),
    "PtruncHolm": P("Truncated evaluation input, Qwen QLoRA", "p_holm"),
    "PoptQwHolm": P("Gemma optimized vs Qwen QLoRA", "p_holm"),
    "OptQwGain": f"{T03.at['Gemma optimized vs Qwen QLoRA', 'diff_points']:.1f}",
    "PsvmFullHolm": P("qwen_fs_full vs TF-IDF SVM", "p_holm"),
    "PgemFewHolm": P("Few-shot benefit, Gemma (bf16)", "p_holm"),
    "PdeviceHolm": P("Device control (compressed few-shot, val)", "p_holm"),
})
# ---- ikinci hakem turu (12 Kasim 2026): T24, T25, COPD, "hep other" tabani ----
assert abs(accx("lr") - RR["checks"]["lr"]["accuracy"]) < 6e-5 and abs(accx("rf") - RR["checks"]["rf"]["accuracy"]) < 6e-5, \
    "T01'deki LR/RF satirlari yeniden kosuyla ayni degil"
T24 = _t("T24_term_stratified_recall.csv").set_index("config")
T25 = _t("T25_unparseable_outputs.csv").set_index("config")
_fs = T25.loc["qwen_fs_full"]
assert _fs.begin_with_based_on == _fs.n_unparseable == _fs.at_least_19_tokens and _fs.name_a_class == 0, \
    "tam few-shot'un ayristirilamayan ciktilari artik tek tip degil: 4.1'deki cumleyi gozden gecirin"
assert int(_fs.n_unparseable) == int(T01.at["qwen_fs_full", "n_invalid"])
assert int(T01.at["qwen_qlora", "n_invalid"]) == 0, "Qwen QLoRA'da ayristirilamayan cikti var: 4.1'i duzeltin"
_ncopd = float(T22["COPD class in the eligible pool, n"]); _n449 = float(T22["of these, J44.9 (no exacerbation coded), n"])
_ap = T24["accuracy_points_attributable"]
N.update({
    "AllOtherAcc": f"{T15.at['OTHER', 'n'] / T15['n'].sum():.3f}",
    "PrevAccMax": f"{max(T16.at[k, 'accuracy_pool'] for k in SIRA):.3f}",
    "GemInvalidMaxN": str(int(max(T01.at[k, 'n_invalid'] for k in T01.index if k.startswith('gemma')))),
    "FullInvalidN": str(int(_fs.n_unparseable)),
    "FullUpperAcc": f"{_fs.accuracy_if_all_unparseable_correct:.3f}",
    "FullSvmGap": f"{100 * (accx('svm') - accx('qwen_fs_full')):.1f}",
    "TermRecN": str(int(T24.at["qwen_qlora", "n_target_records_with_term"])),
    "TermQloraWith": f"{100 * T24.at['qwen_qlora', 'recall_with_term']:.0f}",
    "TermQloraWithout": f"{100 * T24.at['qwen_qlora', 'recall_without_term_same_class_mix']:.0f}",
    "TermPtsQlora": f"{_ap['qwen_qlora']:.1f}",
    "TermPtsMin": f"{_ap.min():.1f}", "TermPtsMax": f"{_ap.max():.1f}", "TermPtsSvm": f"{_ap['svm']:.1f}",
    "TermDeltaAbs": f"{abs(_v('SVM accuracy change, target terms deleted everywhere, points')):.1f}",
    "AkiLabQlora": f"{100 * T24.at['qwen_qlora', 'aki_recall_with_lab_mention']:.0f}",
    "AkiNoLabQlora": f"{100 * T24.at['qwen_qlora', 'aki_recall_without_lab_mention']:.0f}",
    "AkiLabZero": f"{100 * T24.at['qwen_zero', 'aki_recall_with_lab_mention']:.0f}",
    "AkiNoLabZero": f"{100 * T24.at['qwen_zero', 'aki_recall_without_lab_mention']:.0f}",
    "AkiLabSvm": f"{100 * T24.at['svm', 'aki_recall_with_lab_mention']:.0f}",
    "AkiNoLabSvm": f"{100 * T24.at['svm', 'aki_recall_without_lab_mention']:.0f}",
    "CopdUnspecPct": f"{100 * _n449 / _ncopd:.1f}",
})

# ---- sizinti taramasi (T23 <- build_leak_screen_table.py) ----
T23 = _t("T23_leakage_screen.csv").set_index("split")
assert int(T23.at["test", "flagged_final"]) == 0 and int(T23.at["test", "rule_review_needs_human"]) == 0
N.update({
    "LeakTestN": str(int(T23.at["test", "rule_review_kept_historical"]) + int(T23.at["test", "rule_review_records_cleaned"])),
    "LeakKeptN": str(int(T23.at["test", "rule_review_kept_historical"])),
    "LeakRemN": str(int(T23.at["test", "rule_review_records_cleaned"])),
    "LeakTrainN": str(int(T23.at["train_S3", "flagged_final"])),
    "LeakValN": str(int(T23.at["val", "flagged_final"])),
})

# ---- kanit denetimi (13 Kasim 2026) ----
# Tam ve sikistirilmis few-shot istemleri ayni 33 ornegi KULLANMIYOR: sikistirilmis istemin ornekleri
# __run_blok14_v4_reselect_other_3shot.py ile yeniden secildi (ifade denetimi + other dislamasi).
_fj = lambda ad: json.load(open(os.path.join(PROJ, "output", "fewshot", ad), encoding="utf-8"))
_ids = lambda d: {str(e["note_id"]) for v in d.values() for e in (v if isinstance(v, list) else [v])}
_tam, _sik = _ids(_fj("fewshot_v4_3shot_examples.json")), _ids(_fj("fewshot_v4_3shot_ultracompact_examples.json"))
assert len(_tam) == len(_sik) == 33
# Iki kez degerlendirilen yapilandirmalar (results_master): ilk kosu (tez) ve raporlanan ikinci kosu
_rr = [abs(acc("Qwen tam 3-shot") - acc("Qwen tam 3-shot (yeniden)")),
       abs(acc("Gemma QLoRA adil") - acc("Gemma QLoRA adil (yeniden)"))]
N.update({
    "FsSharedN": str(len(_tam & _sik)),
    "FsReplacedN": str(33 - len(_tam & _sik)),
    "RerunMaxDiff": f"{100 * max(_rr):.1f}",
})

with open(os.path.join(BASE, "numbers.tex"), "w", encoding="utf-8") as f:
    f.write("% OTOMATIK URETILDI - make_tables.py. Metinde bu makrolari kullanin,\n"
            "% asla elle rakam yazmayin.\n")
    for k, v in N.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

print("Uretildi:")
print("  tab_results.tex   — ana sonuc tablosu (9 yapilandirma)")
print("  tab_resources.tex — bellek tablosu (iki mimari)")
print(f"  numbers.tex       — {len(N)} makro (metinde elle rakam yok)")
print(f"\n  En iyi acc {en_iyi_acc:.4f} · en iyi MF1 {en_iyi_mf1:.4f} (kalin yazildi)")
print("\n  Rakamlar results_master.csv + memory_table.csv'den okundu.")
print("  Bir sayi degisirse bu script'i tekrar kosturun.")
