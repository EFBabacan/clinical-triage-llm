# -*- coding: utf-8 -*-
"""
C1/C4 — BILDIRI TABLOLARINI DOGRULANMIS CSV'LERDEN URET

Kural: bildiriye elle rakam YAZILMAZ. Bu script results_master.csv ve
memory_table.csv'den LaTeX tablolari uretir; bir sayi degisirse script
tekrar kosturulur ve bildiri kendini gunceller.

Cikti:  tab_results.tex · tab_resources.tex  (main.tex bunlari \input eder)
"""
import math
import os

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(BASE)

rm = pd.read_csv(os.path.join(PROJ, "results_master.csv")).set_index("yapilandirma")
mt = pd.read_csv(os.path.join(PROJ, "memory_table.csv"))
rt = pd.read_csv(os.path.join(PROJ, "resource_table.csv"))


def g(ad, kol):
    return rm.loc[ad, kol]


# ---------------------------------------------------------------- Tablo I
# (etiket, model, yontem, bit, results_master anahtari, 8GB)
SATIRLAR = [
    ("Qwen3.5-4B", "Zero-shot",              "4-bit", "Qwen zero-shot",                   True),
    ("Qwen3.5-4B", "Few-shot, compressed",   "4-bit", "Qwen 3-shot sikistirilmis (A100)", True),
    ("Qwen3.5-4B", "Few-shot, full",         "4-bit", "Qwen tam 3-shot (yeniden)",        False),
    ("Qwen3.5-4B", "QLoRA (fair)",           "4-bit", "Qwen QLoRA adil",                  True),
    ("Gemma-4-E4B", "Zero-shot",             "bf16",  "Gemma zero-shot bf16",             False),
    ("Gemma-4-E4B", "Zero-shot",             "4-bit", "Gemma zero-shot 4bit",             False),
    ("Gemma-4-E4B", "Few-shot, full",        "bf16",  "Gemma tam 3-shot",                 False),
    ("Gemma-4-E4B", "QLoRA (fair)",          "4-bit", "Gemma QLoRA adil (yeniden)",       False),
    ("Gemma-4-E4B", "QLoRA (optimized)",     "4-bit", "Gemma QLoRA optimize",             False),
]

lines = [
    r"% OTOMATIK URETILDI — make_tables.py. ELLE DUZENLEME.",
    r"\begin{table}[t]",
    # KISA TUTUN — 4 sayfa siniri. Adaptor hiperparametreleri zaten Bolum III'te.
    r"\caption{Test set performance ($n=1650$, 150 per class); macro-F1 over the 11",
    r"target classes. \emph{Fair} is the adapter configuration held identical across",
    r"models; \emph{optimized} is a higher-capacity variant run for Gemma only.",
    r"Last column: fit in 8\,GB, measured (Sec.~\ref{sec:resources}).}",
    r"\label{tab:results}",
    r"\centering",
    r"\setlength{\tabcolsep}{3.5pt}",
    r"\begin{tabular}{@{}llcrrc@{}}",
    r"\toprule",
    # "Prec." YAZMA: siniflandirma makalesinde Precision metrigi sanilir.
    # Bu sutun SAYISAL DUYARLIK (4-bit / bf16) tasiyor.
    r"Model & Adaptation & Bits & Acc. & Macro-F1 & 8\,GB \\",
    r"\midrule",
]

prev = None
en_iyi_acc = max(float(g(k, "accuracy")) for *_, k, _ in SATIRLAR)
en_iyi_mf1 = max(float(g(k, "macro_f1_DUZELTILMIS")) for *_, k, _ in SATIRLAR)

for model, yontem, bit, anahtar, sigar in SATIRLAR:
    acc = float(g(anahtar, "accuracy"))
    mf1 = float(g(anahtar, "macro_f1_DUZELTILMIS"))
    a_s = f"\\textbf{{{acc:.3f}}}" if abs(acc - en_iyi_acc) < 1e-9 else f"{acc:.3f}"
    m_s = f"\\textbf{{{mf1:.3f}}}" if abs(mf1 - en_iyi_mf1) < 1e-9 else f"{mf1:.3f}"
    md = model if model != prev else ""
    if model != prev and prev is not None:
        lines.append(r"\midrule")
    prev = model
    # "--" Tablo II'de "uygulanamaz" demek; burada "sigmiyor" demekti.
    # Ayni isaret iki anlama gelmesin diye carpi kullaniliyor.
    isaret = r"\checkmark" if sigar else r"$\times$"
    lines.append(f"{md} & {yontem} & {bit} & {a_s} & {m_s} & {isaret} " + r"\\")

lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
]
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
GE_4T = mem("Gemma-4-E4B", "4-bit", "tepe_cikarim_MiB")
QW_PAY = mem("Qwen3.5-4B", "fp16", "nicelenmeyen_pay_%")
GE_PAY = mem("Gemma-4-E4B", "bf16", "nicelenmeyen_pay_%")
QW_KAZ = mem("Qwen3.5-4B", "4-bit", "niceleme_kazanci_%")
GE_KAZ = mem("Gemma-4-E4B", "4-bit", "niceleme_kazanci_%")
QW_E = mem("Qwen3.5-4B", "fp16", "nicelenmeyen_MiB")
GE_E = mem("Gemma-4-E4B", "bf16", "nicelenmeyen_MiB")
QLORA_TEPE = 7055.0          # qlora egitim gunlugu; tabloda ve metinde ayni kaynak

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
    r"\caption{Measured GPU memory at batch size 1 with 2048-token prompts.",
    r"Rows in MiB are memory; the share row is the fraction of bf16 weights that",
    r"quantization cannot reach. Peak figures are",
    r"\emph{reserved} memory, which also includes activations, the key-value",
    r"cache, and allocator overhead. Gemma exceeds the budget already at",
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
    f"Weights, bf16 (MiB) & {fmt(QW_FULL)} & {fmt(GE_FULL)} \\\\",
    # E satiri EKLENDI: metin "Taking E from Table II" diyordu ama 1,214 ve
    # 6,656 tabloda yoktu; okur sayiyi bulamiyordu. Ayrica %15/%44 payinin
    # neyin payi oldugunu da bu satir acikliyor.
    f"\\textbf{{Embedding tables (MiB)}} & \\textbf{{{fmt(QW_E)}}} & \\textbf{{{fmt(GE_E)}}} \\\\",
    f"Weights, 4-bit NF4 (MiB) & {fmt(QW_4)} & {fmt(GE_4)} \\\\",
    f"Embedding share of bf16 & {QW_PAY:.0f}\\% & {GE_PAY:.0f}\\% \\\\",
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
    return {"DevA": str(x), "DevB": str(y), "DevDisc": str(x + y),
            "Pdevice": f"{binomtest(x, x + y, 0.5).pvalue:.2f}"}


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
    "GemOverVA": f"{round(GE_4) - round(GE_VA) - 8 * 1024:.0f}",
    "GemEmbMiB": f"{GE_E:,.0f}".replace(",", "\\,"),
    # Gomme HARIC boyut: "benzer nominal boyut" iddiasini test eder
    "QwEmbMiB": f"{QW_E:,.0f}".replace(",", "\\,"),
    # Ayni kural: okur tablodaki 8,022 - 1,214 = 6,808 yapar.
    "QwNonEmb": f"{round(QW_FULL) - round(QW_E):,.0f}".replace(",", "\\,"),
    "GemNonEmb": f"{round(GE_FULL) - round(GE_E):,.0f}".replace(",", "\\,"),
    "GemEmbBudget": f"{100 * GE_E / (8 * 1024):.0f}",
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
    "ZeroTok": "865", "CompTok": "2\\,024", "FullTok": "23\\,786",
    # istem uzunluk oranlari ve kesme ablasyonu — hepsi turetilmis
    "CompVsZeroTok": f"{2024/865:.1f}",
    "QwFullTok": f"{QW_FULL_TOK:,.0f}".replace(",", "\\,"),
    "QwTruncTok": f"{QW_TRUNC_TOK:,.0f}".replace(",", "\\,"),
    "TruncPct": f"{100*(QW_FULL_TOK-QW_TRUNC_TOK)/QW_FULL_TOK:.0f}",
    "TruncCost": f"{100*(acc('Qwen QLoRA adil') - acc('Qwen QLoRA adil [kesik 409 tok]')):.1f}",
    # p=0.5'te (en genis) Wilson %95 yari-genisligi, n=TestN
    "MarginCI": f"{100*1.959964*math.sqrt(0.25/1650 + 1.959964**2/(4*1650**2))/(1+1.959964**2/1650):.1f}",
    # istatistik (analyze_b1.py / analyze_b2_b5.py ciktilarindan)
    "PcompFull": "1.3\\times10^{-7}",
    "PfullZero": "1.3\\times10^{-4}",
    "PqwGem": "0.060",
    "Ptrunc": "0.148",
    "Pquant": "2.7\\times10^{-5}",
    "TestN": "1650",
}
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
