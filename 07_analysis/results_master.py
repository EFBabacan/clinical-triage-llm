# -*- coding: utf-8 -*-
"""
A0 — Makro-F1'leri labels=CLASSES ile yeniden uret, results_master.csv olustur.

Her satir icin 3 yoldan biri kullanilir (kaynak sutununda yazar):
  1. PER-KAYIT  : ham CSV'den yeniden hesap (6 satir)
  2. wF1        : per-kayit CSV yok ama eski dosyada wF1 var; dengeli sette wF1 == macroF1(/11)
  3. DEGISMEDI  : parser %100 direct -> INVALID yok -> payda zaten 11, rapor edilen deger dogru

Ayrica sinif bazinda F1 tablosu uretir (Gemma optimize satiri tezde sayisal degildi).
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

BASE = "./workdir"  # workspace/workdir (see README)
OUT = os.path.join(BASE, "output")
DL = os.path.join(OUT, "colab_downloads")  # Colab sonuc dosyalari (5 Kasim 2026: Indirilenler'den projeye tasindi)

CLASSES = ["SEPSIS", "STROKE", "ATRIAL_FIBRILLATION", "AKI", "PNEUMONIA", "UTI",
           "HEART_FAILURE", "GI_BLEEDING", "DIABETES_COMPLICATION",
           "COPD_EXACERBATION", "OTHER"]

# ---------------------------------------------------------------- yardimcilar
def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def boot_mf1(yt, yp, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    yt, yp = np.asarray(yt), np.asarray(yp)
    n = len(yt)
    v = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        v[b] = f1_score(yt[i], yp[i], average="macro", labels=CLASSES, zero_division=0)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


# ---------------------------------------------------------------- tanimlar
# per-kayit CSV'si olan yapilandirmalar
PERREC = [
    # ad, model, yontem, bit, ortam, split, 8gb, path, gold, pred, tezdeki_mf1
    ("Qwen zero-shot", "Qwen3.5-4B", "zero-shot", "4bit", "RTX 4060", "test", "EVET (kostu)",
     os.path.join(OUT, "zeroshot", "zeroshot_v4_test.csv"), "label", "parsed_pred", 0.4681),
    # A6 (6 Agu): tam baglam (2048) — ANA SATIR. Zero-shot ile ayni butce.
    ("Qwen QLoRA adil", "Qwen3.5-4B", "QLoRA adil", "4bit", "RTX 4060", "test", "EVET (kostu)",
     os.path.join(OUT, "qlora_eval_fullctx", "test_predictions.csv"), "gold", "pred", 0.6302),
    # eski kesik kosu — ablasyon/saglamlik kontrolu olarak korunuyor
    ("Qwen QLoRA adil [kesik 409 tok]", "Qwen3.5-4B", "QLoRA adil (kesik)", "4bit",
     "RTX 4060", "test", "EVET (kostu)",
     os.path.join(OUT, "qlora_eval", "v4_s3_full_2epoch_lr1e4", "test_predictions.csv"),
     "gold", "pred", 0.6220),
    ("Gemma zero-shot bf16", "Gemma-4-E4B", "zero-shot", "bf16", "A100", "test", "HAYIR (OOM, denendi)",
     os.path.join(DL, "gemma_bf16_zeroshot_test.csv"), "label", "parsed_pred", 0.5561),
    ("Gemma zero-shot 4bit", "Gemma-4-E4B", "zero-shot", "4bit", "A100", "test", "HAYIR (OOM, denendi)",
     os.path.join(DL, "gemma_4bit_zeroshot_test.csv"), "label", "parsed_pred", 0.4811),
    ("Gemma tam 3-shot", "Gemma-4-E4B", "3-shot tam", "bf16", "A100", "test", "HAYIR (OOM, denendi)",
     os.path.join(OUT, "gemma_full3shot_test.csv"), "label", "parsed_pred", 0.5566),
    ("Gemma QLoRA optimize", "Gemma-4-E4B", "QLoRA optimize", "4bit", "A100", "test", "HAYIR (OOM, denendi)",
     os.path.join(OUT, "gemma_qlora_test_preds.csv"), "output", "pred", 0.6407),
    # B1 (7 Agu): sikistirilmis few-shot TEST, A100 — tam few-shot ile AYNI donanim
    ("Qwen 3-shot sikistirilmis (A100)", "Qwen3.5-4B", "3-shot sikistirilmis", "4bit",
     "A100", "test", "EVET (2024 token)",
     os.path.join(OUT, "b1_compressed_a100", "b1_compressed_test.csv"),
     "label", "parsed_pred", 0.4929),
    # B2 (7 Agu): tam 3-shot yeniden kosuldu -> per-kayit veri artik VAR
    ("Qwen tam 3-shot (yeniden)", "Qwen3.5-4B", "3-shot tam", "4bit",
     "A100", "test", "HAYIR (~23.8k token)",
     os.path.join(OUT, "b2_b5_verify", "b2_full3shot_test.csv"),
     "label", "parsed_pred", 0.5583),
    # B5 (7 Agu): Gemma adil QLoRA yeniden degerlendirildi -> MF1 payda hatasi BURADA DA vardi
    ("Gemma QLoRA adil (yeniden)", "Gemma-4-E4B", "QLoRA adil", "4bit",
     "A100", "test", "HAYIR (OOM, denendi)",
     os.path.join(OUT, "b2_b5_verify", "b5_gemma_qlora_test.csv"),
     "label", "parsed_pred", 0.6130),
    # val — sikistirilmis few-shot (bildirinin 2. iddiasi)
    ("Qwen zero-shot [VAL]", "Qwen3.5-4B", "zero-shot", "4bit", "RTX 4060", "val", "EVET (kostu)",
     os.path.join(OUT, "zeroshot", "zeroshot_v4_val.csv"), "label", "parsed_pred", 0.4367),
    ("Qwen 3-shot SIKISTIRILMIS [VAL]", "Qwen3.5-4B", "3-shot sikistirilmis", "4bit", "RTX 4060", "val", "EVET (kostu)",
     os.path.join(OUT, "fewshot", "fewshot_v4_3shot_ultracompact_val.csv"), "label", "parsed_pred", 0.4307),
    ("Qwen tam 3-shot [VAL]", "Qwen3.5-4B", "3-shot tam", "4bit", "A100", "val", "HAYIR (OOM, denendi)",
     os.path.join(DL, "full_3shot_val.csv"), "label", "parsed_pred", 0.4781),
]

# per-kayit CSV'si OLMAYAN yapilandirmalar -> tureterek doldur
TUREV = [
    # ad, model, yontem, bit, ortam, split, 8gb, n, acc, tezdeki_mf1, duzeltilmis_mf1, parser_direct, kaynak
    ("Qwen tam 3-shot", "Qwen3.5-4B", "3-shot tam", "4bit", "A100", "test", "HAYIR (OOM, denendi)",
     1650, 0.5552, 0.5153, 0.5621, "%95.5",
     "wF1 sutunundan (dengeli set: wF1 == macroF1/11)"),
    ("Gemma QLoRA adil", "Gemma-4-E4B", "QLoRA adil", "4bit", "A100", "test", "HAYIR (OOM, denendi)",
     1650, 0.6285, 0.5653, 0.5653, "%100",
     "DEGISMEDI (parser %100 direct -> INVALID yok -> payda 11)"),
]

rows = []
percls = {}

print("=" * 104)
print("A0 — MAKRO-F1 YENIDEN URETIMI  (labels=CLASSES)")
print("=" * 104)

for (ad, model, yontem, bit, ortam, split, gb8, path, g, p, tez_mf1) in PERREC:
    df = pd.read_csv(path)
    yt = df[g].astype(str).str.strip()
    yp = df[p].fillna("INVALID").astype(str).str.strip()
    n = len(df)
    k = int((yt == yp).sum())
    acc = k / n
    n_inv = int((~yp.isin(CLASSES)).sum())
    mf1 = f1_score(yt, yp, average="macro", labels=CLASSES, zero_division=0)
    mf1_eski = f1_score(yt, yp, average="macro", zero_division=0)
    a_lo, a_hi = wilson(k, n)
    b_lo, b_hi = boot_mf1(yt.values, yp.values)

    rows.append(dict(
        yapilandirma=ad, model=model, yontem=yontem, bit=bit, ortam=ortam,
        split=split, sigdi_8gb=gb8, n=n, n_invalid=n_inv,
        accuracy=round(acc, 4), acc_ci_lo=round(a_lo, 4), acc_ci_hi=round(a_hi, 4),
        macro_f1_TEZDE=round(mf1_eski, 4), macro_f1_DUZELTILMIS=round(mf1, 4),
        mf1_ci_lo=round(b_lo, 4), mf1_ci_hi=round(b_hi, 4),
        payda_tezde=(12 if n_inv > 0 else 11), payda_dogru=11,
        kaynak="PER-KAYIT CSV"))

    # sinif bazinda F1  (YUVARLAMA YOK — makro ortalama buradan tutarli ciksin)
    P, R, F, S = precision_recall_fscore_support(yt, yp, labels=CLASSES, zero_division=0)
    percls[ad] = dict(zip(CLASSES, F))

    delta = mf1 - tez_mf1
    flag = "  <-- DUZELDI" if abs(delta) > 0.0006 else ""
    print(f"{ad:33s} n={n:5d} INV={n_inv:3d}  acc={acc:.4f}  "
          f"MF1 {tez_mf1:.4f} -> {mf1:.4f}{flag}")

for (ad, model, yontem, bit, ortam, split, gb8, n, acc, tez_mf1, fix_mf1, pdirect, kaynak) in TUREV:
    k = int(round(acc * n))
    a_lo, a_hi = wilson(k, n)
    rows.append(dict(
        yapilandirma=ad, model=model, yontem=yontem, bit=bit, ortam=ortam,
        split=split, sigdi_8gb=gb8, n=n, n_invalid="",
        accuracy=round(acc, 4), acc_ci_lo=round(a_lo, 4), acc_ci_hi=round(a_hi, 4),
        macro_f1_TEZDE=tez_mf1, macro_f1_DUZELTILMIS=fix_mf1,
        mf1_ci_lo="", mf1_ci_hi="",
        payda_tezde=(12 if fix_mf1 != tez_mf1 else 11), payda_dogru=11,
        kaynak=kaynak))
    flag = "  <-- DUZELDI" if abs(fix_mf1 - tez_mf1) > 0.0006 else "  (degismedi)"
    print(f"{ad:33s} n={n:5d} INV={'?':>3s}  acc={acc:.4f}  "
          f"MF1 {tez_mf1:.4f} -> {fix_mf1:.4f}{flag}")

master = pd.DataFrame(rows)
dest = os.path.join(BASE, "results_master.csv")
master.to_csv(dest, index=False, encoding="utf-8-sig")

# sinif bazinda F1 tablosu
pc = pd.DataFrame(percls)
pc.index.name = "sinif"
pc.loc["MACRO-F1"] = pc.mean()          # once tam duyarlikta ortala
pc = pc.round(4)                         # sonra yuvarla
dest2 = os.path.join(BASE, "per_class_f1.csv")
pc.to_csv(dest2, encoding="utf-8-sig")

# tutarlilik kontrolu: per-class ortalamasi results_master'daki MF1 ile ayni mi?
print("\n--- TUTARLILIK KONTROLU (per-class ortalamasi <-> results_master MF1) ---")
for ad in percls:
    a = float(pc.loc["MACRO-F1", ad])
    b = float(master.loc[master["yapilandirma"] == ad, "macro_f1_DUZELTILMIS"].iloc[0])
    print(f"   {ad:33s} {a:.4f} vs {b:.4f}  {'OK' if abs(a-b) < 5e-5 else 'FARK!'}")

print("\n" + "=" * 104)
print("Kaydedildi:", dest)
print("Kaydedildi:", dest2)
print("=" * 104)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(master[["yapilandirma", "bit", "ortam", "split", "sigdi_8gb", "n",
              "accuracy", "acc_ci_lo", "acc_ci_hi",
              "macro_f1_TEZDE", "macro_f1_DUZELTILMIS", "kaynak"]].to_string(index=False))

print("\n--- SINIF BAZINDA F1 (test) ---")
print(pc[[c for c in pc.columns if "VAL" not in c]].to_string())
