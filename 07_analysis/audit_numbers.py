# -*- coding: utf-8 -*-
"""
Tez tablosundaki her rakami ham tahmin CSV'lerinden yeniden hesaplar.
Ayrica Wilson CI (accuracy) + bootstrap CI (macro-F1) uretir  -> K5.
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

OUT = r"./workdir/output"
DL = os.path.join(OUT, "colab_downloads")  # Colab sonuc dosyalari (5 Kasim 2026: Indirilenler'den projeye tasindi)

CLASSES = ["SEPSIS", "STROKE", "ATRIAL_FIBRILLATION", "AKI", "PNEUMONIA", "UTI",
           "HEART_FAILURE", "GI_BLEEDING", "DIABETES_COMPLICATION",
           "COPD_EXACERBATION", "OTHER"]

# ad, yol, gold-kolonu, pred-kolonu, raporlanan acc, raporlanan mf1, split
CONFIGS = [
    ("Qwen zero-shot 4bit",        os.path.join(OUT, "zeroshot", "zeroshot_v4_test.csv"),
     "label", "parsed_pred", 0.5152, 0.4681, "test"),
    ("Qwen QLoRA adil 4bit",       os.path.join(OUT, "qlora_eval", "v4_s3_full_2epoch_lr1e4", "test_predictions.csv"),
     "gold", "pred", 0.6345, 0.6220, "test"),
    ("Gemma zero-shot bf16",       os.path.join(DL, "gemma_bf16_zeroshot_test.csv"),
     "label", "parsed_pred", 0.5648, 0.5561, "test"),
    ("Gemma zero-shot 4bit",       os.path.join(DL, "gemma_4bit_zeroshot_test.csv"),
     "label", "parsed_pred", 0.5388, 0.4811, "test"),
    ("Gemma full 3-shot bf16",     os.path.join(OUT, "gemma_full3shot_test.csv"),
     "label", "parsed_pred", 0.5624, 0.5566, "test"),
    ("Gemma QLoRA optimize 4bit",  os.path.join(OUT, "gemma_qlora_test_preds.csv"),
     "output", "pred", 0.6467, 0.6407, "test"),
    # --- val tarafi ---
    ("Qwen zero-shot 4bit [VAL]",  os.path.join(OUT, "zeroshot", "zeroshot_v4_val.csv"),
     "label", "parsed_pred", 0.4818, 0.4367, "val"),
    ("Qwen 3-shot SIKISTIRILMIS [VAL]", os.path.join(OUT, "fewshot", "fewshot_v4_3shot_ultracompact_val.csv"),
     "label", "parsed_pred", 0.4709, 0.4307, "val"),
    ("Qwen full 3-shot [VAL]",     os.path.join(DL, "full_3shot_val.csv"),
     "label", "parsed_pred", 0.5182, 0.4781, "val"),
    ("Gemma zero-shot bf16 [VAL]", os.path.join(DL, "gemma_zeroshot_val.csv"),
     "label", "parsed_pred", 0.5436, 0.5434, "val"),
    ("Gemma zero-shot 4bit [VAL]", os.path.join(DL, "gemma_4bit_zeroshot_val.csv"),
     "label", "parsed_pred", None, None, "val"),
]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def boot_mf1(y_true, y_pred, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = f1_score(yt[idx], yp[idx], average="macro",
                           labels=CLASSES, zero_division=0)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


rows = []
print("=" * 108)
print("HAM CSV'DEN YENIDEN HESAPLAMA  (raporlanan <-> hesaplanan)")
print("=" * 108)

for name, path, gcol, pcol, rep_acc, rep_mf1, split in CONFIGS:
    if not os.path.exists(path):
        print(f"\n[YOK] {name}\n      {path}")
        continue
    df = pd.read_csv(path)
    if gcol not in df.columns or pcol not in df.columns:
        print(f"\n[KOLON YOK] {name} -> kolonlar: {list(df.columns)}")
        continue

    y_true = df[gcol].astype(str).str.strip()
    y_pred_raw = df[pcol]
    n_unparsed = int(y_pred_raw.isna().sum())
    y_pred = y_pred_raw.fillna("UNPARSED").astype(str).str.strip()

    n = len(df)
    k = int((y_true == y_pred).sum())
    acc = k / n
    lo, hi = wilson(k, n)

    # iki farkli macro-F1 tanimi
    mf1_11 = f1_score(y_true, y_pred, average="macro", labels=CLASSES, zero_division=0)
    mf1_union = f1_score(y_true, y_pred, average="macro", zero_division=0)

    b_lo, b_hi = boot_mf1(y_true.values, y_pred.values)

    def mark(calc, rep):
        if rep is None:
            return "  (rapor yok)"
        return "  OK" if abs(calc - rep) < 0.0006 else f"  <-- FARK (rapor {rep:.4f})"

    print(f"\n### {name}   [{split}]  n={n}  parse-edilemeyen={n_unparsed}")
    print(f"    accuracy   = {acc:.4f}{mark(acc, rep_acc)}")
    print(f"      %95 Wilson CI = [{lo:.4f}, {hi:.4f}]  (+/- {100*(hi-lo)/2:.1f} puan)")
    print(f"    macroF1(11 sinif) = {mf1_11:.4f}{mark(mf1_11, rep_mf1)}")
    print(f"    macroF1(union)    = {mf1_union:.4f}{mark(mf1_union, rep_mf1)}")
    print(f"      %95 bootstrap CI (11 sinif, 1000 tekrar) = [{b_lo:.4f}, {b_hi:.4f}]")

    rows.append(dict(config=name, split=split, n=n, unparsed=n_unparsed,
                     acc=round(acc, 4), acc_lo=round(lo, 4), acc_hi=round(hi, 4),
                     macro_f1_11=round(mf1_11, 4), macro_f1_union=round(mf1_union, 4),
                     mf1_lo=round(b_lo, 4), mf1_hi=round(b_hi, 4),
                     rapor_acc=rep_acc, rapor_mf1=rep_mf1))

res = pd.DataFrame(rows)
dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_master_DRAFT.csv")
res.to_csv(dest, index=False, encoding="utf-8-sig")
print("\n" + "=" * 108)
print("Kaydedildi:", dest)
print(res.to_string(index=False))
