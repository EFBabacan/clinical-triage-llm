# -*- coding: utf-8 -*-
"""
1) Her yapilandirmada kac INVALID var -> macro-F1 paydasi 11 mi 12 mi?
2) Payda hatasi duzeltilince iki ANA IDDIA ne oluyor?
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

OUT = "./workdir/output"  # workspace/workdir (see README)
DL = os.path.join(OUT, "colab_downloads")  # Colab sonuc dosyalari (5 Kasim 2026: Indirilenler'den projeye tasindi)

CLASSES = ["SEPSIS", "STROKE", "ATRIAL_FIBRILLATION", "AKI", "PNEUMONIA", "UTI",
           "HEART_FAILURE", "GI_BLEEDING", "DIABETES_COMPLICATION",
           "COPD_EXACERBATION", "OTHER"]
CS = set(CLASSES)

FILES = [
    ("Qwen zero-shot 4bit",        os.path.join(OUT, "zeroshot", "zeroshot_v4_test.csv"), "label", "parsed_pred", "test"),
    ("Qwen QLoRA adil",            os.path.join(OUT, "qlora_eval", "v4_s3_full_2epoch_lr1e4", "test_predictions.csv"), "gold", "pred", "test"),
    ("Gemma zero-shot bf16",       os.path.join(DL, "gemma_bf16_zeroshot_test.csv"), "label", "parsed_pred", "test"),
    ("Gemma zero-shot 4bit",       os.path.join(DL, "gemma_4bit_zeroshot_test.csv"), "label", "parsed_pred", "test"),
    ("Gemma full 3-shot bf16",     os.path.join(OUT, "gemma_full3shot_test.csv"), "label", "parsed_pred", "test"),
    ("Gemma QLoRA optimize",       os.path.join(OUT, "gemma_qlora_test_preds.csv"), "output", "pred", "test"),
    ("Qwen zero-shot [VAL]",       os.path.join(OUT, "zeroshot", "zeroshot_v4_val.csv"), "label", "parsed_pred", "val"),
    ("Qwen 3-shot SIKISTIRILMIS [VAL]", os.path.join(OUT, "fewshot", "fewshot_v4_3shot_ultracompact_val.csv"), "label", "parsed_pred", "val"),
    ("Qwen full 3-shot [VAL]",     os.path.join(DL, "full_3shot_val.csv"), "label", "parsed_pred", "val"),
    ("Gemma zero-shot bf16 [VAL]", os.path.join(DL, "gemma_zeroshot_val.csv"), "label", "parsed_pred", "val"),
]

print("=" * 100)
print("1) INVALID SAYIMI VE MACRO-F1 PAYDASI")
print("=" * 100)
print(f"{'yapilandirma':34s} {'n':>5s} {'INVALID':>8s} {'payda':>6s} {'MF1(/12 rapor)':>15s} {'MF1(/11 dogru)':>15s} {'fark':>7s}")
print("-" * 100)

store = {}
for name, path, g, p, split in FILES:
    if not os.path.exists(path):
        print(f"{name:34s}  DOSYA YOK")
        continue
    df = pd.read_csv(path)
    yt = df[g].astype(str).str.strip()
    yp = df[p].fillna("INVALID").astype(str).str.strip()
    n_inv = int((~yp.isin(CS)).sum())
    denom = len(set(yt) | set(yp))
    mf1_rep = f1_score(yt, yp, average="macro", zero_division=0)          # labels= YOK  (tezdeki hali)
    mf1_fix = f1_score(yt, yp, average="macro", labels=CLASSES, zero_division=0)  # dogru
    acc = float((yt == yp).mean())
    store[name] = dict(acc=acc, mf1_rep=mf1_rep, mf1_fix=mf1_fix, n=len(df), split=split)
    print(f"{name:34s} {len(df):5d} {n_inv:8d} {denom:6d} {mf1_rep:15.4f} {mf1_fix:15.4f} {mf1_fix-mf1_rep:+7.4f}")


def wilson_diff_note(p1, n1, p2, n2):
    """iki oran farkinin yaklasik %95 CI'si (bagimsiz varsayimi -> muhafazakar)"""
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    d = p1 - p2
    return d, d - 1.96 * se, d + 1.96 * se


print("\n" + "=" * 100)
print("2) DUZELTMENIN IKI ANA IDDIAYA ETKISI")
print("=" * 100)

a = store["Gemma zero-shot bf16"]
b = store["Gemma zero-shot 4bit"]
print("\n--- IDDIA 1 (B6): 4-bit niceleme Gemma'da Makro-F1'i dusuruyor ---")
print(f"  Tezdeki hali : MF1 {a['mf1_rep']:.4f} -> {b['mf1_rep']:.4f}  = {100*(b['mf1_rep']-a['mf1_rep']):+.1f} puan")
print(f"  DUZELTILMIS  : MF1 {a['mf1_fix']:.4f} -> {b['mf1_fix']:.4f}  = {100*(b['mf1_fix']-a['mf1_fix']):+.1f} puan")
d, lo, hi = wilson_diff_note(b["acc"], b["n"], a["acc"], a["n"])
print(f"  accuracy farki: {100*d:+.1f} puan, %95 CI [{100*lo:+.1f}, {100*hi:+.1f}]  "
      f"-> {'ANLAMLI' if hi < 0 or lo > 0 else 'GURULTU ICINDE'}")

z = store["Qwen zero-shot [VAL]"]
c = store["Qwen 3-shot SIKISTIRILMIS [VAL]"]
f = store["Qwen full 3-shot [VAL]"]
print("\n--- IDDIA 2 (B1): Sikistirma few-shot'i zero-shot'in ALTINA dusuruyor (val, n=550) ---")
print(f"  Tezdeki hali : zero {z['mf1_rep']:.4f} vs sikistirilmis {c['mf1_rep']:.4f} = {100*(c['mf1_rep']-z['mf1_rep']):+.1f} puan MF1")
print(f"  DUZELTILMIS  : zero {z['mf1_fix']:.4f} vs sikistirilmis {c['mf1_fix']:.4f} = {100*(c['mf1_fix']-z['mf1_fix']):+.1f} puan MF1")
d, lo, hi = wilson_diff_note(c["acc"], c["n"], z["acc"], z["n"])
print(f"  accuracy farki: {100*d:+.1f} puan, %95 CI [{100*lo:+.1f}, {100*hi:+.1f}]  "
      f"-> {'ANLAMLI' if hi < 0 or lo > 0 else 'GURULTU ICINDE'}")
d, lo, hi = wilson_diff_note(f["acc"], f["n"], z["acc"], z["n"])
print(f"  (karsilastirma) TAM few-shot vs zero: {100*d:+.1f} puan, %95 CI [{100*lo:+.1f}, {100*hi:+.1f}]  "
      f"-> {'ANLAMLI' if hi < 0 or lo > 0 else 'GURULTU ICINDE'}")

print("\n--- Yan kontrol: Qwen QLoRA vs Gemma QLoRA-adil farki (devir notu madde 4) ---")
q = store["Qwen QLoRA adil"]
print(f"  Qwen QLoRA acc {q['acc']:.4f} (n={q['n']}) — Gemma adil QLoRA icin per-kayit CSV YOK, "
      f"rapor 0.6285")
d, lo, hi = wilson_diff_note(q["acc"], q["n"], 0.6285, 1650)
print(f"  fark {100*d:+.1f} puan, %95 CI [{100*lo:+.1f}, {100*hi:+.1f}] -> "
      f"{'ANLAMLI' if hi < 0 or lo > 0 else 'GURULTU ICINDE'}")
