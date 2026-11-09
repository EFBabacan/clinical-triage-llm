"""Blok 15A - V4 QLoRA JSON Formati (5 dosya).

discharge_split.parquet -> instruction-input-output JSON formati (V4 only).
Input kolonu: triage_input_v4_timeaware_masked.
Cikti: 5 dosya (S1/S2/S3/val/test).
"""
import pandas as pd
import json
import time
from pathlib import Path
from collections import Counter

OUTPUT_DIR = Path(r"./workdir/output")
SPLIT_YOL  = OUTPUT_DIR / "discharge_split.parquet"
JSON_DIR   = OUTPUT_DIR / "qlora_json"
JSON_DIR.mkdir(parents=True, exist_ok=True)

V4_INPUT_KOLONU = "triage_input_v4_timeaware_masked"

INSTRUCTION = (
    "Classify the following admission note into exactly one of these 11 classes: "
    "SEPSIS, STROKE, ATRIAL_FIBRILLATION, AKI, PNEUMONIA, UTI, HEART_FAILURE, "
    "GI_BLEEDING, DIABETES_COMPLICATION, COPD_EXACERBATION, OTHER. "
    "Output only the class name."
)

print("=" * 70)
print("BLOK 15A - V4 QLoRA JSON FORMATI (5 dosya)")
print("=" * 70)

# 1) YUKLE + FILTRELE
print("\n[1/4] discharge_split.parquet yukleniyor...")
t0 = time.time()
df = pd.read_parquet(SPLIT_YOL)
print(f"   Yuklendi: {len(df):,} satir, {len(df.columns)} kolon")
print(f"   Sure: {time.time()-t0:.1f}s")

assert V4_INPUT_KOLONU in df.columns, f"V4 input kolonu yok: {V4_INPUT_KOLONU}"

test_df     = df[df["split"] == "test"]
val_df      = df[df["split"] == "val"]
train_S3_df = df[df["split"] == "train_S3"]
train_S2_df = df[(df["split"] == "train_S3") & df["in_S2"]]
train_S1_df = df[(df["split"] == "train_S3") & df["in_S1"]]

print(f"\n   Bolum boyutlari:")
print(f"      Test     : {len(test_df):>5,}  (hedef 1.650)")
print(f"      Val      : {len(val_df):>5,}  (hedef   550)")
print(f"      Train S3 : {len(train_S3_df):>5,}  (hedef 4.400)")
print(f"      Train S2 : {len(train_S2_df):>5,}  (hedef 2.200, S3 alt kumesi)")
print(f"      Train S1 : {len(train_S1_df):>5,}  (hedef 1.100, S2 alt kumesi)")

s1_set = set(train_S1_df["note_id"])
s2_set = set(train_S2_df["note_id"])
s3_set = set(train_S3_df["note_id"])
assert s1_set.issubset(s2_set), "S1 S2'nin alt kumesi degil!"
assert s2_set.issubset(s3_set), "S2 S3'un alt kumesi degil!"
print(f"\n   Nested dogrulama: S1 in S2 = True, S2 in S3 = True")

# Bos input kontrolu
for ad, sdf in [("test", test_df), ("val", val_df), ("S3", train_S3_df)]:
    bos = sdf[V4_INPUT_KOLONU].isna().sum() + (sdf[V4_INPUT_KOLONU].astype(str).str.strip() == "").sum()
    print(f"   Bos V4 input ({ad}): {bos}")

# 2) JSON URETICI
def df_to_json_records(subset_df, input_kolonu):
    records = []
    for _, row in subset_df.iterrows():
        records.append({
            "note_id": str(row["note_id"]),
            "subject_id": int(row["subject_id"]),
            "hadm_id": int(row["hadm_id"]),
            "instruction": INSTRUCTION,
            "input": str(row[input_kolonu]) if pd.notna(row[input_kolonu]) else "",
            "output": str(row["label"])
        })
    return records

def json_yaz(records, dosya_adi):
    yol = JSON_DIR / dosya_adi
    with open(yol, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    boyut_mb = yol.stat().st_size / (1024 * 1024)
    return yol, boyut_mb

# 3) 5 DOSYA URET
print("\n[2/4] 5 V4 JSON dosyasi olusturuluyor...")
print("-" * 70)

set_eslemesi = [
    ("train_S1", train_S1_df),
    ("train_S2", train_S2_df),
    ("train_S3", train_S3_df),
    ("val",      val_df),
    ("test",     test_df),
]

tum_dosyalar = []
t0 = time.time()
for set_ad, set_df in set_eslemesi:
    dosya_adi = f"qlora_v4_{set_ad}.json"
    records = df_to_json_records(set_df, V4_INPUT_KOLONU)
    yol, boyut = json_yaz(records, dosya_adi)
    tum_dosyalar.append({"dosya": dosya_adi, "kayit": len(records), "boyut_mb": boyut})
    print(f"   {dosya_adi:<28s} {len(records):>5,} kayit  {boyut:>6.1f} MB")

print(f"\n   Toplam sure: {time.time()-t0:.1f}s")

# 4) AUDIT
print("\n[3/4] Audit raporu...")
print("-" * 70)

print(f"\n   Uretilen V4 dosyalari:")
print(f"   {'Dosya':<28s} {'Kayit':>7s}  {'MB':>6s}")
print("   " + "-" * 46)
toplam_kayit = 0
toplam_mb = 0
for d in tum_dosyalar:
    print(f"   {d['dosya']:<28s} {d['kayit']:>7,}  {d['boyut_mb']:>6.1f}")
    toplam_kayit += d["kayit"]
    toplam_mb += d["boyut_mb"]
print("   " + "-" * 46)
print(f"   {'TOPLAM':<28s} {toplam_kayit:>7,}  {toplam_mb:>6.1f}")

# Test sinif dagilimi
test_v4_yol = JSON_DIR / "qlora_v4_test.json"
with open(test_v4_yol, "r", encoding="utf-8") as f:
    test_v4 = json.load(f)
siniflar = Counter([r["output"] for r in test_v4])
print(f"\n   Test seti V4 sinif dagilimi (hedef 150/sinif):")
for cls in sorted(siniflar.keys()):
    print(f"      {cls:<25s} {siniflar[cls]:>4}")

# Ornek kayit
print(f"\n   Ornek kayit (qlora_v4_test.json[0]):")
ornek = test_v4[0]
print(f"      note_id    : {ornek['note_id']}")
print(f"      subject_id : {ornek['subject_id']}")
print(f"      hadm_id    : {ornek['hadm_id']}")
print(f"      instruction: {ornek['instruction'][:80]}...")
print(f"      input      : {ornek['input'][:200]}...")
print(f"      output     : {ornek['output']}")

# Token uzunluk dagilimi (kabaca char/4)
print(f"\n   Input uzunluk dagilimi (V4 train_S3, char ve ~token tahmini):")
s3_yol = JSON_DIR / "qlora_v4_train_S3.json"
with open(s3_yol, "r", encoding="utf-8") as f:
    s3 = json.load(f)
char_lens = [len(r["input"]) for r in s3]
import statistics as stats
print(f"      char  mean={stats.mean(char_lens):.0f}  median={stats.median(char_lens):.0f}  max={max(char_lens)}")
approx_tok = [c / 4 for c in char_lens]
print(f"      ~tok  mean={stats.mean(approx_tok):.0f}  median={stats.median(approx_tok):.0f}  max={max(approx_tok):.0f}")
print(f"      (max_seq_len=2048 hedefi icin token audit Blok 15B'de yapilacak)")

# 5) FEW-SHOT POOL DOGRULAMA + OZET
print("\n[4/4] Few-shot pool kontrolu (V4)...")
print("-" * 70)
pool_yol = JSON_DIR / "qlora_v4_train_S3.json"
with open(pool_yol, "r", encoding="utf-8") as f:
    pool = json.load(f)
pool_siniflar = Counter([r["output"] for r in pool])
print(f"   {pool_yol.name}: {len(pool):,} kayit")
print(f"   Sinif dagilimi: {dict(sorted(pool_siniflar.items()))}")
assert len(pool) == 4400, f"V4 train_S3 4.400 olmali, {len(pool)}"
assert len(test_v4) == 1650, f"V4 test 1.650 olmali, {len(test_v4)}"

print("\n" + "=" * 70)
print("BLOK 15A TAMAMLANDI.")
print("=" * 70)
print(f"   Cikti klasoru: {JSON_DIR}")
print(f"   Toplam V4 dosyasi: 5")
print(f"   Toplam kayit     : {toplam_kayit:,}")
print(f"   Toplam boyut     : {toplam_mb:.1f} MB")
print("\nSiradaki: Blok 15B - V4 QLoRA-S1 train (1.100 ornek, ~30 dk)")
