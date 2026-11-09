"""Nested split + JSON dosyalari dogrulama.

Kullanici talebi:
- note_id set(S1) subset set(S2)
- note_id set(S2) subset set(S3)
- her split icin sinif dagilimi 11 sinifta dengeli
- mevcut discharge_split.parquet split yapisi korunmus mu (yeniden sampling YOK)
"""
import pandas as pd
import json
from pathlib import Path
from collections import Counter

OUTPUT_DIR = Path(r"./workdir/output")
SPLIT_YOL = OUTPUT_DIR / "discharge_split.parquet"
JSON_DIR = OUTPUT_DIR / "qlora_json"

print("=" * 70)
print("NESTED SPLIT + V4 JSON DOGRULAMA")
print("=" * 70)

# 1) PARQUET KAYNAK DOGRULAMA
df = pd.read_parquet(SPLIT_YOL)
parquet_train_S3 = df[df["split"] == "train_S3"]
parquet_S2 = df[(df["split"] == "train_S3") & df["in_S2"]]
parquet_S1 = df[(df["split"] == "train_S3") & df["in_S1"]]

p_S1 = set(parquet_S1["note_id"])
p_S2 = set(parquet_S2["note_id"])
p_S3 = set(parquet_train_S3["note_id"])

print(f"\n[1/3] discharge_split.parquet kontrolu:")
print(f"   |S1| = {len(p_S1)}, |S2| = {len(p_S2)}, |S3| = {len(p_S3)}")
print(f"   S1 ⊂ S2: {p_S1.issubset(p_S2)}  (eksik: {len(p_S1 - p_S2)})")
print(f"   S2 ⊂ S3: {p_S2.issubset(p_S3)}  (eksik: {len(p_S2 - p_S3)})")
print(f"   S1 ⊂ S3: {p_S1.issubset(p_S3)}  (eksik: {len(p_S1 - p_S3)})")

print(f"\n   Sinif dagilimi (parquet):")
for ad, sdf in [("S1", parquet_S1), ("S2", parquet_S2), ("S3", parquet_train_S3)]:
    counts = sdf["label"].value_counts().sort_index()
    siniflar = sorted(counts.index.tolist())
    n_sinif = len(siniflar)
    minv = counts.min()
    maxv = counts.max()
    print(f"   {ad}: {n_sinif} sinif, min={minv}, max={maxv}, total={counts.sum()}")
    if minv != maxv:
        print(f"      DENGESIZ! {dict(counts)}")
    else:
        print(f"      DENGELI (her sinif {minv})")

# 2) URETILEN JSON DOSYALARI DOGRULAMA
print(f"\n[2/3] qlora_v4_*.json dosyalari kontrolu:")
def yukle(name):
    with open(JSON_DIR / name, "r", encoding="utf-8") as f:
        return json.load(f)

j_S1 = yukle("qlora_v4_train_S1.json")
j_S2 = yukle("qlora_v4_train_S2.json")
j_S3 = yukle("qlora_v4_train_S3.json")
j_val = yukle("qlora_v4_val.json")
j_test = yukle("qlora_v4_test.json")

print(f"   Boyutlar:")
print(f"      S1 JSON   : {len(j_S1):>5,}  (hedef 1.100)")
print(f"      S2 JSON   : {len(j_S2):>5,}  (hedef 2.200)")
print(f"      S3 JSON   : {len(j_S3):>5,}  (hedef 4.400)")
print(f"      val JSON  : {len(j_val):>5,}  (hedef   550)")
print(f"      test JSON : {len(j_test):>5,}  (hedef 1.650)")

# note_id setleri
n_S1 = set(r["note_id"] for r in j_S1)
n_S2 = set(r["note_id"] for r in j_S2)
n_S3 = set(r["note_id"] for r in j_S3)
n_val = set(r["note_id"] for r in j_val)
n_test = set(r["note_id"] for r in j_test)

print(f"\n   JSON nested:")
print(f"      S1 ⊂ S2: {n_S1.issubset(n_S2)}  (eksik: {len(n_S1 - n_S2)})")
print(f"      S2 ⊂ S3: {n_S2.issubset(n_S3)}  (eksik: {len(n_S2 - n_S3)})")
print(f"      S1 ⊂ S3: {n_S1.issubset(n_S3)}  (eksik: {len(n_S1 - n_S3)})")

print(f"\n   JSON disjoint (sızıntı yok):")
print(f"      S3 ∩ val   = {len(n_S3 & n_val)}  (olmali 0)")
print(f"      S3 ∩ test  = {len(n_S3 & n_test)}  (olmali 0)")
print(f"      val ∩ test = {len(n_val & n_test)}  (olmali 0)")

# parquet vs json eslesme
print(f"\n   parquet vs JSON eslesme (yeniden sampling kontrolu):")
print(f"      JSON S1 == parquet S1: {n_S1 == p_S1}  (fark: {len(n_S1 ^ p_S1)})")
print(f"      JSON S2 == parquet S2: {n_S2 == p_S2}  (fark: {len(n_S2 ^ p_S2)})")
print(f"      JSON S3 == parquet S3: {n_S3 == p_S3}  (fark: {len(n_S3 ^ p_S3)})")

# JSON sinif dagilimi (her dosya)
print(f"\n   JSON sinif dagilimi:")
for ad, recs in [("S1", j_S1), ("S2", j_S2), ("S3", j_S3), ("val", j_val), ("test", j_test)]:
    c = Counter([r["output"] for r in recs])
    siniflar = sorted(c.keys())
    minv = min(c.values())
    maxv = max(c.values())
    durum = "DENGELI" if minv == maxv else "DENGESIZ!"
    print(f"      {ad:5s}: {len(siniflar)} sinif, min={minv:>4}, max={maxv:>4}  {durum}")

# 3) SUBJECT_ID AYRIKLIK (bonus, daha onemli)
print(f"\n[3/3] subject_id ayrıklık (data leakage kontrolu):")
sub_S3 = set(parquet_train_S3["subject_id"])
sub_val = set(df[df["split"] == "val"]["subject_id"])
sub_test = set(df[df["split"] == "test"]["subject_id"])
print(f"   S3 ∩ val_subj  = {len(sub_S3 & sub_val)}  (olmali 0)")
print(f"   S3 ∩ test_subj = {len(sub_S3 & sub_test)}  (olmali 0)")
print(f"   val ∩ test_subj = {len(sub_val & sub_test)}  (olmali 0)")

print("\n" + "=" * 70)
print("DOGRULAMA TAMAMLANDI.")
print("=" * 70)
