# =====================================================================
# BLOK 02 — Veriyi Colab'a yukle (files.upload, elle)
# =====================================================================
# YUKLENECEK 3 DOSYA (local'de su yollardan al, surukle-birak):
#   1) discharge_split.parquet   (~83 MB) — val+test queryleri
#      KONUM: workdir/output/discharge_split.parquet
#   2) fewshot_v4_3shot_examples.csv   (~76 KB) — FULL 3-shot ornekleri (11x3=33 satir)
#      KONUM: workdir/output/fewshot/fewshot_v4_3shot_examples.csv
#   3) fewshot_v4_1shot_examples.csv   (~28 KB) — FULL 1-shot (11 satir)
#      KONUM: workdir/output/fewshot/fewshot_v4_1shot_examples.csv
# NOT: CSV de JSON ile ayni veriyi tasir (label + triage_input_v4_timeaware_masked).
# =====================================================================
# KULLANIM: Blogu calistir, cikan "Dosya Sec" ile 3 dosyayi birden sec.
# 83 MB ~1-2 dk surer. Hepsi /content/ altina iner.
# =====================================================================

from google.colab import files
import os, pandas as pd

# Dosyalar zaten yuklendiyse bu satiri yorum yap; tekrar yuklemek istersen ac:
# uploaded = files.upload()

# --- Yollar (calisma dizini /content) ---
SPLIT_PATH      = "discharge_split.parquet"
EX_3SHOT_PATH   = "fewshot_v4_3shot_examples.csv"
EX_1SHOT_PATH   = "fewshot_v4_1shot_examples.csv"

# --- 1) Dogrulama: dosyalar var mi? ---
for p in [SPLIT_PATH, EX_3SHOT_PATH, EX_1SHOT_PATH]:
    assert os.path.exists(p), f"EKSIK: {p} yuklenmedi! Tekrar yukle."
    print(f"OK  {p}  ({os.path.getsize(p)/1e6:.2f} MB)")

# --- 2) Parquet dogrula: split kolonu + V4 input kolonu + val/test sayilari ---
df = pd.read_parquet(SPLIT_PATH)
print(f"\nParquet: {len(df):,} satir, {df.shape[1]} kolon")
assert "split" in df.columns, "split kolonu yok!"
assert "triage_input_v4_timeaware_masked" in df.columns, "V4 input kolonu yok!"
assert "label" in df.columns, "label kolonu yok!"
print("split dagilimi:")
print(df["split"].value_counts())
val_n  = (df["split"]=="val").sum()
test_n = (df["split"]=="test").sum()
print(f"\nval={val_n} (hedef 550), test={test_n} (hedef 1650)")
assert val_n == 550 and test_n == 1650, "val/test sayilari beklenenle uyusmuyor!"

# --- 3) Few-shot ornekleri (CSV -> dict: {sinif: [ornek, ...]}) ---
CLASSES = ["SEPSIS","STROKE","ATRIAL_FIBRILLATION","AKI","PNEUMONIA","UTI",
           "HEART_FAILURE","GI_BLEEDING","DIABETES_COMPLICATION","COPD_EXACERBATION","OTHER"]
V4COL = "triage_input_v4_timeaware_masked"

def csv_to_dict(path):
    d = pd.read_csv(path)
    assert "label" in d.columns and V4COL in d.columns, f"{path}: gerekli kolonlar yok!"
    out = {}
    for cls, grp in d.groupby("label"):
        # rank varsa ona gore sirala (orijinal secim sirasi korunur)
        if "rank" in grp.columns:
            grp = grp.sort_values("rank")
        out[cls] = grp.to_dict("records")
    return out

ex3 = csv_to_dict(EX_3SHOT_PATH)   # {sinif: [3 ornek]}
ex1 = csv_to_dict(EX_1SHOT_PATH)   # {sinif: [1 ornek]}

print(f"\n3-shot: {len(ex3)} sinif, sinif basina {len(ex3['SEPSIS'])} ornek "
      f"(toplam {sum(len(v) for v in ex3.values())})")
assert set(ex3.keys()) == set(CLASSES), "3-shot sinif listesi eksik/farkli!"
assert V4COL in ex3["SEPSIS"][0], "ornekte V4 input alani yok!"
toklens = [e["token_uzunluk"] for v in ex3.values() for e in v if "token_uzunluk" in e]
if toklens:
    print(f"3-shot ornek token uzunlugu: ort {sum(toklens)/len(toklens):.0f}, "
          f"max {max(toklens)} (FULL, sikistirilmamis)")

print("\n✅ BLOK 02 TAMAM — veri hazir.")
print("   Degiskenler: df (split), ex3 (full 3-shot), ex1 (full 1-shot), CLASSES, V4COL")
print("   Sonraki blok: Full 3-shot prompt build + val inference.")
