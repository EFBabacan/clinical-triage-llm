"""hpi_clean_v3'te kalan imaging kalintisini detayli analiz et."""
import pandas as pd
import re
import random
from pathlib import Path

OUTPUT_DIR = Path(r"./workdir/output")
df = pd.read_parquet(OUTPUT_DIR / "discharge_split.parquet")

IMG_RX = re.compile(r'\b(CXR|CT (?:head|chest|abdomen)|CTA|MRI|TTE)\b', re.IGNORECASE)

# Sadece hpi_clean_v3'te imaging kalintisi olan kayitlar
hedef = []
for _, row in df.iterrows():
    hpi = row.get("hpi_clean_v3", "") or ""
    pmh = row.get("pmh_v3", "") or ""
    m_hpi = IMG_RX.search(hpi)
    m_pmh = IMG_RX.search(pmh)
    if m_hpi:  # hpi'da var
        hedef.append({
            "note_id": row["note_id"],
            "label": row["label"],
            "hpi_match": m_hpi.group(),
            "hpi_konum_pct": 100 * m_hpi.start() / max(len(hpi), 1),
            "hpi_only": not m_pmh,
        })

df_h = pd.DataFrame(hedef)
print(f"hpi_clean_v3'te imaging kalintisi: {len(df_h)} kayit")
print(f"  Sadece hpi'da (PMH'de yok)     : {df_h['hpi_only'].sum()}")
print(f"  Hem hpi hem pmh'da             : {(~df_h['hpi_only']).sum()}")

print(f"\nKonum dagilimi (hpi icindeki yuzde):")
print(f"  Ilk %25  : {(df_h['hpi_konum_pct']<25).sum()}")
print(f"  %25-50   : {((df_h['hpi_konum_pct']>=25)&(df_h['hpi_konum_pct']<50)).sum()}")
print(f"  %50-75   : {((df_h['hpi_konum_pct']>=50)&(df_h['hpi_konum_pct']<75)).sum()}")
print(f"  Son %25  : {(df_h['hpi_konum_pct']>=75).sum()}")

print(f"\nMatch terimleri:")
print(df_h['hpi_match'].value_counts().to_dict())

# 5 rastgele ornek
random.seed(0)
ornekler = random.sample(range(len(df_h)), min(5, len(df_h)))
print("\n" + "=" * 70)
print("5 RASTGELE ORNEK (hpi_clean_v3'te imaging kalintisi)")
print("=" * 70)
for i in ornekler:
    row = df_h.iloc[i]
    nid = row["note_id"]
    full_row = df[df["note_id"] == nid].iloc[0]
    hpi = full_row["hpi_clean_v3"]
    m = IMG_RX.search(hpi)
    konum = m.start()
    baslangic = max(0, konum - 200)
    son = min(len(hpi), konum + 200)
    baglam = hpi[baslangic:son]
    eslesen = m.group()
    baglam_vurgu = baglam.replace(eslesen, f">>>>{eslesen}<<<<")

    print(f"\n--- {nid} (label: {row['label']}, match: {eslesen!r} @ {row['hpi_konum_pct']:.0f}%) ---")
    print(f"...{baglam_vurgu}...")
