"""Blok 13A.1: Audit flag'li kayitlardan ornek baglam goster.
Amac: Kalintilarin GECMIS PMH mi yoksa MEVCUT BASVURU workup'i mi oldugunu anla."""

import pandas as pd
import re
import random
from pathlib import Path

OUTPUT_DIR = Path(r"./workdir/output")
SPLIT_YOL  = OUTPUT_DIR / "discharge_split.parquet"

print("=" * 70)
print("BLOK 13A.1 — KALINTI AUDIT (her flag tipinden ornekler)")
print("=" * 70)

df = pd.read_parquet(SPLIT_YOL)
print(f"\n   Toplam: {len(df):,} kayit")

# Flag dagilimi (hatirlatma)
print(f"\n   Flag dagilimi:")
print(f"      temiz                  : {(df['triage_audit_flags']=='temiz').sum():,}")
print(f"      lab_kalinti var        : {df['triage_audit_flags'].str.contains('lab_kalinti', na=False).sum():,}")
print(f"      imaging_kalinti var    : {df['triage_audit_flags'].str.contains('imaging_kalinti', na=False).sum():,}")
print(f"      assessment_kalinti var : {df['triage_audit_flags'].str.contains('assessment_kalinti', na=False).sum():,}")

# Kalinti tipleri ve regex'leri
KALINTILAR = {
    "lab_kalinti":        re.compile(r'\b(WBC \d|Cr \d\.\d|Hgb \d|platelets? \d|INR \d|lactate \d|troponin)\b', re.IGNORECASE),
    "imaging_kalinti":    re.compile(r'\b(CXR|CT (?:head|chest|abdomen)|CTA|MRI|TTE)\b', re.IGNORECASE),
    "assessment_kalinti": re.compile(r'\b(admitted to|diagnosed with|concern for)\b', re.IGNORECASE),
}

# Hangi V3 bolumunde kalinti var? (bolum bazli teshis)
def hangi_bolum_kalinti(row, regex):
    """Hangi V3 alt-bolumunde kalinti var? PMH mi, hpi_clean mi, vitals mi?"""
    bolumler = {
        "cc_v3": row.get("cc_v3", ""),
        "hpi_clean_v3": row.get("hpi_clean_v3", ""),
        "pmh_v3": row.get("pmh_v3", ""),
        "allergies_v3": row.get("allergies_v3", ""),
        "meds_adm_v3": row.get("meds_adm_v3", ""),
        "vitals_v3": row.get("vitals_v3", ""),
    }
    bulgular = {}
    for ad, metin in bolumler.items():
        if isinstance(metin, str) and metin and regex.search(metin):
            bulgular[ad] = regex.search(metin)
    return bulgular

# Her kalinti icin: hangi bolumlerden geliyor?
print("\n" + "=" * 70)
print("KALINTI HANGI BOLUMDEN GELIYOR? (sayim)")
print("=" * 70)

for flag_ad, regex in KALINTILAR.items():
    print(f"\n--- {flag_ad} ---")
    alt = df[df['triage_audit_flags'].str.contains(flag_ad, na=False)]
    print(f"   Toplam kayit: {len(alt):,}")

    bolum_sayim = {"cc_v3": 0, "hpi_clean_v3": 0, "pmh_v3": 0,
                   "allergies_v3": 0, "meds_adm_v3": 0, "vitals_v3": 0}
    for _, row in alt.iterrows():
        bulgular = hangi_bolum_kalinti(row, regex)
        for b in bulgular:
            bolum_sayim[b] += 1

    print(f"   Hangi bolumlerden geliyor (multi count, ayni kayitta birden fazla bolum olabilir):")
    for bolum, n in sorted(bolum_sayim.items(), key=lambda x: -x[1]):
        print(f"      {bolum:<20s} {n:>5,} ({100*n/len(alt):>5.1f}%)")

# 3 örnek/flag tipi - context ile
print("\n" + "=" * 70)
print("ORNEKLER (her flag tipinden 3 ornek + context)")
print("=" * 70)

random.seed(42)

for flag_ad, regex in KALINTILAR.items():
    print("\n" + "#" * 70)
    print(f"# {flag_ad.upper()}")
    print("#" * 70)

    alt = df[df['triage_audit_flags'].str.contains(flag_ad, na=False)].reset_index(drop=True)
    if len(alt) == 0:
        print("   Yok.")
        continue

    # 3 rastgele ornek
    idx_list = random.sample(range(len(alt)), min(3, len(alt)))
    for i, idx in enumerate(idx_list):
        row = alt.iloc[idx]
        print(f"\n  --- Ornek {i+1}/3 (note_id: {row['note_id']}, label: {row['label']}, flags: {row['triage_audit_flags']}) ---")

        bulgular = hangi_bolum_kalinti(row, regex)
        if not bulgular:
            print("   (Kalinti V3'te bulunamadi - regex/flag mismatch?)")
            continue

        for bolum, m in bulgular.items():
            metin = row[bolum]
            konum = m.start()
            eslesen = m.group()
            # ±150 char baglam
            baslangic = max(0, konum - 150)
            son = min(len(metin), konum + len(eslesen) + 150)
            baglam = metin[baslangic:son]
            # Eslesen kelimeyi vurgula
            baglam_vurgu = baglam.replace(eslesen, f">>>>{eslesen}<<<<")

            print(f"\n     Bolum: {bolum}")
            print(f"     Eslesen: {eslesen!r}")
            print(f"     ...{baglam_vurgu}...")

print("\n" + "=" * 70)
print("BLOK 13A.1 TAMAMLANDI")
print("=" * 70)
print("\n   Karar:")
print("   - Eger kalintilar coğunlukla PMH bolumundeyse: gecmis bilgi, KABUL ET")
print("   - Eger kalintilar hpi_clean_v3'te kaliyorsa: cutoff yetersiz, EK PASS gerekli")
