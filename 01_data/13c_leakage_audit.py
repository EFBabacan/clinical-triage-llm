"""Blok 13C: Final leakage audit.
Dogru tahmin edilen ornekler V3 input icinde leakage iceriyor mu?
- Mevcut basvuruya ait riskli ifade -> 'leakage'
- PMH/gecmis kayit -> 'historical_clue'
- Yok -> 'temiz'
"""
import pandas as pd
import re
from pathlib import Path

OUTPUT_DIR = Path(r"./workdir/output")
ZS_DIR     = OUTPUT_DIR / "zeroshot"
SPLIT_YOL  = OUTPUT_DIR / "discharge_split.parquet"

print("=" * 70)
print("BLOK 13C — FINAL LEAKAGE AUDIT (dogru tahminlerde sizinti?)")
print("=" * 70)

# Veri yukle
test = pd.read_csv(ZS_DIR / "zeroshot_v3_test.csv")
df = pd.read_parquet(SPLIT_YOL)
df['note_id'] = df['note_id'].astype(str)
test['note_id'] = test['note_id'].astype(str)

# Triage input ve PMH ayrı bilgisini de cek
test = test.merge(
    df[['note_id', 'triage_input_v3_masked', 'pmh_v3', 'hpi_clean_v3']],
    on='note_id', how='left'
)
test['dogru'] = test['parsed_pred'] == test['label']
print(f"\n   Toplam test    : {len(test):,}")
print(f"   Dogru tahmin   : {test['dogru'].sum():,} ({100*test['dogru'].mean():.1f}%)")

# --- Riskli ifade patternleri ---
RISKY_PATTERNS = {
    "diagnosed_with":      re.compile(r'\bdiagnosed\s+with\b', re.IGNORECASE),
    "admitted_for_to":     re.compile(r'\badmitted\s+(for|to)\b', re.IGNORECASE),
    "found_to_have":       re.compile(r'\bfound\s+to\s+have\b', re.IGNORECASE),
    "workup_showed":       re.compile(r'\bworkup\s+(showed|revealed|notable)\b', re.IGNORECASE),
    "imaging_showed":      re.compile(r'\b(CXR|CT(A)?|MRI|TTE|TEE|US|EKG|ECG|echocardiogram)\b[^.]{0,40}\b(showed|revealed|notable|with|demonstrated|found|consistent)\b', re.IGNORECASE),
    "labs_notable":        re.compile(r'\bLabs?\s+(notable|showed|drawn|remarkable)\b', re.IGNORECASE),
    "received_given":      re.compile(r'\b(received|was\s+given|started\s+on|treating\s+with|treated\s+with)\b', re.IGNORECASE),
    "concern_for":         re.compile(r'\bconcern\s+for\b', re.IGNORECASE),
}

# --- Her dogru tahminden 5 ornek/sinif sec ---
ALLOWED_CLASSES = [
    "SEPSIS", "STROKE", "ATRIAL_FIBRILLATION", "AKI", "PNEUMONIA", "UTI",
    "HEART_FAILURE", "GI_BLEEDING", "DIABETES_COMPLICATION",
    "COPD_EXACERBATION", "OTHER"
]

print("\n[1/3] Her siniftan 5 dogru tahmin ornegi seciliyor...")

secilen = []
for cls in ALLOWED_CLASSES:
    alt = test[(test['label']==cls) & test['dogru']].head(5)
    for _, row in alt.iterrows():
        secilen.append(row)
print(f"   Secilen toplam: {len(secilen)} ornek (11 sinif * 5 = 55 hedef)")

# --- Pattern eslesmelerini bul + kategorile ---
print("\n[2/3] Riskli ifadeler aranan...")

raporlar = []
sinif_sayim = {cls: {"leakage": 0, "historical": 0, "temiz": 0} for cls in ALLOWED_CLASSES}

for row in secilen:
    cls = row['label']
    nid = row['note_id']
    triage = row['triage_input_v3_masked'] or ""
    pmh = row['pmh_v3'] or ""
    hpi = row['hpi_clean_v3'] or ""

    bulgular = {}  # pattern_ad -> list of (eslesen, baglam, kaynak)
    for pat_ad, rx in RISKY_PATTERNS.items():
        for m in rx.finditer(triage):
            konum = m.start()
            eslesen = m.group()
            baslangic = max(0, konum - 100)
            son = min(len(triage), konum + len(eslesen) + 100)
            baglam = triage[baslangic:son]

            # Pattern PMH'de mi yoksa hpi_clean'de mi?
            kaynak = "?"
            if isinstance(pmh, str) and pmh and rx.search(pmh):
                # Sadece PMH'de varsa historical
                if not (isinstance(hpi, str) and hpi and rx.search(hpi)):
                    kaynak = "PMH"
            if kaynak == "?":
                if isinstance(hpi, str) and hpi and rx.search(hpi):
                    kaynak = "HPI"
                else:
                    kaynak = "?"

            bulgular.setdefault(pat_ad, []).append((eslesen, baglam, kaynak))

    # Kategori karari: HPI'da bulunduysa leakage, sadece PMH'de ise historical
    hpi_var = any(kaynak == "HPI" for kayitlar in bulgular.values() for _, _, kaynak in kayitlar)
    pmh_var = any(kaynak == "PMH" for kayitlar in bulgular.values() for _, _, kaynak in kayitlar)

    if hpi_var:
        kategori = "leakage"
    elif pmh_var:
        kategori = "historical"
    else:
        kategori = "temiz"

    sinif_sayim[cls][kategori] += 1
    raporlar.append({
        "note_id": nid, "label": cls, "pred": row['parsed_pred'],
        "kategori": kategori, "bulgular": bulgular,
    })

# --- Detayli rapor ---
print("\n[3/3] Detayli rapor")
print("=" * 70)

for cls in ALLOWED_CLASSES:
    print(f"\n{'#' * 70}")
    print(f"# {cls}  (5 dogru ornek)")
    print(f"#   leakage={sinif_sayim[cls]['leakage']}, historical={sinif_sayim[cls]['historical']}, temiz={sinif_sayim[cls]['temiz']}")
    print(f"{'#' * 70}")
    cls_raporlari = [r for r in raporlar if r['label']==cls]
    for i, r in enumerate(cls_raporlari, 1):
        kat_isaret = {"leakage":"!! LEAKAGE","historical":"** HISTORICAL","temiz":"   temiz"}[r['kategori']]
        print(f"\n  --- Ornek {i}/{len(cls_raporlari)} (note: {r['note_id']}, kategori: {kat_isaret}) ---")
        if not r['bulgular']:
            print(f"     Riskli ifade YOK")
        else:
            for pat_ad, kayitlar in r['bulgular'].items():
                for eslesen, baglam, kaynak in kayitlar[:2]:  # her pattern icin max 2 ornek
                    vurgu = baglam.replace(eslesen, f">>>>{eslesen}<<<<")
                    print(f"     [{pat_ad}] @ {kaynak}: ...{vurgu}...")

# --- Genel ozet ---
print("\n" + "=" * 70)
print("GENEL OZET (her sinif 5 ornek)")
print("=" * 70)
toplam_leak = sum(s['leakage'] for s in sinif_sayim.values())
toplam_hist = sum(s['historical'] for s in sinif_sayim.values())
toplam_temiz = sum(s['temiz'] for s in sinif_sayim.values())
toplam_n = toplam_leak + toplam_hist + toplam_temiz

print(f"\n   {'Sinif':<25s} {'leakage':>8s} {'historic':>9s} {'temiz':>6s}")
print("   " + "-" * 55)
for cls in ALLOWED_CLASSES:
    s = sinif_sayim[cls]
    print(f"   {cls:<25s} {s['leakage']:>8} {s['historical']:>9} {s['temiz']:>6}")
print("   " + "-" * 55)
print(f"   {'TOPLAM':<25s} {toplam_leak:>8} {toplam_hist:>9} {toplam_temiz:>6}")
print(f"\n   Leakage    : {toplam_leak}/{toplam_n} ({100*toplam_leak/max(toplam_n,1):.1f}%)")
print(f"   Historical : {toplam_hist}/{toplam_n} ({100*toplam_hist/max(toplam_n,1):.1f}%)")
print(f"   Temiz      : {toplam_temiz}/{toplam_n} ({100*toplam_temiz/max(toplam_n,1):.1f}%)")

# Kaydet
out_df = pd.DataFrame([
    {"note_id": r["note_id"], "label": r["label"], "pred": r["pred"],
     "kategori": r["kategori"],
     "patternlar": ",".join(r["bulgular"].keys())}
    for r in raporlar
])
out_yol = ZS_DIR / "zeroshot_v3_leakage_audit.csv"
out_df.to_csv(out_yol, index=False)
print(f"\n   Kaydedildi: {out_yol.name}")
