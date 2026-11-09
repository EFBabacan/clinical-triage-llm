"""Leakage severity siniflandirmasi:
- HIGH: mevcut basvuruda 'admitted for X', 'found to have X', 'diagnosed with X' -> direkt etiket sizintisi
- MEDIUM: mevcut basvuruda 'started on X drug' (etiket spesifik tedavi)
- LOW: uzak gecmis ('In ___ she again developed', 'pericardiocentesis') -> sadece anamnez
"""
import pandas as pd
import re
from pathlib import Path

OUTPUT_DIR = Path(r"./workdir/output")
ZS_DIR = OUTPUT_DIR / "zeroshot"
SPLIT_YOL = OUTPUT_DIR / "discharge_split.parquet"

audit = pd.read_csv(ZS_DIR / "zeroshot_v3_leakage_audit.csv")
df = pd.read_parquet(SPLIT_YOL)
df['note_id'] = df['note_id'].astype(str)
audit['note_id'] = audit['note_id'].astype(str)

# Sadece leakage olanlara odaklan
leakages = audit[audit['kategori']=='leakage'].copy()
leakages = leakages.merge(
    df[['note_id', 'triage_input_v3_masked', 'hpi_clean_v3', 'pmh_v3']],
    on='note_id', how='left'
)

print("=" * 70)
print("LEAKAGE SEVERITY SINIFLANDIRMA")
print("=" * 70)
print(f"\n   Leakage isaretli: {len(leakages)} ornek\n")

# Gecmis isaretcileri (zaman ifadeleri)
PAST_MARKERS = re.compile(
    r"\b(in\s+(\d{4}|___|the past|prior|previously|before)|"
    r"history of|h/o|hx of|s/p|status\s*post|"
    r"recent admission|prior admission|last (admission|year|month|week)|"
    r"\d+\s+(years?|months?|weeks?)\s+ago)\b",
    re.IGNORECASE
)

# Spesifik tedavi sinif eslemesi (etiket-spesifik tedavi varsa MEDIUM)
SPESIFIK_TEDAVI = {
    "SEPSIS":      r"\b(vancomycin|zosyn|piperacillin|cefepime|levophed|norepinephrine|vasopressin|broad[-\s]spectrum)",
    "STROKE":      r"\b(tPA|alteplase|aspirin\s+was|thrombectomy)",
    "ATRIAL_FIBRILLATION": r"\b(diltiazem|metoprolol|amiodarone|cardioversion|anticoagulat)",
    "AKI":         r"\b(IV\s+fluids|IVF|fluid\s+bolus|nephrology)",
    "PNEUMONIA":   r"\b(azithromycin|levofloxacin|ceftriaxone|antibiotics)",
    "UTI":         r"\b(ciprofloxacin|nitrofurantoin|levofloxacin|bactrim|cephalexin)",
    "HEART_FAILURE": r"\b(lasix|furosemide|diuresis|metolazone|nitro\b)",
    "GI_BLEEDING": r"\b(transfusion|pRBC|protonix|pantoprazole|EGD|colonoscopy)",
    "DIABETES_COMPLICATION": r"\b(insulin\s+drip|D5|glucagon|D50|dextrose)",
    "COPD_EXACERBATION": r"\b(duonebs|nebulizer|albuterol|prednisone|steroids)",
}

# DIRECT LEAK patterns (Etiket sizintisi olarak kabul edilecek ifade)
DIRECT_LEAK = re.compile(
    r"\b(diagnosed\s+with|admitted\s+(for|to)|found\s+to\s+have|workup\s+showed|workup\s+revealed|concern\s+for)\b",
    re.IGNORECASE
)

# Sınıflandırma
results = []
for _, row in leakages.iterrows():
    cls = row['label']
    nid = row['note_id']
    hpi = row.get('hpi_clean_v3', '') or ''

    severity = "LOW"
    sebep = ""

    # Direkt leak ifadesi var mi (HPI icinde)?
    direct_match = DIRECT_LEAK.search(hpi)
    if direct_match:
        # Cumlede gecmis isaretcisi yoksa HIGH
        konum = direct_match.start()
        baslangic = max(0, konum - 80)
        son = min(len(hpi), konum + 100)
        baglam = hpi[baslangic:son]
        if PAST_MARKERS.search(baglam):
            severity = "LOW"
            sebep = f"direct_leak_pattern={direct_match.group()!r} ama past_marker var"
        else:
            severity = "HIGH"
            sebep = f"direct_leak_pattern={direct_match.group()!r} mevcut basvuruda"

    # Spesifik tedavi var mi?
    if severity == "LOW" and cls in SPESIFIK_TEDAVI:
        rx = re.compile(SPESIFIK_TEDAVI[cls], re.IGNORECASE)
        m = rx.search(hpi)
        if m:
            konum = m.start()
            baslangic = max(0, konum - 80)
            son = min(len(hpi), konum + 100)
            baglam = hpi[baslangic:son]
            if PAST_MARKERS.search(baglam):
                severity = "LOW"
                sebep = f"specific_treatment={m.group()!r} ama past_marker var"
            else:
                severity = "MEDIUM"
                sebep = f"specific_treatment={m.group()!r} mevcut basvuruda"

    if severity == "LOW" and not sebep:
        sebep = "received/given var ama gecmis tedavi gibi gorunuyor"

    results.append({
        "note_id": nid, "label": cls, "severity": severity, "sebep": sebep
    })

# Rapor
print(f"   {'Severity':<10s} {'Sayim':>5s}")
print("   " + "-" * 25)
sev_sayim = pd.Series([r['severity'] for r in results]).value_counts()
for sev in ["HIGH", "MEDIUM", "LOW"]:
    n = sev_sayim.get(sev, 0)
    print(f"   {sev:<10s} {n:>5}")

print(f"\n   Detay:")
print(f"   {'note_id':<22s} {'label':<22s} {'severity':<8s} sebep")
print("   " + "-" * 90)
for r in sorted(results, key=lambda x: {"HIGH":0,"MEDIUM":1,"LOW":2}[x['severity']]):
    print(f"   {r['note_id']:<22s} {r['label']:<22s} {r['severity']:<8s} {r['sebep'][:50]}")

# Detay context (ilk 5 HIGH)
print(f"\n\n   HIGH severity ornekleri (mevcut basvuruda direkt etiket leakage):")
print("   " + "=" * 67)
high_ler = [r for r in results if r['severity']=='HIGH']
for r in high_ler[:5]:
    nid = r['note_id']
    cls = r['label']
    hpi = leakages[leakages['note_id']==nid]['hpi_clean_v3'].iloc[0] or ''
    direct_m = DIRECT_LEAK.search(hpi)
    if direct_m:
        konum = direct_m.start()
        baslangic = max(0, konum - 150)
        son = min(len(hpi), konum + 200)
        baglam = hpi[baslangic:son]
        eslesen = direct_m.group()
        vurgu = baglam.replace(eslesen, f">>>>{eslesen}<<<<")
        print(f"\n   --- {nid} ({cls}) ---")
        print(f"   ...{vurgu}...")

# Final sayim
print("\n\n" + "=" * 70)
print("FINAL SONUC")
print("=" * 70)
toplam_leak = len(results)
high = sev_sayim.get("HIGH", 0)
medium = sev_sayim.get("MEDIUM", 0)
low = sev_sayim.get("LOW", 0)
total_55 = 55  # 11 sinif x 5 ornek
print(f"\n   Toplam ornek (5/sinif): {total_55}")
print(f"   Riskli isaretli       : {toplam_leak} ({100*toplam_leak/total_55:.1f}%)")
print(f"      HIGH (gercek leak) : {high} ({100*high/total_55:.1f}%)")
print(f"      MEDIUM (suspect)   : {medium} ({100*medium/total_55:.1f}%)")
print(f"      LOW (gecmis tedavi): {low} ({100*low/total_55:.1f}%)")
print(f"      Temiz              : {total_55 - toplam_leak} ({100*(total_55-toplam_leak)/total_55:.1f}%)")

# Kaydet
out_df = pd.DataFrame(results)
out_yol = ZS_DIR / "zeroshot_v3_leakage_severity.csv"
out_df.to_csv(out_yol, index=False)
print(f"\n   Kaydedildi: {out_yol.name}")
