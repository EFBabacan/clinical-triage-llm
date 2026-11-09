"""Blok 13E: Refined time-aware leakage detector.
- REVIEW = sadece gercek belirsizlik
- CURRENT_HIGH = current_marker + (lab_value | imaging_finding | direct_dx | treatment)
- 'presenting with cough' CLEAN
- 'multiple CVAs', 'h/o X', 's/p' historical
- Cleanup before, audit after."""
import pandas as pd
import re
import time
import random
from pathlib import Path

OUTPUT_DIR = Path(r"./workdir/output")
SPLIT_YOL  = OUTPUT_DIR / "discharge_split.parquet"

print("=" * 70)
print("BLOK 13E — Refined Time-Aware Leakage Detector")
print("=" * 70)

ALLOWED_CLASSES = [
    "SEPSIS", "STROKE", "ATRIAL_FIBRILLATION", "AKI", "PNEUMONIA", "UTI",
    "HEART_FAILURE", "GI_BLEEDING", "DIABETES_COMPLICATION",
    "COPD_EXACERBATION", "OTHER"
]

# ============================================================
# HISTORICAL MARKERS — gerçek geniş liste
# ============================================================
HISTORICAL = re.compile(
    r"\b("
    r"history\s+of|hx\s+of|h/o|"
    r"PMH|past\s+medical|"
    r"prior|previous|previously|remote|chronic|"
    r"in\s+the\s+past|"
    r"s/p|status\s*post|"
    r"recurrent\s+(UTI|pneumonia|infection)|"
    r"multiple\s+(CVA|stroke|admission|hospitalization)|"
    r"(\d+|few|several|many)\s+(years?|months?|weeks?|days?)\s+ago|"
    r"old(\s+|-)?(imaging|labs?|TTE|CT|MRI|CXR|EF)|"
    r"home\s+(medications?|meds?|O2|insulin|inhaler|nebulizer)|"
    r"on\s+chronic\s+|chronic\s+suppressive|"
    r"baseline\s+(Cr|creatinine|Hgb|HCT|labs?|EF|FEV)|"
    r"last\s+(TTE|EF|CT|MRI|CXR|labs?|admission)|"
    r"prior\s+(TTE|admission|hospitalization|visit|imaging|workup|labs?)|"
    r"recent\s+admission|previous\s+admission|"
    r"discharged\s+(on|in)\s+|recently\s+discharged|"
    r"(in|on)\s+___|admitted\s+(in|on)\s+___|"
    r"originally\s+admitted|initially\s+admitted"
    r")\b",
    re.IGNORECASE
)

# ============================================================
# CURRENT ENCOUNTER MARKERS — net sinyaller (presenting/presented HARIC)
# ============================================================
# 'presenting with X' tek basina current degildir, semptom baslangici
CURRENT = re.compile(
    r"\b("
    r"on\s+arrival|upon\s+arrival|on\s+presentation\s+to|"
    r"in\s+the\s+ED|in\s+ED|ED\s+course|"
    r"initial\s+(labs?|vitals?|workup)|"
    r"upon\s+(arrival|presentation|admission)|"
    r"on\s+admission(?!\s+to\s+rehab)|"
    r"transferred\s+to\s+(MICU|ICU|floor|service|cardiology|medicine|pulmonology)|"
    r"sent\s+to\s+(MICU|ICU|cardiology|medicine)|"
    r"received\s+in\s+(the\s+)?ED"
    r")\b",
    re.IGNORECASE
)

# ============================================================
# CURRENT HIGH PATTERNS — sıkı koşullar
# ============================================================

# 1. Lab result with numeric value (admission/ED context)
LAB_VALUE_NUMERIC = re.compile(
    r"\b(WBC|Hgb|HCT|platelets?|PLT|INR|lactate|troponin|"
    r"BUN|Cr|creatinine|sodium|Na\+?|K\+?|trop)\s*"
    r"(was|of|=|:|\(|<|>|>=|<=)?\s*\d[\d.]*\b",
    re.IGNORECASE
)

# 2. Lab with descriptor (notable/positive/elevated/low/high/normal)
LAB_DESCRIPTOR = re.compile(
    r"\b(WBC|Hgb|HCT|platelets?|PLT|INR|lactate|troponin|"
    r"BUN|Cr|creatinine|sodium|Na\+?|K\+?|trop|UA|labs?)\b[^.]{0,40}"
    r"\b(elevated|high|low|notable|positive|negative|abnormal|"
    r"showed|revealed|drawn|reveal)\b",
    re.IGNORECASE
)

# 3. Imaging finding with verb
IMAGING_FINDING = re.compile(
    r"\b(CXR|CTA?|MRI|US|TTE|TEE|echocardiogram|EKG|ECG|"
    r"ultrasound)\b[^.]{0,80}\b("
    r"showed|revealed|demonstrated|notable|consistent\s+with|"
    r"positive|negative|reveal|reveals|noted|"
    r"impression|read\s+as|w/\s+|with\s+(infiltrate|consolidation|"
    r"effusion|edema|stenosis|ICH|infarct|embolus|hemorrhage|mass|nodule)"
    r")\b",
    re.IGNORECASE
)

# 4. Direct diagnosis (current encounter)
DIRECT_DX = re.compile(
    r"\b(diagnosed\s+with|found\s+to\s+have|"
    r"concern\s+for|consistent\s+with|"
    r"impression\s*:|assessment\s*:)\b",
    re.IGNORECASE
)

# 5. Admitted for X (current decision)
ADMITTED_FOR = re.compile(
    r"\badmitted\s+(for|to)\s+("
    r"workup|management|treatment|further|"
    r"acute|exacerbation|"
    r"sepsis|stroke|pneumonia|HF|CHF|COPD|UTI|"
    r"GI\s+bleed|DKA|AKI"
    r")\b",
    re.IGNORECASE
)

# 6. Current treatment (etiket-spesifik)
CURRENT_TREATMENT = re.compile(
    r"\b(received|was\s+given|started\s+on|treated\s+with|treating\s+with|"
    r"placed\s+on|initiated)\s+[^.]{0,60}\b("
    r"vancomycin|zosyn|piperacillin|cefepime|ceftriaxone|levofloxacin|"
    r"azithromycin|ciprofloxacin|"
    r"levophed|norepinephrine|vasopressin|"
    r"lasix|furosemide|"
    r"insulin\s+drip|glucagon|D5|D50|dextrose|"
    r"duonebs|methylprednisolone|"
    r"tPA|alteplase|"
    r"IV\s+fluids|fluid\s+bolus|crystalloid|"
    r"pRBC|transfusion|"
    r"thrombectomy|cardioversion"
    r")\b",
    re.IGNORECASE
)

# ============================================================
# Sentence classification (REFINED)
# ============================================================
def classify_sentence(cumle):
    """Cumle -> (kategori, [(pattern, eslesen)])"""
    historical = bool(HISTORICAL.search(cumle))
    current = bool(CURRENT.search(cumle))

    bulgular = []

    # Lab numeric
    m = LAB_VALUE_NUMERIC.search(cumle)
    if m: bulgular.append(("lab_numeric", m.group()))
    # Lab descriptor
    m = LAB_DESCRIPTOR.search(cumle)
    if m: bulgular.append(("lab_descriptor", m.group()))
    # Imaging finding
    m = IMAGING_FINDING.search(cumle)
    if m: bulgular.append(("imaging_finding", m.group()))
    # Direct dx
    m = DIRECT_DX.search(cumle)
    if m: bulgular.append(("direct_dx", m.group()))
    # Admitted for
    m = ADMITTED_FOR.search(cumle)
    if m: bulgular.append(("admitted_for", m.group()))
    # Current treatment
    m = CURRENT_TREATMENT.search(cumle)
    if m: bulgular.append(("current_treatment", m.group()))

    # Risk yoksa CLEAN (whether historical or not)
    if not bulgular:
        if historical:
            return "HISTORICAL_ALLOWED", []  # historical context but no risk pattern
        return "CLEAN", []

    # Risk pattern var.
    # 1. Historical marker varsa VE current marker yoksa -> HISTORICAL_ALLOWED
    if historical and not current:
        return "HISTORICAL_ALLOWED", bulgular

    # 2. Lab numeric direkt -> CURRENT_HIGH (cunku sayisal sonuc current encounter sinyali)
    has_lab_numeric = any(p == "lab_numeric" for p, _ in bulgular)
    has_imaging = any(p == "imaging_finding" for p, _ in bulgular)
    has_direct_dx = any(p == "direct_dx" for p, _ in bulgular)
    has_admitted = any(p == "admitted_for" for p, _ in bulgular)
    has_treatment = any(p == "current_treatment" for p, _ in bulgular)

    # Sıkı koşul: current_marker var ya da lab_numeric/imaging/direct_dx/admitted/treatment
    if current:
        if has_lab_numeric or has_imaging or has_direct_dx or has_admitted:
            return "CURRENT_HIGH", bulgular
        if has_treatment:
            return "CURRENT_MEDIUM", bulgular
        # Sadece lab_descriptor (sayisiz) ve current marker -> REVIEW
        return "REVIEW_UNCERTAIN", bulgular

    # Current marker yok ama:
    # - Lab numeric tek basina -> CURRENT_HIGH (gercek sayi sonucu)
    if has_lab_numeric:
        return "CURRENT_HIGH", bulgular
    # - Imaging finding tek basina -> CURRENT_HIGH
    if has_imaging:
        return "CURRENT_HIGH", bulgular
    # - Direct dx tek basina (historical de yok) -> CURRENT_HIGH
    if has_direct_dx:
        return "CURRENT_HIGH", bulgular
    # - Admitted for tek basina -> CURRENT_HIGH
    if has_admitted:
        return "CURRENT_HIGH", bulgular
    # - Treatment tek basina -> CURRENT_MEDIUM
    if has_treatment:
        return "CURRENT_MEDIUM", bulgular
    # - Sadece lab_descriptor (sayisiz) -> REVIEW
    return "REVIEW_UNCERTAIN", bulgular


def cumlelere_bol(metin):
    if not metin: return []
    cs = re.split(r"(?<=[.!?])\s+(?=[A-Z\d])", metin)
    return [c.strip() for c in cs if c.strip()]


def clean_hpi(hpi_text):
    """Cumle bazli: HIGH/MEDIUM cumleleri at. CLEAN/HISTORICAL/REVIEW tut."""
    if not isinstance(hpi_text, str) or not hpi_text:
        return hpi_text
    ros_m = re.search(r"(?ims)(\n\s*(?:Review of Systems?|ROS)\s*:)", hpi_text)
    if ros_m:
        on, ros_kismi = hpi_text[:ros_m.start()], hpi_text[ros_m.start():]
    else:
        on, ros_kismi = hpi_text, ""

    cumleler = cumlelere_bol(on)
    korunan = []
    for c in cumleler:
        kat, _ = classify_sentence(c)
        if kat in ("CURRENT_HIGH", "CURRENT_MEDIUM"):
            continue  # at
        korunan.append(c)
    yeni = " ".join(korunan).strip()
    if ros_kismi:
        return yeni + "\n\n" + ros_kismi.strip()
    return yeni


def clean_pmh(pmh_text):
    """PMH genelde tum historical. Sadece CURRENT_HIGH/MEDIUM olanlari at."""
    if not isinstance(pmh_text, str) or not pmh_text:
        return pmh_text
    cumleler = cumlelere_bol(pmh_text)
    korunan = []
    for c in cumleler:
        kat, _ = classify_sentence(c)
        if kat in ("CURRENT_HIGH", "CURRENT_MEDIUM"):
            continue
        korunan.append(c)
    return " ".join(korunan).strip()


def audit_after(v4_text):
    """V4 final text uzerinde audit. En yuksek severity."""
    if not isinstance(v4_text, str) or not v4_text:
        return "CLEAN", []
    cumleler = cumlelere_bol(v4_text)
    severities = []
    reasons = []
    for c in cumleler:
        kat, b = classify_sentence(c)
        if kat != "CLEAN":
            severities.append(kat)
            for ad, eslesen in b:
                reasons.append(f"{kat}:{ad}={eslesen[:30]}")

    if not severities:
        return "CLEAN", []
    oncelik = {"CURRENT_HIGH":4, "CURRENT_MEDIUM":3, "REVIEW_UNCERTAIN":2, "HISTORICAL_ALLOWED":1}
    en_yuksek = max(severities, key=lambda x: oncelik.get(x, 0))
    return en_yuksek, reasons[:8]


# ============================================================
# MAIN
# ============================================================
print("\n[1/5] Veri yukleniyor...")
df = pd.read_parquet(SPLIT_YOL)
print(f"   N={len(df):,}, kolon={len(df.columns)}")

HPI_KAYNAK = "hpi_clean_v3_precleanup" if "hpi_clean_v3_precleanup" in df.columns else "hpi_clean_v3"
print(f"   HPI kaynak: {HPI_KAYNAK}")

# CC dokunulmaz
df["cc_v4_timeaware"] = df["cc_v3"]

print("\n[2/5] HPI temizleme...")
t0 = time.time()
df["hpi_clean_v4_timeaware"] = df[HPI_KAYNAK].apply(clean_hpi)
print(f"   Sure: {time.time()-t0:.1f}s")

print("\n[3/5] PMH temizleme...")
t0 = time.time()
df["pmh_v4_timeaware"] = df["pmh_v3"].apply(clean_pmh)
print(f"   Sure: {time.time()-t0:.1f}s")

# Compose V4
def compose_v4(row):
    parts = []
    if row.get("cc_v4_timeaware"):
        parts.append(f"Chief Complaint:\n{row['cc_v4_timeaware']}")
    if row.get("hpi_clean_v4_timeaware"):
        parts.append(f"Symptom History:\n{row['hpi_clean_v4_timeaware']}")
    if row.get("pmh_v4_timeaware"):
        parts.append(f"Past Medical History:\n{row['pmh_v4_timeaware']}")
    if row.get("allergies_v3"):
        parts.append(f"Allergies:\n{row['allergies_v3']}")
    if row.get("meds_adm_v3"):
        parts.append(f"Medications on Admission:\n{row['meds_adm_v3']}")
    if row.get("vitals_v3"):
        parts.append(f"Vital Signs:\n{row['vitals_v3']}")
    return "\n\n".join(parts)

df["triage_input_v4_timeaware"] = df.apply(compose_v4, axis=1)

# Maskeleme
DIAG_KW = {
    'PNEUMONIA': r'\b(pneumon\w*|bronchopneumon\w*)\b',
    'HEART_FAILURE': r'\b(CHF|congestive heart failure|heart failure|HFrEF|HFpEF|systolic heart failure|diastolic heart failure)\b',
    'SEPSIS': r'\b(sepsis|septic|septicemia|bacteremia|septicaemia)\b',
    'AKI': r'\b(AKI|acute kidney injury|acute renal failure|ARF|acute renal insufficiency)\b',
    'UTI': r'\b(UTI|urinary tract infection|pyelonephritis|cystitis)\b',
    'STROKE': r'\b(stroke|CVA|cerebrovascular accident|ischemic stroke|hemorrhagic stroke|intracerebral hemorrhage|intracranial hemorrhage)\b',
    'GI_BLEEDING': r'\b(GI bleed\w*|gastrointestinal bleed\w*|UGIB|LGIB|hematemesis|melena|hematochezia)\b',
    'COPD_EXACERBATION': r'\b(COPD exacerbation|AECOPD|chronic obstructive|COPD)\b',
    'DIABETES_COMPLICATION': r'\b(DKA|diabetic ketoacidosis|HHS|hyperosmolar|hyperglycemic crisis|diabetic coma)\b',
    'ATRIAL_FIBRILLATION': r'\b(atrial fibrillation|AFib|A-fib|atrial flutter|paroxysmal AF)\b',
}
BIRLESIK = re.compile("|".join(f"(?:{p})" for p in DIAG_KW.values()), re.IGNORECASE)
df["triage_input_v4_timeaware_masked"] = df["triage_input_v4_timeaware"].apply(
    lambda m: BIRLESIK.sub("[DIAG_MASKED]", m) if m else m
)

print("\n[4/5] V4 final text uzerinde AFTER-CLEANUP audit...")
t0 = time.time()
sevs, reasons = [], []
for v4 in df["triage_input_v4_timeaware"]:
    s, r = audit_after(v4)
    sevs.append(s)
    reasons.append(",".join(r) if r else "no_leak")
df["strict_current_leakage_severity"] = sevs
df["strict_current_leakage_reason"] = reasons
df["strict_current_leakage_flag"] = df["strict_current_leakage_severity"] != "CLEAN"
print(f"   Sure: {time.time()-t0:.1f}s")

# ============================================================
# AUDIT RAPORU
# ============================================================
print("\n[5/5] AUDIT")
print("=" * 70)

for split_ad, alt in [("ALL", df), ("test", df[df["split"]=="test"]),
                       ("val", df[df["split"]=="val"]),
                       ("train_S3", df[df["split"]=="train_S3"])]:
    n = len(alt)
    print(f"\n--- {split_ad} (N={n:,}) ---")
    sev_say = alt["strict_current_leakage_severity"].value_counts().to_dict()
    for sev in ["CLEAN", "HISTORICAL_ALLOWED", "REVIEW_UNCERTAIN", "CURRENT_MEDIUM", "CURRENT_HIGH"]:
        c = sev_say.get(sev, 0)
        print(f"   {sev:<25s} {c:>5} ({100*c/max(n,1):>5.1f}%)")

# Test ozet
test_alt = df[df["split"]=="test"]
print(f"\n   TEST ozet:")
for sev in ["CLEAN", "HISTORICAL_ALLOWED", "REVIEW_UNCERTAIN", "CURRENT_MEDIUM", "CURRENT_HIGH"]:
    c = (test_alt["strict_current_leakage_severity"]==sev).sum()
    print(f"      {sev:<25s} {c}")

# Manuel false-positive sanity check
print(f"\n   Sanity check: 'multiple CVAs' / 'presenting with cough' CLEAN olmali")
fp_test = [
    ("multiple CVAs", "Mr. X is a man with history of multiple CVAs and dementia."),
    ("presenting with cough", "Patient presenting with cough, wheezing and SOB for 2 days."),
    ("h/o pneumonia", "She has h/o pneumonia and recurrent UTI on chronic suppressive."),
    ("multiple CVAs+presented current", "Patient with multiple CVAs presented with new headache."),
    ("LAB CURRENT", "Upon arrival, WBC 14, Cr 2.1, lactate 3.2."),
    ("IMAGING CURRENT", "CXR showed bilateral infiltrates."),
    ("DX CURRENT", "He was diagnosed with sepsis and admitted for management."),
    ("HISTORICAL EF", "Last TTE showed EF 35% in 2020."),
    ("baseline Cr", "CKD with baseline Cr 1.8."),
]
for ad, c in fp_test:
    kat, _ = classify_sentence(c)
    print(f"      {ad:<35s} -> {kat}")

# ============================================================
# 10 manuel ornek
# ============================================================
print("\n" + "=" * 70)
print("ORNEKLER (test setinden)")
print("=" * 70)
random.seed(42)
test_df = df[df["split"]=="test"].reset_index(drop=True)

ornek_dagilim = [("CLEAN", 3), ("HISTORICAL_ALLOWED", 3), ("REVIEW_UNCERTAIN", 2),
                 ("CURRENT_MEDIUM", 2), ("CURRENT_HIGH", 2)]
for kategori, sayi in ornek_dagilim:
    alt = test_df[test_df["strict_current_leakage_severity"]==kategori]
    if len(alt) == 0:
        print(f"\n--- {kategori}: 0 ornek ---")
        continue
    n_ornek = min(sayi, len(alt))
    secilen = alt.sample(n=n_ornek, random_state=42)
    print(f"\n{'#'*70}")
    print(f"# {kategori} ({len(alt)} kayit, gosterilen {n_ornek})")
    print(f"{'#'*70}")
    for _, row in secilen.iterrows():
        print(f"\n  --- note={row['note_id']} label={row['label']} ---")
        v4 = row["triage_input_v4_timeaware"] or ""
        print(f"  V4 ({len(v4)} char):")
        print(f"    {v4[:600]}")
        print(f"  reasons: {row['strict_current_leakage_reason'][:200]}")

# Acceptance
print("\n" + "=" * 70)
print("ACCEPTANCE")
print("=" * 70)
hi = (test_alt["strict_current_leakage_severity"]=="CURRENT_HIGH").sum()
md = (test_alt["strict_current_leakage_severity"]=="CURRENT_MEDIUM").sum()
rv = (test_alt["strict_current_leakage_severity"]=="REVIEW_UNCERTAIN").sum()
cl = (test_alt["strict_current_leakage_severity"]=="CLEAN").sum()
hist = (test_alt["strict_current_leakage_severity"]=="HISTORICAL_ALLOWED").sum()

print(f"   Test CLEAN              = {cl}  ({100*cl/len(test_alt):.1f}%)")
print(f"   Test HISTORICAL_ALLOWED = {hist}  ({100*hist/len(test_alt):.1f}%)")
print(f"   Test REVIEW             = {rv}  ({100*rv/len(test_alt):.1f}%) (hedef <%5)")
print(f"   Test CURRENT_MEDIUM     = {md}  (hedef: 0)")
print(f"   Test CURRENT_HIGH       = {hi}  (hedef: 0)")

if hi == 0 and md == 0 and rv < 0.05*len(test_alt):
    print(f"\n   ✅ ACCEPT: Hedeflere ulasildi.")
else:
    if hi > 0 or md > 0:
        print(f"\n   ⚠️  HIGH={hi} MEDIUM={md} -> manuel review gerekli.")
    if rv >= 0.05*len(test_alt):
        print(f"\n   ⚠️  REVIEW orani {100*rv/len(test_alt):.1f}% (<%5 hedefin ustunde)")

# Kaydet
df.to_parquet(SPLIT_YOL, compression="snappy", index=False)
print(f"\n   Parquet guncellendi ({SPLIT_YOL.stat().st_size/1024**2:.1f} MB, {len(df.columns)} kolon)")

print("\n" + "=" * 70)
print("BLOK 13E TAMAMLANDI")
print("=" * 70)
