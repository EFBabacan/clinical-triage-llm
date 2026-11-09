"""Blok 13E v2: Audit'i sadece HPI + PMH uzerinde yap, medications/allergies/vitals haric.
False positive: ilac dozajlari (Enoxaparin Sodium 80mg, Nifedipine CR 30mg) lab sanildi."""
import pandas as pd
import re
import time
import random
from pathlib import Path

OUTPUT_DIR = Path(r"./workdir/output")
SPLIT_YOL  = OUTPUT_DIR / "discharge_split.parquet"

print("=" * 70)
print("BLOK 13E v2 — Refined detector + audit scope (HPI/PMH only)")
print("=" * 70)

ALLOWED_CLASSES = ["SEPSIS","STROKE","ATRIAL_FIBRILLATION","AKI","PNEUMONIA","UTI",
    "HEART_FAILURE","GI_BLEEDING","DIABETES_COMPLICATION","COPD_EXACERBATION","OTHER"]

# Patterns (13E ile ayni)
HISTORICAL = re.compile(
    r"\b(history\s+of|hx\s+of|h/o|PMH|past\s+medical|prior|previous|previously|remote|chronic|"
    r"in\s+the\s+past|s/p|status\s*post|recurrent\s+(UTI|pneumonia|infection)|"
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
    r"originally\s+admitted|initially\s+admitted)\b", re.IGNORECASE)

CURRENT = re.compile(
    r"\b(on\s+arrival|upon\s+arrival|on\s+presentation\s+to|"
    r"in\s+the\s+ED|in\s+ED|ED\s+course|"
    r"initial\s+(labs?|vitals?|workup)|upon\s+(arrival|presentation|admission)|"
    r"on\s+admission(?!\s+to\s+rehab)|"
    r"transferred\s+to\s+(MICU|ICU|floor|service|cardiology|medicine|pulmonology)|"
    r"sent\s+to\s+(MICU|ICU|cardiology|medicine)|"
    r"received\s+in\s+(the\s+)?ED)\b", re.IGNORECASE)

# Lab numeric — sadece sayinin yaninda 'mg' DEGIL ise (ilac dozaji degil)
# Negative lookahead: sayidan sonra ' mg' veya ' mcg' veya ' tab' gelirse -> ilac dozaji, lab degil
LAB_VALUE_NUMERIC = re.compile(
    r"\b(WBC|Hgb|HCT|platelets?|PLT|INR|lactate|troponin|BUN|"
    r"creatinine|trop)\s*(was|of|=|:|\(|<|>)?\s*\d[\d.]*\b(?!\s*(mg|mcg|g|tab|cap|ml|unit))",
    re.IGNORECASE)
# 'sodium', 'Na', 'K', 'Cr' gibi tek harfliler kelime sinirinda olmali ve mg/g vb. olmamali
LAB_VALUE_SHORT = re.compile(
    r"(?:^|[^a-zA-Z])(Cr|Na\+?|K\+?|sodium|potassium|chloride)\s*(was|of|=|:|<|>|\()?\s*"
    r"\d[\d.]*\b(?!\s*(mg|mcg|g|tab|cap|ml|unit|chloride|sulfate|bicarb))",
    re.IGNORECASE)

LAB_DESCRIPTOR = re.compile(
    r"\b(WBC|Hgb|HCT|platelets?|PLT|INR|lactate|troponin|BUN|Cr|creatinine|"
    r"sodium|Na\+?|K\+?|trop|UA|labs?)\b[^.]{0,40}"
    r"\b(elevated|high|low|notable|positive|negative|abnormal|"
    r"showed|revealed|drawn|reveal)\b", re.IGNORECASE)

IMAGING_FINDING = re.compile(
    r"\b(CXR|CTA?|MRI|US|TTE|TEE|echocardiogram|EKG|ECG|ultrasound)\b[^.]{0,80}\b("
    r"showed|revealed|demonstrated|notable|consistent\s+with|"
    r"positive|negative|reveal|reveals|noted|impression|read\s+as|"
    r"w/\s+|with\s+(infiltrate|consolidation|effusion|edema|stenosis|ICH|infarct|embolus|hemorrhage|mass|nodule))\b",
    re.IGNORECASE)

DIRECT_DX = re.compile(
    r"\b(diagnosed\s+with|found\s+to\s+have|concern\s+for|consistent\s+with|"
    r"impression\s*:|assessment\s*:)\b", re.IGNORECASE)

ADMITTED_FOR = re.compile(
    r"\badmitted\s+(for|to)\s+(workup|management|treatment|further|"
    r"acute|exacerbation|sepsis|stroke|pneumonia|HF|CHF|COPD|UTI|"
    r"GI\s+bleed|DKA|AKI)\b", re.IGNORECASE)

CURRENT_TREATMENT = re.compile(
    r"\b(received|was\s+given|started\s+on|treated\s+with|treating\s+with|"
    r"placed\s+on|initiated)\s+[^.]{0,60}\b("
    r"vancomycin|zosyn|piperacillin|cefepime|ceftriaxone|levofloxacin|"
    r"azithromycin|ciprofloxacin|levophed|norepinephrine|vasopressin|"
    r"lasix|furosemide|insulin\s+drip|glucagon|D5|D50|dextrose|"
    r"duonebs|methylprednisolone|tPA|alteplase|"
    r"IV\s+fluids|fluid\s+bolus|crystalloid|pRBC|transfusion|"
    r"thrombectomy|cardioversion)\b", re.IGNORECASE)


def classify_sentence(c):
    historical = bool(HISTORICAL.search(c))
    current = bool(CURRENT.search(c))
    bulgular = []
    if LAB_VALUE_NUMERIC.search(c) or LAB_VALUE_SHORT.search(c):
        m = LAB_VALUE_NUMERIC.search(c) or LAB_VALUE_SHORT.search(c)
        bulgular.append(("lab_numeric", m.group()))
    if LAB_DESCRIPTOR.search(c):
        bulgular.append(("lab_descriptor", LAB_DESCRIPTOR.search(c).group()))
    if IMAGING_FINDING.search(c):
        bulgular.append(("imaging_finding", IMAGING_FINDING.search(c).group()))
    if DIRECT_DX.search(c):
        bulgular.append(("direct_dx", DIRECT_DX.search(c).group()))
    if ADMITTED_FOR.search(c):
        bulgular.append(("admitted_for", ADMITTED_FOR.search(c).group()))
    if CURRENT_TREATMENT.search(c):
        bulgular.append(("current_treatment", CURRENT_TREATMENT.search(c).group()))

    if not bulgular:
        return ("HISTORICAL_ALLOWED" if historical else "CLEAN"), []

    if historical and not current:
        return "HISTORICAL_ALLOWED", bulgular

    has_lab_n = any(p == "lab_numeric" for p, _ in bulgular)
    has_imaging = any(p == "imaging_finding" for p, _ in bulgular)
    has_dx = any(p == "direct_dx" for p, _ in bulgular)
    has_adm = any(p == "admitted_for" for p, _ in bulgular)
    has_tx = any(p == "current_treatment" for p, _ in bulgular)

    if current:
        if has_lab_n or has_imaging or has_dx or has_adm:
            return "CURRENT_HIGH", bulgular
        if has_tx:
            return "CURRENT_MEDIUM", bulgular
        return "REVIEW_UNCERTAIN", bulgular

    if has_lab_n: return "CURRENT_HIGH", bulgular
    if has_imaging: return "CURRENT_HIGH", bulgular
    if has_dx: return "CURRENT_HIGH", bulgular
    if has_adm: return "CURRENT_HIGH", bulgular
    if has_tx: return "CURRENT_MEDIUM", bulgular
    return "REVIEW_UNCERTAIN", bulgular


def cumlelere_bol(t):
    if not t: return []
    return [c.strip() for c in re.split(r"(?<=[.!?])\s+(?=[A-Z\d])", t) if c.strip()]


def clean_hpi(t):
    if not isinstance(t, str) or not t: return t
    ros_m = re.search(r"(?ims)(\n\s*(?:Review of Systems?|ROS)\s*:)", t)
    on, ros_kismi = (t[:ros_m.start()], t[ros_m.start():]) if ros_m else (t, "")
    cumleler = cumlelere_bol(on)
    korunan = [c for c in cumleler if classify_sentence(c)[0] not in ("CURRENT_HIGH","CURRENT_MEDIUM")]
    yeni = " ".join(korunan).strip()
    return yeni + ("\n\n" + ros_kismi.strip() if ros_kismi else "")


def clean_pmh(t):
    if not isinstance(t, str) or not t: return t
    cumleler = cumlelere_bol(t)
    korunan = [c for c in cumleler if classify_sentence(c)[0] not in ("CURRENT_HIGH","CURRENT_MEDIUM")]
    return " ".join(korunan).strip()


def audit_after(cc, hpi, pmh):
    """Audit sadece CC + HPI + PMH uzerinde (medications/allergies/vitals dahil DEGIL)."""
    text = "\n".join([cc or "", hpi or "", pmh or ""])
    if not text.strip():
        return "CLEAN", []
    cumleler = cumlelere_bol(text)
    severities = []
    reasons = []
    for c in cumleler:
        kat, b = classify_sentence(c)
        if kat in ("CURRENT_HIGH", "CURRENT_MEDIUM", "REVIEW_UNCERTAIN"):
            severities.append(kat)
            for ad, eslesen in b:
                reasons.append(f"{kat}:{ad}={eslesen[:30]}")
        elif kat == "HISTORICAL_ALLOWED":
            severities.append(kat)

    if not severities:
        return "CLEAN", []
    oncelik = {"CURRENT_HIGH":4,"CURRENT_MEDIUM":3,"REVIEW_UNCERTAIN":2,"HISTORICAL_ALLOWED":1}
    en_yuksek = max(severities, key=lambda x: oncelik.get(x, 0))
    return en_yuksek, reasons[:8]


# MAIN
print("\n[1/5] Veri yukleniyor...")
df = pd.read_parquet(SPLIT_YOL)
HPI_KAYNAK = "hpi_clean_v3_precleanup" if "hpi_clean_v3_precleanup" in df.columns else "hpi_clean_v3"
print(f"   N={len(df):,}, HPI kaynak: {HPI_KAYNAK}")

df["cc_v4_timeaware"] = df["cc_v3"]

print("\n[2/5] HPI clean...")
df["hpi_clean_v4_timeaware"] = df[HPI_KAYNAK].apply(clean_hpi)
print("\n[3/5] PMH clean...")
df["pmh_v4_timeaware"] = df["pmh_v3"].apply(clean_pmh)

# Compose V4
def compose_v4(row):
    parts = []
    if row.get("cc_v4_timeaware"): parts.append(f"Chief Complaint:\n{row['cc_v4_timeaware']}")
    if row.get("hpi_clean_v4_timeaware"): parts.append(f"Symptom History:\n{row['hpi_clean_v4_timeaware']}")
    if row.get("pmh_v4_timeaware"): parts.append(f"Past Medical History:\n{row['pmh_v4_timeaware']}")
    if row.get("allergies_v3"): parts.append(f"Allergies:\n{row['allergies_v3']}")
    if row.get("meds_adm_v3"): parts.append(f"Medications on Admission:\n{row['meds_adm_v3']}")
    if row.get("vitals_v3"): parts.append(f"Vital Signs:\n{row['vitals_v3']}")
    return "\n\n".join(parts)

df["triage_input_v4_timeaware"] = df.apply(compose_v4, axis=1)

# Maskele
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
    lambda m: BIRLESIK.sub("[DIAG_MASKED]", m) if m else m)

print("\n[4/5] AFTER-CLEANUP audit (sadece CC+HPI+PMH uzerinde)...")
sevs, reasons = [], []
for _, row in df.iterrows():
    s, r = audit_after(row["cc_v4_timeaware"], row["hpi_clean_v4_timeaware"], row["pmh_v4_timeaware"])
    sevs.append(s); reasons.append(",".join(r) if r else "no_leak")
df["strict_current_leakage_severity"] = sevs
df["strict_current_leakage_reason"] = reasons
df["strict_current_leakage_flag"] = df["strict_current_leakage_severity"] != "CLEAN"

# Audit raporu
print("\n[5/5] AUDIT")
print("=" * 70)
for split_ad, alt in [("ALL", df), ("test", df[df["split"]=="test"]),
                      ("val", df[df["split"]=="val"]),
                      ("train_S3", df[df["split"]=="train_S3"])]:
    n = len(alt)
    print(f"\n--- {split_ad} (N={n:,}) ---")
    sev_say = alt["strict_current_leakage_severity"].value_counts().to_dict()
    for sev in ["CLEAN","HISTORICAL_ALLOWED","REVIEW_UNCERTAIN","CURRENT_MEDIUM","CURRENT_HIGH"]:
        c = sev_say.get(sev, 0)
        print(f"   {sev:<25s} {c:>5} ({100*c/max(n,1):>5.1f}%)")

# Sanity
print("\n   Sanity check:")
fp_test = [
    ("multiple CVAs+history", "Mr. X is a man with history of multiple CVAs and dementia."),
    ("presenting with cough", "Patient presenting with cough, wheezing for 2 days."),
    ("h/o pneumonia", "She has h/o pneumonia and recurrent UTI on chronic suppressive."),
    ("LAB CURRENT lab+sayi", "Upon arrival, WBC 14, Cr 2.1, lactate 3.2."),
    ("IMAGING CURRENT", "CXR showed bilateral infiltrates."),
    ("DX CURRENT", "He was diagnosed with sepsis and admitted for management."),
    ("HISTORICAL EF", "Last TTE showed EF 35% in 2020."),
    ("baseline Cr", "CKD with baseline Cr 1.8."),
    ("Enoxaparin Sodium 80mg", "8. Enoxaparin Sodium 80 mg SC Q12H"),
    ("Nifedipine CR 30 mg", "6. NIFEdipine CR 30 mg PO DAILY"),
    ("Calcium 500 mg", "Calcium Carbonate 500 mg PO DAILY"),
]
for ad, c in fp_test:
    kat, b = classify_sentence(c)
    print(f"      {ad:<35s} -> {kat}")

# Test ozet
test_alt = df[df["split"]=="test"]
print(f"\n   TEST ozet (N=1.650):")
for sev in ["CLEAN","HISTORICAL_ALLOWED","REVIEW_UNCERTAIN","CURRENT_MEDIUM","CURRENT_HIGH"]:
    c = (test_alt["strict_current_leakage_severity"]==sev).sum()
    print(f"      {sev:<25s} {c}")

# 10 ornek
print("\n" + "=" * 70)
print("ORNEKLER")
print("=" * 70)
random.seed(42)
test_df = df[df["split"]=="test"].reset_index(drop=True)
ornek = [("CLEAN", 3), ("HISTORICAL_ALLOWED", 3), ("REVIEW_UNCERTAIN", 2),
         ("CURRENT_MEDIUM", 1), ("CURRENT_HIGH", 2)]
for kat, sayi in ornek:
    alt = test_df[test_df["strict_current_leakage_severity"]==kat]
    if len(alt) == 0:
        print(f"\n--- {kat}: 0 ornek ---")
        continue
    n_ornek = min(sayi, len(alt))
    secilen = alt.sample(n=n_ornek, random_state=42)
    print(f"\n{'#'*70}\n# {kat} ({len(alt)} kayit)\n{'#'*70}")
    for _, row in secilen.iterrows():
        v4 = row["triage_input_v4_timeaware"] or ""
        print(f"\n  --- note={row['note_id']} label={row['label']} ---")
        print(f"  V4 ({len(v4)} char):")
        print(f"    {v4[:550]}")
        print(f"  reasons: {row['strict_current_leakage_reason'][:200]}")

# Acceptance
print("\n" + "=" * 70)
print("ACCEPTANCE")
print("=" * 70)
hi = (test_alt["strict_current_leakage_severity"]=="CURRENT_HIGH").sum()
md = (test_alt["strict_current_leakage_severity"]=="CURRENT_MEDIUM").sum()
rv = (test_alt["strict_current_leakage_severity"]=="REVIEW_UNCERTAIN").sum()
print(f"   Test HIGH={hi}, MEDIUM={md}, REVIEW={rv}")
if hi == 0 and md == 0 and rv < 0.05*len(test_alt):
    print(f"   ✅ ACCEPT")
else:
    print(f"   ⚠️  Hedef: HIGH=0, MEDIUM=0, REVIEW<5%")

df.to_parquet(SPLIT_YOL, compression="snappy", index=False)
print(f"\n   Parquet ({SPLIT_YOL.stat().st_size/1024**2:.1f} MB)")
print("\n" + "=" * 70)
print("BLOK 13E v2 TAMAMLANDI")
