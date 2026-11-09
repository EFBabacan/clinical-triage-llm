"""Blok 13D: Time-aware triage input (V4) + leakage audit.
Historical EHR ALLOWED. Current encounter workup/treatment FORBIDDEN.
Inference YOK."""
import pandas as pd
import re
import time
import random
from pathlib import Path

OUTPUT_DIR = Path(r"./workdir/output")
SPLIT_YOL  = OUTPUT_DIR / "discharge_split.parquet"

print("=" * 70)
print("BLOK 13D — Time-Aware Triage Input (V4)")
print("=" * 70)

ALLOWED_CLASSES = [
    "SEPSIS", "STROKE", "ATRIAL_FIBRILLATION", "AKI", "PNEUMONIA", "UTI",
    "HEART_FAILURE", "GI_BLEEDING", "DIABETES_COMPLICATION",
    "COPD_EXACERBATION", "OTHER"
]

# ============================================================
# HISTORICAL MARKERS (bunlar bir cumlede varsa cumle historical, KORUNUR)
# ============================================================
HISTORICAL_MARKERS = re.compile(
    r"\b("
    r"history\s+of|hx\s+of|h/o|"
    r"prior|previously|remote|chronic|baseline|"
    r"in\s+the\s+past|past\s+medical|"
    r"s/p|status\s+post|"
    r"recently\s+admitted\s+(in|on|to)|"  # 'recently admitted to OSH for X' historical
    r"prior\s+(admission|hospitalization|visit|imaging|workup|labs?)|"
    r"(\d+|few|several|many)\s+(years?|months?|weeks?|days?)\s+ago|"
    r"old(\s+|-)?(imaging|labs?|TTE|CT|MRI|CXR)|"
    r"home\s+(medications?|meds?|O2|insulin|inhaler|nebulizer)|"
    r"on\s+chronic|chronic\s+suppressive|"
    r"baseline\s+(Cr|creatinine|Hgb|labs?|EF|FEV)"
    r")\b",
    re.IGNORECASE
)

# Bir cumlede yil ifadesi varsa (___ yer tutucu MIMIC) historical sinyal
YEAR_MARKER = re.compile(r"\b(in\s+___|in\s+\d{4}|on\s+___|admitted\s+on\s+___)\b", re.IGNORECASE)

# ============================================================
# CURRENT ENCOUNTER MARKERS (bunlar varsa ZAMANSAL sinyal current)
# ============================================================
CURRENT_MARKERS = re.compile(
    r"\b("
    r"on\s+arrival|upon\s+arrival|on\s+presentation|presented\s+to\s+(the\s+)?ED|"
    r"in\s+the\s+ED|in\s+ED|ED\s+course|"
    r"initial\s+(labs?|vitals)|"
    r"on\s+admission(?!\s+to\s+rehab)|admitted\s+to\s+(MICU|ICU|floor|service|cardiology|medicine)|"
    r"transferred\s+to|sent\s+to\s+(MICU|ICU|floor|cardiology|medicine|pulmonology)|"
    r"received\s+(in\s+ED|in\s+the\s+ED)"
    r")\b",
    re.IGNORECASE
)

# ============================================================
# LEAKAGE PATTERNS — HIGH (current diagnosis/imaging/lab)
# ============================================================
HIGH_LEAK_PATTERNS = {
    # Imaging finding (mevcut basvuru)
    "imaging_showed": re.compile(
        r"\b(CXR|CTA?|MRI|US|TTE|TEE|echocardiogram|EKG|ECG|"
        r"ultrasound|sonography)\b[^.]{0,80}\b("
        r"showed|revealed|demonstrated|notable|consistent|"
        r"with|w/|positive|negative|reveal|reveals|noted|"
        r"found|impression|read\s+as"
        r")\b",
        re.IGNORECASE
    ),
    # Diagnosis assertion (current)
    "diagnosed_with": re.compile(r"\bdiagnosed\s+with\b", re.IGNORECASE),
    "found_to_have": re.compile(r"\bfound\s+to\s+have\b", re.IGNORECASE),
    "concern_for": re.compile(r"\bconcern\s+for\b", re.IGNORECASE),
    "consistent_with_dx": re.compile(
        r"\bconsistent\s+with\b[^.]{0,40}\b("
        r"sepsis|stroke|pneumonia|UTI|infarct|hemorrhage|embolus|"
        r"effusion|infiltrate|consolidation|edema|stenosis|ICH"
        r")\b",
        re.IGNORECASE
    ),
    "admitted_for": re.compile(
        r"\badmitted\s+(for|to)\b[^.]{0,50}\b("
        r"workup|management|treatment|further|acute|exacerbation|"
        r"sepsis|stroke|pneumonia|HF|CHF|COPD|UTI|GI\s+bleed|DKA"
        r")\b",
        re.IGNORECASE
    ),
    # Lab values (current encounter)
    "lab_value_numeric": re.compile(
        r"\b(WBC|Hgb|Hct|platelets?|PLT|INR|lactate|troponin|"
        r"BUN|Cr|creatinine|sodium|Na\+?|K\+?)\s*[<>=]?\s*\d[\d.]*",
        re.IGNORECASE
    ),
    "trop_negative": re.compile(r"\btrop(onin)?\s+(negative|positive|elevated|<|>)", re.IGNORECASE),
    "lactate_value": re.compile(r"\blactate\s+(of\s+)?\d", re.IGNORECASE),
    "labs_notable": re.compile(r"\blabs?\s+(notable|showed|drawn|remarkable|reveal)", re.IGNORECASE),
    "ua_result": re.compile(r"\bUA\s+(showed|notable|positive|negative|with)", re.IGNORECASE),
    "cultures": re.compile(r"\b(blood|urine|sputum)\s+cultures?\s+(grew|positive|notable)", re.IGNORECASE),
    "workup_showed": re.compile(r"\bworkup\s+(showed|revealed|positive|negative|notable)", re.IGNORECASE),
}

# ============================================================
# LEAKAGE PATTERNS — MEDIUM (current treatment / etiket-spesifik tedavi)
# ============================================================
# Bunlar etiket-spesifik tedavi: vancomycin -> sepsis, glucagon -> hypoglycemia
MEDIUM_TREATMENT_PATTERNS = {
    "received_treatment": re.compile(
        r"\b(received|was\s+given|started\s+on|treated\s+with|treating\s+with)\b[^.]{0,60}\b("
        # Antibiotics
        r"vancomycin|zosyn|piperacillin|cefepime|ceftriaxone|levofloxacin|"
        r"azithromycin|ciprofloxacin|nitrofurantoin|bactrim|cephalexin|"
        # Pressors
        r"levophed|norepinephrine|vasopressin|epinephrine|phenylephrine|"
        # Diuretics
        r"lasix|furosemide|bumetanide|metolazone|"
        # GI
        r"protonix|pantoprazole|octreotide|"
        # Endocrine
        r"insulin\s+drip|glucagon|D5|D50|dextrose|"
        # Pulmonary
        r"duonebs|albuterol\s+nebs?|nebulizer|prednisone\s+\d|methylprednisolone|"
        r"tPA|alteplase|"
        r"IV\s+fluids|IVF|fluid\s+bolus|crystalloid|"
        r"pRBC|transfusion|"
        r"thrombectomy|cardioversion"
        r")\b",
        re.IGNORECASE
    ),
}

# ============================================================
# Sentence is HISTORICAL?
# ============================================================
def is_historical_sentence(cumle):
    """Cumle historical EHR mi? (ALLOWED)"""
    if HISTORICAL_MARKERS.search(cumle):
        return True
    if YEAR_MARKER.search(cumle):
        return True
    return False

def is_current_sentence(cumle):
    """Cumle current encounter mi?"""
    return bool(CURRENT_MARKERS.search(cumle))

# ============================================================
# Sentence classification
# ============================================================
def classify_sentence(cumle):
    """Bir cumle icin: CLEAN | HISTORICAL_ALLOWED | REVIEW_UNCERTAIN | CURRENT_HIGH | CURRENT_MEDIUM
       Her durumda hangi pattern tetikledi de doner."""
    bulgular_high = []
    for ad, rx in HIGH_LEAK_PATTERNS.items():
        m = rx.search(cumle)
        if m:
            bulgular_high.append((ad, m.group()))

    bulgular_med = []
    for ad, rx in MEDIUM_TREATMENT_PATTERNS.items():
        m = rx.search(cumle)
        if m:
            bulgular_med.append((ad, m.group()))

    # Risk yoksa CLEAN
    if not bulgular_high and not bulgular_med:
        return "CLEAN", []

    historical = is_historical_sentence(cumle)
    current = is_current_sentence(cumle)

    # Eger historical marker varsa VE current marker yoksa -> HISTORICAL_ALLOWED
    if historical and not current:
        return "HISTORICAL_ALLOWED", bulgular_high + bulgular_med

    # Eger current marker varsa -> CURRENT
    if current:
        if bulgular_high:
            return "CURRENT_HIGH", bulgular_high
        if bulgular_med:
            return "CURRENT_MEDIUM", bulgular_med

    # Hicbir tarafta marker yok ama risk pattern var -> REVIEW
    # Risk pattern HIGH olsa bile, marker yoksa belirsiz
    if bulgular_high:
        # Lab values + numerical + no historical marker = muhtemelen current
        # but to be safe, REVIEW
        return "REVIEW_UNCERTAIN", bulgular_high
    return "REVIEW_UNCERTAIN", bulgular_med

# ============================================================
# Cumlelere bol
# ============================================================
def cumlelere_bol(metin):
    if not metin: return []
    cs = re.split(r"(?<=[.!?])\s+(?=[A-Z\d])", metin)
    return [c.strip() for c in cs if c.strip()]

# ============================================================
# HPI/CC time-aware temizleme
# ============================================================
def clean_hpi_time_aware(hpi_text):
    """Cumle bazli: CURRENT_HIGH ve CURRENT_MEDIUM cumleleri at.
    HISTORICAL_ALLOWED ve CLEAN cumleleri tut. REVIEW_UNCERTAIN: kaydet ama at."""
    if not isinstance(hpi_text, str) or not hpi_text:
        return hpi_text, [], []

    # ROS koru
    ros_m = re.search(r"(?ims)(\n\s*(?:Review of Systems?|ROS)\s*:)", hpi_text)
    if ros_m:
        on = hpi_text[:ros_m.start()]
        ros_kismi = hpi_text[ros_m.start():]
    else:
        on = hpi_text
        ros_kismi = ""

    cumleler = cumlelere_bol(on)
    korunan = []
    atilan_log = []  # [(cumle, kategori, bulgular)]
    review_log = []

    for c in cumleler:
        kat, bulgular = classify_sentence(c)
        if kat == "CLEAN":
            korunan.append(c)
        elif kat == "HISTORICAL_ALLOWED":
            korunan.append(c)
        elif kat == "REVIEW_UNCERTAIN":
            review_log.append((c, kat, bulgular))
            # Conservative: REVIEW'i at ama kaydet
            atilan_log.append((c, kat, bulgular))
        elif kat in ("CURRENT_HIGH", "CURRENT_MEDIUM"):
            atilan_log.append((c, kat, bulgular))

    yeni_on = " ".join(korunan).strip()
    if ros_kismi:
        return yeni_on + "\n\n" + ros_kismi.strip(), atilan_log, review_log
    return yeni_on, atilan_log, review_log

# ============================================================
# PMH/Meds: sadece "current encounter" yapan cumleleri at, kalan tum historical TUTULUR
# ============================================================
def clean_pmh_time_aware(pmh_text):
    """PMH genelde tum historical. Sadece is_current_sentence olanlari at."""
    if not isinstance(pmh_text, str) or not pmh_text:
        return pmh_text, []
    cumleler = cumlelere_bol(pmh_text)
    korunan = []
    atilan = []
    for c in cumleler:
        if is_current_sentence(c):
            kat, b = classify_sentence(c)
            if kat in ("CURRENT_HIGH", "CURRENT_MEDIUM"):
                atilan.append((c, kat, b))
                continue
        korunan.append(c)
    return " ".join(korunan).strip(), atilan

# ============================================================
# Audit overall: V4 input'unun tum kelimelerinde leakage flag
# ============================================================
def audit_v4_input(v4_text):
    """V4 birlestirilmis input'un cumlelerini bir kez daha klasifiye et.
    Toplam: en yuksek severity flag + reason listesi."""
    if not isinstance(v4_text, str) or not v4_text:
        return "CLEAN", "EMPTY", []
    cumleler = cumlelere_bol(v4_text)
    severities = []
    reasons = []
    for c in cumleler:
        kat, bulgular = classify_sentence(c)
        if kat != "CLEAN":
            severities.append(kat)
            for ad, eslesen in bulgular:
                reasons.append(f"{kat}:{ad}={eslesen[:30]}")

    if not severities:
        return "CLEAN", "no_leak", []

    # En yuksek severity
    oncelik = {"CURRENT_HIGH": 4, "CURRENT_MEDIUM": 3, "REVIEW_UNCERTAIN": 2, "HISTORICAL_ALLOWED": 1}
    en_yuksek = max(severities, key=lambda x: oncelik.get(x, 0))
    return en_yuksek, ",".join(set(severities)), reasons[:10]

# ============================================================
# V4 input compose
# ============================================================
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

# ============================================================
# MAIN
# ============================================================
print("\n[1/6] Veri yukleniyor...")
df = pd.read_parquet(SPLIT_YOL)
print(f"   Toplam: {len(df):,} satir, {len(df.columns)} kolon")

# Kaynak: hpi_clean_v3_precleanup (en az silinmis hali) varsa onu kullan, yoksa hpi_clean_v3
HPI_KAYNAK = "hpi_clean_v3_precleanup" if "hpi_clean_v3_precleanup" in df.columns else "hpi_clean_v3"
print(f"   HPI kaynak: {HPI_KAYNAK}")

# CC: dokunulmaz (zaten kisa, hasta ifadesi)
df["cc_v4_timeaware"] = df["cc_v3"].copy()

# ============================================================
# HPI temizle (her satir)
# ============================================================
print("\n[2/6] HPI time-aware temizleme...")
t0 = time.time()
hpi_v4_list = []
hpi_atilan_logs = []  # liste of (atilan, review)
for hpi in df[HPI_KAYNAK]:
    yeni_hpi, atilan, review = clean_hpi_time_aware(hpi)
    hpi_v4_list.append(yeni_hpi)
    hpi_atilan_logs.append((atilan, review))
df["hpi_clean_v4_timeaware"] = hpi_v4_list
print(f"   Sure: {time.time()-t0:.1f}s")

# ============================================================
# PMH temizle
# ============================================================
print("\n[3/6] PMH time-aware temizleme (sadece current cumleleri at)...")
t0 = time.time()
pmh_v4_list = []
pmh_atilan_logs = []
for pmh in df["pmh_v3"]:
    yeni_pmh, atilan = clean_pmh_time_aware(pmh)
    pmh_v4_list.append(yeni_pmh)
    pmh_atilan_logs.append(atilan)
df["pmh_v4_timeaware"] = pmh_v4_list
print(f"   Sure: {time.time()-t0:.1f}s")

# ============================================================
# V4 compose + audit
# ============================================================
print("\n[4/6] V4 compose + final audit...")
t0 = time.time()
df["triage_input_v4_timeaware"] = df.apply(compose_v4, axis=1)

# Maskeleme (Blok 4 ile ayni keyword)
DIAG_KEYWORDS = {
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
BIRLESIK = re.compile("|".join(f"(?:{p})" for p in DIAG_KEYWORDS.values()), re.IGNORECASE)
df["triage_input_v4_timeaware_masked"] = df["triage_input_v4_timeaware"].apply(
    lambda m: BIRLESIK.sub("[DIAG_MASKED]", m) if m else m
)

# Audit her satir icin
audit_flag, audit_reason, audit_severity = [], [], []
for v4 in df["triage_input_v4_timeaware"]:
    sev, reason, _ = audit_v4_input(v4)
    audit_flag.append(reason)
    audit_severity.append(sev)
    audit_reason.append(reason)
df["strict_current_leakage_severity"] = audit_severity
df["strict_current_leakage_flag"] = audit_severity != "CLEAN"
df["strict_current_leakage_reason"] = audit_reason
print(f"   Sure: {time.time()-t0:.1f}s")

# ============================================================
# Audit raporu (split bazli)
# ============================================================
print("\n[5/6] AUDIT RAPORU")
print("=" * 70)

for split_ad, alt in [("ALL", df), ("test", df[df["split"]=="test"]),
                       ("val", df[df["split"]=="val"]),
                       ("train_S3", df[df["split"]=="train_S3"])]:
    n = len(alt)
    print(f"\n--- {split_ad} (N={n:,}) ---")
    sev_say = alt["strict_current_leakage_severity"].value_counts().to_dict()
    for sev in ["CLEAN", "HISTORICAL_ALLOWED", "REVIEW_UNCERTAIN", "CURRENT_MEDIUM", "CURRENT_HIGH"]:
        c = sev_say.get(sev, 0)
        if c > 0 or sev in ("CURRENT_HIGH", "CURRENT_MEDIUM", "REVIEW_UNCERTAIN"):
            print(f"   {sev:<25s} {c:>5} ({100*c/max(n,1):>5.1f}%)")

# Test ozeti
print(f"\n   TEST OZET (N=1.650):")
test_alt = df[df["split"]=="test"]
print(f"      full N             : {len(test_alt)}")
print(f"      CLEAN              : {(test_alt['strict_current_leakage_severity']=='CLEAN').sum()}")
print(f"      HISTORICAL_ALLOWED : {(test_alt['strict_current_leakage_severity']=='HISTORICAL_ALLOWED').sum()}")
print(f"      REVIEW_UNCERTAIN   : {(test_alt['strict_current_leakage_severity']=='REVIEW_UNCERTAIN').sum()}")
print(f"      CURRENT_MEDIUM     : {(test_alt['strict_current_leakage_severity']=='CURRENT_MEDIUM').sum()}")
print(f"      CURRENT_HIGH       : {(test_alt['strict_current_leakage_severity']=='CURRENT_HIGH').sum()}")

# Uzunluk karsilastirma
v3_lens = df["triage_input_v3"].fillna("").str.len() if "triage_input_v3" in df.columns else None
v4_lens = df["triage_input_v4_timeaware"].fillna("").str.len()
print(f"\n   Uzunluk: V4 ort={v4_lens.mean():.0f}, max={v4_lens.max()}")
if v3_lens is not None:
    print(f"            V3 ort={v3_lens.mean():.0f}, V4 daha {('uzun' if v4_lens.mean()>v3_lens.mean() else 'kisa')}")

# ============================================================
# Ornek yazdir (3'er ornek)
# ============================================================
print("\n[6/6] ORNEKLER (her kategoriden 3 ornek)")
print("=" * 70)

random.seed(42)
test_df = df[df["split"]=="test"].reset_index(drop=True)

for kategori in ["CLEAN", "HISTORICAL_ALLOWED", "REVIEW_UNCERTAIN", "CURRENT_MEDIUM", "CURRENT_HIGH"]:
    alt = test_df[test_df["strict_current_leakage_severity"]==kategori]
    if len(alt) == 0:
        print(f"\n--- {kategori} --- 0 ornek")
        continue
    n_ornek = min(3, len(alt))
    secilen = alt.sample(n=n_ornek, random_state=42).index.tolist()
    print(f"\n{'#'*70}")
    print(f"# {kategori} ({len(alt)} kayit, {n_ornek} ornek)")
    print(f"{'#'*70}")
    for idx in secilen:
        row = alt.loc[idx]
        print(f"\n   --- note={row['note_id']} label={row['label']} ---")
        print(f"   V4 input ({len(row['triage_input_v4_timeaware'])} char) PREVIEW:")
        print(f"     {(row['triage_input_v4_timeaware'] or '')[:600]}")
        print(f"   reasons: {row['strict_current_leakage_reason'][:200]}")

# ============================================================
# Acceptance kriteri
# ============================================================
print("\n" + "=" * 70)
print("ACCEPTANCE")
print("=" * 70)
high_test = (test_df["strict_current_leakage_severity"]=="CURRENT_HIGH").sum()
med_test = (test_df["strict_current_leakage_severity"]=="CURRENT_MEDIUM").sum()
rev_test = (test_df["strict_current_leakage_severity"]=="REVIEW_UNCERTAIN").sum()

print(f"   Test CURRENT_HIGH   = {high_test}  (hedef: 0)")
print(f"   Test CURRENT_MEDIUM = {med_test}  (hedef: 0)")
print(f"   Test REVIEW         = {rev_test}  (hedef: 0)")

if high_test == 0 and med_test == 0 and rev_test == 0:
    print(f"\n   ✅ ACCEPT: Test'te HIGH/MEDIUM/REVIEW yok. V4 final.")
else:
    print(f"\n   ⚠️  HIGH/MEDIUM/REVIEW > 0. Insan karari gerekli.")
    print(f"      Yukaridaki ornekleri inceleyip patternlari refine et.")

# Parquet'e yaz
print("\nKAYDETME...")
df.to_parquet(SPLIT_YOL, compression="snappy", index=False)
print(f"   {SPLIT_YOL.name} ({SPLIT_YOL.stat().st_size/1024**2:.1f} MB, {len(df.columns)} kolon)")

# Yeni kolonlar
yeni_kolonlar = ["cc_v4_timeaware", "hpi_clean_v4_timeaware", "pmh_v4_timeaware",
                 "triage_input_v4_timeaware", "triage_input_v4_timeaware_masked",
                 "strict_current_leakage_flag", "strict_current_leakage_reason",
                 "strict_current_leakage_severity"]
print(f"   Yeni kolonlar: {yeni_kolonlar}")

print("\n" + "=" * 70)
print("BLOK 13D TAMAMLANDI - inference YOK")
print("=" * 70)
