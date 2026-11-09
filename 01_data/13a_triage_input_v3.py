"""Blok 13A'yi standalone calistir - notebook execute YERINE direkt python.
Notebook'a hucreleri ekledik ama tum notebook'u baştan execute etmek istemiyoruz
(8B inference 30dk kayip olur). Direkt Python'da ayni mantigi calistir."""

import pandas as pd
import numpy as np
import re
import time
from pathlib import Path

OUTPUT_DIR = Path(r"./workdir/output")
SPLIT_YOL  = OUTPUT_DIR / "discharge_split.parquet"
FULL_YOL   = OUTPUT_DIR / "discharge_etiketli_full.parquet"

print("=" * 70)
print("BLOK 13A STANDALONE - TRIAGE_INPUT/V3 URETIM + AUDIT")
print("=" * 70)

HEADER_PATTERN = (
    r"(?:Name|Admission Date|Date of Birth|Service|Allergies|Attending|"
    r"Chief Complaint|Major Surgical[^:]*|History of Present Illness|"
    r"Past Medical History|Social History|Family History|Physical Exam|"
    r"ADMISSION PHYSICAL EXAM|Discharge PE|Pertinent Results|"
    r"Brief Hospital Course|Medications on Admission|Discharge Medications|"
    r"Discharge Disposition|Facility|Discharge Diagnosis|Discharge Condition|"
    r"Discharge Instructions|Followup Instructions|SUBSTANCE ABUSE HISTORY)"
)

def bolum_cikar(text, baslik_pattern, durdur_pattern=None):
    if not isinstance(text, str):
        return ""
    if durdur_pattern is None:
        durdur_pattern = HEADER_PATTERN
    pat = rf"(?ims)^\s*(?:{baslik_pattern}):\s*\n?(.+?)(?=\n\s*(?:{durdur_pattern}):|\Z)"
    m = re.search(pat, text)
    if m:
        return m.group(1).strip()
    return ""

ED_CUTOFF_PATTERNS = [
    r'\bIn the ED\b', r'\bIn the emergency (department|room)\b',
    r'\bED course\b', r'\bUpon arrival to (the )?ED\b',
    r'\bInitial vitals (in the ED|were|on arrival)\b',
    r'\bUpon (admission|arrival)\b', r'\bOn presentation\b(?=.{0,80}(BP|HR|T |labs))',
    r'\bLabs? (notable|showed|were|are|drawn|remarkable|significant)\b',
    r'\bLab(s|oratory) results?\b', r'\bWBC \d', r'\bCr \d\.\d', r'\bHgb \d',
    r'\bplatelets? \d', r'\bINR \d', r'\blactate \d', r'\btroponin\b',
    r'\bUA (showed|with|notable|positive|negative)\b',
    r'\bCXR (with|showed|was|notable|negative|positive|revealed)\b',
    r'\bCT (head|chest|abdomen|of|scan|abd|brain)\b', r'\bCTA\b',
    r'\bMRI (showed|of|brain|head|revealed)\b', r'\bUS (showed|revealed|notable)\b',
    r'\b(echocardiogram|TTE|TEE) (showed|revealed|with|demonstrated|notable)\b',
    r'\bEKG (showed|revealed|notable|with)\b', r'\bECG (showed|revealed|notable)\b',
    r'\b(He|She|Pt|Patient) was (admitted|started on|given|transferred|treated|placed)\b',
    r'\bAdmitted to\b', r'\bwas referred to\b',
    r'\breceived (IV|PO|po|im|sc|sq) ', r'\bgiven (IV|PO|po) ',
    r'\bdiagnosed with\b', r'\bconcern for\b', r'\bworkup\b',
    r'\bfound to have\b', r'\bnoted to have\b',
    r'\bstarted on (IV|PO|antibiotics|abx|heparin|insulin)\b',
]
ED_CUTOFF_REGEX = re.compile('|'.join(f'(?:{p})' for p in ED_CUTOFF_PATTERNS), re.IGNORECASE)

def hpi_temizle(hpi_text):
    if not isinstance(hpi_text, str) or not hpi_text:
        return "", False
    cutoff_uygulandi = False
    m = ED_CUTOFF_REGEX.search(hpi_text)
    if m:
        kesim_konum = m.start()
        snippet = hpi_text[:kesim_konum]
        son_nokta = max(snippet.rfind('. '), snippet.rfind('.\n'), snippet.rfind('\n\n'))
        if son_nokta > kesim_konum * 0.5:
            kesim_konum = son_nokta + 1
        on_kisim = hpi_text[:kesim_konum].rstrip()
        cutoff_uygulandi = True
    else:
        on_kisim = hpi_text.strip()

    ros_eklenmis = ""
    ros_pat = re.compile(
        r'(?ims)(^\s*(?:Review of Systems?|ROS)\s*:?[\s\S]+?)'
        r'(?=\n\s*\n[A-Z][a-z]+ [A-Z]|\Z)'
    )
    ros_m = ros_pat.search(hpi_text)
    if ros_m:
        ros_text = ros_m.group(1).strip()
        if 'ROS' not in on_kisim and 'Review of Systems' not in on_kisim:
            ros_eklenmis = '\n\n' + ros_text

    return on_kisim + ros_eklenmis, cutoff_uygulandi

def vital_satir_cikar(phys_exam):
    if not isinstance(phys_exam, str) or not phys_exam:
        return ""
    satirlar = phys_exam.split('\n')
    vital_satirlari = []
    for s in satirlar[:5]:
        s_strip = s.strip()
        if not s_strip:
            continue
        if re.search(r'\b(VS|T |BP|HR|RR|O2|SaO2|SpO2|temp|temperature|pulse)\b', s_strip, re.IGNORECASE):
            vital_satirlari.append(s_strip)
        elif re.match(r'^\d{2,3}[/.\s]', s_strip):
            vital_satirlari.append(s_strip)
        elif vital_satirlari:
            break
    return ' '.join(vital_satirlari)

# --- Veri yukle ---
print("\n[1/5] Veri yukleniyor...")
t0 = time.time()
df_split = pd.read_parquet(SPLIT_YOL)
df_full = pd.read_parquet(FULL_YOL)
print(f"   split: {len(df_split):,} satir, {len(df_split.columns)} kolon")

df_split['note_id'] = df_split['note_id'].astype(str)
df_full['note_id'] = df_full['note_id'].astype(str)
df_text = df_full[['note_id', 'text']]
df_split = df_split.merge(df_text, on='note_id', how='left')
assert df_split['text'].notna().all(), "Bazi text eksik!"
print(f"   Text birlestirildi. Sure: {time.time()-t0:.1f}s")

# --- Parse ---
print("\n[2/5] Bolum parse + HPI cutoff (6.600 kayit)...")
t0 = time.time()

cc_list, hpi_clean_list, cutoff_list = [], [], []
pmh_list, allergies_list, meds_list, vitals_list = [], [], [], []
audit_flags_list = []

for i, row in df_split.iterrows():
    text = row['text']
    cc        = bolum_cikar(text, r"Chief Complaint")
    hpi       = bolum_cikar(text, r"History of Present Illness")
    pmh       = bolum_cikar(text, r"Past Medical History")
    allergies = bolum_cikar(text, r"Allergies")
    meds      = bolum_cikar(text, r"Medications on Admission")
    phys_exam = bolum_cikar(text, r"(?:ADMISSION )?Physical Exam")

    hpi_clean, cutoff_uygulandi = hpi_temizle(hpi)
    vitals = vital_satir_cikar(phys_exam)

    flags = []
    v3_compose = ' '.join([cc, hpi_clean, pmh, allergies, meds, vitals])
    if re.search(r'\bWBC \d|\bCr \d\.\d|\bHgb \d', v3_compose, re.IGNORECASE):
        flags.append('lab_kalinti')
    if re.search(r'\bCXR\b|\bCT (head|chest|abdomen)|\bCTA\b|\bMRI\b|\bTTE\b', v3_compose, re.IGNORECASE):
        flags.append('imaging_kalinti')
    if re.search(r'\badmitted to\b|\bdiagnosed with\b|\bconcern for\b', v3_compose, re.IGNORECASE):
        flags.append('assessment_kalinti')

    cc_list.append(cc)
    hpi_clean_list.append(hpi_clean)
    cutoff_list.append(cutoff_uygulandi)
    pmh_list.append(pmh)
    allergies_list.append(allergies)
    meds_list.append(meds)
    vitals_list.append(vitals)
    audit_flags_list.append(','.join(flags) if flags else 'temiz')

df_split['cc_v3']        = cc_list
df_split['hpi_clean_v3'] = hpi_clean_list
df_split['pmh_v3']       = pmh_list
df_split['allergies_v3'] = allergies_list
df_split['meds_adm_v3']  = meds_list
df_split['vitals_v3']    = vitals_list
df_split['hpi_cutoff_applied'] = cutoff_list
df_split['triage_audit_flags'] = audit_flags_list

print(f"   Sure: {time.time()-t0:.1f}s")
print(f"   Cutoff uygulandi: {sum(cutoff_list):,}/{len(df_split):,} ({100*sum(cutoff_list)/len(df_split):.1f}%)")

# --- Compose + maskele ---
print("\n[3/5] V3 compose + maskeleme...")
def compose_v3(row):
    parcalar = []
    if row['cc_v3']:
        parcalar.append(f"Chief Complaint:\n{row['cc_v3']}")
    if row['hpi_clean_v3']:
        parcalar.append(f"Symptom History:\n{row['hpi_clean_v3']}")
    if row['pmh_v3']:
        parcalar.append(f"Past Medical History:\n{row['pmh_v3']}")
    if row['allergies_v3']:
        parcalar.append(f"Allergies:\n{row['allergies_v3']}")
    if row['meds_adm_v3']:
        parcalar.append(f"Medications on Admission:\n{row['meds_adm_v3']}")
    if row['vitals_v3']:
        parcalar.append(f"Vital Signs:\n{row['vitals_v3']}")
    return '\n\n'.join(parcalar)

df_split['triage_input_v3'] = df_split.apply(compose_v3, axis=1)

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
BIRLESIK = re.compile('|'.join(f'(?:{p})' for p in DIAG_KEYWORDS.values()), re.IGNORECASE)
CLASS_REGEXES = {cls: re.compile(pat, re.IGNORECASE) for cls, pat in DIAG_KEYWORDS.items()}

df_split['triage_input_v3_masked'] = df_split['triage_input_v3'].apply(
    lambda m: BIRLESIK.sub('[DIAG_MASKED]', m) if m else m
)

def kendi_keyword(metin, label):
    if not isinstance(metin, str) or not metin:
        return False
    rx = CLASS_REGEXES.get(label)
    if rx is None:
        return False
    return rx.search(metin) is not None

df_split['has_own_label_term_v3'] = [
    kendi_keyword(t, l) for t, l in zip(df_split['triage_input_v3'], df_split['label'])
]

print(f"   V3 ortalama uzunluk:        {df_split['triage_input_v3'].str.len().mean():>7,.0f} char")
print(f"   V3 maskeli ort. uzunluk:    {df_split['triage_input_v3_masked'].str.len().mean():>7,.0f} char")
print(f"   has_own_label_term_v3 oran: {100*df_split['has_own_label_term_v3'].mean():.1f}%")

# --- Audit ---
print("\n[4/5] Audit raporu")
print("=" * 70)
v1_uzun = df_split['model_input_v1'].str.len().fillna(0)
v3_uzun = df_split['triage_input_v3'].str.len().fillna(0)
print(f"\n   V1 vs V3 uzunluk:")
print(f"      {'':>10s}  {'V1':>10s}  {'V3':>10s}  {'V3/V1':>7s}")
print(f"      {'ortalama':>10s}  {v1_uzun.mean():>10,.0f}  {v3_uzun.mean():>10,.0f}  {100*v3_uzun.mean()/max(v1_uzun.mean(),1):>6.1f}%")
print(f"      {'medyan':>10s}  {v1_uzun.median():>10,.0f}  {v3_uzun.median():>10,.0f}  {100*v3_uzun.median()/max(v1_uzun.median(),1):>6.1f}%")
print(f"      {'min':>10s}  {v1_uzun.min():>10,.0f}  {v3_uzun.min():>10,.0f}")
print(f"      {'max':>10s}  {v1_uzun.max():>10,.0f}  {v3_uzun.max():>10,.0f}")

print(f"\n   HPI cutoff uygulandi: {df_split['hpi_cutoff_applied'].sum():,}/{len(df_split):,} ({100*df_split['hpi_cutoff_applied'].mean():.1f}%)")

print(f"\n   Audit flag dagilimi:")
flag_sayim = df_split['triage_audit_flags'].value_counts()
for flag, n in flag_sayim.head(10).items():
    print(f"      {flag:<40s} {n:>5,} ({100*n/len(df_split):>4.1f}%)")

lab_kal = df_split['triage_audit_flags'].str.contains('lab_kalinti', na=False).sum()
img_kal = df_split['triage_audit_flags'].str.contains('imaging_kalinti', na=False).sum()
ass_kal = df_split['triage_audit_flags'].str.contains('assessment_kalinti', na=False).sum()
temiz   = (df_split['triage_audit_flags'] == 'temiz').sum()
print(f"\n   Ozet kalinti:")
print(f"      Lab kalintisi      : {lab_kal:>5,} ({100*lab_kal/len(df_split):.1f}%)")
print(f"      Imaging kalintisi  : {img_kal:>5,} ({100*img_kal/len(df_split):.1f}%)")
print(f"      Assessment kalintisi: {ass_kal:>5,} ({100*ass_kal/len(df_split):.1f}%)")
print(f"      Tamamen temiz      : {temiz:>5,} ({100*temiz/len(df_split):.1f}%)")

print(f"\n   Sinif bazli (V3 ort uzunluk + cutoff% + has_own%):")
print(f"   {'Sinif':<25s} {'V3 ort':>8s} {'V1 ort':>8s} {'cutoff%':>9s} {'has_own%':>10s}")
print("   " + "-" * 65)
for cls in sorted(df_split['label'].unique()):
    alt = df_split[df_split['label'] == cls]
    v3_o = alt['triage_input_v3'].str.len().mean()
    v1_o = alt['model_input_v1'].str.len().mean()
    cut = 100 * alt['hpi_cutoff_applied'].mean()
    own = 100 * alt['has_own_label_term_v3'].mean()
    print(f"   {cls:<25s} {v3_o:>8,.0f} {v1_o:>8,.0f} {cut:>8.1f}% {own:>9.1f}%")

print(f"\n   Clean test boyutu (V3, has_own_label_term_v3=False):")
test_df = df_split[df_split['split']=='test']
clean_v3 = (~test_df['has_own_label_term_v3']).sum()
print(f"      Test toplam       : {len(test_df):>4,}")
print(f"      V3 clean (yeni)   : {clean_v3:>4,} ({100*clean_v3/len(test_df):.1f}%)")
print(f"      V1 clean (referans): {(~test_df['has_own_label_term_v1']).sum():>4,} ({100*(~test_df['has_own_label_term_v1']).mean():.1f}%)")

# --- Sınıf başına 2 örnek ---
print("\n[5/5] Sinif basina 2 ornek (V3 maskeli, val'dan):")
print("=" * 70)
siralama = ["SEPSIS", "STROKE", "ATRIAL_FIBRILLATION", "AKI", "PNEUMONIA", "UTI",
            "HEART_FAILURE", "GI_BLEEDING", "DIABETES_COMPLICATION", "COPD_EXACERBATION", "OTHER"]
val_df = df_split[df_split['split']=='val']
for cls in siralama:
    alt = val_df[val_df['label']==cls]
    if len(alt) < 2: continue
    print("\n" + "#" * 70)
    print(f"# {cls}")
    print("#" * 70)
    for j in range(2):
        ornek = alt.iloc[j]
        v3 = ornek['triage_input_v3_masked'] or ""
        print(f"\n  --- Ornek {j+1} (note_id: {ornek['note_id']}, has_own: {ornek['has_own_label_term_v3']}, cutoff: {ornek['hpi_cutoff_applied']}, flags: {ornek['triage_audit_flags']}) ---")
        print(f"  Uzunluk: {len(v3)} char")
        print(f"  --- BASLANGIC ---")
        print(v3[:1500])
        if len(v3) > 1500:
            print(f"  ... ({len(v3)-1500} char daha) ...")
        print(f"  --- SON ---")

# --- Kaydet ---
print("\n" + "=" * 70)
print("KAYDETME")
print("=" * 70)
if 'text' in df_split.columns:
    df_split = df_split.drop(columns=['text'])

t0 = time.time()
df_split.to_parquet(SPLIT_YOL, compression='snappy', index=False)
boyut = SPLIT_YOL.stat().st_size / (1024*1024)
print(f"   Yazildi: {SPLIT_YOL.name}")
print(f"   Boyut  : {boyut:.1f} MB")
print(f"   Kolon  : {len(df_split.columns)}")
print(f"   Sure   : {time.time()-t0:.1f}s")

print("\n" + "=" * 70)
print("BLOK 13A TAMAMLANDI")
print("=" * 70)
