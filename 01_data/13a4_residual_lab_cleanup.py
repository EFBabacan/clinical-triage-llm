"""Blok 13A.4: Final residual lab cleanup. Vital korunuyor. Sadece hpi_clean_v3."""
import pandas as pd
import re
import time
from pathlib import Path

OUTPUT_DIR = Path(r"./workdir/output")
SPLIT_YOL  = OUTPUT_DIR / "discharge_split.parquet"

print("=" * 70)
print("BLOK 13A.4 — RESIDUAL LAB CLEANUP (vital koruyor, sadece hpi_clean_v3)")
print("=" * 70)

df = pd.read_parquet(SPLIT_YOL)
print(f"\n   Yuklendi: {len(df):,} satir, {len(df.columns)} kolon")
assert 'hpi_clean_v3_precleanup' in df.columns, "Yedek yok!"
assert 'hpi_clean_v3' in df.columns, "hpi_clean_v3 yok!"

# --- Vital koruma regex (DOKUNULMAZ) ---
VITAL_KEEP_PATTERNS = [
    r'\bhypotens\w*\b', r'\bhypertens\w*\b',
    r'\btachycardi\w*\b', r'\bbradycardi\w*\b',
    r'\btachypne\w*\b', r'\bbradypne\w*\b',
    r'\bfebrile\b', r'\bafebrile\b',
    r'\bhypoxic?\b', r'\bhypoxia\b',
    r'\bhypoglycem\w*\b', r'\bhyperglycem\w*\b',
    r'\bdesaturat\w*\b', r'\bdesat\b',
    r'\bSpO2\b', r'\bSaO2\b', r'\bO2\s*sat\w*\b',
    r'\bglucose\b', r'\bblood\s*sugar\b', r'\bfingerstick\b',
    r'\b(BP|SBP|DBP)\s*\d',
    r'\bHR\s*\d', r'\bRR\s*\d', r'\bpulse\s*\d',
    r'\bT\s*\d{2,3}\.\d', r'\btemp(erature)?\s*\d',
    r'\bfever\b',
]
VITAL_KEEP_REGEX = re.compile('|'.join(f'(?:{p})' for p in VITAL_KEEP_PATTERNS), re.IGNORECASE)

# --- LAB FRAGMENT patterns (cumlenin parcasi olarak silinir, vital varsa cumle korunur) ---
# 1) Direkt sayisal lab degerleri
LAB_FRAGMENT_PATTERNS = [
    # "Trop negative", "troponin negative/positive"
    re.compile(r'(?<![A-Za-z])Trop(onin)?\s+(negative|positive|elevated|<\s*\d|\d[\d.]*)', re.IGNORECASE),
    # "Lactate of 2.6", "lactate 2.6", "lactate elevated"
    re.compile(r'\blactate(\s+of)?\s+\d[\d.]*', re.IGNORECASE),
    re.compile(r'\blactate\s+(elevated|normal|negative)', re.IGNORECASE),
    # "low sodium (119)", "high sodium (155)"
    re.compile(r'\b(low|high|elevated|decreased)\s+(sodium|potassium|chloride|bicarb|magnesium|calcium|phosphate|albumin|bilirubin|creatinine|BUN)(\s*\([^)]+\))?', re.IGNORECASE),
    # "Na 130", "Na+ 130", "K 4.5", "Cr 1.7", "BUN 35"
    re.compile(r'\b(Na\+?|K\+?|Cl|HCO3|Mg|Ca|Phos|BUN|Cr|creatinine|sodium|potassium)\s*[<>=]?\s*\d[\d.]*', re.IGNORECASE),
    # "Hgb 11.3", "HCT 14", "WBC 14", "platelets 200", "PLT 200", "INR 1.3"
    re.compile(r'\b(Hgb|HCT|WBC|PLT|platelets?|INR|MCV|RDW|ANC)\s*[<>=]?\s*\d[\d.]*', re.IGNORECASE),
    # "albumin <2", "albumin 2.1"
    re.compile(r'\balbumin\s*[<>=]?\s*\d[\d.]*', re.IGNORECASE),
    # "UA negative/positive/notable"
    re.compile(r'\bUA\s+(negative|positive|notable|with|showed|unremarkable|clean)', re.IGNORECASE),
    # "Labs done/showed/notable" - cumle pre-fix temizligi
    re.compile(r'\bLabs?\s+(done|drawn|notable|showed|remarkable|significant|reveal)\b', re.IGNORECASE),
    # "labs were unchanged from baseline with Na 130" - bu fragment formati
    re.compile(r'\blabs?\s+(were|are)\s+[^.]*?\b(Na\+?|K\+?|Cr|BUN|Hgb|HCT|WBC)\s*\d[\d.]*', re.IGNORECASE),
    # "Outside laboratory data notable for"
    re.compile(r'\b(outside\s+)?laboratory\s+data\s+(notable|showed|reveal)', re.IGNORECASE),
]

# Cumle drop patterns (lab dominantli cumleler, vital yoksa silinir)
LAB_SENTENCE_DROP_PATTERNS = [
    # "Labs notable for X, Y, Z" tipinde lab-only cumle
    re.compile(r'\bLabs?\b[^.]{0,80}\b(notable|showed|drawn|remarkable|reveal)', re.IGNORECASE),
    re.compile(r'\b(outside\s+)?laboratory\s+data\s+(notable|showed)', re.IGNORECASE),
    # "UA done/notable/showed/positive/negative"
    re.compile(r'\bUA\b[^.]{0,40}\b(done|notable|showed|positive|negative|unremarkable)\b', re.IGNORECASE),
]

def lab_fragment_temizle(cumle):
    """Cumle icindeki lab fragmentlerini sil."""
    yeni = cumle
    for rx in LAB_FRAGMENT_PATTERNS:
        yeni = rx.sub('', yeni)
    # Tipografik temizlik
    yeni = re.sub(r'\s*,\s*,', ',', yeni)
    yeni = re.sub(r'\s*,\s*\.', '.', yeni)
    yeni = re.sub(r'\s+,', ',', yeni)
    yeni = re.sub(r'\s+\.', '.', yeni)
    yeni = re.sub(r'\(\s*\)', '', yeni)  # bos parantez
    yeni = re.sub(r'\s{2,}', ' ', yeni)
    return yeni.strip()

def cumle_temizle(metin):
    if not isinstance(metin, str) or not metin:
        return metin, 0, 0

    # ROS koru
    ros_match = re.search(r'(?ims)(\n\s*(?:Review of Systems?|ROS)\s*:)', metin)
    if ros_match:
        on_kisim = metin[:ros_match.start()]
        ros_kismi = metin[ros_match.start():]
    else:
        on_kisim = metin
        ros_kismi = ""

    cumleler = re.split(r'(?<=[.!?])\s+(?=[A-Z\d])', on_kisim)

    atilan = 0
    fragment_temizlenen = 0
    korunan = []

    for cumle in cumleler:
        c = cumle.strip()
        if not c:
            continue

        # Cumle drop kontrolu (lab-only cumle, vital yoksa at)
        lab_dom = any(rx.search(c) for rx in LAB_SENTENCE_DROP_PATTERNS)
        vital = bool(VITAL_KEEP_REGEX.search(c))

        if lab_dom and not vital:
            atilan += 1
            continue
        elif lab_dom and vital:
            # vital korunsun, fragment temizle
            yeni_c = lab_fragment_temizle(cumle)
            if yeni_c.strip():
                korunan.append(yeni_c)
                fragment_temizlenen += 1
            continue

        # Lab fragment temizligi (her cumle icin)
        yeni_c = lab_fragment_temizle(cumle)
        if yeni_c != cumle.strip():
            fragment_temizlenen += 1
        if yeni_c.strip():
            korunan.append(yeni_c)

    yeni_on = ' '.join(korunan).strip()
    if ros_kismi:
        return yeni_on + '\n\n' + ros_kismi.strip(), atilan, fragment_temizlenen
    return yeni_on, atilan, fragment_temizlenen

# --- Cleanup uygula (mevcut hpi_clean_v3 uzerinde, daha fazla pass) ---
print("\n[1/4] Residual lab cleanup uygulaniyor (hpi_clean_v3 uzerinde)...")
t0 = time.time()
yeni_hpi = []
ek_atilan = []
ek_fragment = []
for hpi in df['hpi_clean_v3']:
    yeni, atilan, fragment = cumle_temizle(hpi)
    yeni_hpi.append(yeni)
    ek_atilan.append(atilan)
    ek_fragment.append(fragment)

# Eski sayilara ekle
df['cumle_atilan_sayi'] = df['cumle_atilan_sayi'].fillna(0).astype(int) + pd.Series(ek_atilan, index=df.index)
df['fragment_temizlenen_sayi'] = df['fragment_temizlenen_sayi'].fillna(0).astype(int) + pd.Series(ek_fragment, index=df.index)
once_hpi = df['hpi_clean_v3'].copy()  # 13A.3 sonrasi
df['hpi_clean_v3'] = yeni_hpi

print(f"   Sure: {time.time()-t0:.1f}s")
print(f"   13A.4 ekstra cumle atilan      : {sum(ek_atilan):,}")
print(f"   13A.4 ekstra fragment temizlenen: {sum(ek_fragment):,}")
print(f"   Etkilenen kayit (yeni cumle)   : {sum(1 for x in ek_atilan if x>0):,}")
print(f"   Etkilenen kayit (yeni fragment): {sum(1 for x in ek_fragment if x>0):,}")

# --- V3 yeniden compose + maskele ---
print("\n[2/4] V3 yeniden compose + maskeleme...")

def compose_v3(row):
    parcalar = []
    if row['cc_v3']:        parcalar.append(f"Chief Complaint:\n{row['cc_v3']}")
    if row['hpi_clean_v3']: parcalar.append(f"Symptom History:\n{row['hpi_clean_v3']}")
    if row['pmh_v3']:       parcalar.append(f"Past Medical History:\n{row['pmh_v3']}")
    if row['allergies_v3']: parcalar.append(f"Allergies:\n{row['allergies_v3']}")
    if row['meds_adm_v3']:  parcalar.append(f"Medications on Admission:\n{row['meds_adm_v3']}")
    if row['vitals_v3']:    parcalar.append(f"Vital Signs:\n{row['vitals_v3']}")
    return '\n\n'.join(parcalar)

df['triage_input_v3'] = df.apply(compose_v3, axis=1)

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

df['triage_input_v3_masked'] = df['triage_input_v3'].apply(
    lambda m: BIRLESIK.sub('[DIAG_MASKED]', m) if m else m
)

def kendi_keyword(metin, label):
    if not isinstance(metin, str) or not metin: return False
    rx = CLASS_REGEXES.get(label)
    if rx is None: return False
    return rx.search(metin) is not None

df['has_own_label_term_v3'] = [
    kendi_keyword(t, l) for t, l in zip(df['triage_input_v3'], df['label'])
]

def yeni_flags(row):
    flags = []
    v3 = ' '.join([str(row.get(k, '') or '') for k in
                   ['cc_v3','hpi_clean_v3','pmh_v3','allergies_v3','meds_adm_v3','vitals_v3']])
    if re.search(r'\bWBC \d|\bCr \d\.\d|\bHgb \d', v3, re.IGNORECASE):
        flags.append('lab_kalinti')
    if re.search(r'\bCXR\b|\bCT (head|chest|abdomen)|\bCTA\b|\bMRI\b|\bTTE\b', v3, re.IGNORECASE):
        flags.append('imaging_kalinti')
    if re.search(r'\badmitted to\b|\bdiagnosed with\b|\bconcern for\b', v3, re.IGNORECASE):
        flags.append('assessment_kalinti')
    return ','.join(flags) if flags else 'temiz'

df['triage_audit_flags'] = df.apply(yeni_flags, axis=1)

# --- Audit ---
print("\n[3/4] FINAL Audit raporu (precleanup vs 13A.4)")
print("=" * 70)

# hpi_clean_v3 icindeki kalintilar
def hpi_kalinti_sayilari(seri):
    img_rx = re.compile(r'\bCXR\b|\bCT (head|chest|abdomen)|\bCTA\b|\bMRI\b|\bTTE\b', re.IGNORECASE)
    lab_rx = re.compile(r'\bWBC \d|\bCr \d\.\d|\bHgb \d|\blactate\s+\d|\bNa\s*\d{2,3}|\btroponin\b|\bUA\s+(negative|positive)', re.IGNORECASE)
    img = sum(1 for h in seri if isinstance(h,str) and img_rx.search(h))
    lab = sum(1 for h in seri if isinstance(h,str) and lab_rx.search(h))
    vital = sum(1 for h in seri if isinstance(h,str) and VITAL_KEEP_REGEX.search(h))
    return img, lab, vital

pre_img, pre_lab, pre_vit = hpi_kalinti_sayilari(df['hpi_clean_v3_precleanup'])
mid_img, mid_lab, mid_vit = hpi_kalinti_sayilari(once_hpi)  # 13A.3 sonu
post_img, post_lab, post_vit = hpi_kalinti_sayilari(df['hpi_clean_v3'])  # 13A.4 sonu

print(f"\n   hpi_clean_v3 icinde kalanlar:")
print(f"   {'Tip':<22s} {'precleanup':>11s} {'13A.3':>7s} {'13A.4':>7s} {'fark':>7s}")
print("   " + "-" * 60)
print(f"   {'imaging':<22s} {pre_img:>11,} {mid_img:>7,} {post_img:>7,} {post_img-pre_img:>+7,}")
print(f"   {'lab (genis)':<22s} {pre_lab:>11,} {mid_lab:>7,} {post_lab:>7,} {post_lab-pre_lab:>+7,}")
print(f"   {'vital/bedside':<22s} {pre_vit:>11,} {mid_vit:>7,} {post_vit:>7,} {post_vit-pre_vit:>+7,} ← KORUNMALI")

# Genel audit_flags
print(f"\n   Toplam V3 audit_flags (PMH dahil):")
for flag, n in df['triage_audit_flags'].value_counts().head(8).items():
    print(f"      {flag:<40s} {n:>5,} ({100*n/len(df):.1f}%)")

# Uzunluk
hpi_post = df['hpi_clean_v3'].fillna('').str.len()
v3_post = df['triage_input_v3'].fillna('').str.len()
hpi_pre = df['hpi_clean_v3_precleanup'].fillna('').str.len()
print(f"\n   Uzunluk:")
print(f"      hpi precleanup ort   : {hpi_pre.mean():>6,.0f}")
print(f"      hpi 13A.4 sonra ort  : {hpi_post.mean():>6,.0f}")
print(f"      Toplam kayip         : {(hpi_pre-hpi_post).mean():>6,.0f} char ({100*(hpi_pre-hpi_post).mean()/max(hpi_pre.mean(),1):.1f}%)")
print(f"      V3 final ort         : {v3_post.mean():>6,.0f}")
print(f"   has_own_label_term_v3 oran: {100*df['has_own_label_term_v3'].mean():.1f}%")

# --- Ornekler ---
print("\n" + "=" * 70)
print("ORNEKLER (13A.3 -> 13A.4 fark + sikayet/semptom korundu mu?)")
print("=" * 70)

siralama = ["SEPSIS","STROKE","ATRIAL_FIBRILLATION","AKI","PNEUMONIA","UTI",
            "HEART_FAILURE","GI_BLEEDING","DIABETES_COMPLICATION","COPD_EXACERBATION","OTHER"]

# Kontrol: semptom kelimeleri korundu mu (chest pain, SOB, fatigue, weakness, vs)?
SEMPTOM_RX = re.compile(r'\b(chest\s+pain|shortness\s+of\s+breath|SOB|fatigue|weakness|cough|fever|pain|nausea|vomit\w*|dizz\w*|confusion|edema|dyspnea|wheezing|bleeding|melena|hemato\w*)\b', re.IGNORECASE)

semptom_korunan = 0
semptom_kayip = 0
for once_h, sonra_h in zip(once_hpi, df['hpi_clean_v3']):
    if not isinstance(once_h, str) or not once_h: continue
    once_set = set(m.group().lower() for m in SEMPTOM_RX.finditer(once_h))
    sonra_set = set(m.group().lower() for m in SEMPTOM_RX.finditer(sonra_h or ''))
    if once_set:
        kayip = once_set - sonra_set
        if kayip:
            semptom_kayip += len(kayip)
        semptom_korunan += len(once_set & sonra_set)

print(f"\n   Semptom kelimesi koruma orani:")
print(f"      Korunan match: {semptom_korunan:,}")
print(f"      Kayip match  : {semptom_kayip:,}")
print(f"      Koruma orani : {100*semptom_korunan/max(semptom_korunan+semptom_kayip,1):.1f}%")

# Ornekler
for cls in siralama:
    alt = df[df['label']==cls]
    # 13A.3 sonu vs 13A.4 sonu farki var olan ornek
    secim = None
    for idx, row in alt.iterrows():
        once = once_hpi.loc[idx] if idx in once_hpi.index else ""
        sonra = row['hpi_clean_v3']
        if once and sonra and once != sonra:
            secim = (idx, row, once, sonra)
            break
    if secim is None:
        # Fark yoksa rastgele
        if len(alt) > 0:
            row = alt.iloc[0]
            secim = (row.name, row, once_hpi.loc[row.name] if row.name in once_hpi.index else "", row['hpi_clean_v3'])
        else:
            print(f"\n--- {cls} --- (yok)")
            continue

    idx, row, once, sonra = secim
    print(f"\n--- {cls} (note: {row['note_id']}) ---")
    print(f"   13A.3 ({len(once or '')}):")
    print(f"     {(once or '')[:500]}")
    print(f"   13A.4 ({len(sonra or '')}):")
    print(f"     {(sonra or '')[:500]}")
    once_v = bool(VITAL_KEEP_REGEX.search(once or ''))
    sonra_v = bool(VITAL_KEEP_REGEX.search(sonra or ''))
    once_s = list(SEMPTOM_RX.findall(once or ''))[:5]
    sonra_s = list(SEMPTOM_RX.findall(sonra or ''))[:5]
    print(f"   Vital  : 13A.3={once_v}, 13A.4={sonra_v}")
    print(f"   Semptom: 13A.3={once_s}")
    print(f"            13A.4={sonra_s}")

# --- Kaydet ---
print("\n" + "=" * 70)
print("KAYDETME")
print("=" * 70)

t0 = time.time()
df.to_parquet(SPLIT_YOL, compression='snappy', index=False)
boyut = SPLIT_YOL.stat().st_size / (1024*1024)
print(f"   Yazildi: {SPLIT_YOL.name}")
print(f"   Boyut  : {boyut:.1f} MB")
print(f"   Kolon  : {len(df.columns)}")

print("\n" + "=" * 70)
print("BLOK 13A.4 TAMAMLANDI")
print("=" * 70)
