"""Blok 13A.3: Vital/basic olcum koruyan cleanup. precleanup'tan yeniden uret."""
import pandas as pd
import re
import time
from pathlib import Path

OUTPUT_DIR = Path(r"./workdir/output")
SPLIT_YOL  = OUTPUT_DIR / "discharge_split.parquet"

print("=" * 70)
print("BLOK 13A.3 — VITAL KORUYAN CLEANUP (precleanup'tan yeniden uret)")
print("=" * 70)

df = pd.read_parquet(SPLIT_YOL)
print(f"\n   Yuklendi: {len(df):,} satir, {len(df.columns)} kolon")

assert 'hpi_clean_v3_precleanup' in df.columns, "Yedek yok!"

# --- Vital/bedside whitelist patterns (BU CUMLELER ATILMAZ) ---
# Bunlar hastadan ya da bedside olcumden alinabilen bilgiler.
VITAL_KEEP_PATTERNS = [
    r'\bhypotens\w*\b', r'\bhypertens\w*\b',
    r'\btachycardi\w*\b', r'\bbradycardi\w*\b',
    r'\btachypne\w*\b', r'\bbradypne\w*\b',
    r'\bfebrile\b', r'\bafebrile\b',
    r'\bhypoxic?\b', r'\bhypoxia\b',
    r'\bhypoglycem\w*\b', r'\bhyperglycem\w*\b',
    r'\bdesaturat\w*\b', r'\bdesat\b',
    r'\bSpO2\b', r'\bSaO2\b', r'\bO2\s*sat\w*\b',
    r'\bglucose\b', r'\bblood\s*sugar\b',
    r'\b(BP|SBP|DBP)\s*\d',  # tansiyon olcumu (BP 120/80)
    r'\bHR\s*\d', r'\bRR\s*\d',  # nabiz, solunum sayisi
    r'\bT\s*\d{2,3}\.\d',  # ates: T 99.5
    r'\btemp(erature)?\s*\d',
    r'\bpulse\s*\d',
]
VITAL_KEEP_REGEX = re.compile('|'.join(f'(?:{p})' for p in VITAL_KEEP_PATTERNS), re.IGNORECASE)

# --- Atilacak patterns (LAB + IMAGING + TANI SIZINTISI) ---
# Bir cumle eger bunlardan birini iceriyorsa atilir, AMA vital varsa korunur.
DROP_PATTERNS = [
    # Imaging finding
    re.compile(r'\b(CXR|CTA?|MRI|US|TTE|TEE|EKG|ECG|echocardiogram)\b[^.]{0,80}\b(showed|revealed|notable|with|demonstrated|w/|consistent|positive|negative|reveal|reveals|noted|found|w\.|c/w)\b', re.IGNORECASE),
    re.compile(r'\b(showed|revealed|notable|demonstrated|consistent|positive|negative)\b[^.]{0,40}\b(CXR|CTA?|MRI|US|TTE|TEE|EKG|ECG|echocardiogram)\b', re.IGNORECASE),
    re.compile(r'\b(got|had|received|underwent|taken\sto|sent\sto|brought\sto)\b[^.]{0,60}\b(CXR|CTA?|MRI|US|TTE|TEE|EKG|ECG|echocardiogram)\b', re.IGNORECASE),
    # Imaging + finding direkt
    re.compile(r'\b(CXR|CTA?|MRI|US|TTE|TEE)\b[^.]{0,60}\b(infiltrate|consolidation|effusion|edema|stenosis|ICH|infarct|embolus|clot|thrombus|hemorrhage|mass|nodule|opacity|cardiomegaly|pneumonia|stroke)\b', re.IGNORECASE),

    # Lab paneli (sayisal degerli)
    re.compile(r'\bLabs?\b[^.]{0,80}\b(notable|showed|drawn|remarkable|significant|reveal|positive|negative|with)\b', re.IGNORECASE),
    re.compile(r'\b(WBC|Hgb|HCT|platelets?|PLT|INR|lactate|troponin|BUN|creatinine)\s*[<>=]?\s*\d', re.IGNORECASE),
    re.compile(r'\bCr\s*\d\.\d', re.IGNORECASE),  # Creatinine deger
    re.compile(r'\bUA\b[^.]{0,40}\b(showed|with|notable|positive|negative)\b', re.IGNORECASE),

    # Tani sizintisi (bulgu + tani)
    re.compile(r'\bfound\sto\shave\b[^.]{0,80}\b(CXR|CTA?|MRI|US|TTE|infiltrate|consolidation|effusion|edema|stenosis|ICH|infarct|embolus|clot|thrombus|hemorrhage|stroke|pneumonia|sepsis|stenosis)\b', re.IGNORECASE),
    re.compile(r'\bworkup\b[^.]{0,40}\b(positive|negative|notable|showed|revealed)\b', re.IGNORECASE),
]

# --- Lab fragmenti silme regex'leri (tum cumle silinmeden, parca silme) ---
# Eger bir cumlede vital VE lab var ise, sadece lab parcasini cikar.
LAB_FRAGMENT_PATTERNS = [
    # ", WBC 14, Hgb 11, Cr 1.5" gibi virgul/colon ile ayrilmis lab listeleri
    re.compile(r',\s*(?:WBC|Hgb|HCT|platelets?|PLT|INR|lactate|troponin|BUN|Cr)\s*[<>=]?\s*\d[\d.]*\s*(?=,|\.|$)', re.IGNORECASE),
    re.compile(r'\b(?:with|w/)\s*(?:WBC|Hgb|HCT|platelets?|PLT|INR|lactate|troponin|BUN|Cr)\s*[<>=]?\s*\d[\d.]*\b', re.IGNORECASE),
    # "with normal lactate" gibi
    re.compile(r',\s*(?:with\s+)?(?:normal|elevated|low|high)\s+(?:WBC|Hgb|HCT|platelets?|PLT|INR|lactate|troponin|BUN|creatinine|Cr)\b', re.IGNORECASE),
]

def lab_fragment_temizle(cumle):
    """Cumle icindeki lab fragmentlerini sil, vital korunsun."""
    yeni = cumle
    for rx in LAB_FRAGMENT_PATTERNS:
        yeni = rx.sub('', yeni)
    # Cift bosluk/virgul temizligi
    yeni = re.sub(r'\s+,', ',', yeni)
    yeni = re.sub(r',\s*,', ',', yeni)
    yeni = re.sub(r'\s+\.', '.', yeni)
    yeni = re.sub(r'\s{2,}', ' ', yeni)
    return yeni.strip()

def cumle_temizle(metin):
    """Cumlelere bol. Her cumle icin:
    1) Drop pattern eslesiyor mu?
       - Eslesirse VE vital_keep_regex de eslesirse: lab_fragment_temizle ile parca sil, cumleyi koru
       - Eslesirse VE vital yoksa: cumleyi at
    2) Drop pattern yoksa: koru (ama lab fragmenti varsa onu temizle)
    """
    if not isinstance(metin, str) or not metin:
        return metin, 0, 0

    # ROS'u koru
    ros_match = re.search(r'(?ims)(\n\s*(?:Review of Systems?|ROS)\s*:)', metin)
    if ros_match:
        on_kisim = metin[:ros_match.start()]
        ros_kismi = metin[ros_match.start():]
    else:
        on_kisim = metin
        ros_kismi = ""

    cumleler = re.split(r'(?<=[.!?])\s+(?=[A-Z\d])', on_kisim)

    atilan = 0
    fragmenti_temizlenen = 0
    korunan = []
    for cumle in cumleler:
        c = cumle.strip()
        if not c:
            continue

        drop_match = any(rx.search(c) for rx in DROP_PATTERNS)
        vital_match = VITAL_KEEP_REGEX.search(c)

        if drop_match and vital_match:
            # Vital korunsun, sadece lab/imaging fragmentlerini sil
            yeni_c = lab_fragment_temizle(cumle)
            if yeni_c.strip():
                korunan.append(yeni_c)
                fragmenti_temizlenen += 1
        elif drop_match and not vital_match:
            # Tum cumleyi at
            atilan += 1
        else:
            # Cumlede drop yok - yine de fragment temizligi yap (lab listesi olabilir)
            yeni_c = lab_fragment_temizle(cumle)
            korunan.append(yeni_c if yeni_c else cumle)
            if yeni_c != cumle.strip():
                fragmenti_temizlenen += 1

    yeni_on = ' '.join(korunan).strip()
    if ros_kismi:
        return yeni_on + '\n\n' + ros_kismi.strip(), atilan, fragmenti_temizlenen
    return yeni_on, atilan, fragmenti_temizlenen

# --- precleanup'tan yeniden uret ---
print("\n[1/4] precleanup'tan yeniden cleanup uygulaniyor...")
t0 = time.time()
yeni_hpi = []
toplam_atilan = []
toplam_fragment = []
for hpi in df['hpi_clean_v3_precleanup']:
    yeni, atilan, fragment = cumle_temizle(hpi)
    yeni_hpi.append(yeni)
    toplam_atilan.append(atilan)
    toplam_fragment.append(fragment)

df['hpi_clean_v3'] = yeni_hpi
df['cumle_atilan_sayi'] = toplam_atilan
df['fragment_temizlenen_sayi'] = toplam_fragment

print(f"   Sure: {time.time()-t0:.1f}s")
print(f"   Cumle atilan toplam        : {sum(toplam_atilan):,}")
print(f"   Fragment temizlenen toplam : {sum(toplam_fragment):,}")
print(f"   Etkilenen kayit (cumle)    : {sum(1 for x in toplam_atilan if x > 0):,}")
print(f"   Etkilenen kayit (fragment) : {sum(1 for x in toplam_fragment if x > 0):,}")

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

# --- audit_flags yeniden hesap ---
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

# Hpi-bazli kalinti (hpi_clean_v3'te kalan)
def hpi_kalintilari(hpi):
    if not isinstance(hpi, str): return {}
    out = {}
    out['imaging'] = bool(re.search(r'\bCXR\b|\bCT (head|chest|abdomen)|\bCTA\b|\bMRI\b|\bTTE\b', hpi, re.IGNORECASE))
    out['lab']     = bool(re.search(r'\bWBC \d|\bCr \d\.\d|\bHgb \d', hpi, re.IGNORECASE))
    out['vital']   = bool(VITAL_KEEP_REGEX.search(hpi))
    return out

# --- Audit ---
print("\n[3/4] Audit raporu — 13A.2 vs 13A.3 + vital koruma kontrolu")
print("=" * 70)

# 13A.2 sonrasi degerleri precleanup uzerinden yeniden simule edemiyoruz
# ama mevcut/yeni karsilastirmasi icin precleanup'a uygulanan ESKI 13A.2 patterni
# bilmiyoruz. Yerine: precleanup vs SONRA karsilastirma yapiyoruz.

# Precleanup uzerinden hpi-icindeki kalinti
once_imaging = sum(1 for h in df['hpi_clean_v3_precleanup'] if isinstance(h,str) and re.search(r'\bCXR\b|\bCT (head|chest|abdomen)|\bCTA\b|\bMRI\b|\bTTE\b', h, re.IGNORECASE))
sonra_imaging = sum(1 for h in df['hpi_clean_v3'] if isinstance(h,str) and re.search(r'\bCXR\b|\bCT (head|chest|abdomen)|\bCTA\b|\bMRI\b|\bTTE\b', h, re.IGNORECASE))
once_lab = sum(1 for h in df['hpi_clean_v3_precleanup'] if isinstance(h,str) and re.search(r'\bWBC \d|\bCr \d\.\d|\bHgb \d', h, re.IGNORECASE))
sonra_lab = sum(1 for h in df['hpi_clean_v3'] if isinstance(h,str) and re.search(r'\bWBC \d|\bCr \d\.\d|\bHgb \d', h, re.IGNORECASE))
once_vital = sum(1 for h in df['hpi_clean_v3_precleanup'] if isinstance(h,str) and VITAL_KEEP_REGEX.search(h))
sonra_vital = sum(1 for h in df['hpi_clean_v3'] if isinstance(h,str) and VITAL_KEEP_REGEX.search(h))

print(f"\n   hpi_clean_v3 icindeki kalintilar (PRECLEANUP -> 13A.3 SONRASI):")
print(f"      imaging          : {once_imaging:>5,} -> {sonra_imaging:>5,}  ({once_imaging-sonra_imaging:+,})")
print(f"      lab (sayili)     : {once_lab:>5,} -> {sonra_lab:>5,}  ({once_lab-sonra_lab:+,})")
print(f"      vital/bedside    : {once_vital:>5,} -> {sonra_vital:>5,}  ({once_vital-sonra_vital:+,}) ← KORUNMALI")

# Sinifa gore audit_flags
print(f"\n   Sinif bazli imaging_kalinti (V3 toplam — pmh dahil):")
print(f"   {'Sinif':<25s} {'imaging':>8s} {'lab':>5s} {'temiz':>6s}")
print("   " + "-" * 50)
for cls in sorted(df['label'].unique()):
    alt = df[df['label']==cls]
    img = alt['triage_audit_flags'].str.contains('imaging_kalinti', na=False).sum()
    lab = alt['triage_audit_flags'].str.contains('lab_kalinti', na=False).sum()
    tem = (alt['triage_audit_flags']=='temiz').sum()
    print(f"   {cls:<25s} {img:>8,} {lab:>5,} {tem:>6,}")

# Genel ozet
print(f"\n   triage_audit_flags genel:")
for flag, n in df['triage_audit_flags'].value_counts().head(8).items():
    print(f"      {flag:<40s} {n:>5,} ({100*n/len(df):.1f}%)")

# Uzunluk
hpi_pre = df['hpi_clean_v3_precleanup'].fillna('').str.len()
hpi_post = df['hpi_clean_v3'].fillna('').str.len()
v3_post = df['triage_input_v3'].fillna('').str.len()
print(f"\n   Uzunluk:")
print(f"      hpi precleanup ort : {hpi_pre.mean():>6,.0f}")
print(f"      hpi sonra ort      : {hpi_post.mean():>6,.0f}")
print(f"      hpi kayip ort      : {(hpi_pre-hpi_post).mean():>6,.0f} char ({100*(hpi_pre-hpi_post).mean()/max(hpi_pre.mean(),1):.1f}%)")
print(f"      V3 final ort       : {v3_post.mean():>6,.0f}")

print(f"\n   has_own_label_term_v3 oran: {100*df['has_own_label_term_v3'].mean():.1f}%")

# --- 11 sınıf örneği: cumle atilanlar + vital iceren ---
print("\n" + "=" * 70)
print("ORNEKLER (cumle atilan + vital iceren - hpi ONCE/SONRA karsilastirma)")
print("=" * 70)

siralama = ["SEPSIS","STROKE","ATRIAL_FIBRILLATION","AKI","PNEUMONIA","UTI",
            "HEART_FAILURE","GI_BLEEDING","DIABETES_COMPLICATION","COPD_EXACERBATION","OTHER"]

for cls in siralama:
    # Vital iceren ve cumle atilan ornekleri tercih et
    alt = df[(df['label']==cls) & (df['cumle_atilan_sayi']>0)]
    # Once vital iceren bul
    secim = None
    for _, row in alt.iterrows():
        once_h = row['hpi_clean_v3_precleanup'] or ""
        if VITAL_KEEP_REGEX.search(once_h):
            secim = row
            break
    if secim is None and len(alt) > 0:
        secim = alt.iloc[0]
    if secim is None:
        print(f"\n--- {cls} --- (atilan cumle yok)")
        continue

    print(f"\n--- {cls} (note: {secim['note_id']}, atilan: {secim['cumle_atilan_sayi']}, fragment: {secim['fragment_temizlenen_sayi']}) ---")
    once = secim['hpi_clean_v3_precleanup'] or ""
    sonra = secim['hpi_clean_v3'] or ""
    print(f"   ONCE  ({len(once)} char):")
    print(f"     {once[:600]}")
    print(f"   SONRA ({len(sonra)} char):")
    print(f"     {sonra[:600]}")
    # Vital korundu mu?
    once_v = bool(VITAL_KEEP_REGEX.search(once))
    sonra_v = bool(VITAL_KEEP_REGEX.search(sonra))
    print(f"   Vital korundu: ONCE={once_v}, SONRA={sonra_v}")

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
print(f"   Sure   : {time.time()-t0:.1f}s")

print("\n" + "=" * 70)
print("BLOK 13A.3 TAMAMLANDI")
print("=" * 70)
