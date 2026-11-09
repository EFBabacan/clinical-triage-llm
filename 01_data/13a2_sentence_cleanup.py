"""Blok 13A.2: Conservative cumle-bazli cleanup (sadece hpi_clean_v3)."""
import pandas as pd
import re
import time
from pathlib import Path

OUTPUT_DIR = Path(r"./workdir/output")
SPLIT_YOL  = OUTPUT_DIR / "discharge_split.parquet"

print("=" * 70)
print("BLOK 13A.2 — CONSERVATIVE CUMLE-BAZLI CLEANUP (sadece hpi_clean_v3)")
print("=" * 70)

df = pd.read_parquet(SPLIT_YOL)
print(f"\n   Yuklendi: {len(df):,} satir, {len(df.columns)} kolon")

# --- Yedek ---
if 'hpi_clean_v3_precleanup' not in df.columns:
    df['hpi_clean_v3_precleanup'] = df['hpi_clean_v3'].copy()
    print("   hpi_clean_v3_precleanup yedek olusturuldu.")
else:
    print("   hpi_clean_v3_precleanup yedek zaten var, dokunulmadi.")

# --- Cumle drop patterns (CONSERVATIVE) ---
# Bir cumle eger asagidaki patternlerden BIRINI iceriyorsa, o cumle atilir.
SENTENCE_DROP_PATTERNS = [
    # 1. Imaging + verb (showed/revealed/notable/with)
    re.compile(r'\b(CXR|CTA?|MRI|US|TTE|TEE|EKG|ECG)\b[^.]{0,80}\b(showed|revealed|notable|with|demonstrated|w/|consistent|positive|negative|w\.|reveal|reveals|noted|notable\sfor)\b', re.IGNORECASE),
    re.compile(r'\b(showed|revealed|notable|demonstrated|consistent|w/|positive|negative)\b[^.]{0,40}\b(CXR|CTA?|MRI|US|TTE|TEE|EKG|ECG)\b', re.IGNORECASE),
    # 2. got/had/underwent/taken to + imaging
    re.compile(r'\b(got|had|received|underwent|taken\sto|sent\sto|brought\sto)\b[^.]{0,60}\b(CXR|CTA?|MRI|US|TTE|TEE|EKG|ECG)\b', re.IGNORECASE),
    # 3. Labs notable/showed/drawn
    re.compile(r'\bLabs?\b[^.]{0,60}\b(notable|showed|drawn|remarkable|significant|reveal|positive|negative)\b', re.IGNORECASE),
    re.compile(r'\b(WBC|Cr|Hgb|platelets?|INR|lactate|troponin|BUN|Na|K\+?)\s*\d', re.IGNORECASE),
    # 4. found to have + imaging/lab finding (current admission)
    re.compile(r'\bfound\sto\sbe\b[^.]{0,40}\b(elevated|hypotensive|hypertensive|tachy|brady|febrile|septic)\b', re.IGNORECASE),
    re.compile(r'\bfound\sto\shave\b[^.]{0,80}\b(CXR|CTA?|MRI|US|TTE|infiltrate|consolidation|effusion|edema|stenosis|ICH|infarct|embolus|clot|thrombus|hemorrhage|stroke|pneumonia|sepsis)\b', re.IGNORECASE),
    # 5. UA showed
    re.compile(r'\bUA\b[^.]{0,40}\b(showed|with|notable|positive|negative)\b', re.IGNORECASE),
    # 6. CT/MRI/CXR + finding (etiket sizintisi olusturan kalip)
    re.compile(r'\b(CXR|CTA?|MRI|US|TTE|TEE)\b[^.]{0,50}\b(infiltrate|consolidation|effusion|edema|stenosis|ICH|infarct|embolus|clot|thrombus|hemorrhage|mass|nodule|opacity|cardiomegaly|pneumonia)\b', re.IGNORECASE),
    # 7. Workup + finding
    re.compile(r'\bworkup\b[^.]{0,40}\b(positive|negative|notable|showed|revealed)\b', re.IGNORECASE),
]

def cumle_temizle(metin):
    """Cumlelere bol, drop pattern eslesenleri at, kalanlari birlestir.
    Cumle ayraci: . ? ! veya \\n\\n (paragraf)."""
    if not isinstance(metin, str) or not metin:
        return metin, 0

    # ROS bolumunu koru — cumle bolme yapma
    ros_match = re.search(r'(?ims)(\n\s*(?:Review of Systems?|ROS)\s*:)', metin)
    if ros_match:
        on_kisim = metin[:ros_match.start()]
        ros_kismi = metin[ros_match.start():]
    else:
        on_kisim = metin
        ros_kismi = ""

    # Sadece on_kisim'i temizle
    # Cumlelere bol: . ? ! ile, ama "Mr. ___" gibi durumlari koru
    # Basit yaklasim: nokta+bosluk veya nokta+yenisatir
    cumleler = re.split(r'(?<=[.!?])\s+(?=[A-Z\d])', on_kisim)

    atilan_sayi = 0
    korunan = []
    for cumle in cumleler:
        c = cumle.strip()
        if not c:
            continue
        # Drop pattern check
        atildi = False
        for rx in SENTENCE_DROP_PATTERNS:
            if rx.search(c):
                atildi = True
                atilan_sayi += 1
                break
        if not atildi:
            korunan.append(cumle)

    yeni_on = ' '.join(korunan).strip()
    if ros_kismi:
        return yeni_on + '\n\n' + ros_kismi.strip(), atilan_sayi
    return yeni_on, atilan_sayi

# --- Cleanup uygula ---
print("\n[1/4] Cumle-bazli cleanup uygulaniyor (sadece hpi_clean_v3)...")
t0 = time.time()
yeni_hpi = []
toplam_atilan = []
for hpi in df['hpi_clean_v3_precleanup']:
    yeni, atilan = cumle_temizle(hpi)
    yeni_hpi.append(yeni)
    toplam_atilan.append(atilan)

df['hpi_clean_v3'] = yeni_hpi
df['cumle_atilan_sayi'] = toplam_atilan

print(f"   Sure: {time.time()-t0:.1f}s")
print(f"   Toplam cumle atilmis: {sum(toplam_atilan):,}")
print(f"   Etkilenen kayit: {sum(1 for x in toplam_atilan if x > 0):,} / {len(df):,}")

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
print("\n[3/4] Audit flag yeniden hesaplaniyor...")

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

# --- AUDIT: ONCE/SONRA karsilastirma ---
print("\n[4/4] Audit raporu (ONCE vs SONRA)")
print("=" * 70)

# Once degerlerini precleanup uzerinden hesaplayalim
def eski_flags(row):
    flags = []
    v3 = ' '.join([
        str(row.get('cc_v3','') or ''),
        str(row.get('hpi_clean_v3_precleanup','') or ''),
        str(row.get('pmh_v3','') or ''),
        str(row.get('allergies_v3','') or ''),
        str(row.get('meds_adm_v3','') or ''),
        str(row.get('vitals_v3','') or ''),
    ])
    if re.search(r'\bWBC \d|\bCr \d\.\d|\bHgb \d', v3, re.IGNORECASE):
        flags.append('lab_kalinti')
    if re.search(r'\bCXR\b|\bCT (head|chest|abdomen)|\bCTA\b|\bMRI\b|\bTTE\b', v3, re.IGNORECASE):
        flags.append('imaging_kalinti')
    if re.search(r'\badmitted to\b|\bdiagnosed with\b|\bconcern for\b', v3, re.IGNORECASE):
        flags.append('assessment_kalinti')
    return ','.join(flags) if flags else 'temiz'

df['_eski_flags'] = df.apply(eski_flags, axis=1)

# Karsilastirma - kalinti sayilari
print("\n   Kalinti karsilastirmasi (ONCE -> SONRA):")
print(f"   {'Tip':<22s} {'ONCE':>7s} {'SONRA':>7s} {'DUSTU':>7s}")
print("   " + "-" * 50)
for tip in ['lab_kalinti', 'imaging_kalinti', 'assessment_kalinti']:
    once = df['_eski_flags'].str.contains(tip, na=False).sum()
    sonra = df['triage_audit_flags'].str.contains(tip, na=False).sum()
    print(f"   {tip:<22s} {once:>7,} {sonra:>7,} {once-sonra:>+7,}")

# Tamamen temiz orani
once_temiz = (df['_eski_flags'] == 'temiz').sum()
sonra_temiz = (df['triage_audit_flags'] == 'temiz').sum()
print(f"   {'tamamen temiz':<22s} {once_temiz:>7,} {sonra_temiz:>7,} {sonra_temiz-once_temiz:>+7,}")

# Uzunluk degisimi
once_uzun = df['hpi_clean_v3_precleanup'].fillna('').str.len()
sonra_uzun = df['hpi_clean_v3'].fillna('').str.len()
v3_once = df.apply(lambda r: len(r['triage_input_v3'] or '') + (len(str(r['hpi_clean_v3_precleanup'] or '')) - len(str(r['hpi_clean_v3'] or ''))), axis=1)
# Daha temiz yaklasim:
v3_sonra_uzun = df['triage_input_v3'].str.len()
hpi_kayip = once_uzun - sonra_uzun

print(f"\n   hpi_clean_v3 uzunluk:")
print(f"      ONCE  ortalama: {once_uzun.mean():>7,.0f} char")
print(f"      SONRA ortalama: {sonra_uzun.mean():>7,.0f} char")
print(f"      Kayip ortalama: {hpi_kayip.mean():>7,.0f} char ({100*hpi_kayip.mean()/max(once_uzun.mean(),1):.1f}%)")
print(f"      Etkilenen kayit (cumle atilanmis): {(df['cumle_atilan_sayi']>0).sum():,}")

print(f"\n   triage_input_v3 (final) ortalama uzunluk: {v3_sonra_uzun.mean():,.0f} char")

# has_own_label_term_v3 degisimi
print(f"\n   has_own_label_term_v3 oran: {100*df['has_own_label_term_v3'].mean():.1f}%")

# Sınıf bazlı flag karşılaştırması
print("\n   Sinif bazli imaging_kalinti (ONCE -> SONRA):")
print(f"   {'Sinif':<25s} {'once':>5s} {'sonra':>5s}")
print("   " + "-" * 45)
for cls in sorted(df['label'].unique()):
    alt = df[df['label']==cls]
    o = alt['_eski_flags'].str.contains('imaging_kalinti', na=False).sum()
    s = alt['triage_audit_flags'].str.contains('imaging_kalinti', na=False).sum()
    print(f"   {cls:<25s} {o:>5,} {s:>5,}")

# Geçici kolonu sil
df = df.drop(columns=['_eski_flags'])

# --- 11 sınıftan 1 örnek (cumle atılmış olanlardan) ---
print("\n" + "=" * 70)
print("ORNEKLER (cumle atilanlar - hpi_clean_v3 ONCE/SONRA)")
print("=" * 70)

siralama = ["SEPSIS","STROKE","ATRIAL_FIBRILLATION","AKI","PNEUMONIA","UTI",
            "HEART_FAILURE","GI_BLEEDING","DIABETES_COMPLICATION","COPD_EXACERBATION","OTHER"]

for cls in siralama:
    alt = df[(df['label']==cls) & (df['cumle_atilan_sayi']>0)]
    if len(alt) == 0:
        # Cumle atilmamis ornek goster (kalinti var miydi?)
        print(f"\n--- {cls} --- (cumle atilan ornek yok)")
        continue

    ornek = alt.iloc[0]
    print(f"\n--- {cls} (note_id: {ornek['note_id']}, atilan: {ornek['cumle_atilan_sayi']} cumle) ---")
    once = ornek['hpi_clean_v3_precleanup'] or ""
    sonra = ornek['hpi_clean_v3'] or ""
    print(f"   ONCE  ({len(once)} char): ...{once[:600]}...")
    print(f"   SONRA ({len(sonra)} char): ...{sonra[:600]}...")

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
print(f"\n   YEDEK: hpi_clean_v3_precleanup kolonu korundu (rollback icin)")

print("\n" + "=" * 70)
print("BLOK 13A.2 TAMAMLANDI")
print("=" * 70)
