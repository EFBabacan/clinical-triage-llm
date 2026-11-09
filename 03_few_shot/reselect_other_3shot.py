"""Blok 14V4-RESELECT-OTHER-AND-3SHOT-CLEAN.
1. OTHER icin smart selection (target-class benzeri ornekleri disla)
2. 3-shot/class compact clean (medoid + MMR diversity)
3. Compact audit: forbidden phrase + leakage severity
Inference YOK."""
import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer

OUTPUT_DIR = Path(r"./workdir/output")
SPLIT_YOL  = OUTPUT_DIR / "discharge_split.parquet"
FS_DIR     = OUTPUT_DIR / "fewshot"

print("=" * 70)
print("BLOK 14V4-RESELECT-OTHER-AND-3SHOT-CLEAN")
print("=" * 70)

ALLOWED_CLASSES = ["SEPSIS","STROKE","ATRIAL_FIBRILLATION","AKI","PNEUMONIA","UTI",
    "HEART_FAILURE","GI_BLEEDING","DIABETES_COMPLICATION","COPD_EXACERBATION","OTHER"]

# ============================================================
# OTHER icin DISLAMA pattern listesi (target sinif benzeri)
# ============================================================
OTHER_EXCLUDE_PATTERNS = re.compile(
    # Respiratory (PNEUMONIA, COPD, HF benzeri)
    r"\b(dyspnea|SOB|shortness\s+of\s+breath|wheezing|wheez\w*|hypox\w*|"
    r"productive\s+cough|cough\s+productive|pneumonia|"
    # GI bleeding
    r"melena|hematochezia|BRBPR|hematemesis|GI\s+bleed|"
    # UTI
    r"dysuria|urinary\s+(tract|frequency|urgency)|UTI|pyelonephritis|cystitis|"
    # Stroke-like
    r"facial\s+droop|aphasia|hemiparesis|hemiplegi|slurred\s+speech|"
    r"acute\s+onset\s+(weakness|numbness)|"
    # AFib
    r"palpitation|atrial\s+fib|AFib|A-fib|atrial\s+flutter|"
    # Diabetes
    r"hyperglyc\w*|hypoglyc\w*|DKA|HHS|hyperosmolar|"
    # AKI
    r"elevated\s+(Cr|creatinine)|acute\s+kidney|AKI|"
    # Sepsis / infection
    r"sepsis|septic|bacteremia|"
    # CHF
    r"CHF|congestive\s+heart\s+failure|HFrEF|HFpEF|volume\s+overload|"
    r"orthopnea|PND|paroxysmal\s+nocturnal)\b",
    re.IGNORECASE
)

# OTHER icin TERCIH pattern (gercek non-target)
OTHER_PREFER_PATTERNS = re.compile(
    r"\b(fall|trauma|fracture|MVA|motor\s+vehicle|"
    r"laceration|burn|abrasion|"
    r"intoxication|alcohol\s+(use|intoxication)|"
    r"psychiatric|depression|anxiety|suicidal|"
    r"malignancy|cancer|metasta|chemotherapy|"
    r"abdominal\s+pain|RUQ|LUQ|pelvic|"
    r"post-?op|procedure|elective|"
    r"electrolyte\s+disturbance|hyponatremia(?!.*HF)|hypernatremia|"
    r"vertigo(?!.*stroke)|syncope(?!.*HF))\b",
    re.IGNORECASE
)

# ============================================================
# Forbidden compact phrases (14V4-CLEAN-FEWSHOT ile ayni)
# ============================================================
FORBIDDEN_PATTERNS = [
    ("code_stroke", re.compile(r"\bcode\s+(stroke|MI|sepsis|trauma)\b", re.IGNORECASE)),
    ("ct_scanning", re.compile(r"\b(CT scanning|reviewed images|read images|reviewed CT|reviewed scan)\b", re.IGNORECASE)),
    ("micu_admission", re.compile(r"\b(REASON FOR (MICU|ICU) ADMISSION|MICU admission|ICU admission)\b", re.IGNORECASE)),
    ("scheduled_proc", re.compile(r"\bscheduled\s+(TEE|DCCV|EGD|colonoscopy|cardioversion|TTE)\b", re.IGNORECASE)),
    ("elevated_lab", re.compile(r"\b(elevated\s+(Cr|creatinine|WBC|Hgb|HCT|lactate|troponin|BUN|sodium|Na)|abnormal\s+labs?)\b", re.IGNORECASE)),
    ("found_in_ed", re.compile(r"\b(found\s+in\s+ED|found\s+to\s+have|found\s+down)\b", re.IGNORECASE)),
    ("given_treatment", re.compile(r"\b(given|received|started\s+on|placed\s+on|treated\s+with)\s+[^.]{0,40}\b(oxygen|CPAP|BiPAP|high\s*flow|nebulizer|nebs|albuterol|duonebs|antibiotics|abx|vancomycin|zosyn|lasix|insulin\s+drip|glucagon|tPA|pressors|levophed|norepinephrine|vasopressin|crystalloid|IV\s+fluids|fluid\s+bolus)\b", re.IGNORECASE)),
    ("admitted_for", re.compile(r"\b(admitted\s+(for|to)|MICU\s+admission|ED\s+course)\b", re.IGNORECASE)),
    ("imaging_showed", re.compile(r"\b(CXR|CTA?|MRI|TTE|TEE|US|echocardiogram)\b[^.]{0,40}\b(showed|revealed|demonstrated|consistent\s+with|notable|w/|with\s+(infiltrate|consolidation|effusion|edema|ICH|infarct|hemorrhage))\b", re.IGNORECASE)),
    ("labs_showed", re.compile(r"\bLabs?\b[^.]{0,40}\b(showed|notable|revealed|drawn|remarkable)\b", re.IGNORECASE)),
    ("presented_workup", re.compile(r"\bpresent(ed|s|ing)\s+to\s+(the\s+)?ED\s+(for|with)\s+(workup|further\s+evaluation|management|admission|treatment)\b", re.IGNORECASE)),
]

def find_forbidden(text):
    if not isinstance(text, str) or not text: return []
    bulgular = []
    for ad, rx in FORBIDDEN_PATTERNS:
        for m in rx.finditer(text):
            bulgular.append((ad, m.group()))
    return bulgular

# ============================================================
# Section-aware compaction
# ============================================================
SECTION_RE = re.compile(
    r"(Chief Complaint|Symptom History|Past Medical History|Allergies|"
    r"Medications on Admission|Vital Signs):\s*\n?", re.IGNORECASE)
SYMPTOM_KEYWORDS = re.compile(
    r"\b(presents?|reports?|complains?|denies|developed|noted|noticed|started|began|"
    r"pain|cough|fever|dyspnea|SOB|fatigue|weakness|nausea|vomit|diarrhea|bleeding|"
    r"dizz|confusion|swelling|edema|chest|abdomen|headache|hypoxic|hypotens|tachycard|"
    r"palpitations|orthopnea|wheez|melena|hemato|dysuria|frequen|urgency)\b", re.IGNORECASE)

def parse_sections(text):
    if not text: return {}
    parts = SECTION_RE.split(text)
    s = {}
    for i in range(1, len(parts) - 1, 2):
        s[parts[i].strip()] = parts[i+1].strip()
    return s

def cumlelere_bol(t):
    if not t: return []
    return [c.strip() for c in re.split(r"(?<=[.!?])\s+(?=[A-Z\d])", t) if c.strip()]

def hasta_skor(c):
    return len(SYMPTOM_KEYWORDS.findall(c))

def compact_for_3shot(full_text, max_token_target=130, tok=None):
    """3-shot icin daha agresif compact: 100-130 token hedef."""
    sections = parse_sections(full_text)
    parts = []
    cc = sections.get("Chief Complaint", "").strip()
    if cc: parts.append(f"Chief Complaint: {cc}")
    sh = sections.get("Symptom History", "").strip()
    if sh:
        ros = re.search(r"(?im)^\s*(Review of Systems?|ROS)\s*:", sh)
        if ros: sh = sh[:ros.start()]
        cumleler = cumlelere_bol(sh)
        skorli = [(i, c, hasta_skor(c)) for i, c in enumerate(cumleler[:5])]
        secilen = [c for i, c, s in skorli if s > 0][:2]  # 3-shot icin max 2 cumle
        if not secilen:
            secilen = [c for i, c, s in skorli][:1]
        if secilen: parts.append(f"Key Symptoms: {' '.join(secilen)}")
    vs = sections.get("Vital Signs", "").strip()
    if vs:
        # Vitals'i kisalt: 100 char max
        parts.append(f"Vitals: {vs[:120]}")
    pmh = sections.get("Past Medical History", "").strip()
    if pmh:
        pmh_satirlar = [s.strip() for s in re.split(r"[\n;]+", pmh) if s.strip()]
        kisa = [s for s in pmh_satirlar if 5 < len(s) < 80][:2]  # 3-shot icin 2 PMH
        if not kisa:
            kisa = [c for c in cumlelere_bol(pmh)[:1] if len(c) < 150]
        if kisa: parts.append(f"Relevant History: " + "; ".join(kisa))
    meds = sections.get("Medications on Admission", "").strip()
    if meds:
        ilac_satirlar = re.findall(r"\d+\.\s*([A-Z][A-Za-z][^.\n]{2,60})", meds)
        if ilac_satirlar:
            kisa_ilaclar = []
            for s in ilac_satirlar[:3]:  # 3-shot icin max 3 ilac
                ilac = s.split()[:2]
                kisa_ilaclar.append(" ".join(ilac))
            parts.append(f"Medications: " + ", ".join(kisa_ilaclar))
    return "\n".join(parts)

# ============================================================
# Veri ve pool
# ============================================================
print("\n[1/6] Veri yukleme ve pool filtre...")
df = pd.read_parquet(SPLIT_YOL)
df["note_id"] = df["note_id"].astype(str)
pool_full = df[df["split"]=="train_S3"].reset_index(drop=True)

LEAKAGE_KOLON = "strict_current_leakage_severity_final"
if LEAKAGE_KOLON not in pool_full.columns:
    LEAKAGE_KOLON = "strict_current_leakage_severity"
clean_mask = pool_full[LEAKAGE_KOLON].isin(["CLEAN","HISTORICAL_ALLOWED"])
pool = pool_full[clean_mask].reset_index(drop=True)
print(f"   train_S3 pool (V4 clean): {len(pool):,}")

vec = TfidfVectorizer(max_features=20000, ngram_range=(1,2), min_df=2,
                      stop_words="english", sublinear_tf=True)
X = vec.fit_transform(pool["triage_input_v4_timeaware_masked"].fillna("").tolist())
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", local_files_only=True)

# ============================================================
# OTHER icin SMART selection
# ============================================================
print("\n[2/6] OTHER icin smart re-selection...")
print("=" * 70)

other_idx = np.where(pool["label"].values == "OTHER")[0]
print(f"   OTHER pool: {len(other_idx)} ornek")

# OTHER adaylarini skorla
other_kayitlar = []
for lokal_idx in other_idx:
    row = pool.iloc[lokal_idx]
    text = row["triage_input_v4_timeaware_masked"] or ""
    excl_count = len(OTHER_EXCLUDE_PATTERNS.findall(text))
    pref_count = len(OTHER_PREFER_PATTERNS.findall(text))
    other_kayitlar.append({
        "lokal_idx": int(lokal_idx),
        "note_id": str(row["note_id"]),
        "leakage": str(row[LEAKAGE_KOLON]),
        "excl": excl_count,
        "pref": pref_count,
        "skor": pref_count - excl_count,
        "text": text,
    })

# Filtreleme: excl=0 olanlar, pref>=1 olanlar tercih
other_uygun = [o for o in other_kayitlar if o["excl"] == 0]
print(f"   Excl=0 (target benzeri yok): {len(other_uygun)}")
other_yuksek_pref = [o for o in other_uygun if o["pref"] >= 1]
print(f"   Pref>=1 (gercek non-target): {len(other_yuksek_pref)}")

if len(other_yuksek_pref) >= 3:
    aday_havuz_idx = [o["lokal_idx"] for o in other_yuksek_pref]
    print(f"   -> Pref>=1 havuzu kullanilacak ({len(aday_havuz_idx)})")
elif len(other_uygun) >= 3:
    aday_havuz_idx = [o["lokal_idx"] for o in other_uygun]
    print(f"   -> Excl=0 havuzu kullanilacak ({len(aday_havuz_idx)})")
else:
    # Geri donus: en dusuk excl skoru olanlardan al
    other_kayitlar.sort(key=lambda x: x["excl"])
    aday_havuz_idx = [o["lokal_idx"] for o in other_kayitlar[:50]]
    print(f"   -> En dusuk excl havuzu kullanilacak ({len(aday_havuz_idx)})")

print(f"\n   ORNEK reddedilen OTHER (ilk 5):")
for o in other_kayitlar[:5]:
    if o["excl"] > 0:
        # Hangi pattern eslesti
        eslesen = OTHER_EXCLUDE_PATTERNS.search(o["text"])
        m_str = eslesen.group() if eslesen else "?"
        print(f"      {o['note_id']}: excl={o['excl']} pref={o['pref']} (eslesen: {m_str!r})")

print(f"\n   ORNEK secilen OTHER aday (ilk 5):")
for o in other_yuksek_pref[:5]:
    eslesen = OTHER_PREFER_PATTERNS.search(o["text"])
    m_str = eslesen.group() if eslesen else "?"
    print(f"      {o['note_id']}: pref={o['pref']} (eslesen: {m_str!r})")

# ============================================================
# 3-shot/class secim (medoid + MMR)
# ============================================================
print("\n[3/6] 3-shot/class secim (medoid + MMR diversity)...")
print("=" * 70)

ALPHA = 0.7
BETA = 0.3
MAX_RETRIES = 20  # forbidden cikarsa kac aday daha dene

shot3_kayitlar = []

for cls in ALLOWED_CLASSES:
    if cls == "OTHER":
        # Smart aday havuzu
        cls_idx_arr = np.array(aday_havuz_idx)
    else:
        cls_idx_arr = np.where(pool["label"].values == cls)[0]

    if len(cls_idx_arr) < 3:
        print(f"   UYARI: {cls} icin {len(cls_idx_arr)} aday var")
        continue

    X_cls = X[cls_idx_arr]
    centroid = np.asarray(X_cls.mean(axis=0)).reshape(1,-1)
    sim_cent = cosine_similarity(X_cls, centroid).flatten()

    # 3 ornek sec, MMR + forbidden filter
    secilen_lokal_arr = []  # cls_idx_arr icindeki indexler
    rank_taranan = 0

    # Aday sirasi: rank by score (mmr style)
    for hedef_sira in range(3):
        if hedef_sira == 0:
            skor = sim_cent.copy()
        else:
            secilen_X = X_cls[secilen_lokal_arr]
            sim_sel = cosine_similarity(X_cls, secilen_X)
            max_sim_sel = sim_sel.max(axis=1)
            skor = ALPHA * sim_cent - BETA * max_sim_sel
            for s in secilen_lokal_arr: skor[s] = -np.inf

        # Sirala
        rank = np.argsort(-skor)
        bulundu = False
        for r_local, lokal in enumerate(rank):
            if int(lokal) in secilen_lokal_arr: continue
            global_idx = cls_idx_arr[int(lokal)]
            row = pool.iloc[global_idx]
            full_text = row["triage_input_v4_timeaware_masked"] or ""
            compact_aday = compact_for_3shot(full_text, tok=tok)
            bulg = find_forbidden(compact_aday)
            if not bulg and len(compact_aday) > 50:
                secilen_lokal_arr.append(int(lokal))
                bulundu = True
                rank_taranan = r_local
                break
            else:
                rank_taranan = r_local
                if r_local >= MAX_RETRIES: break

        if not bulundu:
            print(f"   UYARI: {cls} sira {hedef_sira+1} icin temiz aday bulunamadi (taranan: {rank_taranan+1})")

    # Kaydet
    for sira, lokal in enumerate(secilen_lokal_arr, 1):
        global_idx = cls_idx_arr[lokal]
        row = pool.iloc[global_idx]
        full_text = row["triage_input_v4_timeaware_masked"] or ""
        compact_text = compact_for_3shot(full_text, tok=tok)

        if sira == 1:
            max_sim_prev = 0.0
        else:
            secilen_X = X_cls[secilen_lokal_arr[:sira-1]]
            max_sim_prev = float(cosine_similarity(X_cls[lokal:lokal+1], secilen_X).max())

        n_token = len(tok.encode(compact_text, add_special_tokens=False))
        bulg = find_forbidden(compact_text)
        shot3_kayitlar.append({
            "label": cls,
            "rank": sira,
            "note_id": str(row["note_id"]),
            "subject_id": int(row["subject_id"]),
            "hadm_id": int(row["hadm_id"]),
            "sim_to_centroid": round(float(sim_cent[lokal]), 4),
            "max_sim_to_prev": round(max_sim_prev, 4),
            "leakage_severity": str(row[LEAKAGE_KOLON]),
            "compact_token": n_token,
            "compact_char": len(compact_text),
            "compact_text": compact_text,
            "forbidden_count": len(bulg),
        })

    sims = [shot3_kayitlar[-3]["sim_to_centroid"],
            shot3_kayitlar[-2]["sim_to_centroid"],
            shot3_kayitlar[-1]["sim_to_centroid"]]
    toks = [shot3_kayitlar[-3]["compact_token"],
            shot3_kayitlar[-2]["compact_token"],
            shot3_kayitlar[-1]["compact_token"]]
    print(f"   {cls:<25s} sim=[{sims[0]:.3f},{sims[1]:.3f},{sims[2]:.3f}] tok=[{toks[0]},{toks[1]},{toks[2]}]")

# ============================================================
# Audit
# ============================================================
print("\n[4/6] FINAL AUDIT")
print("=" * 70)

shot3_df = pd.DataFrame(shot3_kayitlar)
toplam_token = shot3_df["compact_token"].sum()
toplam_kirli = (shot3_df["forbidden_count"] > 0).sum()

print(f"\n   Secilen toplam: {len(shot3_df)}/33")
print(f"   Toplam token  : {toplam_token:,} (avg {shot3_df['compact_token'].mean():.0f}/ornek)")
print(f"   Hala kirli    : {toplam_kirli}")
print(f"   Severity dist : {shot3_df['leakage_severity'].value_counts().to_dict()}")

# Token bütçe hesabi
HEADER_TOKEN = 280
EX_WRAPPER = 25
sabit_total = HEADER_TOKEN + toplam_token + 33 * EX_WRAPPER
print(f"\n   Sabit prompt (header+33 example+wrapper): ~{sabit_total:,} token")
print(f"   Test query avg (~580): toplam ~{sabit_total + 580:,}")
print(f"   Test query max (~2024): toplam ~{sabit_total + 2024:,}")
print(f"   8192 sigar mi? {'EVET' if sabit_total + 2024 < 8192 else 'HAYIR (kesilir)'}")

# MMR diversity
print(f"\n   MMR diversity (rank 2+):")
for sira in [2, 3]:
    alt = shot3_df[shot3_df["rank"]==sira]
    print(f"      rank {sira}: max_sim_prev ort={alt['max_sim_to_prev'].mean():.3f}, max={alt['max_sim_to_prev'].max():.3f}")

# Sinif bazli detay
print(f"\n   Sinif bazli detay:")
print(f"   {'Sinif':<25s} {'rank':>4s} {'note':<22s} {'sim':>6s} {'mmr':>6s} {'tok':>4s} {'fb':>3s}")
print("   " + "-" * 80)
for _, r in shot3_df.iterrows():
    print(f"   {r['label']:<25s} {r['rank']:>4} {r['note_id']:<22s} {r['sim_to_centroid']:>6.3f} {r['max_sim_to_prev']:>6.3f} {r['compact_token']:>4} {r['forbidden_count']:>3}")

# ============================================================
# Kaydet
# ============================================================
print("\n[5/6] Kaydetme...")
shot3_df.to_csv(FS_DIR / "fewshot_v4_3shot_clean_examples.csv", index=False)

shot3_json = {}
for cls in ALLOWED_CLASSES:
    shot3_json[cls] = []
    for _, r in shot3_df[shot3_df["label"]==cls].iterrows():
        shot3_json[cls].append({
            "rank": int(r["rank"]),
            "note_id": r["note_id"], "subject_id": r["subject_id"], "hadm_id": r["hadm_id"],
            "sim_to_centroid": float(r["sim_to_centroid"]),
            "max_sim_to_prev": float(r["max_sim_to_prev"]),
            "leakage_severity": r["leakage_severity"],
            "compact_token": int(r["compact_token"]),
            "compact_text": r["compact_text"],
        })
with open(FS_DIR / "fewshot_v4_3shot_clean_examples.json", "w", encoding="utf-8") as f:
    json.dump(shot3_json, f, ensure_ascii=False, indent=2)

print(f"   fewshot_v4_3shot_clean_examples.csv ({(FS_DIR/'fewshot_v4_3shot_clean_examples.csv').stat().st_size/1024:.1f} KB)")
print(f"   fewshot_v4_3shot_clean_examples.json ({(FS_DIR/'fewshot_v4_3shot_clean_examples.json').stat().st_size/1024:.1f} KB)")

# ============================================================
# Preview
# ============================================================
print("\n[6/6] PREVIEW (her sinif rank 1)")
print("=" * 70)
for cls in ALLOWED_CLASSES:
    if cls not in shot3_json or not shot3_json[cls]: continue
    ex = shot3_json[cls][0]
    print(f"\n   {cls} (rank 1, note: {ex['note_id']}, tok: {ex['compact_token']}, sev: {ex['leakage_severity']}):")
    print(f"   {ex['compact_text'][:400]}")

print("\n" + "=" * 70)
print("BLOK 14V4-RESELECT-OTHER-AND-3SHOT-CLEAN TAMAMLANDI - inference YOK")
print("=" * 70)
