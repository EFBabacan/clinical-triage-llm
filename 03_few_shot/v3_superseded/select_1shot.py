"""Blok 14A-SELECT-1SHOT: Class-wise TF-IDF clean medoid (11 ornek).
Centroid'e en yakin temiz medoid. Riskli ranklar reddedilir, sonraki temize gec."""
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
FS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("BLOK 14A-SELECT-1SHOT — Class-wise TF-IDF CLEAN medoid (11 ornek)")
print("=" * 70)

ALLOWED_CLASSES = [
    "SEPSIS", "STROKE", "ATRIAL_FIBRILLATION", "AKI", "PNEUMONIA", "UTI",
    "HEART_FAILURE", "GI_BLEEDING", "DIABETES_COMPLICATION",
    "COPD_EXACERBATION", "OTHER"
]

# --- Riskli leakage patterns (aday reddetme kosullari) ---
LEAKAGE_PATTERNS = {
    "elevated_cr":         re.compile(r'\belevated\s+(creatinine|Cr)\b', re.IGNORECASE),
    "lab_value":           re.compile(r'\b(WBC|Hgb|HCT|lactate|troponin|sodium|Na\+?|BUN|Cr)\s*[<>=]?\s*\d', re.IGNORECASE),
    "imaging_showed":      re.compile(r'\b(CXR|CTA?|MRI|TTE|TEE|US|EKG|ECG|echocardiogram)\b[^.]{0,40}\b(showed|revealed|notable|with|demonstrated|found|consistent)\b', re.IGNORECASE),
    "diagnosed_with":      re.compile(r'\bdiagnosed\s+with\b', re.IGNORECASE),
    "admitted_for_to":     re.compile(r'\badmitted\s+(for|to)\b', re.IGNORECASE),
    "found_to_have":       re.compile(r'\bfound\s+to\s+have\b', re.IGNORECASE),
    "labs_notable":        re.compile(r'\bLabs?\s+(notable|showed|drawn|remarkable)\b', re.IGNORECASE),
    "concern_for":         re.compile(r'\bconcern\s+for\b', re.IGNORECASE),
    "received_started":    re.compile(r'\b(received|was\s+given|started\s+on|treating\s+with|treated\s+with)\b', re.IGNORECASE),
}
# 'received_started' icin past marker varsa kabul (sadece mevcut basvuruda riskli)
PAST_MARKERS = re.compile(
    r"\b(in\s+(\d{4}|___|the past|prior|previously|before)|"
    r"history of|h/o|hx of|s/p|status\s*post|"
    r"recent admission|prior admission|last (admission|year|month|week)|"
    r"\d+\s+(years?|months?|weeks?)\s+ago)\b",
    re.IGNORECASE
)

def leakage_kontrol(metin):
    """Riskli pattern var mi? Hangileri?"""
    if not isinstance(metin, str) or not metin:
        return [], "EMPTY"

    bulgular = []
    for pat_ad, rx in LEAKAGE_PATTERNS.items():
        m = rx.search(metin)
        if m:
            # received_started icin past marker kontrolu
            if pat_ad == "received_started":
                # Match cevresinde past marker var mi?
                konum = m.start()
                baslangic = max(0, konum - 100)
                son = min(len(metin), konum + len(m.group()) + 100)
                baglam = metin[baslangic:son]
                if PAST_MARKERS.search(baglam):
                    continue  # past marker var, kabul (gecmis)
            bulgular.append((pat_ad, m.group()))

    if bulgular:
        return bulgular, "DIRTY"
    return [], "CLEAN"

# --- Veri ---
print("\n[1/5] Veri yukleniyor...")
df = pd.read_parquet(SPLIT_YOL)
df['note_id'] = df['note_id'].astype(str)
pool = df[df['split'] == 'train_S3'].reset_index(drop=True)
print(f"   train_S3 pool: {len(pool):,} ornek")

# --- TF-IDF fit ---
print("\n[2/5] TF-IDF fit (sadece train_S3)...")
docs = pool['triage_input_v3_masked'].fillna('').tolist()
vec = TfidfVectorizer(
    max_features=20000, ngram_range=(1, 2),
    min_df=2, stop_words='english', sublinear_tf=True,
)
X = vec.fit_transform(docs)
print(f"   Shape: {X.shape}")

# --- Class-wise temiz medoid ---
print("\n[3/5] Her sinif icin temiz medoid arama...")

tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B', local_files_only=True)

secilen = []
for cls in ALLOWED_CLASSES:
    cls_idx = np.where(pool['label'].values == cls)[0]
    X_cls = X[cls_idx]
    centroid = X_cls.mean(axis=0)
    centroid_2d = np.asarray(centroid).reshape(1, -1)
    sim_cent = cosine_similarity(X_cls, centroid_2d).flatten()

    # Rank: yuksekten dusuge
    rank_lokal = np.argsort(-sim_cent)

    # Temizden ilkini bul
    secilen_lokal = None
    rank_no = None
    reddedilen_aday = []  # rank_no, note_id, sebep

    for r, lokal in enumerate(rank_lokal, 1):
        global_idx = cls_idx[int(lokal)]
        row = pool.iloc[global_idx]
        triage = row['triage_input_v3_masked'] or ""
        bulgular, durum = leakage_kontrol(triage)

        if durum == "CLEAN":
            secilen_lokal = int(lokal)
            rank_no = r
            break
        else:
            sebep = ", ".join(f"{p}={m!r}" for p, m in bulgular[:3])  # ilk 3 bulgu
            reddedilen_aday.append({
                "rank": r, "note_id": str(row['note_id']),
                "sim_cent": float(sim_cent[lokal]),
                "sebep": sebep
            })

    if secilen_lokal is None:
        print(f"   UYARI: {cls} icin temiz aday BULUNAMADI! (ilk {len(rank_lokal)} aday hep dirty)")
        continue

    global_idx = cls_idx[secilen_lokal]
    row = pool.iloc[global_idx]
    triage = row['triage_input_v3_masked'] or ""
    n_token = len(tok.encode(triage, add_special_tokens=False))

    secilen.append({
        "label": cls,
        "note_id": str(row['note_id']),
        "subject_id": int(row['subject_id']),
        "hadm_id": int(row['hadm_id']),
        "secilen_rank": rank_no,
        "sim_to_centroid": round(float(sim_cent[secilen_lokal]), 4),
        "char_uzunluk": len(triage),
        "token_uzunluk": n_token,
        "leakage_flag": "CLEAN",
        "reddedilen_aday_sayisi": len(reddedilen_aday),
        "reddedilen_aday_detay": reddedilen_aday,
        "preview": triage[:250].replace('\n', ' '),
        "triage_input_v3_masked": triage,
    })

    print(f"   {cls:<25s} rank={rank_no:<3} note={row['note_id']:<20s} sim={sim_cent[secilen_lokal]:.3f} reddedilen={len(reddedilen_aday)}")

# --- Audit raporu ---
print("\n[4/5] Audit raporu")
print("=" * 70)

sec_df = pd.DataFrame([{k:v for k,v in s.items() if k!='reddedilen_aday_detay' and k!='triage_input_v3_masked'} for s in secilen])
print(f"\n   Toplam secilen: {len(sec_df)}/11")
print(f"\n   Token uzunluk istatistigi:")
print(f"      Toplam (11 ornek): {sec_df['token_uzunluk'].sum():,}")
print(f"      Ort  : {sec_df['token_uzunluk'].mean():.0f}")
print(f"      Min  : {sec_df['token_uzunluk'].min()}")
print(f"      Max  : {sec_df['token_uzunluk'].max()}")

print(f"\n   Char uzunluk:")
print(f"      Toplam: {sec_df['char_uzunluk'].sum():,}")
print(f"      Ort   : {sec_df['char_uzunluk'].mean():.0f}")

print(f"\n   Rank dagilimi (kacinci aday secildi):")
for r in sorted(sec_df['secilen_rank'].unique()):
    n = (sec_df['secilen_rank']==r).sum()
    print(f"      rank {r}: {n} sinif")

# Reddedilen aday istatistigi
print(f"\n   Reddedilen aday ortalama: {sec_df['reddedilen_aday_sayisi'].mean():.1f}/sinif")

# Detay tablo
print(f"\n   Detay:")
print(f"   {'Sinif':<25s} {'note_id':<22s} {'rank':>5s} {'sim_cent':>9s} {'char':>5s} {'token':>6s} {'red':>4s}")
print("   " + "-" * 85)
for _, r in sec_df.iterrows():
    print(f"   {r['label']:<25s} {r['note_id']:<22s} {r['secilen_rank']:>5} "
          f"{r['sim_to_centroid']:>9.3f} {r['char_uzunluk']:>5} {r['token_uzunluk']:>6} {r['reddedilen_aday_sayisi']:>4}")

# Reddedilen ornek detay (her sinif icin ilk reddedilen)
print(f"\n   Ilk reddedilen aday detayi (her sinif icin):")
print(f"   {'Sinif':<25s} {'rank':>4s} {'note_id':<22s} sebep")
print("   " + "-" * 90)
for s in secilen:
    if s['reddedilen_aday_detay']:
        d = s['reddedilen_aday_detay'][0]
        print(f"   {s['label']:<25s} {d['rank']:>4} {d['note_id']:<22s} {d['sebep'][:55]}")
    else:
        print(f"   {s['label']:<25s} {'-':>4} {'-':<22s} (rank 1 zaten temiz)")

# --- Preview ---
print("\n[5/5] Sinif basina preview:")
print("=" * 70)
for s in secilen:
    print(f"\n   --- {s['label']} (note: {s['note_id']}, rank: {s['secilen_rank']}, char: {s['char_uzunluk']}, token: {s['token_uzunluk']}) ---")
    print(f"   {s['preview'][:300]}...")

# --- Kaydet ---
print("\n" + "=" * 70)
print("KAYDETME")
print("=" * 70)

csv_yol = FS_DIR / "fewshot_v3_fixed_1shot_examples.csv"
csv_kayit = []
for s in secilen:
    csv_kayit.append({
        "label": s['label'], "note_id": s['note_id'],
        "subject_id": s['subject_id'], "hadm_id": s['hadm_id'],
        "secilen_rank": s['secilen_rank'],
        "sim_to_centroid": s['sim_to_centroid'],
        "char_uzunluk": s['char_uzunluk'],
        "token_uzunluk": s['token_uzunluk'],
        "leakage_flag": s['leakage_flag'],
        "reddedilen_aday_sayisi": s['reddedilen_aday_sayisi'],
        "preview": s['preview'],
        "triage_input_v3_masked": s['triage_input_v3_masked'],
    })
pd.DataFrame(csv_kayit).to_csv(csv_yol, index=False)
print(f"   CSV: {csv_yol.name} ({csv_yol.stat().st_size/1024:.1f} KB)")

json_yapi = {}
for s in secilen:
    json_yapi[s['label']] = {
        "note_id": s['note_id'],
        "subject_id": s['subject_id'],
        "hadm_id": s['hadm_id'],
        "secilen_rank": s['secilen_rank'],
        "sim_to_centroid": s['sim_to_centroid'],
        "char_uzunluk": s['char_uzunluk'],
        "token_uzunluk": s['token_uzunluk'],
        "leakage_flag": s['leakage_flag'],
        "reddedilen_aday_sayisi": s['reddedilen_aday_sayisi'],
        "reddedilen_aday_detay": s['reddedilen_aday_detay'],
        "triage_input_v3_masked": s['triage_input_v3_masked'],
    }
json_yol = FS_DIR / "fewshot_v3_fixed_1shot_examples.json"
with open(json_yol, "w", encoding="utf-8") as f:
    json.dump(json_yapi, f, ensure_ascii=False, indent=2)
print(f"   JSON: {json_yol.name} ({json_yol.stat().st_size/1024:.1f} KB)")

# Final summary
print(f"\n   FINAL: 11 ornek, toplam {sec_df['token_uzunluk'].sum():,} token (sadece icerik)")
print(f"   Inference YOK. Onay sonrasi 14-PROMPT + 14-VAL + 14-TEST.")
print("\n" + "=" * 70)
print("BLOK 14A-SELECT-1SHOT TAMAMLANDI")
print("=" * 70)
