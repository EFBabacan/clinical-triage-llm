"""Blok 14-V4-SELECT: V4 1-shot ve 3-shot medoid + MMR seclim.
Sadece train_S3 + leakage_severity_final in [CLEAN, HISTORICAL_ALLOWED].
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
FS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("BLOK 14-V4-SELECT — V4 1-shot + 3-shot medoid + MMR")
print("=" * 70)

ALLOWED_CLASSES = [
    "SEPSIS","STROKE","ATRIAL_FIBRILLATION","AKI","PNEUMONIA","UTI",
    "HEART_FAILURE","GI_BLEEDING","DIABETES_COMPLICATION","COPD_EXACERBATION","OTHER"
]

# --- Veri ---
print("\n[1/6] Veri yukleniyor...")
df = pd.read_parquet(SPLIT_YOL)
df["note_id"] = df["note_id"].astype(str)
print(f"   N={len(df):,}, kolon={len(df.columns)}")

# Pool: train_S3 + leakage clean
pool_full = df[df["split"]=="train_S3"].reset_index(drop=True)
print(f"   train_S3 pool: {len(pool_full):,}")

# Filtre: leakage_severity_final in [CLEAN, HISTORICAL_ALLOWED]
LEAKAGE_KOLON = "strict_current_leakage_severity_final"
if LEAKAGE_KOLON not in pool_full.columns:
    print(f"   UYARI: {LEAKAGE_KOLON} yok, {pool_full.columns.tolist()}")
    LEAKAGE_KOLON = "strict_current_leakage_severity"

clean_mask = pool_full[LEAKAGE_KOLON].isin(["CLEAN","HISTORICAL_ALLOWED"])
pool = pool_full[clean_mask].reset_index(drop=True)
print(f"   Filtrelenmis pool (CLEAN/HISTORICAL_ALLOWED): {len(pool):,}")
print(f"   Disardakiler: {(~clean_mask).sum()} (REVIEW/HIGH/MEDIUM)")

# Sinif dagilimi
print(f"\n   Pool sinif dagilimi (V4 clean):")
for cls in ALLOWED_CLASSES:
    n = (pool["label"]==cls).sum()
    n_orig = (pool_full["label"]==cls).sum()
    print(f"      {cls:<25s} {n:>3} (V4 clean) / {n_orig:>3} (train_S3 toplam)")

# --- TF-IDF fit ---
print("\n[2/6] TF-IDF fit (sadece V4 clean pool)...")
docs = pool["triage_input_v4_timeaware_masked"].fillna("").tolist()
vec = TfidfVectorizer(
    max_features=20000, ngram_range=(1, 2),
    min_df=2, stop_words="english", sublinear_tf=True
)
X = vec.fit_transform(docs)
print(f"   Shape: {X.shape}, vocab: {len(vec.vocabulary_)}")

# Tokenizer (Qwen)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", local_files_only=True)

def token_say(metin):
    if not isinstance(metin, str) or not metin: return 0
    return len(tok.encode(metin, add_special_tokens=False))

# --- 1-shot/class secim (medoid, en yakin centroid) ---
print("\n[3/6] V4 1-shot/class medoid seclim...")
shot1_kayitlar = []

for cls in ALLOWED_CLASSES:
    cls_idx = np.where(pool["label"].values == cls)[0]
    if len(cls_idx) == 0:
        print(f"   UYARI: {cls} icin pool'da clean ornek yok!")
        continue

    X_cls = X[cls_idx]
    centroid = np.asarray(X_cls.mean(axis=0)).reshape(1, -1)
    sim_cent = cosine_similarity(X_cls, centroid).flatten()
    rank_lokal = np.argsort(-sim_cent)

    # En yakin (rank 1) — leakage filtresi zaten pool'da uygulanmis
    secilen = int(rank_lokal[0])
    global_idx = cls_idx[secilen]
    row = pool.iloc[global_idx]

    triage = row["triage_input_v4_timeaware_masked"] or ""
    n_token = token_say(triage)

    shot1_kayitlar.append({
        "label": cls,
        "rank": 1,
        "note_id": str(row["note_id"]),
        "subject_id": int(row["subject_id"]),
        "hadm_id": int(row["hadm_id"]),
        "sim_to_centroid": round(float(sim_cent[secilen]), 4),
        "leakage_severity": str(row[LEAKAGE_KOLON]),
        "char_uzunluk": len(triage),
        "token_uzunluk": n_token,
        "preview": triage[:200].replace("\n", " "),
        "triage_input_v4_timeaware_masked": triage,
    })
    print(f"   {cls:<25s} note={row['note_id']:<22s} sim={sim_cent[secilen]:.3f} sev={str(row[LEAKAGE_KOLON]):<22s} tok={n_token}")

# --- 3-shot/class secim (medoid + MMR) ---
print("\n[4/6] V4 3-shot/class secim (centroid + MMR diversity)...")
ALPHA = 0.7  # centroid agirlik
BETA  = 0.3  # diversity penalty

shot3_kayitlar = []

for cls in ALLOWED_CLASSES:
    cls_idx = np.where(pool["label"].values == cls)[0]
    if len(cls_idx) < 3:
        print(f"   UYARI: {cls} icin clean pool'da {len(cls_idx)} ornek var (3 gerekli)")
        continue

    X_cls = X[cls_idx]
    centroid = np.asarray(X_cls.mean(axis=0)).reshape(1, -1)
    sim_cent = cosine_similarity(X_cls, centroid).flatten()

    # 1. ornek: en yakin centroid
    secilen_lokal = [int(np.argmax(sim_cent))]

    # 2. ve 3. ornek: MMR
    for _ in range(2):
        secilen_X = X_cls[secilen_lokal]
        sim_to_sel = cosine_similarity(X_cls, secilen_X)
        max_sim_sel = sim_to_sel.max(axis=1)
        skor = ALPHA * sim_cent - BETA * max_sim_sel
        for s in secilen_lokal: skor[s] = -np.inf
        secilen_lokal.append(int(np.argmax(skor)))

    for sira, lokal_idx in enumerate(secilen_lokal, 1):
        global_idx = cls_idx[lokal_idx]
        row = pool.iloc[global_idx]
        # Max sim to previously selected
        if sira == 1:
            max_sim_prev = 0.0
        else:
            secilen_oncekiler = X_cls[secilen_lokal[:sira-1]]
            max_sim_prev = float(cosine_similarity(X_cls[lokal_idx:lokal_idx+1], secilen_oncekiler).max())

        triage = row["triage_input_v4_timeaware_masked"] or ""
        n_token = token_say(triage)

        shot3_kayitlar.append({
            "label": cls,
            "rank": sira,
            "note_id": str(row["note_id"]),
            "subject_id": int(row["subject_id"]),
            "hadm_id": int(row["hadm_id"]),
            "sim_to_centroid": round(float(sim_cent[lokal_idx]), 4),
            "max_sim_to_prev_selected": round(max_sim_prev, 4),
            "leakage_severity": str(row[LEAKAGE_KOLON]),
            "char_uzunluk": len(triage),
            "token_uzunluk": n_token,
            "preview": triage[:200].replace("\n", " "),
            "triage_input_v4_timeaware_masked": triage,
        })

    sims = [shot3_kayitlar[-3]["sim_to_centroid"],
            shot3_kayitlar[-2]["sim_to_centroid"],
            shot3_kayitlar[-1]["sim_to_centroid"]]
    print(f"   {cls:<25s} sim=[{sims[0]:.3f}, {sims[1]:.3f}, {sims[2]:.3f}]")

# --- Audit ---
print("\n[5/6] AUDIT")
print("=" * 70)

shot1_df = pd.DataFrame(shot1_kayitlar)
shot3_df = pd.DataFrame(shot3_kayitlar)

# 1-shot ozeti
print(f"\n1-SHOT SECILEN (11 ornek):")
print(f"   {'Sinif':<25s} {'note_id':<22s} {'sim_cent':>9s} {'sev':<22s} {'tok':>5s}")
print("   " + "-" * 90)
for _, r in shot1_df.iterrows():
    print(f"   {r['label']:<25s} {r['note_id']:<22s} {r['sim_to_centroid']:>9.3f} {r['leakage_severity']:<22s} {r['token_uzunluk']:>5}")

print(f"\n   1-shot token toplami: {shot1_df['token_uzunluk'].sum():,}")
print(f"   1-shot ortalama: {shot1_df['token_uzunluk'].mean():.0f}")
print(f"   1-shot min/max : {shot1_df['token_uzunluk'].min()} / {shot1_df['token_uzunluk'].max()}")

# 3-shot ozeti
print(f"\n3-SHOT SECILEN (33 ornek):")
print(f"   {'Sinif':<25s} {'rank':>4s} {'note_id':<22s} {'sim_cent':>9s} {'max_sim_prev':>12s} {'sev':<22s} {'tok':>5s}")
print("   " + "-" * 110)
for _, r in shot3_df.iterrows():
    print(f"   {r['label']:<25s} {r['rank']:>4} {r['note_id']:<22s} {r['sim_to_centroid']:>9.3f} {r['max_sim_to_prev_selected']:>12.3f} {r['leakage_severity']:<22s} {r['token_uzunluk']:>5}")

print(f"\n   3-shot token toplami: {shot3_df['token_uzunluk'].sum():,}")
print(f"   3-shot ortalama: {shot3_df['token_uzunluk'].mean():.0f}")
print(f"   3-shot min/max : {shot3_df['token_uzunluk'].min()} / {shot3_df['token_uzunluk'].max()}")

# MMR diversity check
print(f"\n   3-shot MMR diversity (rank 2+ icin max_sim_to_prev):")
for sira in [2, 3]:
    alt = shot3_df[shot3_df["rank"]==sira]
    print(f"      rank {sira}: ort={alt['max_sim_to_prev_selected'].mean():.3f}, max={alt['max_sim_to_prev_selected'].max():.3f}")

# --- Token butce hesabi ---
print(f"\n[6/6] TOKEN BUTCESI HESAPLAMA")
print("=" * 70)

# Sabit kisim (header + sinif tanimlari + trailer wrapper) — ~280 token
HEADER_TOKEN = 280
EX_WRAPPER_TOKEN = 25  # "EXAMPLE i: ... PRIMARY DIAGNOSIS:" wrapper

# 1-shot full
shot1_full = HEADER_TOKEN + shot1_df["token_uzunluk"].sum() + 11 * EX_WRAPPER_TOKEN
# 1-shot compact (180-220 token/example hedef, var olan 14D mantigi)
SHOT1_COMPACT_TARGET = 200
shot1_compact = HEADER_TOKEN + 11 * SHOT1_COMPACT_TARGET + 11 * EX_WRAPPER_TOKEN

# 3-shot full
shot3_full = HEADER_TOKEN + shot3_df["token_uzunluk"].sum() + 33 * EX_WRAPPER_TOKEN
# 3-shot compact (daha agresif, 100-120 token/example hedef cunku 33 var)
SHOT3_COMPACT_TARGET = 120
shot3_compact = HEADER_TOKEN + 33 * SHOT3_COMPACT_TARGET + 33 * EX_WRAPPER_TOKEN

print(f"\n   {'Yontem':<22s} {'Full prompt':>12s} {'Compact (target)':>17s}")
print("   " + "-" * 55)
print(f"   {'1-shot (11 ornek)':<22s} {shot1_full:>12,} {shot1_compact:>17,}")
print(f"   {'3-shot (33 ornek)':<22s} {shot3_full:>12,} {shot3_compact:>17,}")

print(f"\n   Test input ortalamasi (V4): ~580 token")
print(f"   1-shot full + test_avg     : ~{shot1_full + 580:,}")
print(f"   1-shot compact + test_avg  : ~{shot1_compact + 580:,}  (max_length=4096 sigar)")
print(f"   3-shot full + test_avg     : ~{shot3_full + 580:,}  (8192 sigmaz)")
print(f"   3-shot compact + test_avg  : ~{shot3_compact + 580:,}  (max_length=8192 hedeflenir)")

# Leakage severity dagilimi (secilen)
print(f"\n   Leakage severity (secilen ornekler):")
print(f"   1-shot: {shot1_df['leakage_severity'].value_counts().to_dict()}")
print(f"   3-shot: {shot3_df['leakage_severity'].value_counts().to_dict()}")

# Acceptance: sadece CLEAN/HISTORICAL olmali
shot1_kabul = shot1_df["leakage_severity"].isin(["CLEAN","HISTORICAL_ALLOWED"]).all()
shot3_kabul = shot3_df["leakage_severity"].isin(["CLEAN","HISTORICAL_ALLOWED"]).all()
print(f"\n   1-shot tum CLEAN/HISTORICAL: {'✅' if shot1_kabul else '❌'}")
print(f"   3-shot tum CLEAN/HISTORICAL: {'✅' if shot3_kabul else '❌'}")

# Kaydet
print("\nKAYDETME...")
shot1_csv = FS_DIR / "fewshot_v4_1shot_examples.csv"
shot1_json = FS_DIR / "fewshot_v4_1shot_examples.json"
shot3_csv = FS_DIR / "fewshot_v4_3shot_examples.csv"
shot3_json = FS_DIR / "fewshot_v4_3shot_examples.json"

shot1_df.to_csv(shot1_csv, index=False)
shot3_df.to_csv(shot3_csv, index=False)

# JSON: label -> (1-shot tek dict, 3-shot list of 3)
shot1_json_yapi = {}
for _, r in shot1_df.iterrows():
    shot1_json_yapi[r["label"]] = {
        "note_id": r["note_id"], "subject_id": r["subject_id"], "hadm_id": r["hadm_id"],
        "sim_to_centroid": float(r["sim_to_centroid"]),
        "leakage_severity": r["leakage_severity"],
        "char_uzunluk": int(r["char_uzunluk"]),
        "token_uzunluk": int(r["token_uzunluk"]),
        "triage_input_v4_timeaware_masked": r["triage_input_v4_timeaware_masked"],
    }

shot3_json_yapi = {}
for cls in ALLOWED_CLASSES:
    shot3_json_yapi[cls] = []
    for _, r in shot3_df[shot3_df["label"]==cls].iterrows():
        shot3_json_yapi[cls].append({
            "rank": int(r["rank"]),
            "note_id": r["note_id"], "subject_id": r["subject_id"], "hadm_id": r["hadm_id"],
            "sim_to_centroid": float(r["sim_to_centroid"]),
            "max_sim_to_prev_selected": float(r["max_sim_to_prev_selected"]),
            "leakage_severity": r["leakage_severity"],
            "char_uzunluk": int(r["char_uzunluk"]),
            "token_uzunluk": int(r["token_uzunluk"]),
            "triage_input_v4_timeaware_masked": r["triage_input_v4_timeaware_masked"],
        })

with open(shot1_json, "w", encoding="utf-8") as f:
    json.dump(shot1_json_yapi, f, ensure_ascii=False, indent=2)
with open(shot3_json, "w", encoding="utf-8") as f:
    json.dump(shot3_json_yapi, f, ensure_ascii=False, indent=2)

print(f"   {shot1_csv.name} ({shot1_csv.stat().st_size/1024:.1f} KB)")
print(f"   {shot1_json.name} ({shot1_json.stat().st_size/1024:.1f} KB)")
print(f"   {shot3_csv.name} ({shot3_csv.stat().st_size/1024:.1f} KB)")
print(f"   {shot3_json.name} ({shot3_json.stat().st_size/1024:.1f} KB)")

# Preview
print("\n" + "=" * 70)
print("PREVIEW (1-shot)")
print("=" * 70)
for cls in ALLOWED_CLASSES:
    if cls not in shot1_json_yapi: continue
    ex = shot1_json_yapi[cls]
    print(f"\n   {cls} (note: {ex['note_id']}, tok: {ex['token_uzunluk']}, sev: {ex['leakage_severity']})")
    print(f"   {ex['triage_input_v4_timeaware_masked'][:250]}...")

print("\n" + "=" * 70)
print("BLOK 14-V4-SELECT TAMAMLANDI - inference YOK")
print("=" * 70)
