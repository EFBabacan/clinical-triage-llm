"""Blok 14E-COMPACT-QUERY: Val/test input'u section-aware compact + smoke + audit."""
import pandas as pd
import json
import re
import time
import torch
import numpy as np
import random
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

OUTPUT_DIR = Path(r"./workdir/output")
SPLIT_YOL  = OUTPUT_DIR / "discharge_split.parquet"
FS_DIR     = OUTPUT_DIR / "fewshot"

print("=" * 70)
print("BLOK 14E-COMPACT-QUERY — Val/test input compact + smoke audit")
print("=" * 70)

ALLOWED_CLASSES = [
    "SEPSIS", "STROKE", "ATRIAL_FIBRILLATION", "AKI", "PNEUMONIA", "UTI",
    "HEART_FAILURE", "GI_BLEEDING", "DIABETES_COMPLICATION",
    "COPD_EXACERBATION", "OTHER"
]

with open(FS_DIR / "fewshot_v3_compact_1shot_examples.json", 'r', encoding='utf-8') as f:
    compact_data = json.load(f)

# --- Sentence/keyword helpers (14D ile ayni mantik, structured V3 kolonlardan compact) ---
SYMPTOM_KEYWORDS = re.compile(
    r'\b(presents?|reports?|complains?|denies|developed|noted|noticed|started|began|'
    r'pain|cough|fever|dyspnea|SOB|fatigue|weakness|nausea|vomit|diarrhea|bleeding|'
    r'dizz|confusion|swelling|edema|chest|abdomen|headache|hypoxic|hypotens|tachycard|'
    r'palpitations|orthopnea|wheez|melena|hemato|dysuria|frequen|urgency)\b',
    re.IGNORECASE
)
def cumlelere_bol(metin):
    if not metin: return []
    cs = re.split(r'(?<=[.!?])\s+(?=[A-Z\d])', metin)
    return [c.strip() for c in cs if c.strip()]
def hasta_skor(c):
    return len(SYMPTOM_KEYWORDS.findall(c))

# Yapilandirilmis kolonlardan input compact uret (cc_v3, hpi_clean_v3, pmh_v3, ...)
def compact_from_columns(row):
    parts = []
    cc = (row.get('cc_v3') or '').strip()
    if cc:
        parts.append(f"Chief Complaint: {cc}")

    hpi = (row.get('hpi_clean_v3') or '').strip()
    if hpi:
        # ROS varsa kes
        ros_idx = re.search(r'(?im)^\s*(Review of Systems?|ROS)\s*:', hpi)
        if ros_idx:
            hpi = hpi[:ros_idx.start()]
        cumleler = cumlelere_bol(hpi)
        ilk_5 = [(i, c, hasta_skor(c)) for i, c in enumerate(cumleler[:5])]
        secilen = [c for i, c, s in ilk_5 if s > 0][:3]
        if not secilen:
            secilen = [c for i, c, s in ilk_5][:2]
        else:
            if cumleler and cumleler[0] not in secilen:
                secilen = [cumleler[0]] + secilen
                secilen = secilen[:3]
        if secilen:
            parts.append(f"Key Symptoms: {' '.join(secilen)}")

    vs = (row.get('vitals_v3') or '').strip()
    if vs:
        parts.append(f"Vitals: {vs}")

    pmh = (row.get('pmh_v3') or '').strip()
    if pmh:
        pmh_satirlar = [s.strip() for s in re.split(r'[\n;]+', pmh) if s.strip()]
        kisa = [s for s in pmh_satirlar if 5 < len(s) < 100][:3]
        if not kisa:
            cumleler = cumlelere_bol(pmh)[:2]
            kisa = [c for c in cumleler if len(c) < 200]
        if kisa:
            parts.append(f"Relevant History: " + "; ".join(kisa))

    al = (row.get('allergies_v3') or '').strip()
    if al:
        al_kisa = al.split('\n')[0].strip()[:100]
        if al_kisa and al_kisa.upper() not in ('NKDA', 'NONE', 'NKA'):
            parts.append(f"Allergies: {al_kisa}")

    meds = (row.get('meds_adm_v3') or '').strip()
    if meds:
        ilac_satirlar = re.findall(r'\d+\.\s*([A-Z][A-Za-z][^.\n]{2,80})', meds)
        if ilac_satirlar:
            kisa_ilaclar = []
            for s in ilac_satirlar[:5]:
                ilac = s.split()[:3]
                kisa_ilaclar.append(' '.join(ilac))
            parts.append(f"Medications: " + ", ".join(kisa_ilaclar))

    return '\n'.join(parts)

# --- Veri yukle + compact uret ---
print("\n[1/5] Veri yukleme + compact uretim (val + test)...")
df = pd.read_parquet(SPLIT_YOL)

t0 = time.time()
df['triage_input_v3_compact'] = df.apply(compact_from_columns, axis=1)
print(f"   Compact uretildi: {time.time()-t0:.1f}s")

val_df = df[df['split']=='val'].reset_index(drop=True)
test_df = df[df['split']=='test'].reset_index(drop=True)
print(f"   Val: {len(val_df):,}, Test: {len(test_df):,}")

# --- Token audit ---
print("\n[2/5] Token audit (val + test)...")
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B', local_files_only=True)

def token_say(texts, n_sample=None):
    if n_sample and len(texts) > n_sample:
        random.seed(0)
        idx = random.sample(range(len(texts)), n_sample)
        texts = [texts[i] for i in idx]
    return [len(tok.encode(t or '', add_special_tokens=False)) for t in texts]

val_full_tok   = token_say(val_df['triage_input_v3_masked'].tolist())
val_comp_tok   = token_say(val_df['triage_input_v3_compact'].tolist())
test_full_tok  = token_say(test_df['triage_input_v3_masked'].tolist(), n_sample=300)
test_comp_tok  = token_say(test_df['triage_input_v3_compact'].tolist(), n_sample=300)

print(f"\n   {'':<30s} {'Ort':>6s} {'Med':>6s} {'P95':>6s} {'Max':>6s}")
print("   " + "-" * 60)
print(f"   {'Val full (V3 masked)':<30s} {np.mean(val_full_tok):>6.0f} {np.median(val_full_tok):>6.0f} {np.percentile(val_full_tok,95):>6.0f} {np.max(val_full_tok):>6.0f}")
print(f"   {'Val compact (V3 compact)':<30s} {np.mean(val_comp_tok):>6.0f} {np.median(val_comp_tok):>6.0f} {np.percentile(val_comp_tok,95):>6.0f} {np.max(val_comp_tok):>6.0f}")
print(f"   {'Test full (V3 masked)':<30s} {np.mean(test_full_tok):>6.0f} {np.median(test_full_tok):>6.0f} {np.percentile(test_full_tok,95):>6.0f} {np.max(test_full_tok):>6.0f}")
print(f"   {'Test compact (V3 compact)':<30s} {np.mean(test_comp_tok):>6.0f} {np.median(test_comp_tok):>6.0f} {np.percentile(test_comp_tok,95):>6.0f} {np.max(test_comp_tok):>6.0f}")

# Hedef kontrol
hedef_max = 900
val_asan = sum(1 for n in val_comp_tok if n > hedef_max)
print(f"\n   Hedef: <{hedef_max} token. Val asan: {val_asan}/{len(val_comp_tok)} ({100*val_asan/len(val_comp_tok):.1f}%)")

# --- Prompt sabit boyutu ---
HEADER = """You are a clinical NLP system. Given a patient's triage information collected at first contact (chief complaint, symptoms, brief history, medications, allergies, vital signs), classify the most likely PRIMARY reason for admission into ONE of these 11 categories:

SEPSIS - Systemic infection, septic shock, bacteremia
STROKE - Ischemic or hemorrhagic stroke, CVA
ATRIAL_FIBRILLATION - Atrial fibrillation/flutter (acute)
AKI - Acute kidney injury, acute renal failure
PNEUMONIA - Lung infection (bacterial, viral, aspiration)
UTI - Urinary tract infection, pyelonephritis, cystitis
HEART_FAILURE - Congestive heart failure, decompensated CHF
GI_BLEEDING - Upper or lower GI hemorrhage
DIABETES_COMPLICATION - DKA, HHS, severe hyper/hypoglycemia
COPD_EXACERBATION - Acute exacerbation of COPD
OTHER - None of the above

Rules:
- Output exactly one class label from the list, using the exact uppercase label string.
- Choose the SINGLE most likely PRIMARY reason for admission.

Below are 11 compact example triage notes (one per class):
"""
EX_TMPL = "\nEXAMPLE {i} ({label}):\n{input}\nPRIMARY DIAGNOSIS: {label}\n"
TT = "\nNow classify the following triage note:\nTRIAGE NOTE:\n{input}\nPRIMARY DIAGNOSIS:"

sabit = HEADER
for i, cls in enumerate(ALLOWED_CLASSES, 1):
    sabit += EX_TMPL.format(i=i, label=cls, input=compact_data[cls]['compact_text'])
sabit_n = len(tok.encode(sabit + TT.format(input=""), add_special_tokens=False))
print(f"\n   Sabit prompt (header + 11 medoid + trailer wrap): {sabit_n} token")
print(f"   Toplam (sabit + max compact val): {sabit_n + max(val_comp_tok)}")
print(f"   Toplam (sabit + max compact test): {sabit_n + max(test_comp_tok)}")

# 4096 kapsama
val_4096_kapsam = sum(1 for n in val_comp_tok if sabit_n + n + 50 <= 4096) / len(val_comp_tok) * 100
test_4096_kapsam = sum(1 for n in test_comp_tok if sabit_n + n + 50 <= 4096) / len(test_comp_tok) * 100
print(f"\n   max_length=4096 kapsama:")
print(f"      Val   : {val_4096_kapsam:.1f}%")
print(f"      Test  : {test_4096_kapsam:.1f}%")

# --- Model yukle ---
print("\n[3/5] Model yukleniyor...")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", quantization_config=bnb,
    device_map="auto", torch_dtype=torch.float16, local_files_only=True)
model.eval()
if tok.pad_token is None: tok.pad_token = tok.eos_token
print(f"   Yuklendi {time.time()-t0:.1f}s, GPU: {torch.cuda.memory_allocated()/1024**2:.0f}MB")

MAX_LEN = 4096

def inference(test_input):
    prompt = sabit + TT.format(input=test_input)
    messages = [{"role":"user","content":prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to("cuda")
    in_n = inputs.input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tok.eos_token_id)
    raw = tok.decode(out[0][in_n:], skip_special_tokens=True).strip()
    return raw, in_n

def parse_class(raw):
    if not raw: return "INVALID"
    text = raw.strip().upper()
    for c in sorted(ALLOWED_CLASSES, key=len, reverse=True):
        if c in text[:80]: return c
    return "INVALID"

# --- En uzun 5 val + random 5 ---
print("\n[4/5] Smoke: en uzun 5 + random 5 val ornegi...")
val_with_n = [(i, val_comp_tok[i]) for i in range(len(val_df))]
val_with_n.sort(key=lambda x: -x[1])
en_uzun_5 = [i for i,_ in val_with_n[:5]]
random.seed(42)
random_5 = random.sample([i for i in range(len(val_df)) if i not in en_uzun_5], 5)

print(f"\n   En uzun 5 val (compact tok):")
print(f"   {'idx':<5s} {'token':>6s} {'note':<22s} {'label':<22s} {'sure':>7s} {'peak':>7s} {'raw':<40s} {'parsed':<22s} {'OK':>3s}")
print("   " + "-" * 130)

basari_say = 0
think_say = 0
for kategori, idx_list in [("UZUN", en_uzun_5), ("RAND", random_5)]:
    if kategori == "RAND":
        print(f"\n   Random 5 val:")
        print(f"   {'idx':<5s} {'token':>6s} {'note':<22s} {'label':<22s} {'sure':>7s} {'peak':>7s} {'raw':<40s} {'parsed':<22s} {'OK':>3s}")
        print("   " + "-" * 130)

    for idx in idx_list:
        torch.cuda.reset_peak_memory_stats()
        row = val_df.iloc[idx]
        compact_input = row['triage_input_v3_compact']
        try:
            t1 = time.time()
            raw, in_n = inference(compact_input)
            sure = time.time() - t1
            peak = torch.cuda.max_memory_allocated() / 1024**2
            parsed = parse_class(raw)
            dogru = "OK" if parsed == row['label'] else "X"
            if dogru == "OK": basari_say += 1
            if "<think>" in raw or "think" in raw[:50].lower(): think_say += 1
            raw_short = raw[:38].replace('\n', ' ')
            print(f"   {idx:<5d} {in_n:>6d} {row['note_id']:<22s} {row['label']:<22s} {sure:>6.1f}s {peak:>6.0f}M {raw_short!r:<42s} {parsed:<22s} {dogru:>3s}")
        except torch.cuda.OutOfMemoryError:
            print(f"   {idx:<5d} OOM!")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"   {idx:<5d} ERROR: {type(e).__name__}: {str(e)[:80]}")

# --- Karar ---
print("\n" + "=" * 70)
print("[5/5] AUDIT KARAR")
print("=" * 70)
print(f"   Smoke 10 ornek: dogru={basari_say}/10, <think> govde={think_say}/10")
peak_max_mb = torch.cuda.max_memory_allocated() / 1024**2
print(f"   Peak GPU: {peak_max_mb:.0f}MB ({100*peak_max_mb/8188:.1f}%)")

# Compact kolon kalici mi?
print(f"\n   triage_input_v3_compact kolonu uretildi (val + test). Parquet'e yazma henuz YOK.")

# Compact'i parquet'e yazalim
df.to_parquet(SPLIT_YOL, compression='snappy', index=False)
print(f"   Parquet guncellendi: triage_input_v3_compact kolonu eklendi.")
print(f"   Toplam kolon: {len(df.columns)}")

print("\n" + "=" * 70)
print("BLOK 14E TAMAMLANDI - Otomatik gecis YOK")
if think_say == 0 and basari_say >= 5:
    print(f"   ✅ Smoke OK: think yok, dogruluk makul. 14C-VAL'i tetikleyebiliriz.")
else:
    print(f"   ⚠️ Sorun var: think={think_say}, dogru={basari_say}/10")
print("=" * 70)
