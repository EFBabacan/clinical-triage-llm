"""Blok 14D: Section-aware compaction + 5 ornek audit + 3 smoke inference.
Mevcut 11 medoid'i sikistirip 180-220 token hedefine indir."""
import pandas as pd
import json
import re
import time
from pathlib import Path
from transformers import AutoTokenizer

OUTPUT_DIR = Path(r"./workdir/output")
SPLIT_YOL  = OUTPUT_DIR / "discharge_split.parquet"
FS_DIR     = OUTPUT_DIR / "fewshot"

print("=" * 70)
print("BLOK 14D — SECTION-AWARE COMPACTION + AUDIT")
print("=" * 70)

# 11 medoid yukle
with open(FS_DIR / "fewshot_v3_fixed_1shot_examples.json", 'r', encoding='utf-8') as f:
    medoid_data = json.load(f)

ALLOWED_CLASSES = [
    "SEPSIS", "STROKE", "ATRIAL_FIBRILLATION", "AKI", "PNEUMONIA", "UTI",
    "HEART_FAILURE", "GI_BLEEDING", "DIABETES_COMPLICATION",
    "COPD_EXACERBATION", "OTHER"
]

# --- Section parser (V3 input zaten 'Chief Complaint:', 'Symptom History:' vb. headerli) ---
SECTION_RE = re.compile(
    r'(Chief Complaint|Symptom History|Past Medical History|Allergies|Medications on Admission|Vital Signs):\s*\n?',
    re.IGNORECASE
)

def parse_sections(text):
    """V3 input -> dict of sections."""
    if not text:
        return {}
    parts = SECTION_RE.split(text)
    # parts: ['', 'Chief Complaint', 'cc text', 'Symptom History', 'symptom text', ...]
    sections = {}
    for i in range(1, len(parts) - 1, 2):
        ad = parts[i].strip()
        icerik = parts[i+1].strip()
        sections[ad] = icerik
    return sections

# --- Sentence splitter ---
def cumlelere_bol(metin):
    if not metin: return []
    cumleler = re.split(r'(?<=[.!?])\s+(?=[A-Z\d])', metin)
    return [c.strip() for c in cumleler if c.strip()]

# Hasta semptomu/sikayeti olan cumleleri tercih (basit heuristic)
SYMPTOM_KEYWORDS = re.compile(
    r'\b(presents?|reports?|complains?|denies|developed|noted|noticed|started|began|'
    r'pain|cough|fever|dyspnea|SOB|fatigue|weakness|nausea|vomit|diarrhea|bleeding|'
    r'dizz|confusion|swelling|edema|chest|abdomen|headache|hypoxic|hypotens|tachycard|'
    r'palpitations|orthopnea|wheez|melena|hemato|dysuria|frequen|urgency)\b',
    re.IGNORECASE
)

def hasta_cumlesi_skor(cumle):
    return len(SYMPTOM_KEYWORDS.findall(cumle))

def compact_input(full_text):
    """Section-aware sikistirma: CC tam + symptom 2-3 cumle + vitals tam +
    PMH 2-3 kisa ifade + meds en fazla 5 ilac."""
    sections = parse_sections(full_text)

    parts = []

    # 1. Chief Complaint - TAM
    cc = sections.get('Chief Complaint', '').strip()
    if cc:
        parts.append(f"Chief Complaint: {cc}")

    # 2. Symptom History - hasta semptomlu ilk 2-3 cumle
    sh = sections.get('Symptom History', '').strip()
    if sh:
        # ROS varsa onu kes (compact'e gerek yok)
        ros_idx = re.search(r'(?im)^\s*(Review of Systems?|ROS)\s*:', sh)
        if ros_idx:
            sh = sh[:ros_idx.start()]
        cumleler = cumlelere_bol(sh)
        # Skor + sira
        skorli = [(i, c, hasta_cumlesi_skor(c)) for i, c in enumerate(cumleler)]
        # Ilk 5 cumle icinden en yuksek skorlu 3'u (siralama korunur)
        ilk_5 = skorli[:5]
        # Skoru >0 olanlardan ilk 3'unu al, sira korunur
        secilen = [c for i, c, s in ilk_5 if s > 0][:3]
        # Eger hic skor yoksa ilk 2'yi al
        if not secilen:
            secilen = [c for i, c, s in ilk_5][:2]
        else:
            # Hep ilk cumleyi ekle (meta context)
            if cumleler and cumleler[0] not in secilen:
                secilen = [cumleler[0]] + secilen
                secilen = secilen[:3]
        if secilen:
            parts.append(f"Key Symptoms: {' '.join(secilen)}")

    # 3. Vital Signs - TAM (hep kisa)
    vs = sections.get('Vital Signs', '').strip()
    if vs:
        parts.append(f"Vitals: {vs}")

    # 4. PMH - kisa (ilk 2-3 onemli)
    pmh = sections.get('Past Medical History', '').strip()
    if pmh:
        # Liste/kisa formatlari tercih
        # Numarali liste varsa (1. 2. 3. veya - x - y)
        pmh_satirlar = [s.strip() for s in re.split(r'[\n;]+', pmh) if s.strip()]
        # Ilk 3 kisa madde
        kisa = [s for s in pmh_satirlar if 5 < len(s) < 100][:3]
        if not kisa:
            # Cumle bazli al
            cumleler = cumlelere_bol(pmh)[:2]
            kisa = [c for c in cumleler if len(c) < 200]
        if kisa:
            parts.append(f"Relevant History: " + "; ".join(kisa))

    # 5. Allergies - kisa
    al = sections.get('Allergies', '').strip()
    if al:
        # Ilk satir
        al_kisa = al.split('\n')[0].strip()[:100]
        if al_kisa and al_kisa.upper() not in ('NKDA', 'NONE', 'NKA'):
            parts.append(f"Allergies: {al_kisa}")

    # 6. Medications - en fazla 5 ilac, kisa
    meds = sections.get('Medications on Admission', '').strip()
    if meds:
        # Numarali liste pattern: '1. Drug', '2. Drug'
        ilac_satirlar = re.findall(r'\d+\.\s*([A-Z][A-Za-z][^.\n]{2,80})', meds)
        if ilac_satirlar:
            # Sadece ilac adi (ilk 1-2 kelime)
            kisa_ilaclar = []
            for s in ilac_satirlar[:5]:
                ilac = s.split()[:3]  # ilk 3 kelime (ilac+doz)
                kisa_ilaclar.append(' '.join(ilac))
            parts.append(f"Medications: " + ", ".join(kisa_ilaclar))

    return '\n'.join(parts)

# --- Compact uygula 11 medoide ---
print("\n[1/4] 11 medoidi compact'la...")
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B', local_files_only=True)

compact_data = {}
for cls in ALLOWED_CLASSES:
    full = medoid_data[cls]['triage_input_v3_masked']
    compact = compact_input(full)
    n_tok_full = len(tok.encode(full, add_special_tokens=False))
    n_tok_comp = len(tok.encode(compact, add_special_tokens=False))
    compact_data[cls] = {
        "note_id": medoid_data[cls]['note_id'],
        "full_token": n_tok_full,
        "compact_token": n_tok_comp,
        "compact_char": len(compact),
        "compact_text": compact,
    }
    print(f"   {cls:<25s} full={n_tok_full:>4} -> compact={n_tok_comp:>3} ({100*n_tok_comp/n_tok_full:.0f}%)")

toplam_compact = sum(d['compact_token'] for d in compact_data.values())
toplam_full = sum(d['full_token'] for d in compact_data.values())
print(f"\n   Toplam: full={toplam_full:,} -> compact={toplam_compact:,} ({100*toplam_compact/toplam_full:.0f}%)")
print(f"   Hedef: 11x 180-220 = 1.980-2.420 token")

# Kaydet
compact_yol = FS_DIR / "fewshot_v3_compact_1shot_examples.json"
with open(compact_yol, 'w', encoding='utf-8') as f:
    json.dump(compact_data, f, ensure_ascii=False, indent=2)
print(f"   Kaydedildi: {compact_yol.name}")

# Ornek goster
print("\n[2/4] Compact ornek (HEART_FAILURE):")
print("=" * 70)
print(compact_data['HEART_FAILURE']['compact_text'])
print("=" * 70)

# --- Token bütçesi tüm test seti ---
print("\n[3/4] Yeni token bütcesi (compact 11 ornek)...")

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

EXAMPLE_TEMPLATE = "\nEXAMPLE {i} ({label}):\n{input}\nPRIMARY DIAGNOSIS: {label}\n"
TEST_TRAILER = "\nNow classify the following triage note:\nTRIAGE NOTE:\n{input}\nPRIMARY DIAGNOSIS:"

# Sabit prompt
sabit = HEADER
for i, cls in enumerate(ALLOWED_CLASSES, 1):
    sabit += EXAMPLE_TEMPLATE.format(i=i, label=cls, input=compact_data[cls]['compact_text'])
sabit_with_test_wrap = sabit + TEST_TRAILER.format(input="")
sabit_token = len(tok.encode(sabit_with_test_wrap, add_special_tokens=False))
print(f"   Sabit kisim (header + 11 compact + trailer wrapper): {sabit_token:,} token")

# Test inputs
df = pd.read_parquet(SPLIT_YOL)
val_df = df[df["split"]=="val"]
import random, numpy as np
random.seed(0)
sample_idx = random.sample(range(len(val_df)), 5)
test_token_sample = []
for i in sample_idx:
    t = val_df.iloc[i]['triage_input_v3_masked'] or ""
    test_token_sample.append(len(tok.encode(t, add_special_tokens=False)))

print(f"\n   5 val ornek input tokenleri: {test_token_sample}")
print(f"   TOPLAM tahmini (sabit + ortalama input): {sabit_token + int(np.mean(test_token_sample)):,}")
print(f"   TOPLAM tahmini (sabit + max input):     {sabit_token + max(test_token_sample):,}")

# Tum val'da
val_input_tok = []
for _, row in val_df.iterrows():
    t = row['triage_input_v3_masked'] or ""
    val_input_tok.append(len(tok.encode(t, add_special_tokens=False)))

print(f"\n   Val tum 550 ornek input tokenleri:")
print(f"      Ort  : {np.mean(val_input_tok):.0f}")
print(f"      P95  : {np.percentile(val_input_tok, 95):.0f}")
print(f"      Max  : {np.max(val_input_tok)}")
print(f"\n   Final: sabit + max input = {sabit_token + int(np.max(val_input_tok)):,} token")

# max_length onerisi
print(f"\n   max_length onerisi (8 GB GPU):")
print(f"      4096 -> %{100*sum(1 for t in val_input_tok if sabit_token + t <= 4096)/len(val_input_tok):.1f} kapsama")
print(f"      6144 -> %{100*sum(1 for t in val_input_tok if sabit_token + t <= 6144)/len(val_input_tok):.1f} kapsama")
print(f"      8192 -> %{100*sum(1 for t in val_input_tok if sabit_token + t <= 8192)/len(val_input_tok):.1f} kapsama")

# --- 3 smoke inference ---
print("\n[4/4] 3 ornek SMOKE INFERENCE (OOM kontrolu)...")
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
print("   Model yukleniyor...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", quantization_config=bnb,
    device_map="auto", torch_dtype=torch.float16, local_files_only=True)
model.eval()
if tok.pad_token is None: tok.pad_token = tok.eos_token
print(f"   Yuklendi {time.time()-t0:.1f}s, GPU alloc: {torch.cuda.memory_allocated()/1024**2:.0f}MB")

MAX_LEN = 4096

# 3 farkli boy test ornegi
test_idxs = [0, len(val_df)//2, len(val_df)-1]
for idx in test_idxs:
    row = val_df.iloc[idx]
    test_input = row['triage_input_v3_masked'] or ""
    prompt = sabit + TEST_TRAILER.format(input=test_input)
    messages = [{"role":"user","content":prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to("cuda")
    in_n = inputs.input_ids.shape[1]
    print(f"\n   --- val[{idx}] (label={row['label']}, input_token={in_n}) ---")
    try:
        t1 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tok.eos_token_id)
        sure = time.time()-t1
        raw = tok.decode(out[0][in_n:], skip_special_tokens=True).strip()
        peak = torch.cuda.max_memory_allocated()/1024**2
        print(f"      OK sure={sure:.2f}s peak_GPU={peak:.0f}MB")
        print(f"      raw={raw!r}")
        torch.cuda.reset_peak_memory_stats()
    except torch.cuda.OutOfMemoryError as e:
        print(f"      OOM! {str(e)[:200]}")
        break
    except Exception as e:
        print(f"      ERROR: {e}")
        break

print("\n" + "=" * 70)
print("14D AUDIT TAMAMLANDI. OOM yoksa 14C-VAL'i bu compact ornekleriyle calistir.")
print("=" * 70)
