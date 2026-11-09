"""Blok 14V4-3SHOT-ULTRACOMPACT: Ultra compact 3-shot examples (60-80 token).
Hedef: 8 GB GPU'ya uygun, total prompt < 4000 token."""
import pandas as pd
import numpy as np
import torch
import json
import re
import time
import random
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

OUTPUT_DIR = Path(r"./workdir/output")
SPLIT_YOL  = OUTPUT_DIR / "discharge_split.parquet"
FS_DIR     = OUTPUT_DIR / "fewshot"

print("=" * 70)
print("BLOK 14V4-3SHOT-ULTRACOMPACT — 60-80 token/example, 8GB-uyumlu")
print("=" * 70)

# 33 clean examples (mevcut)
with open(FS_DIR / "fewshot_v4_3shot_clean_examples.json", "r", encoding="utf-8") as f:
    shot3_data = json.load(f)

ALLOWED_CLASSES = ["SEPSIS","STROKE","ATRIAL_FIBRILLATION","AKI","PNEUMONIA","UTI",
    "HEART_FAILURE","GI_BLEEDING","DIABETES_COMPLICATION","COPD_EXACERBATION","OTHER"]

# ============================================================
# Forbidden phrases (audit)
# ============================================================
FORBIDDEN_PATTERNS = [
    ("code_stroke", re.compile(r"\bcode\s+(stroke|MI|sepsis|trauma)\b", re.IGNORECASE)),
    ("ct_scanning", re.compile(r"\b(CT scanning|reviewed images|read images|reviewed CT|reviewed scan)\b", re.IGNORECASE)),
    ("micu_admission", re.compile(r"\b(REASON FOR (MICU|ICU) ADMISSION|MICU admission|ICU admission)\b", re.IGNORECASE)),
    ("scheduled_proc", re.compile(r"\bscheduled\s+(TEE|DCCV|EGD|colonoscopy|cardioversion|TTE)\b", re.IGNORECASE)),
    ("elevated_lab", re.compile(r"\b(elevated\s+(Cr|creatinine|WBC|Hgb|HCT|lactate|troponin|BUN|sodium|Na)|abnormal\s+labs?)\b", re.IGNORECASE)),
    ("found_in_ed", re.compile(r"\b(found\s+in\s+ED|found\s+to\s+have|found\s+down)\b", re.IGNORECASE)),
    ("given_treatment", re.compile(r"\b(given|received|started\s+on|placed\s+on|treated\s+with)\s+[^.]{0,40}\b(oxygen|CPAP|BiPAP|nebulizer|nebs|albuterol|duonebs|antibiotics|abx|vancomycin|zosyn|lasix|insulin\s+drip|glucagon|tPA|pressors|levophed|crystalloid|IV\s+fluids|fluid\s+bolus)\b", re.IGNORECASE)),
    ("admitted_for", re.compile(r"\b(admitted\s+(for|to)|MICU\s+admission|ED\s+course)\b", re.IGNORECASE)),
    ("imaging_showed", re.compile(r"\b(CXR|CTA?|MRI|TTE|TEE|US|echocardiogram)\b[^.]{0,40}\b(showed|revealed|demonstrated|consistent\s+with|notable|w/|with\s+(infiltrate|consolidation|effusion|edema|ICH|infarct|hemorrhage))\b", re.IGNORECASE)),
    ("labs_showed", re.compile(r"\bLabs?\b[^.]{0,40}\b(showed|notable|revealed|drawn|remarkable)\b", re.IGNORECASE)),
]

def find_forbidden(text):
    if not isinstance(text, str) or not text: return []
    return [(ad, m.group()) for ad, rx in FORBIDDEN_PATTERNS for m in rx.finditer(text)]

# ============================================================
# ULTRA-COMPACT: CC: ... / Sx: 1 cumle / Hx: 1 phrase / VS: kisa
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

def ultra_compact(full_text):
    """Ultra compact: CC + 1 sx cumle + 1 hx + kisa VS. Meds/allergies YOK."""
    sections = parse_sections(full_text)
    parts = []

    # CC: tam (zaten kisa)
    cc = sections.get("Chief Complaint", "").strip()
    if cc:
        # CC'i 80 char max
        cc_kisa = cc[:80].strip()
        parts.append(f"CC: {cc_kisa}")

    # Sx: 1 ana semptom cumle (en yuksek skorlu, max 200 char)
    sh = sections.get("Symptom History", "").strip()
    if sh:
        ros = re.search(r"(?im)^\s*(Review of Systems?|ROS)\s*:", sh)
        if ros: sh = sh[:ros.start()]
        cumleler = cumlelere_bol(sh)
        if cumleler:
            # Skoru en yuksek olan ilk cumleyi al (ya da ilk cumle)
            skorlu = sorted([(i, c, hasta_skor(c)) for i, c in enumerate(cumleler[:4])],
                            key=lambda x: (-x[2], x[0]))
            ana = skorlu[0][1] if skorlu else cumleler[0]
            ana_kisa = ana[:200].strip()
            parts.append(f"Sx: {ana_kisa}")

    # Hx: 1 onemli kisa phrase (PMH'den, max 80 char)
    pmh = sections.get("Past Medical History", "").strip()
    if pmh:
        pmh_satirlar = [s.strip() for s in re.split(r"[\n;,]+", pmh) if s.strip()]
        # Onemli liste eleman (5-60 char)
        kisa = [s for s in pmh_satirlar if 5 < len(s) < 60][:1]
        if kisa:
            parts.append(f"Hx: {kisa[0]}")

    # VS: sadece kisa vital ozeti
    vs = sections.get("Vital Signs", "").strip()
    if vs:
        # VS satirini 60 char max
        vs_kisa = vs.replace("\n"," ")[:60].strip()
        if vs_kisa:
            parts.append(f"VS: {vs_kisa}")

    return "\n".join(parts)

# ============================================================
# Ultra compact tum 33 ornek
# ============================================================
print("\n[1/5] 33 ornek ultra compact...")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", local_files_only=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

ultra_data = {}
toplam_kirli = 0
toplam_token = 0

for cls in ALLOWED_CLASSES:
    ultra_data[cls] = []
    for ex in shot3_data[cls]:
        # Compact'i parse edip ultra hale getir
        # Ama compact_text section header'i farkli (Chief Complaint:, Key Symptoms:)
        # ultra_compact full V4 input istiyor — eski compact'tan re-parse et
        # Compact_text'te zaten section header var, parse_sections calisir
        ultra_text = ultra_compact(ex["compact_text"])
        # Eger ultra_compact bos donduriyorsa (parse failed), full V4'e bak
        if not ultra_text or len(ultra_text) < 30:
            # df'den note_id ile full V4 al
            df = pd.read_parquet(SPLIT_YOL)
            row_match = df[df["note_id"]==ex["note_id"]]
            if len(row_match) > 0:
                v4_full = row_match.iloc[0]["triage_input_v4_timeaware_masked"] or ""
                ultra_text = ultra_compact(v4_full)

        n_tok = len(tok.encode(ultra_text, add_special_tokens=False))
        bulg = find_forbidden(ultra_text)
        toplam_kirli += len(bulg)
        toplam_token += n_tok
        ultra_data[cls].append({
            "rank": ex["rank"],
            "note_id": ex["note_id"],
            "leakage_severity": ex["leakage_severity"],
            "ultra_token": n_tok,
            "ultra_char": len(ultra_text),
            "ultra_text": ultra_text,
            "forbidden_count": len(bulg),
        })

# Audit
print(f"\n   Ultra compact 33 ornek:")
print(f"   {'Sinif':<25s} {'rank':>4s} {'note':<22s} {'ultra_tok':>10s} {'fb':>3s}")
print("   " + "-" * 70)
for cls in ALLOWED_CLASSES:
    for ex in ultra_data[cls]:
        print(f"   {cls:<25s} {ex['rank']:>4} {ex['note_id']:<22s} {ex['ultra_token']:>10} {ex['forbidden_count']:>3}")

print(f"\n   Toplam token: {toplam_token:,}")
print(f"   Avg/example : {toplam_token/33:.0f} (hedef 60-80)")
print(f"   Toplam kirli: {toplam_kirli}")

# ============================================================
# Prompt build
# ============================================================
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
- OTHER is NOT an uncertainty label. Choose OTHER ONLY when none of the 10 named diagnoses is clinically supported.
- If one of the 10 named classes is plausible, choose that named class instead of OTHER.

Below are 33 ultra-compact example triage notes (3 per class):
"""
EX_TMPL = "\nEx{i} ({label}): {input}\n=> {label}\n"
TT = "\nNow classify:\n{input}\n=> "

SABIT = HEADER
ex_idx = 1
for cls in ALLOWED_CLASSES:
    for ex in ultra_data[cls]:
        SABIT += EX_TMPL.format(i=ex_idx, label=cls, input=ex["ultra_text"].replace("\n"," | "))
        ex_idx += 1

# Sabit token
def chat_token_count(prompt):
    msgs = [{"role":"user","content":prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return len(tok.encode(text, add_special_tokens=False)), text

print("\n[2/5] Sabit prompt token audit...")
fixed_prompt = SABIT + TT.format(input="")
fixed_n, _ = chat_token_count(fixed_prompt)
print(f"   Sabit prompt (header + 33 ultra example + trailer wrap): {fixed_n} token")

# ============================================================
# Val query token istatistigi
# ============================================================
print("\n[3/5] Val query token istatistigi (V4 compact query, runtime)...")

# Mevcut V3 compact query mantigi (14C-V4-VAL ile ayni)
def compact_v4_query(full_text):
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
        secilen = [c for i, c, s in skorli if s > 0][:3]
        if not secilen:
            secilen = [c for i, c, s in skorli][:2]
        else:
            if cumleler and cumleler[0] not in secilen:
                secilen = [cumleler[0]] + secilen
                secilen = secilen[:3]
        if secilen: parts.append(f"Key Symptoms: {' '.join(secilen)}")
    vs = sections.get("Vital Signs", "").strip()
    if vs: parts.append(f"Vitals: {vs}")
    pmh = sections.get("Past Medical History", "").strip()
    if pmh:
        pmh_satirlar = [s.strip() for s in re.split(r"[\n;]+", pmh) if s.strip()]
        kisa = [s for s in pmh_satirlar if 5 < len(s) < 100][:3]
        if not kisa:
            kisa = [c for c in cumlelere_bol(pmh)[:2] if len(c) < 200]
        if kisa: parts.append(f"Relevant History: " + "; ".join(kisa))
    al = sections.get("Allergies", "").strip()
    if al:
        al_kisa = al.split("\n")[0].strip()[:100]
        if al_kisa and al_kisa.upper() not in ("NKDA","NONE","NKA"):
            parts.append(f"Allergies: {al_kisa}")
    meds = sections.get("Medications on Admission", "").strip()
    if meds:
        ilac_satirlar = re.findall(r"\d+\.\s*([A-Z][A-Za-z][^.\n]{2,80})", meds)
        if ilac_satirlar:
            kisa = [" ".join(s.split()[:3]) for s in ilac_satirlar[:5]]
            parts.append(f"Medications: " + ", ".join(kisa))
    return "\n".join(parts)

df = pd.read_parquet(SPLIT_YOL)
val_df = df[df["split"]=="val"].reset_index(drop=True)

total_tokens = []
query_tokens = []
for _, row in val_df.iterrows():
    v4_full = row["triage_input_v4_timeaware_masked"] or ""
    inp = compact_v4_query(v4_full)
    full_prompt = SABIT + TT.format(input=inp)
    total_n, _ = chat_token_count(full_prompt)
    query_tokens.append(total_n - fixed_n)
    total_tokens.append(total_n)

ta = np.array(total_tokens)
qa = np.array(query_tokens)
print(f"\n   Query (sadece V4 compact query):")
print(f"      Avg: {qa.mean():.0f}  P50: {np.median(qa):.0f}  P95: {np.percentile(qa,95):.0f}  Max: {qa.max():.0f}")
print(f"\n   Total prompt:")
print(f"      Avg: {ta.mean():.0f}  P50: {np.median(ta):.0f}  P95: {np.percentile(ta,95):.0f}  Max: {ta.max():.0f}")
print(f"      4000 ustu: {(ta > 4000).sum()}")
print(f"      4096 ustu: {(ta > 4096).sum()}")
print(f"      8192 ustu: {(ta > 8192).sum()}")

# ============================================================
# Smoke test
# ============================================================
print("\n[4/5] Smoke test (3 longest + 3 random)...")

val_with_idx = [(i, ta[i]) for i in range(len(val_df))]
val_with_idx.sort(key=lambda x: -x[1])
longest_3 = [i for i,_ in val_with_idx[:3]]
random.seed(42)
remaining = [i for i in range(len(val_df)) if i not in longest_3]
random_3 = random.sample(remaining, 3)

print("\n   Model yukleniyor...")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", quantization_config=bnb,
    device_map="auto", torch_dtype=torch.float16, local_files_only=True)
model.eval()
print(f"   Yuklendi {time.time()-t0:.1f}s, GPU {torch.cuda.memory_allocated()/1024**2:.0f}MB")

MAX_LEN = 4096

def parse_class_simple(raw):
    if not raw: return "INVALID"
    text = raw.strip().upper()
    for cls in sorted(ALLOWED_CLASSES, key=len, reverse=True):
        if cls in text[:80]: return cls
    return "INVALID"

print(f"\n   {'idx':<5s} {'cat':<6s} {'tok':>5s} {'sure':>7s} {'peak_GB':>8s} {'<think>':>7s} {'parsed':<22s} {'OK':>3s}")
print("   " + "-" * 90)

for kategori, idx_list in [("UZUN", longest_3), ("RAND", random_3)]:
    for idx in idx_list:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        row = val_df.iloc[idx]
        v4_full = row["triage_input_v4_timeaware_masked"] or ""
        inp = compact_v4_query(v4_full)
        prompt = SABIT + TT.format(input=inp)
        msgs = [{"role":"user","content":prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to("cuda")
        in_n = inputs.input_ids.shape[1]

        try:
            t1 = time.time()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tok.eos_token_id)
            sure = time.time() - t1
            peak_gb = torch.cuda.max_memory_allocated()/1024**3
            raw = tok.decode(out[0][in_n:], skip_special_tokens=True).strip()
            parsed = parse_class_simple(raw)
            think_var = "<think>" in raw or "</think>" in raw
            dogru = "OK" if parsed == row["label"] else "X"
            print(f"   {idx:<5d} {kategori:<6s} {in_n:>5d} {sure:>6.1f}s {peak_gb:>7.2f}G {str(think_var):>7s} {parsed:<22s} {dogru:>3s}")
            print(f"      raw={raw[:30]!r}, label={row['label']}")
        except torch.cuda.OutOfMemoryError:
            print(f"   {idx:<5d} OOM!")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"   {idx:<5d} ERROR: {type(e).__name__}: {str(e)[:100]}")

# ============================================================
# Acceptance + Kaydet
# ============================================================
print("\n[5/5] ACCEPTANCE + KAYDETME")
print("=" * 70)
print(f"   Hedef: total prompt avg < 4000")
print(f"   Sonuc: avg={ta.mean():.0f}, max={ta.max():.0f}")
print(f"   Hedef: smoke sure < 12 s/ornek (yukarida)")
print(f"   Hedef: peak GPU shared'a tasmasin (8 GB altinda)")
print(f"   Hedef: <think> yok (yukarida)")
print(f"   Hedef: forbidden 0/33: {'EVET' if toplam_kirli == 0 else f'HAYIR ({toplam_kirli})'}")

# JSON kaydet
out_yol = FS_DIR / "fewshot_v4_3shot_ultracompact_examples.json"
with open(out_yol, "w", encoding="utf-8") as f:
    json.dump(ultra_data, f, ensure_ascii=False, indent=2)
print(f"\n   Kaydedildi: {out_yol.name}")

# Preview
print("\n" + "=" * 70)
print("PREVIEW (her sinif rank 1)")
print("=" * 70)
for cls in ALLOWED_CLASSES:
    if not ultra_data[cls]: continue
    ex = ultra_data[cls][0]
    print(f"\n   {cls} (note: {ex['note_id']}, tok: {ex['ultra_token']}):")
    print(f"   {ex['ultra_text']}")

print("\n" + "=" * 70)
print("BLOK 14V4-3SHOT-ULTRACOMPACT TAMAMLANDI - val'e otomatik gecis YOK")
print("=" * 70)
