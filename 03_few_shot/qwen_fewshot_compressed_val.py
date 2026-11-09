"""Blok 14C-V4-3SHOT-ULTRA-VAL: V4 3-shot ULTRA-COMPACT few-shot val (550 ornek).
- 33 ultra-compact medoid (avg 29 tok)
- Query V4 compact (runtime)
- max_length=4096, sure ~25-35 dk
- Test'e otomatik gecmez."""
import pandas as pd
import numpy as np
import torch
import json
import re
import time
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

OUTPUT_DIR = Path(r"./workdir/output")
SPLIT_YOL  = OUTPUT_DIR / "discharge_split.parquet"
FS_DIR     = OUTPUT_DIR / "fewshot"

print("=" * 70)
print("BLOK 14C-V4-3SHOT-ULTRA-VAL — V4 3-shot ULTRA-COMPACT val")
print("=" * 70)
print("   KURAL: Sadece val. Test'e otomatik gecmez.")

# Ultra-compact 33 medoid
with open(FS_DIR / "fewshot_v4_3shot_ultracompact_examples.json", "r", encoding="utf-8") as f:
    ultra_data = json.load(f)

ALLOWED_CLASSES = ["SEPSIS","STROKE","ATRIAL_FIBRILLATION","AKI","PNEUMONIA","UTI",
    "HEART_FAILURE","GI_BLEEDING","DIABETES_COMPLICATION","COPD_EXACERBATION","OTHER"]

# ============================================================
# V4 query compact (smoke ile ayni)
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

# ============================================================
# Prompt build (smoke ile ayni format)
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

FUZZY_MAP = {
    "SEPS":"SEPSIS","SEPT":"SEPSIS","PNEUM":"PNEUMONIA",
    "STROK":"STROKE","CVA":"STROKE",
    "ATRIAL":"ATRIAL_FIBRILLATION","AFIB":"ATRIAL_FIBRILLATION","FIBRIL":"ATRIAL_FIBRILLATION",
    "AKI":"AKI","KIDNEY":"AKI","RENAL":"AKI",
    "UTI":"UTI","URIN":"UTI",
    "HEART":"HEART_FAILURE","CHF":"HEART_FAILURE",
    "GI":"GI_BLEEDING","BLEED":"GI_BLEEDING",
    "DIAB":"DIABETES_COMPLICATION","DKA":"DIABETES_COMPLICATION",
    "COPD":"COPD_EXACERBATION","OTHER":"OTHER","NONE":"OTHER",
}

def parse_class(raw):
    if not raw: return "INVALID","empty"
    text = raw.strip().upper()
    for cls in sorted(ALLOWED_CLASSES, key=len, reverse=True):
        if cls in text[:80]: return cls,"direct"
    for k,v in FUZZY_MAP.items():
        if k in text[:80]: return v,"fuzzy"
    return "INVALID","invalid"

# ============================================================
# Model
# ============================================================
print("\n[1/4] Model yukleme...")
t0 = time.time()
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", local_files_only=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", quantization_config=bnb,
    device_map="auto", torch_dtype=torch.float16, local_files_only=True)
model.eval()
print(f"   Yuklendi {time.time()-t0:.1f}s, GPU {torch.cuda.memory_allocated()/1024**2:.0f}MB")

MAX_LEN = 4096

def inference(query):
    prompt = SABIT + TT.format(input=query)
    msgs = [{"role":"user","content":prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tok.eos_token_id)
    raw = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return raw, inputs.input_ids.shape[1]

# ============================================================
# Veri ve inference
# ============================================================
print("\n[2/4] Val seti...")
df = pd.read_parquet(SPLIT_YOL)
val_df = df[df["split"]=="val"].reset_index(drop=True)
print(f"   Val: {len(val_df):,}")

print("\n[3/4] Inference (checkpoint/resume)...")
cp_yol = FS_DIR / "fewshot_v4_3shot_ultracompact_val_checkpoint.csv"
final_yol = FS_DIR / "fewshot_v4_3shot_ultracompact_val.csv"

if cp_yol.exists():
    cp = pd.read_csv(cp_yol)
    islenen = set(cp["note_id"].astype(str))
    kayitlar = cp.to_dict("records")
    print(f"   Resume: {len(islenen)} kayit zaten var.")
else:
    islenen = set()
    kayitlar = []

t_baslangic = time.time()
sayac_yeni = 0

for i, row in val_df.iterrows():
    nid = str(row["note_id"])
    if nid in islenen: continue
    # NaN guvenligi
    v4_full = row["triage_input_v4_timeaware_masked"]
    if not isinstance(v4_full, str):
        v4_full = ""
    inp = compact_v4_query(v4_full)

    t0 = time.time()
    raw, in_n = inference(inp)
    sure = time.time() - t0
    parsed, kat = parse_class(raw)
    kayitlar.append({
        "note_id": nid, "subject_id": int(row["subject_id"]),
        "hadm_id": int(row["hadm_id"]), "label": row["label"],
        "raw_output": raw, "parsed_pred": parsed,
        "parse_kategori": kat, "in_tokens": int(in_n),
        "sure_s": round(sure, 3),
        "has_think": ("<think>" in raw or "</think>" in raw),
        "leakage_severity_final": str(row.get("strict_current_leakage_severity_final","UNK")),
    })
    sayac_yeni += 1
    if len(kayitlar) % 25 == 0:
        pd.DataFrame(kayitlar).to_csv(cp_yol, index=False)
        gecen = time.time() - t_baslangic
        ort = gecen / sayac_yeni if sayac_yeni > 0 else 0
        kalan = (len(val_df) - len(kayitlar)) * ort
        print(f"   [{len(kayitlar):>3}/{len(val_df)}] gecen {gecen/60:.1f}dk ort {ort:.2f}s kalan ~{kalan/60:.1f}dk")

df_son = pd.DataFrame(kayitlar)
df_son.to_csv(final_yol, index=False)
df_son.to_csv(cp_yol, index=False)
toplam = time.time() - t_baslangic
print(f"   BITTI. {len(kayitlar)} kayit, {toplam/60:.1f}dk")

# ============================================================
# AUDIT
# ============================================================
print("\n[4/4] V4 3-SHOT ULTRA VAL AUDIT")
print("=" * 70)
n = len(df_son)
ks = df_son["parse_kategori"].value_counts().to_dict()
direct = ks.get("direct", 0); fuzzy = ks.get("fuzzy", 0)
invalid = ks.get("invalid", 0) + ks.get("empty", 0)
print(f"\nParser: direct={direct} ({100*direct/n:.1f}%) fuzzy={fuzzy} invalid={invalid} ({100*invalid/n:.1f}%)")

# Think + token istatistik
think_count = df_son["has_think"].sum() if "has_think" in df_son.columns else 0
print(f"Think output: {think_count}/{n} ({100*think_count/n:.1f}%)")
print(f"Input token avg/max: {df_son['in_tokens'].mean():.0f} / {df_son['in_tokens'].max()}")

y_true = df_son["label"].values
y_pred = df_son["parsed_pred"].values
valid = y_pred != "INVALID"
acc_full = accuracy_score(y_true, y_pred)
mf1_full = f1_score(y_true, y_pred, average="macro", zero_division=0)
acc_v = accuracy_score(y_true[valid], y_pred[valid]) if valid.sum() else 0.0
mf1_v = f1_score(y_true[valid], y_pred[valid], average="macro", zero_division=0) if valid.sum() else 0.0

print(f"\nV4 3-SHOT ULTRA Val:")
print(f"   Full        n={n}    acc={acc_full:.4f} macroF1={mf1_full:.4f}")
print(f"   Valid only  n={valid.sum()}  acc={acc_v:.4f} macroF1={mf1_v:.4f}")

# Per-class
labels = sorted(set(y_true) | (set(y_pred) - {"INVALID"}))
p, r, f, supp = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
print(f"\nPer-class:")
for cls, pi, ri, fi, si in zip(labels, p, r, f, supp):
    print(f"   {cls:<25s} P={pi:.3f} R={ri:.3f} F1={fi:.3f} N={si}")

# Bias
print(f"\nTahmin dagilimi (bias, ozellikle OTHER ve HEART_FAILURE):")
for cls, sayi in df_son["parsed_pred"].value_counts().items():
    print(f"   {cls:<25s} {sayi:>4} ({100*sayi/n:>5.1f}%)")

# Karsilastirma
other_pct = 100*(df_son['parsed_pred']=='OTHER').sum()/n
hf_pct = 100*(df_son['parsed_pred']=='HEART_FAILURE').sum()/n
print(f"\nKARSILASTIRMA (val):")
print(f"   {'':<32s} {'Acc':>8s} {'MacroF1':>10s} {'OTHER%':>8s} {'HF%':>7s}")
print(f"   {'V4 zero-shot':<32s} {0.4818:>8.4f} {0.4367:>10.4f} {15.8:>7.1f}% {20.2:>6.1f}%")
print(f"   {'V4 1-shot CLEAN':<32s} {0.4727:>8.4f} {0.4479:>10.4f} {'?':>7s} {'?':>6s}")
print(f"   {'V4 3-shot ULTRA (this)':<32s} {acc_full:>8.4f} {mf1_full:>10.4f} {other_pct:>7.1f}% {hf_pct:>6.1f}%")

# Summary
ozet = pd.DataFrame({
    "subset":["full","valid_only"],
    "n":[n, valid.sum()],
    "accuracy":[acc_full, acc_v],
    "macro_f1":[mf1_full, mf1_v],
})
ozet.to_csv(FS_DIR / "fewshot_v4_3shot_ultracompact_val_results_summary.csv", index=False)

print("\n" + "=" * 70)
print("BLOK 14C-V4-3SHOT-ULTRA-VAL TAMAMLANDI - test'e otomatik gecis YOK")
print("=" * 70)
