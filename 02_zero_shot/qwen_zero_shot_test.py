"""Blok 13B-V4-TEST: V4 (TIME-AWARE) zero-shot TEST. Final V4 sonucu."""
import pandas as pd
import numpy as np
import torch
import time
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

OUTPUT_DIR = Path(r"./workdir/output")
SPLIT_YOL  = OUTPUT_DIR / "discharge_split.parquet"
ZS_DIR     = OUTPUT_DIR / "zeroshot"

print("=" * 70)
print("BLOK 13B-V4-TEST — V4 TIME-AWARE Zero-shot TEST (FINAL)")
print("=" * 70)

ALLOWED_CLASSES = [
    "SEPSIS","STROKE","ATRIAL_FIBRILLATION","AKI","PNEUMONIA","UTI",
    "HEART_FAILURE","GI_BLEEDING","DIABETES_COMPLICATION","COPD_EXACERBATION","OTHER"
]

PROMPT_TEMPLATE = """You are a clinical NLP system. Given a patient's triage information collected at first contact (chief complaint, symptoms, brief history, medications, allergies, vital signs), classify the most likely PRIMARY reason for admission into ONE of these 11 categories:

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

TRIAGE NOTE:
{input}

PRIMARY DIAGNOSIS:"""

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

# Model
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

def inference(input_text):
    prompt = PROMPT_TEMPLATE.format(input=input_text)
    msgs = [{"role":"user","content":prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tok.eos_token_id)
    raw = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return raw, inputs.input_ids.shape[1]

# Veri
print("\n[2/4] V4 test seti...")
df = pd.read_parquet(SPLIT_YOL)
test_df = df[df["split"]=="test"].reset_index(drop=True)
print(f"   Test: {len(test_df):,}")

# Inference
print("\n[3/4] V4 TEST inference (checkpoint/resume)...")
cp_yol  = ZS_DIR / "zeroshot_v4_test_checkpoint.csv"
final_yol = ZS_DIR / "zeroshot_v4_test.csv"

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

for i, row in test_df.iterrows():
    nid = str(row["note_id"])
    if nid in islenen: continue
    inp = row["triage_input_v4_timeaware_masked"] if pd.notna(row["triage_input_v4_timeaware_masked"]) else ""
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
        "leakage_severity_final": str(row.get("strict_current_leakage_severity_final","UNK")),
        "has_own_label_term_v3": bool(row.get("has_own_label_term_v3", False)),
    })
    sayac_yeni += 1
    if len(kayitlar) % 50 == 0:
        pd.DataFrame(kayitlar).to_csv(cp_yol, index=False)
        gecen = time.time() - t_baslangic
        ort = gecen / sayac_yeni if sayac_yeni > 0 else 0
        kalan = (len(test_df) - len(kayitlar)) * ort
        print(f"   [{len(kayitlar):>4}/{len(test_df)}] gecen {gecen/60:.1f}dk ort {ort:.2f}s kalan ~{kalan/60:.1f}dk")

df_son = pd.DataFrame(kayitlar)
df_son.to_csv(final_yol, index=False)
df_son.to_csv(cp_yol, index=False)
toplam = time.time() - t_baslangic
print(f"   BITTI. {len(kayitlar)} kayit, {toplam/60:.1f}dk")

# AUDIT
print("\n[4/4] V4 TEST FINAL AUDIT")
print("=" * 70)
n = len(df_son)
ks = df_son["parse_kategori"].value_counts().to_dict()
direct = ks.get("direct", 0); fuzzy = ks.get("fuzzy", 0)
invalid = ks.get("invalid", 0) + ks.get("empty", 0)
print(f"\nParser: direct={direct} ({100*direct/n:.1f}%) fuzzy={fuzzy} invalid={invalid} ({100*invalid/n:.1f}%)")

y_true = df_son["label"].values
y_pred = df_son["parsed_pred"].values
valid = y_pred != "INVALID"
clean_v3_subset = (~df_son["has_own_label_term_v3"]).values
timeaware_clean = df_son["leakage_severity_final"].isin(["CLEAN","HISTORICAL_ALLOWED"]).values

acc_full = accuracy_score(y_true, y_pred)
mf1_full = f1_score(y_true, y_pred, average="macro", zero_division=0)
wf1_full = f1_score(y_true, y_pred, average="weighted", zero_division=0)
acc_v = accuracy_score(y_true[valid], y_pred[valid]) if valid.sum() else 0.0
mf1_v = f1_score(y_true[valid], y_pred[valid], average="macro", zero_division=0) if valid.sum() else 0.0
wf1_v = f1_score(y_true[valid], y_pred[valid], average="weighted", zero_division=0) if valid.sum() else 0.0

# Time-aware clean alt-set (testte zaten 1650/1650)
acc_ta = accuracy_score(y_true[timeaware_clean], y_pred[timeaware_clean]) if timeaware_clean.sum() else 0.0
mf1_ta = f1_score(y_true[timeaware_clean], y_pred[timeaware_clean], average="macro", zero_division=0) if timeaware_clean.sum() else 0.0

# V3 keyword-clean alt-set (V4 ile karsilastirma icin)
acc_kw = accuracy_score(y_true[clean_v3_subset], y_pred[clean_v3_subset]) if clean_v3_subset.sum() else 0.0
mf1_kw = f1_score(y_true[clean_v3_subset], y_pred[clean_v3_subset], average="macro", zero_division=0) if clean_v3_subset.sum() else 0.0

print(f"\nV4 TEST METRIKLERI:")
print(f"   {'Subset':<26s} {'N':>5s} {'Acc':>8s} {'MacroF1':>10s} {'WF1':>8s}")
print("-" * 65)
print(f"   {'Full':<26s} {n:>5} {acc_full:>8.4f} {mf1_full:>10.4f} {wf1_full:>8.4f}")
print(f"   {'Valid only':<26s} {valid.sum():>5} {acc_v:>8.4f} {mf1_v:>10.4f} {wf1_v:>8.4f}")
print(f"   {'Time-aware clean':<26s} {timeaware_clean.sum():>5} {acc_ta:>8.4f} {mf1_ta:>10.4f}")
print(f"   {'Keyword-clean (V3 ref)':<26s} {clean_v3_subset.sum():>5} {acc_kw:>8.4f} {mf1_kw:>10.4f}")

# Per-class
labels = sorted(set(y_true) | (set(y_pred) - {"INVALID"}))
p, r, f, supp = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
print(f"\nPer-class:")
for cls, pi, ri, fi, si in zip(labels, p, r, f, supp):
    print(f"   {cls:<25s} P={pi:.3f} R={ri:.3f} F1={fi:.3f} N={si}")

# Confusion matrix
labels_full = ALLOWED_CLASSES + ["INVALID"]
cm = confusion_matrix(y_true, y_pred, labels=labels_full)
ks_short = {"SEPSIS":"SEP","STROKE":"STR","ATRIAL_FIBRILLATION":"AFB","AKI":"AKI",
    "PNEUMONIA":"PNE","UTI":"UTI","HEART_FAILURE":"HF","GI_BLEEDING":"GIB",
    "DIABETES_COMPLICATION":"DIA","COPD_EXACERBATION":"CPD","OTHER":"OTH","INVALID":"INV"}
print(f"\nConfusion matrix:")
print("gercek\\pred  " + " ".join(f"{ks_short[c]:>4}" for c in labels_full))
for i, g in enumerate(labels_full):
    if cm[i].sum() == 0: continue
    print(f"{ks_short[g]:<13s}" + " ".join(f"{cm[i][j]:>4}" for j in range(len(labels_full))))

# Bias
print(f"\nTahmin dagilimi (bias):")
for cls, sayi in df_son["parsed_pred"].value_counts().items():
    print(f"   {cls:<25s} {sayi:>5} ({100*sayi/n:>5.1f}%)")

# V3 vs V4 (test) karsilastirma
print(f"\nV3 vs V4 (test):")
v3_sum_yol = ZS_DIR / "zeroshot_v3_test_results_summary.csv"
if v3_sum_yol.exists():
    v3 = pd.read_csv(v3_sum_yol)
    v3_full = float(v3[v3["subset"]=="full"]["accuracy"].iloc[0])
    v3_mf1 = float(v3[v3["subset"]=="full"]["macro_f1"].iloc[0])
    v3_kw = float(v3[v3["subset"]=="clean_own"]["accuracy"].iloc[0])
    v3_kw_mf1 = float(v3[v3["subset"]=="clean_own"]["macro_f1"].iloc[0])
    print(f"   {'':<26s} {'V3':>8s} {'V4':>8s} {'Delta':>8s}")
    print(f"   {'Full Acc':<26s} {v3_full:>8.4f} {acc_full:>8.4f} {acc_full-v3_full:>+8.4f}")
    print(f"   {'Full MacroF1':<26s} {v3_mf1:>8.4f} {mf1_full:>8.4f} {mf1_full-v3_mf1:>+8.4f}")
    print(f"   {'Keyword-clean Acc':<26s} {v3_kw:>8.4f} {acc_kw:>8.4f} {acc_kw-v3_kw:>+8.4f}")
    print(f"   {'Keyword-clean MacroF1':<26s} {v3_kw_mf1:>8.4f} {mf1_kw:>8.4f} {mf1_kw-v3_kw_mf1:>+8.4f}")

# Val-test consistency
print(f"\nV4 Val vs Test consistency:")
v4_val_yol = ZS_DIR / "zeroshot_v4_val_results_summary.csv"
if v4_val_yol.exists():
    val_s = pd.read_csv(v4_val_yol)
    val_full = float(val_s[val_s["subset"]=="full"]["accuracy"].iloc[0])
    val_mf1 = float(val_s[val_s["subset"]=="full"]["macro_f1"].iloc[0])
    print(f"   Val Acc  : {val_full:.4f} (test: {acc_full:.4f}, delta: {acc_full-val_full:+.4f})")
    print(f"   Val MF1  : {val_mf1:.4f} (test: {mf1_full:.4f}, delta: {mf1_full-val_mf1:+.4f})")

# Summary kaydet
ozet = pd.DataFrame({
    "subset": ["full","valid_only","timeaware_clean","keyword_clean"],
    "n": [n, valid.sum(), timeaware_clean.sum(), clean_v3_subset.sum()],
    "accuracy": [acc_full, acc_v, acc_ta, acc_kw],
    "macro_f1": [mf1_full, mf1_v, mf1_ta, mf1_kw],
    "weighted_f1": [wf1_full, wf1_v, 0.0, 0.0],
})
ozet.to_csv(ZS_DIR / "zeroshot_v4_test_results_summary.csv", index=False)

audit_yol = ZS_DIR / "zeroshot_v4_test_audit.txt"
with open(audit_yol, "w", encoding="utf-8") as fh:
    fh.write("BLOK 13B-V4-TEST FINAL AUDIT (TIME-AWARE)\n")
    fh.write("=" * 70 + "\n")
    fh.write(f"Test n: {n}\n")
    fh.write(f"Parser: direct={direct}, fuzzy={fuzzy}, invalid={invalid}\n\n")
    fh.write(f"Full          : acc={acc_full:.4f}, macroF1={mf1_full:.4f}, wF1={wf1_full:.4f}\n")
    fh.write(f"Valid only    : acc={acc_v:.4f}, macroF1={mf1_v:.4f}\n")
    fh.write(f"Time-aware clean: acc={acc_ta:.4f}, macroF1={mf1_ta:.4f}, n={timeaware_clean.sum()}\n")
    fh.write(f"Keyword-clean : acc={acc_kw:.4f}, macroF1={mf1_kw:.4f}, n={clean_v3_subset.sum()}\n\n")
    fh.write("Per-class:\n")
    for cls, pi, ri, fi, si in zip(labels, p, r, f, supp):
        fh.write(f"   {cls:<25s} P={pi:.3f} R={ri:.3f} F1={fi:.3f} N={si}\n")

print("\n" + "=" * 70)
print("V4 TEST TAMAMLANDI - FINAL")
print("=" * 70)
