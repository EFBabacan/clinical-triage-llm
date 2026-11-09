# =====================================================================
# BLOK G02 — Gemma-4-E4B-it ZERO-SHOT val (550) — Colab A100
# =====================================================================
# Qwen zero-shot (__run_blok13b_v4_val.py) ile BIREBIR ayni:
#   ayni PROMPT_TEMPLATE, ayni parse_class+FUZZY_MAP, ayni veri/kolon/val seti.
#   TEK FARK: model = Gemma (g_model/g_proc/g_tok), Qwen degil.
# -> Adil "ikinci model" karsilastirmasi.
# Onceki bloklardan (G01): g_model, g_proc, g_tok
# Blok 02'den: df, CLASSES, V4COL  (yoksa asagida parquet'ten okunur)
# Sure: zero-shot prompt kisa (~500 tok) -> Gemma'da hizli, ~10-20dk val.
# =====================================================================
import torch, time, os, gc
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ALLOWED_CLASSES = [
    "SEPSIS","STROKE","ATRIAL_FIBRILLATION","AKI","PNEUMONIA","UTI",
    "HEART_FAILURE","GI_BLEEDING","DIABETES_COMPLICATION","COPD_EXACERBATION","OTHER"
]
V4COL = "triage_input_v4_timeaware_masked"

# Veri (df yoksa oku)
try:
    df
except NameError:
    df = pd.read_parquet("discharge_split.parquet")

# ---------- PROMPT (Qwen zero-shot ile BIREBIR ayni) ----------
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

# ---------- Parser (Qwen ile BIREBIR ayni) ----------
FUZZY_MAP = {
    "SEPS":"SEPSIS","SEPT":"SEPSIS","PNEUM":"PNEUMONIA","STROK":"STROKE","CVA":"STROKE",
    "ATRIAL":"ATRIAL_FIBRILLATION","AFIB":"ATRIAL_FIBRILLATION","FIBRIL":"ATRIAL_FIBRILLATION",
    "AKI":"AKI","KIDNEY":"AKI","RENAL":"AKI","UTI":"UTI","URIN":"UTI",
    "HEART":"HEART_FAILURE","CHF":"HEART_FAILURE","GI":"GI_BLEEDING","BLEED":"GI_BLEEDING",
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

# ---------- Gemma inference (chat template + enable_thinking=False + greedy) ----------
BATCH_SIZE = 16   # zero-shot prompt kisa -> Gemma'da 16 rahat. OOM olursa 8.
MAX_LEN    = 4096
g_tok.padding_side = "left"
if g_tok.pad_token is None:
    g_tok.pad_token = g_tok.eos_token
trunc_sayac = {"kesilen": 0}

def build_text(input_text):
    prompt = PROMPT_TEMPLATE.format(input=input_text)
    try:
        return g_proc.apply_chat_template([{"role":"user","content":prompt}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return g_proc.apply_chat_template([{"role":"user","content":prompt}],
                    tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def inference_batch(inputs_list):
    texts = [build_text(x) for x in inputs_list]
    for t in texts:
        if len(g_tok(t).input_ids) > MAX_LEN: trunc_sayac["kesilen"] += 1
    enc = g_tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(g_model.device)
    out = g_model.generate(**enc, max_new_tokens=20, do_sample=False, pad_token_id=g_tok.eos_token_id)
    gen = out[:, enc.input_ids.shape[1]:]
    raws = [g_tok.decode(gen[j], skip_special_tokens=True).strip() for j in range(len(inputs_list))]
    in_lens = [int(enc.attention_mask[j].sum().item()) for j in range(len(inputs_list))]
    return raws, in_lens

# ---------- Veri: VAL ----------
val_df = df[df["split"]=="val"].reset_index(drop=True)
print(f"Gemma zero-shot VAL: {len(val_df)} ornek, BATCH_SIZE={BATCH_SIZE}")

# ---------- Checkpoint/resume ----------
OUT_DIR = Path("/content/gemma_out"); OUT_DIR.mkdir(exist_ok=True)
cp_yol    = OUT_DIR / "gemma_zeroshot_val_checkpoint.csv"
final_yol = OUT_DIR / "gemma_zeroshot_val.csv"
if cp_yol.exists():
    cp = pd.read_csv(cp_yol); islenen = set(cp["note_id"].astype(str)); kayitlar = cp.to_dict("records")
    print(f"RESUME: {len(islenen)} kayit var.")
else:
    islenen = set(); kayitlar = []

rows = val_df.to_dict("records")
kalan = [r for r in rows if str(r["note_id"]) not in islenen]
print(f"Islenecek: {len(kalan)}")

t_bas = time.time(); yeni = 0
for b in range(0, len(kalan), BATCH_SIZE):
    grup = kalan[b:b+BATCH_SIZE]
    inps = [r[V4COL] if isinstance(r[V4COL], str) else "" for r in grup]
    raws, in_lens = inference_batch(inps)
    for r, raw, in_n in zip(grup, raws, in_lens):
        parsed, kat = parse_class(raw)
        kayitlar.append({
            "note_id": str(r["note_id"]), "subject_id": int(r["subject_id"]),
            "hadm_id": int(r["hadm_id"]), "label": r["label"],
            "raw_output": raw, "parsed_pred": parsed, "parse_kategori": kat,
            "in_tokens": in_n, "has_think": ("<think>" in raw or "channel" in raw.lower()),
        })
    yeni += len(grup)
    pd.DataFrame(kayitlar).to_csv(cp_yol, index=False)
    gc.collect(); torch.cuda.empty_cache()
    gecen = time.time()-t_bas; ort = gecen/yeni
    print(f"   [{len(kayitlar):>3}/{len(rows)}] gecen {gecen/60:.1f}dk ort {ort:.2f}s kalan ~{(len(kalan)-yeni)*ort/60:.1f}dk")

df_son = pd.DataFrame(kayitlar)
df_son.to_csv(final_yol, index=False)
print(f"\nBITTI. {len(kayitlar)} kayit, {(time.time()-t_bas)/60:.1f}dk")

# ---------- AUDIT ----------
print("\n" + "="*70 + "\nGEMMA ZERO-SHOT VAL AUDIT\n" + "="*70)
n = len(df_son)
ks = df_son["parse_kategori"].value_counts().to_dict()
direct = ks.get("direct",0); fuzzy = ks.get("fuzzy",0); invalid = ks.get("invalid",0)+ks.get("empty",0)
print(f"Parser: direct={direct} ({100*direct/n:.1f}%) fuzzy={fuzzy} invalid={invalid} ({100*invalid/n:.1f}%)")
print(f"Think/channel output: {df_son['has_think'].sum()}/{n}")
print(f"Input token avg/max: {df_son['in_tokens'].mean():.0f} / {df_son['in_tokens'].max()} (MAX_LEN={MAX_LEN})")
print("✅ Truncation YOK" if trunc_sayac["kesilen"]==0 else f"🔴 {trunc_sayac['kesilen']} ornek KESILDI!")

y_true = df_son["label"].values; y_pred = df_son["parsed_pred"].values
valid = y_pred != "INVALID"
acc_full = accuracy_score(y_true, y_pred); mf1_full = f1_score(y_true, y_pred, average="macro", zero_division=0)
acc_v = accuracy_score(y_true[valid], y_pred[valid]) if valid.sum() else 0.0
mf1_v = f1_score(y_true[valid], y_pred[valid], average="macro", zero_division=0) if valid.sum() else 0.0
print(f"\nGEMMA zero-shot Val:")
print(f"   Full       n={n}   acc={acc_full:.4f} macroF1={mf1_full:.4f}")
print(f"   Valid only n={valid.sum()} acc={acc_v:.4f} macroF1={mf1_v:.4f}")

labels = sorted(set(y_true) | (set(y_pred)-{"INVALID"}))
p,r,f,supp = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
print("\nPer-class:")
for cls,pi,ri,fi,si in zip(labels,p,r,f,supp):
    print(f"   {cls:<25s} P={pi:.3f} R={ri:.3f} F1={fi:.3f} N={si}")

print(f"\nKARSILASTIRMA (val, IKI MODEL):")
print(f"   {'':<28s} {'Acc':>8s} {'MacroF1':>9s}")
print(f"   {'Qwen3.5-4B zero-shot':<28s} {0.4818:>8.4f} {0.4367:>9.4f}")
print(f"   {'Qwen3.5-4B 3-shot FULL':<28s} {0.5182:>8.4f} {0.4781:>9.4f}")
print(f"   {'Gemma-4-E4B zero-shot (this)':<28s} {acc_full:>8.4f} {mf1_full:>9.4f}")
print("\n✅ BLOK G02 TAMAM. Cikti: /content/gemma_out/gemma_zeroshot_val.csv")
