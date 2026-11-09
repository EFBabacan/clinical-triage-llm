# =====================================================================
# BLOK G03 — Gemma-4-E4B-it 4-BIT zero-shot val (550) — Colab A100
# =====================================================================
# "Ikisini de dene": G01 full bf16'ydi, bu 4-bit NF4.
# Amac: (1) Qwen ile ESIT kosul (her ikisi 4-bit), (2) bellek/hiz/accuracy karsilastirmasi.
# Ayni PROMPT/parser/veri/val -> sadece model 4-bit yuklenir.
#
# DIKKAT: Gemma PLE mimarisinde 4-bit embedding'i quantize ETMEZ; bellek dususu
# Qwen'deki kadar dramatik olmayabilir. Accuracy bf16'ya cok yakin beklenir.
#
# Bu blok KENDI BASINA calisir (G02'ye bagimli degil) ama Blok 02'den df/V4COL gelmeli.
# Eski full-bf16 g_model'i bellekten silip 4-bit yukler (cakismasin).
# =====================================================================
import torch, time, os, gc
import pandas as pd
from pathlib import Path
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
GEMMA_ID = "google/gemma-4-E4B-it"

# ---------- Eski full-bf16 modeli bellekten temizle ----------
for v in ["g_model"]:
    if v in dir():
        try: del globals()[v]
        except Exception: pass
gc.collect(); torch.cuda.empty_cache()
print(f"Temizlik sonrasi GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ---------- 4-bit yukleme (Qwen ile ayni NF4 ayarlari) ----------
print("\n[1/2] Gemma 4-bit NF4 yukleniyor...")
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,   # Gemma bf16 tabanli (Qwen fp16'ydi; Gemma'da bf16 dogru)
    bnb_4bit_use_double_quant=True,
)
g_proc = AutoProcessor.from_pretrained(GEMMA_ID)
g_tok = g_proc.tokenizer if hasattr(g_proc, "tokenizer") else g_proc
if g_tok.pad_token is None: g_tok.pad_token = g_tok.eos_token
g_tok.padding_side = "left"

t0 = time.time()
g_model = AutoModelForCausalLM.from_pretrained(
    GEMMA_ID, quantization_config=bnb, device_map="auto", dtype=torch.bfloat16,
)
g_model.eval()
print(f"   Yuklendi {time.time()-t0:.0f}s")
print(f"   GPU bellek (4-bit): {torch.cuda.memory_allocated()/1e9:.2f} GB  "
      f"(full bf16 ~16GB idi — karsilastir)")

# ---------- PROMPT + parser (G02 / Qwen ile BIREBIR ayni) ----------
ALLOWED_CLASSES = ["SEPSIS","STROKE","ATRIAL_FIBRILLATION","AKI","PNEUMONIA","UTI",
    "HEART_FAILURE","GI_BLEEDING","DIABETES_COMPLICATION","COPD_EXACERBATION","OTHER"]
V4COL = "triage_input_v4_timeaware_masked"
try: df
except NameError: df = pd.read_parquet("discharge_split.parquet")

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

BATCH_SIZE = 16; MAX_LEN = 4096; trunc_sayac={"kesilen":0}
def build_text(x):
    p = PROMPT_TEMPLATE.format(input=x)
    try:
        return g_proc.apply_chat_template([{"role":"user","content":p}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return g_proc.apply_chat_template([{"role":"user","content":p}],
                    tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def inference_batch(xs):
    texts=[build_text(x) for x in xs]
    for t in texts:
        if len(g_tok(t).input_ids) > MAX_LEN: trunc_sayac["kesilen"]+=1
    enc=g_tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(g_model.device)
    out=g_model.generate(**enc, max_new_tokens=20, do_sample=False, pad_token_id=g_tok.eos_token_id)
    gen=out[:, enc.input_ids.shape[1]:]
    raws=[g_tok.decode(gen[j], skip_special_tokens=True).strip() for j in range(len(xs))]
    in_lens=[int(enc.attention_mask[j].sum().item()) for j in range(len(xs))]
    return raws, in_lens

# ---------- VAL inference ----------
val_df = df[df["split"]=="val"].reset_index(drop=True)
print(f"\n[2/2] Gemma 4-bit zero-shot VAL: {len(val_df)} ornek")
OUT_DIR=Path("/content/gemma_out"); OUT_DIR.mkdir(exist_ok=True)
final_yol=OUT_DIR/"gemma_4bit_zeroshot_val.csv"

kayitlar=[]; t_bas=time.time(); rows=val_df.to_dict("records")
for b in range(0, len(rows), BATCH_SIZE):
    grup=rows[b:b+BATCH_SIZE]
    xs=[r[V4COL] if isinstance(r[V4COL],str) else "" for r in grup]
    raws,in_lens=inference_batch(xs)
    for r,raw,in_n in zip(grup,raws,in_lens):
        parsed,kat=parse_class(raw)
        kayitlar.append({"note_id":str(r["note_id"]),"subject_id":int(r["subject_id"]),
            "hadm_id":int(r["hadm_id"]),"label":r["label"],"raw_output":raw,
            "parsed_pred":parsed,"parse_kategori":kat,"in_tokens":in_n})
    gc.collect(); torch.cuda.empty_cache()
df_son=pd.DataFrame(kayitlar); df_son.to_csv(final_yol, index=False)
sure_4bit=(time.time()-t_bas)
print(f"BITTI. {len(kayitlar)} kayit, {sure_4bit/60:.1f}dk")

# ---------- AUDIT ----------
print("\n" + "="*70 + "\nGEMMA 4-BIT ZERO-SHOT VAL AUDIT\n" + "="*70)
n=len(df_son)
ks=df_son["parse_kategori"].value_counts().to_dict()
direct=ks.get("direct",0); invalid=ks.get("invalid",0)+ks.get("empty",0)
print(f"Parser: direct={direct} ({100*direct/n:.1f}%) invalid={invalid} ({100*invalid/n:.1f}%)")
print("✅ Truncation YOK" if trunc_sayac["kesilen"]==0 else f"🔴 {trunc_sayac['kesilen']} KESILDI!")

y_true=df_son["label"].values; y_pred=df_son["parsed_pred"].values
acc=accuracy_score(y_true,y_pred); mf1=f1_score(y_true,y_pred,average="macro",zero_division=0)
print(f"\nGEMMA 4-bit zero-shot Val: acc={acc:.4f} macroF1={mf1:.4f}")

labels=sorted(set(y_true)|(set(y_pred)-{"INVALID"}))
p,r,f,supp=precision_recall_fscore_support(y_true,y_pred,labels=labels,zero_division=0)
print("\nPer-class:")
for cls,pi,ri,fi,si in zip(labels,p,r,f,supp):
    print(f"   {cls:<25s} P={pi:.3f} R={ri:.3f} F1={fi:.3f} N={si}")

print(f"\nKARSILASTIRMA (val):")
print(f"   {'':<32s} {'Acc':>8s} {'MacroF1':>9s} {'Sure':>8s}")
print(f"   {'Qwen3.5-4B zero-shot (4bit)':<32s} {0.4818:>8.4f} {0.4367:>9.4f} {'-':>8s}")
print(f"   {'Gemma-4-E4B zero-shot (bf16)':<32s} {0.5436:>8.4f} {0.5434:>9.4f} {'1.5dk':>8s}")
print(f"   {'Gemma-4-E4B zero-shot (4bit)':<32s} {acc:>8.4f} {mf1:>9.4f} {f'{sure_4bit/60:.1f}dk':>8s}")
print("\n>>> 4-bit vs bf16 Gemma farki: accuracy yakin mi? (PLE -> beklenen: cok yakin)")
print("✅ BLOK G03 TAMAM. Cikti: /content/gemma_out/gemma_4bit_zeroshot_val.csv")
