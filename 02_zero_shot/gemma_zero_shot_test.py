# =====================================================================
# BLOK G04 — Gemma-4-E4B-it ZERO-SHOT TEST (1650) — Colab A100
# =====================================================================
# Local Qwen ile BIREBIR AYNI veri/prompt/parser:
#   - discharge_split.parquet  (ayni dosya, sen upload ettin)
#   - split=="test" -> 1650 (Qwen test'iyle ayni satirlar)
#   - triage_input_v4_timeaware_masked (ayni kolon)
#   - PROMPT_TEMPLATE + parse_class + FUZZY_MAP (Qwen ile birebir)
# TEK FARK: model = Gemma.
#
# QUANT secimi: asagidaki QUANT degiskeni "bf16" veya "4bit".
#   ONCE bf16 calistir -> sonuc al. Sonra QUANT="4bit" yapip tekrar calistir.
# Her ikisi ayri cikti dosyasina yazar, checkpoint/resume var.
# Sure: Gemma hizli -> bf16 test ~5-7 dk, 4-bit ~7-9 dk.
# =====================================================================
import torch, time, os, gc
import pandas as pd
from pathlib import Path
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

# >>>>>>>>>>>>>>>>>>>>>>  BURAYI DEGISTIR  <<<<<<<<<<<<<<<<<<<<<<
QUANT = "bf16"      # ONCE "bf16", sonra "4bit"
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
GEMMA_ID = "google/gemma-4-E4B-it"

# ---------- Onceki modeli temizle, yeniden yukle ----------
if "g_model" in dir():
    try: del g_model
    except Exception: pass
gc.collect(); torch.cuda.empty_cache()
print(f"GPU temizlik sonrasi: {torch.cuda.memory_allocated()/1e9:.2f} GB")

print(f"\n[1/2] Gemma yukleniyor (QUANT={QUANT})...")
g_proc = AutoProcessor.from_pretrained(GEMMA_ID)
g_tok = g_proc.tokenizer if hasattr(g_proc, "tokenizer") else g_proc
if g_tok.pad_token is None: g_tok.pad_token = g_tok.eos_token
g_tok.padding_side = "left"

t0 = time.time()
if QUANT == "4bit":
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    g_model = AutoModelForCausalLM.from_pretrained(GEMMA_ID, quantization_config=bnb,
        device_map="auto", dtype=torch.bfloat16)
else:  # bf16
    g_model = AutoModelForCausalLM.from_pretrained(GEMMA_ID, dtype=torch.bfloat16,
        device_map="auto")
g_model.eval()
print(f"   Yuklendi {time.time()-t0:.0f}s, GPU bellek: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ---------- PROMPT + parser (Qwen ile BIREBIR ayni) ----------
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

# ---------- TEST inference (checkpoint/resume) ----------
test_df = df[df["split"]=="test"].reset_index(drop=True)
print(f"\n[2/2] Gemma {QUANT} zero-shot TEST: {len(test_df)} ornek")
OUT_DIR=Path("/content/gemma_out"); OUT_DIR.mkdir(exist_ok=True)
cp_yol    = OUT_DIR / f"gemma_{QUANT}_zeroshot_test_checkpoint.csv"
final_yol = OUT_DIR / f"gemma_{QUANT}_zeroshot_test.csv"

if cp_yol.exists():
    cp=pd.read_csv(cp_yol); islenen=set(cp["note_id"].astype(str)); kayitlar=cp.to_dict("records")
    print(f"RESUME: {len(islenen)} kayit var.")
else:
    islenen=set(); kayitlar=[]
rows=test_df.to_dict("records")
kalan=[r for r in rows if str(r["note_id"]) not in islenen]
print(f"Islenecek: {len(kalan)}")

t_bas=time.time(); yeni=0
for b in range(0, len(kalan), BATCH_SIZE):
    grup=kalan[b:b+BATCH_SIZE]
    xs=[r[V4COL] if isinstance(r[V4COL],str) else "" for r in grup]
    raws,in_lens=inference_batch(xs)
    for r,raw,in_n in zip(grup,raws,in_lens):
        parsed,kat=parse_class(raw)
        kayitlar.append({"note_id":str(r["note_id"]),"subject_id":int(r["subject_id"]),
            "hadm_id":int(r["hadm_id"]),"label":r["label"],"raw_output":raw,
            "parsed_pred":parsed,"parse_kategori":kat,"in_tokens":in_n})
    yeni+=len(grup)
    pd.DataFrame(kayitlar).to_csv(cp_yol, index=False)
    gc.collect(); torch.cuda.empty_cache()
    if (b//BATCH_SIZE) % 5 == 0:
        gecen=time.time()-t_bas; ort=gecen/yeni
        print(f"   [{len(kayitlar):>4}/{len(rows)}] gecen {gecen/60:.1f}dk kalan ~{(len(kalan)-yeni)*ort/60:.1f}dk")

df_son=pd.DataFrame(kayitlar); df_son.to_csv(final_yol, index=False)
print(f"\nBITTI. {len(kayitlar)} kayit, bu oturum {(time.time()-t_bas)/60:.1f}dk")

# ---------- AUDIT ----------
print("\n" + "="*70 + f"\nGEMMA {QUANT.upper()} ZERO-SHOT TEST AUDIT\n" + "="*70)
n=len(df_son)
ks=df_son["parse_kategori"].value_counts().to_dict()
direct=ks.get("direct",0); invalid=ks.get("invalid",0)+ks.get("empty",0)
print(f"Parser: direct={direct} ({100*direct/n:.1f}%) invalid={invalid} ({100*invalid/n:.1f}%)")
print("✅ Truncation YOK" if trunc_sayac["kesilen"]==0 else f"🔴 {trunc_sayac['kesilen']} KESILDI!")

y_true=df_son["label"].values; y_pred=df_son["parsed_pred"].values
acc=accuracy_score(y_true,y_pred)
mf1=f1_score(y_true,y_pred,average="macro",zero_division=0)
wf1=f1_score(y_true,y_pred,average="weighted",zero_division=0)
print(f"\nGEMMA {QUANT} zero-shot TEST: acc={acc:.4f} macroF1={mf1:.4f} wF1={wf1:.4f}")

labels=sorted(set(y_true)|(set(y_pred)-{"INVALID"}))
p,r,f,supp=precision_recall_fscore_support(y_true,y_pred,labels=labels,zero_division=0)
print("\nPer-class:")
for cls,pi,ri,fi,si in zip(labels,p,r,f,supp):
    print(f"   {cls:<25s} P={pi:.3f} R={ri:.3f} F1={fi:.3f} N={si}")

print(f"\nKARSILASTIRMA (TEST 1650, ayni veri):")
print(f"   {'':<32s} {'Acc':>8s} {'MacroF1':>9s}")
print(f"   {'Qwen3.5-4B zero-shot (local)':<32s} {0.5152:>8.4f} {0.4681:>9.4f}")
print(f"   {'Qwen3.5-4B QLoRA-S3 (local)':<32s} {0.6345:>8.4f} {0.6220:>9.4f}")
print(f"   {'Gemma-4-E4B zero-shot ('+QUANT+')':<32s} {acc:>8.4f} {mf1:>9.4f}")
print(f"\n✅ BLOK G04 ({QUANT}) TAMAM. Cikti: {final_yol}")
print("   !! INDIR (files.download) — Colab kapaninca silinir!")
