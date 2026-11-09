# PROVENANCE NOTE (added for the public release)
# This cell evaluates the model held in memory (`g_model`) in the current Colab session.
# The "optimized" Gemma result was produced by running gemma_qlora_train_optimized.py and then
# this script in the same session; its output (gemma_qlora_test_preds.csv) overwrote the earlier
# fair-adapter predictions. The fair row in the paper comes from colab_B5_gemma_qlora_eval.py.

# =====================================================================
# BLOK G06c — Gemma QLoRA-S3 EVAL (val + test) — Colab A100
# =====================================================================
# G06b'de egitilen adapter ile val(550)+test(1650) eval.
# Qwen QLoRA eval ile ayni mantik: instruction + input -> sinif, greedy, parser.
# Onceki bloklardan: g_model (G06b'den, PEFT-li), g_proc, g_tok, val_recs, test_recs.
#   (Oturum koptu ise: adapter'i diskten yukleyen alternatif asagida yorum satirinda.)
# Sure: ~10-15 dk.
# =====================================================================
import torch, time, json
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

ALLOWED_CLASSES = ["SEPSIS","STROKE","ATRIAL_FIBRILLATION","AKI","PNEUMONIA","UTI",
    "HEART_FAILURE","GI_BLEEDING","DIABETES_COMPLICATION","COPD_EXACERBATION","OTHER"]

# --- Oturum KOPTUYSA adapter'i diskten yukle (normalde gerek yok, G06b'den g_model var) ---
# from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
# from peft import PeftModel
# bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
# g_proc=AutoProcessor.from_pretrained("google/gemma-4-E4B-it")
# g_tok=g_proc.tokenizer
# base=AutoModelForCausalLM.from_pretrained("google/gemma-4-E4B-it",quantization_config=bnb,
#     device_map="auto",dtype=torch.bfloat16)
# g_model=PeftModel.from_pretrained(base,"/content/gemma_out/gemma_qlora_s3/adapter")

assert "g_model" in dir(), "g_model yok! G06b'yi calistir ya da ustteki adapter-yukle blogunu ac."
assert "test_recs" in dir() and "val_recs" in dir(), "veri yok! G06a calistir."

# inference moduna al
g_model.config.use_cache = True
g_model.eval()
g_tok.padding_side = "left"   # inference icin left padding
if g_tok.pad_token is None: g_tok.pad_token = g_tok.eos_token

INSTRUCTION = train_recs[0]["instruction"] if "train_recs" in dir() else val_recs[0].get("instruction",
    "Classify the following admission note into exactly one of these 11 classes: "
    "SEPSIS, STROKE, ATRIAL_FIBRILLATION, AKI, PNEUMONIA, UTI, HEART_FAILURE, "
    "GI_BLEEDING, DIABETES_COMPLICATION, COPD_EXACERBATION, OTHER. Output only the class name.")

FUZZY_MAP = {"SEPS":"SEPSIS","SEPT":"SEPSIS","PNEUM":"PNEUMONIA","STROK":"STROKE","CVA":"STROKE",
    "ATRIAL":"ATRIAL_FIBRILLATION","AFIB":"ATRIAL_FIBRILLATION","FIBRIL":"ATRIAL_FIBRILLATION",
    "AKI":"AKI","KIDNEY":"AKI","RENAL":"AKI","UTI":"UTI","URIN":"UTI",
    "HEART":"HEART_FAILURE","CHF":"HEART_FAILURE","GI":"GI_BLEEDING","BLEED":"GI_BLEEDING",
    "DIAB":"DIABETES_COMPLICATION","DKA":"DIABETES_COMPLICATION",
    "COPD":"COPD_EXACERBATION","OTHER":"OTHER","NONE":"OTHER"}
def parse_class(raw):
    if not raw: return "INVALID"
    t=raw.strip().upper()
    for c in sorted(ALLOWED_CLASSES,key=len,reverse=True):
        if c in t[:80]: return c
    for k,v in FUZZY_MAP.items():
        if k in t[:80]: return v
    return "INVALID"

@torch.no_grad()
def eval_split(recs, isim, BATCH=8):
    preds=[]; t0=time.time()
    for b in range(0,len(recs),BATCH):
        grup=recs[b:b+BATCH]
        texts=[]
        for r in grup:
            msgs=[{"role":"user","content":f"{INSTRUCTION}\n\n{r['input']}"}]
            try:
                texts.append(g_proc.apply_chat_template(msgs,tokenize=False,
                    add_generation_prompt=True, enable_thinking=False))
            except TypeError:
                texts.append(g_proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True))
        enc=g_tok(texts,return_tensors="pt",padding=True,truncation=True,max_length=2048).to(g_model.device)
        out=g_model.generate(**enc,max_new_tokens=20,do_sample=False,pad_token_id=g_tok.eos_token_id)
        gen=out[:,enc.input_ids.shape[1]:]
        for j in range(len(grup)):
            preds.append(parse_class(g_tok.decode(gen[j],skip_special_tokens=True).strip()))
    y=[r["output"] for r in recs]
    acc=accuracy_score(y,preds); mf1=f1_score(y,preds,average="macro",zero_division=0)
    inv=sum(1 for p in preds if p=="INVALID")
    print(f"   {isim}: acc={acc:.4f} macroF1={mf1:.4f} invalid={inv} ({(time.time()-t0)/60:.1f}dk)")
    return preds,acc,mf1,y

print("GEMMA QLoRA-S3 EVAL")
vp,v_acc,v_mf1,vy = eval_split(val_recs, "VAL (550)")
tp,t_acc,t_mf1,ty = eval_split(test_recs,"TEST (1650)")

OUT=Path("/content/gemma_out"); OUT.mkdir(exist_ok=True)
pd.DataFrame({"output":ty,"pred":tp}).to_csv(OUT/"gemma_qlora_test_preds.csv",index=False)
pd.DataFrame({"subset":["val","test"],"acc":[v_acc,t_acc],"macro_f1":[v_mf1,t_mf1]}
    ).to_csv(OUT/"gemma_qlora_summary.csv",index=False)

# per-class test
print("\nPer-class (TEST):")
labels=sorted(set(ty)|(set(tp)-{"INVALID"}))
p,r,f,supp=precision_recall_fscore_support(ty,tp,labels=labels,zero_division=0)
for c,pi,ri,fi,si in zip(labels,p,r,f,supp):
    print(f"   {c:<25s} P={pi:.3f} R={ri:.3f} F1={fi:.3f} N={si}")

print("\n" + "="*72 + "\nTUM MODELLER — FINAL TEST TABLOSU (1650, ayni veri)\n" + "="*72)
print(f"   {'Model / Yontem':<34s} {'Bit':>6s} {'Acc':>8s} {'MacroF1':>9s}")
print(f"   {'Qwen3.5-4B zero-shot':<34s} {'4bit':>6s} {0.5152:>8.4f} {0.4681:>9.4f}")
print(f"   {'Qwen3.5-4B QLoRA-S3':<34s} {'4bit':>6s} {0.6345:>8.4f} {0.6220:>9.4f}")
print(f"   {'Gemma-4-E4B zero-shot':<34s} {'bf16':>6s} {0.5648:>8.4f} {0.5561:>9.4f}")
print(f"   {'Gemma-4-E4B zero-shot':<34s} {'4bit':>6s} {0.5388:>8.4f} {0.4811:>9.4f}")
print(f"   {'Gemma-4-E4B QLoRA-S3 (this)':<34s} {'4bit':>6s} {t_acc:>8.4f} {t_mf1:>9.4f}")
print(f"\n>>> Gemma QLoRA, Qwen QLoRA'yi (0.6345) ve Gemma zero-shot'u (0.5648) gecti mi?")
print("✅ BLOK G06c TAMAM. Cikti: gemma_qlora_test_preds.csv + gemma_qlora_summary.csv — INDIR!")
