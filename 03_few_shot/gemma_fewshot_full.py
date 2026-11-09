# =====================================================================
# BLOK G05 — Gemma-4-E4B-it FULL 3-shot (val + test) — Colab A100
# =====================================================================
# Qwen full 3-shot (blok03/04) ile BIREBIR ayni: ayni 33 FULL ornek (ex3),
# ayni HEADER, ayni prompt format, ayni parse_class/FUZZY_MAP, ayni veri.
# TEK FARK: model = Gemma bf16 (g_model/g_proc/g_tok).
#
# SPLIT degiskeni: "val" (550) once, sonra "test" (1650).
# Gemma bf16, full prompt ~23k token -> batch=4 (Qwen ile ayni). OOM olursa 2.
# Onceki bloklardan: ex3, CLASSES, V4COL (Blok 02), g_model/g_proc/g_tok (G01)
#   g_model bf16 olmali! 4-bit'ten geldiyse G01'i (bf16) tekrar calistir.
# Sure: val ~15-25dk, test ~45-80dk (ilk batch'te netlesir).
# =====================================================================
import torch, time, os, gc
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

# >>>>>>>>>>>>>>>>>>>>>>  BURAYI DEGISTIR  <<<<<<<<<<<<<<<<<<<<<<
SPLIT = "val"      # ONCE "val", sonra "test"
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
BATCH_SIZE = 4
MAX_LEN    = 32768

# ---------- Self-contained: df / CLASSES / V4COL / ex3 kur (Blok 02'ye bagimli degil) ----------
CLASSES = ["SEPSIS","STROKE","ATRIAL_FIBRILLATION","AKI","PNEUMONIA","UTI",
    "HEART_FAILURE","GI_BLEEDING","DIABETES_COMPLICATION","COPD_EXACERBATION","OTHER"]
V4COL = "triage_input_v4_timeaware_masked"
ALLOWED_CLASSES = CLASSES

try:
    df
except NameError:
    df = pd.read_parquet("discharge_split.parquet")

# ex3 (full 3-shot ornekleri) yoksa CSV'den kur
try:
    ex3
except NameError:
    _e = pd.read_csv("fewshot_v4_3shot_examples.csv")
    ex3 = {}
    for cls, grp in _e.groupby("label"):
        if "rank" in grp.columns: grp = grp.sort_values("rank")
        ex3[cls] = grp.to_dict("records")
    print(f"ex3 CSV'den kuruldu: {sum(len(v) for v in ex3.values())} ornek")

# ---------- HEADER (Qwen full 3-shot ile BIREBIR ayni) ----------
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

Below are 33 example triage notes (3 per class):
"""
EX_TMPL = "\nExample {i} ({label}):\n{input}\n=> {label}\n"
TT = "\nNow classify:\n{input}\n=> "

# ---------- SABIT prompt: 33 FULL ornek (ex3 — Qwen ile ayni) ----------
SABIT = HEADER
ex_idx = 1
for cls in ALLOWED_CLASSES:
    for ex in ex3[cls]:
        full_text = str(ex[V4COL]) if ex.get(V4COL) is not None else ""
        SABIT += EX_TMPL.format(i=ex_idx, label=cls, input=full_text.strip())
        ex_idx += 1
print(f"SABIT prompt: {ex_idx-1} ornek, ~{len(g_tok(SABIT).input_ids)} token (Gemma tokenizer)")

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

# ---------- Gemma batch inference (sol-padding) ----------
g_tok.padding_side = "left"
if g_tok.pad_token is None: g_tok.pad_token = g_tok.eos_token
trunc_sayac = {"kesilen":0}

def build_text(query):
    prompt = SABIT + TT.format(input=query)
    try:
        return g_proc.apply_chat_template([{"role":"user","content":prompt}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return g_proc.apply_chat_template([{"role":"user","content":prompt}],
                    tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def inference_batch(queries):
    texts=[build_text(q) for q in queries]
    for t in texts:
        if len(g_tok(t).input_ids) > MAX_LEN: trunc_sayac["kesilen"]+=1
    enc=g_tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(g_model.device)
    out=g_model.generate(**enc, max_new_tokens=20, do_sample=False, pad_token_id=g_tok.eos_token_id)
    gen=out[:, enc.input_ids.shape[1]:]
    raws=[g_tok.decode(gen[j], skip_special_tokens=True).strip() for j in range(len(queries))]
    in_lens=[int(enc.attention_mask[j].sum().item()) for j in range(len(queries))]
    return raws, in_lens

# ---------- Veri ----------
sub_df = df[df["split"]==SPLIT].reset_index(drop=True)
print(f"\nGemma full 3-shot {SPLIT.upper()}: {len(sub_df)} ornek, BATCH={BATCH_SIZE}")
OUT_DIR=Path("/content/gemma_out"); OUT_DIR.mkdir(exist_ok=True)
cp_yol    = OUT_DIR / f"gemma_full3shot_{SPLIT}_checkpoint.csv"
final_yol = OUT_DIR / f"gemma_full3shot_{SPLIT}.csv"

if cp_yol.exists():
    cp=pd.read_csv(cp_yol); islenen=set(cp["note_id"].astype(str)); kayitlar=cp.to_dict("records")
    print(f"RESUME: {len(islenen)} kayit var.")
else:
    islenen=set(); kayitlar=[]
rows=sub_df.to_dict("records")
kalan=[r for r in rows if str(r["note_id"]) not in islenen]
print(f"Islenecek: {len(kalan)}")

t_bas=time.time(); yeni=0
for b in range(0, len(kalan), BATCH_SIZE):
    grup=kalan[b:b+BATCH_SIZE]
    qs=[r[V4COL] if isinstance(r[V4COL],str) else "" for r in grup]
    raws,in_lens=inference_batch(qs)
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
        print(f"   [{len(kayitlar):>4}/{len(rows)}] gecen {gecen/60:.1f}dk ort {ort:.2f}s kalan ~{(len(kalan)-yeni)*ort/60:.1f}dk")

df_son=pd.DataFrame(kayitlar); df_son.to_csv(final_yol, index=False)
print(f"\nBITTI. {len(kayitlar)} kayit, {(time.time()-t_bas)/60:.1f}dk")

# ---------- AUDIT ----------
print("\n" + "="*70 + f"\nGEMMA FULL 3-SHOT {SPLIT.upper()} AUDIT\n" + "="*70)
n=len(df_son)
ks=df_son["parse_kategori"].value_counts().to_dict()
direct=ks.get("direct",0); invalid=ks.get("invalid",0)+ks.get("empty",0)
print(f"Parser: direct={direct} ({100*direct/n:.1f}%) invalid={invalid} ({100*invalid/n:.1f}%)")
print(f"Input token avg/max: {df_son['in_tokens'].mean():.0f} / {df_son['in_tokens'].max()} (MAX_LEN={MAX_LEN})")
print("✅ Truncation YOK" if trunc_sayac["kesilen"]==0 else f"🔴 {trunc_sayac['kesilen']} KESILDI!")

y_true=df_son["label"].values; y_pred=df_son["parsed_pred"].values
acc=accuracy_score(y_true,y_pred); mf1=f1_score(y_true,y_pred,average="macro",zero_division=0)
wf1=f1_score(y_true,y_pred,average="weighted",zero_division=0)
print(f"\nGEMMA full 3-shot {SPLIT}: acc={acc:.4f} macroF1={mf1:.4f} wF1={wf1:.4f}")

labels=sorted(set(y_true)|(set(y_pred)-{"INVALID"}))
p,r,f,supp=precision_recall_fscore_support(y_true,y_pred,labels=labels,zero_division=0)
print("\nPer-class:")
for cls,pi,ri,fi,si in zip(labels,p,r,f,supp):
    print(f"   {cls:<25s} P={pi:.3f} R={ri:.3f} F1={fi:.3f} N={si}")

print(f"\nKARSILASTIRMA ({SPLIT}):")
print(f"   {'':<32s} {'Acc':>8s} {'MacroF1':>9s}")
if SPLIT=="val":
    print(f"   {'Qwen zero-shot':<32s} {0.4818:>8.4f} {0.4367:>9.4f}")
    print(f"   {'Qwen 3-shot FULL':<32s} {0.5182:>8.4f} {0.4781:>9.4f}")
    print(f"   {'Gemma zero-shot (bf16)':<32s} {0.5436:>8.4f} {0.5434:>9.4f}")
else:
    print(f"   {'Qwen zero-shot':<32s} {0.5152:>8.4f} {0.4681:>9.4f}")
    print(f"   {'Gemma zero-shot (bf16)':<32s} {0.5648:>8.4f} {0.5561:>9.4f}")
    print(f"   {'Qwen QLoRA-S3':<32s} {0.6345:>8.4f} {0.6220:>9.4f}")
print(f"   {'Gemma 3-shot FULL (this)':<32s} {acc:>8.4f} {mf1:>9.4f}")
print(f"\n✅ BLOK G05 ({SPLIT}) TAMAM. Cikti: {final_yol}  — INDIR!")
