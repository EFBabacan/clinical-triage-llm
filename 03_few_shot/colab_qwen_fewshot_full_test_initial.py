# =====================================================================
# BLOK 04 — FULL 3-shot few-shot, TEST (1650) — Colab A100
# =====================================================================
# Val'da zero-shot'u GECTI (0.5182 > 0.4818) -> kural geregi test'e gecis.
# Blok 03 BATCH ile AYNI kod, sadece split="test" ve checkpoint adlari farkli.
# Sure: ~2.5 saat (batch=4, ort 5.1s/ornek x 1650).
# !!! CHECKPOINT/RESUME GUCLU: oturum kopsa AYNI blogu tekrar calistir,
#     kaldigi yerden devam eder (eski checkpoint SILINMEZ, resume eder).
# Onceki bloklardan: model, tok, df, ex3, CLASSES, V4COL
# =====================================================================
import torch, time, os, gc
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ALLOWED_CLASSES = CLASSES
BATCH_SIZE = 4           # gated-delta-rule + 23k prompt: 4 guvenli. OOM olursa 2.
MAX_LEN    = 32768

# ---------- HEADER (Blok 03 / local ile BIREBIR ayni) ----------
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

# ---------- SABIT prompt: 33 FULL ornek ----------
SABIT = HEADER
ex_idx = 1
for cls in ALLOWED_CLASSES:
    for ex in ex3[cls]:
        full_text = str(ex[V4COL]) if ex.get(V4COL) is not None else ""
        SABIT += EX_TMPL.format(i=ex_idx, label=cls, input=full_text.strip())
        ex_idx += 1
print(f"SABIT prompt: {ex_idx-1} ornek, ~{len(tok(SABIT).input_ids)} token")

# ---------- Parser (local ile BIREBIR ayni) ----------
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

# ---------- BATCH inference (sol-padding) ----------
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
trunc_sayac = {"kesilen": 0}

def build_text(query):
    prompt = SABIT + TT.format(input=query)
    return tok.apply_chat_template([{"role":"user","content":prompt}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)

@torch.no_grad()
def inference_batch(queries):
    texts = [build_text(q) for q in queries]
    for t in texts:
        if len(tok(t).input_ids) > MAX_LEN: trunc_sayac["kesilen"] += 1
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to("cuda")
    out = model.generate(**enc, max_new_tokens=20, do_sample=False, pad_token_id=tok.eos_token_id)
    gen = out[:, enc.input_ids.shape[1]:]
    raws = [tok.decode(gen[j], skip_special_tokens=True).strip() for j in range(len(queries))]
    in_lens = [int(enc.attention_mask[j].sum().item()) for j in range(len(queries))]
    return raws, in_lens

# ---------- Veri: TEST ----------
test_df = df[df["split"]=="test"].reset_index(drop=True)
print(f"TEST: {len(test_df)} ornek, BATCH_SIZE={BATCH_SIZE}")

# ---------- Checkpoint/RESUME (TEST icin: SILME! kaldigi yerden devam) ----------
OUT_DIR = Path("/content/fs_out"); OUT_DIR.mkdir(exist_ok=True)
cp_yol    = OUT_DIR / "full_3shot_test_checkpoint.csv"
final_yol = OUT_DIR / "full_3shot_test.csv"

if cp_yol.exists():
    cp = pd.read_csv(cp_yol)
    islenen = set(cp["note_id"].astype(str))
    kayitlar = cp.to_dict("records")
    print(f"RESUME: {len(islenen)} kayit zaten var, kaldigi yerden devam.")
else:
    islenen = set(); kayitlar = []
    print("Sifirdan baslaniyor.")

rows = test_df.to_dict("records")
# islenmemis satirlari filtrele
kalan_rows = [r for r in rows if str(r["note_id"]) not in islenen]
print(f"Islenecek: {len(kalan_rows)} (toplam {len(rows)})")

t_bas = time.time(); yeni = 0
for b in range(0, len(kalan_rows), BATCH_SIZE):
    grup = kalan_rows[b:b+BATCH_SIZE]
    queries = [r[V4COL] if isinstance(r[V4COL], str) else "" for r in grup]
    raws, in_lens = inference_batch(queries)
    for r, raw, in_n in zip(grup, raws, in_lens):
        parsed, kat = parse_class(raw)
        kayitlar.append({
            "note_id": str(r["note_id"]), "subject_id": int(r["subject_id"]),
            "hadm_id": int(r["hadm_id"]), "label": r["label"],
            "raw_output": raw, "parsed_pred": parsed, "parse_kategori": kat,
            "in_tokens": in_n, "has_think": ("<think>" in raw or "</think>" in raw),
        })
    yeni += len(grup)
    pd.DataFrame(kayitlar).to_csv(cp_yol, index=False)   # HER batch'te kaydet (resume guvenligi)
    gc.collect(); torch.cuda.empty_cache()
    gecen = time.time()-t_bas; ort = gecen/yeni
    kalan = (len(kalan_rows)-yeni)*ort
    print(f"   [{len(kayitlar):>4}/{len(rows)}] gecen {gecen/60:.1f}dk ort {ort:.2f}s kalan ~{kalan/60:.1f}dk")

df_son = pd.DataFrame(kayitlar)
df_son.to_csv(final_yol, index=False)
print(f"\nBITTI. {len(kayitlar)} kayit, bu oturum {(time.time()-t_bas)/60:.1f}dk")

# ---------- AUDIT ----------
print("\n" + "="*70 + "\nFULL 3-SHOT TEST AUDIT\n" + "="*70)
n = len(df_son)
ks = df_son["parse_kategori"].value_counts().to_dict()
direct = ks.get("direct",0); fuzzy = ks.get("fuzzy",0); invalid = ks.get("invalid",0)+ks.get("empty",0)
print(f"Parser: direct={direct} ({100*direct/n:.1f}%) fuzzy={fuzzy} invalid={invalid} ({100*invalid/n:.1f}%)")
print(f"Think output: {df_son['has_think'].sum()}/{n}")
print(f"Input token avg/max: {df_son['in_tokens'].mean():.0f} / {df_son['in_tokens'].max()} (MAX_LEN={MAX_LEN})")
if trunc_sayac["kesilen"] > 0:
    print(f"🔴 UYARI: {trunc_sayac['kesilen']} ornekte QUERY KESILDI! MAX_LEN artir.")
else:
    print(f"✅ Truncation YOK.")

y_true = df_son["label"].values; y_pred = df_son["parsed_pred"].values
valid = y_pred != "INVALID"
acc_full = accuracy_score(y_true, y_pred)
mf1_full = f1_score(y_true, y_pred, average="macro", zero_division=0)
wf1_full = f1_score(y_true, y_pred, average="weighted", zero_division=0)
acc_v = accuracy_score(y_true[valid], y_pred[valid]) if valid.sum() else 0.0
mf1_v = f1_score(y_true[valid], y_pred[valid], average="macro", zero_division=0) if valid.sum() else 0.0
print(f"\nFULL 3-SHOT Test:")
print(f"   Full        n={n}   acc={acc_full:.4f} macroF1={mf1_full:.4f} wF1={wf1_full:.4f}")
print(f"   Valid only  n={valid.sum()} acc={acc_v:.4f} macroF1={mf1_v:.4f}")

labels = sorted(set(y_true) | (set(y_pred)-{"INVALID"}))
p,r,f,supp = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
print("\nPer-class:")
for cls,pi,ri,fi,si in zip(labels,p,r,f,supp):
    print(f"   {cls:<25s} P={pi:.3f} R={ri:.3f} F1={fi:.3f} N={si}")

other_pct = 100*(df_son['parsed_pred']=='OTHER').sum()/n
hf_pct = 100*(df_son['parsed_pred']=='HEART_FAILURE').sum()/n
print(f"\nKARSILASTIRMA (TEST):")
print(f"   {'':<30s} {'Acc':>8s} {'MacroF1':>9s} {'OTHER%':>8s} {'HF%':>7s}")
print(f"   {'V4 zero-shot (local)':<30s} {0.5152:>8.4f} {0.4681:>9.4f} {'-':>7s} {20.8:>6.1f}%")
print(f"   {'V4 QLoRA-S3 (local)':<30s} {0.6345:>8.4f} {0.6220:>9.4f} {'-':>7s} {'-':>6s}")
print(f"   {'V4 3-shot FULL (Colab,this)':<30s} {acc_full:>8.4f} {mf1_full:>9.4f} {other_pct:>7.1f}% {hf_pct:>6.1f}%")

# Summary kaydet
pd.DataFrame({"subset":["full","valid_only"],"n":[n,valid.sum()],
    "accuracy":[acc_full,acc_v],"macro_f1":[mf1_full,mf1_v],
    "weighted_f1":[wf1_full,0.0]}).to_csv(OUT_DIR/"full_3shot_test_results_summary.csv", index=False)
print("\n✅ BLOK 04 TAMAM. Cikti: /content/fs_out/full_3shot_test.csv + _results_summary.csv")
print("   !! Bu dosyalari Colab'dan INDIR (files.download) ya da Drive'a kopyala — oturum kapaninca silinir!")
