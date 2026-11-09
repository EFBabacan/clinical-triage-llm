# =====================================================================
# BLOK G07 — Gemma QLoRA OPTIMIZE varyant (Gemma'nin tavanini ara) — A100
# =====================================================================
# AMAC: Adil kiyas (G06, q+v r8 2ep) Qwen'le esitti. Bu varyant Gemma'yi ZORLAR:
#   - r=16 (G06: 8)
#   - target = TUM linear: q,k,v,o + gate,up,down (G06: sadece q,v)
#   - 3 epoch (G06: 2)
# G06'nin CALISAN mantigi korunur: tokenize=True (Gemma4 token bug cozumu) + completion-only.
# Bu, "adil ayar" DEGIL "Gemma-optimize" -> tezde AYRI varyant olarak sunulur.
# Onceki: train_recs, val_recs, test_recs (G06a). Model bu blokta yuklenir.
# Sure: ~2.5-3 saat (3 epoch + cok modul). SMOKE ile once dogrula.
# =====================================================================
import importlib.util as _ilu
if _ilu.find_spec("bitsandbytes") is None or _ilu.find_spec("peft") is None:
    import subprocess
    subprocess.run(["pip","-q","install","-U","bitsandbytes>=0.46.1","peft","accelerate"], check=True)

import json, time, os, gc, shutil, torch
from pathlib import Path
from torch.utils.data import Dataset
from transformers import (AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig,
                          Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# !!! DRIVE'A KALICI KAYIT (gecen sefer /content silindi, adapter gitti).
# Drive mount -> adapter Drive'a yazilir, oturum kapansa bile DURUR.
from google.colab import drive
drive.mount("/content/drive")
DRIVE_OUT = Path("/content/drive/MyDrive/tez_gemma_qlora")  # Drive'da kalici klasor
DRIVE_OUT.mkdir(parents=True, exist_ok=True)
print(f"Drive kayit yolu: {DRIVE_OUT}")

# >>>>>>>>>>>>>>  AYAR  <<<<<<<<<<<<<<
SMOKE = False   # full train (smoke gecmisti, loss dustu 0.088). Bitince DRIVE'a kaydeder.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
GEMMA_ID = "google/gemma-4-E4B-it"
MAX_SEQ_LEN = 2048
SEED = 42
# OPTIMIZE ayarlari:
LORA_R = 16
LORA_ALPHA = 32          # alpha = 2*r konvansiyonu
EPOCHS = 3
# tum linear leaf isimleri (attention + MLP)
HEDEF_LEAF = ("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")

assert "train_recs" in dir() and "val_recs" in dir(), "Once G06a calistir!"

# ===== 0) Temizlik (ESKI G06 adapter'ini SILME! ayri klasore yaz) =====
print("[0/6] Temizlik...")
for v in ["trainer","g_model"]:
    if v in dir():
        try: del globals()[v]
        except Exception: pass
gc.collect(); torch.cuda.empty_cache()
ADIR = Path("/content/gemma_out/gemma_qlora_s3_OPTIMIZE")   # G06'dan AYRI klasor
if ADIR.exists(): shutil.rmtree(ADIR)
ADIR.mkdir(parents=True, exist_ok=True)
print(f"   GPU {torch.cuda.memory_allocated()/1e9:.2f}GB | cikti: {ADIR}")

# ===== 1) Model =====
print("\n[1/6] Gemma 4-bit...")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
g_proc = AutoProcessor.from_pretrained(GEMMA_ID)
g_tok = g_proc.tokenizer if hasattr(g_proc, "tokenizer") else g_proc
if g_tok.pad_token is None: g_tok.pad_token = g_tok.eos_token
g_tok.padding_side = "right"
g_model = AutoModelForCausalLM.from_pretrained(GEMMA_ID, quantization_config=bnb,
    device_map="auto", dtype=torch.bfloat16)
print(f"   GPU {torch.cuda.memory_allocated()/1e9:.2f}GB")

# ===== 2) target_modules: TUM linear (language_model, vision/audio haric) =====
target_modules = [n for n,m in g_model.named_modules()
    if m.__class__.__name__ in ("Linear4bit","Linear","Linear8bitLt")
    and n.split(".")[-1] in HEDEF_LEAF
    and "language_model" in n and "vision" not in n and "audio" not in n]
print(f"\n[2/6] target_modules: {len(target_modules)} (TUM linear: q,k,v,o,gate,up,down)")
print(f"   ornek: {target_modules[0]}")
from collections import Counter
leaf_say = Counter(n.split('.')[-1] for n in target_modules)
print(f"   leaf dagilimi: {dict(leaf_say)}")
assert target_modules

# ===== 3) Dataset (G06 ile AYNI: tokenize=True + completion-only) =====
print("\n[3/6] Dataset (tokenize=True, completion-only)...")
def to_ids(msgs, gen):
    x = g_proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=gen)
    if hasattr(x,"tolist"): x = x.tolist()
    if x and isinstance(x[0], list): x = x[0]
    return x
def build(r):
    user = f"{r['instruction']}\n\n{r['input']}"
    p = to_ids([{"role":"user","content":user}], gen=True)
    f = to_ids([{"role":"user","content":user},{"role":"assistant","content":r["output"]}], gen=False)
    labels = [-100]*len(p) + list(f[len(p):])
    ids = list(f)
    if len(labels)!=len(ids): labels = ([-100]*len(p)+list(ids[len(p):]))[:len(ids)]
    if len(ids)>MAX_SEQ_LEN:
        k=len(ids)-MAX_SEQ_LEN; ids=ids[k:]; labels=labels[k:]
    return ids, labels
class DS(Dataset):
    def __init__(s,recs): s.recs=recs
    def __len__(s): return len(s.recs)
    def __getitem__(s,i):
        ids,labels=build(s.recs[i]); pad=MAX_SEQ_LEN-len(ids)
        if pad>0:
            ids=ids+[g_tok.pad_token_id]*pad; labels=labels+[-100]*pad; am=[1]*(MAX_SEQ_LEN-pad)+[0]*pad
        else: ids=ids[:MAX_SEQ_LEN]; labels=labels[:MAX_SEQ_LEN]; am=[1]*MAX_SEQ_LEN
        return {"input_ids":torch.tensor(ids),"attention_mask":torch.tensor(am),"labels":torch.tensor(labels)}
def collate(f): return {k: torch.stack([x[k] for x in f]) for k in f[0]}
train_ds, val_ds = DS(train_recs), DS(val_recs)
_lt=(train_ds[0]["labels"]!=-100).sum().item()
print(f"   train={len(train_ds)} val={len(val_ds)} | loss token: {_lt} (~5-10 cevap)")
assert 2<=_lt<=40

# ===== 4) LoRA (OPTIMIZE: r=16, tum linear) =====
print(f"\n[4/6] LoRA r={LORA_R} alpha={LORA_ALPHA} (OPTIMIZE)...")
g_model = prepare_model_for_kbit_training(g_model, use_gradient_checkpointing=True)
g_model.config.use_cache = False
if hasattr(g_model,"enable_input_require_grads"): g_model.enable_input_require_grads()
g_model = get_peft_model(g_model, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA,
    target_modules=target_modules, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"))
g_model.print_trainable_parameters()

# ===== 5) TRAIN =====
print(f"\n[5/6] {'SMOKE (30 adim)' if SMOKE else f'FULL ({EPOCHS} epoch)'} train...")
args = TrainingArguments(
    output_dir=str(ADIR/"checkpoints"),
    num_train_epochs=(1 if SMOKE else EPOCHS), max_steps=(30 if SMOKE else -1),
    per_device_train_batch_size=4, per_device_eval_batch_size=4,
    gradient_accumulation_steps=4, gradient_checkpointing=True,
    learning_rate=1e-4, warmup_ratio=0.03, lr_scheduler_type="cosine",
    optim="paged_adamw_8bit", bf16=True, logging_steps=2,
    eval_strategy=("no" if SMOKE else "steps"), eval_steps=50,
    save_strategy=("no" if SMOKE else "steps"), save_steps=50, save_total_limit=1,
    load_best_model_at_end=(not SMOKE), metric_for_best_model="eval_loss",
    greater_is_better=False, report_to=[], seed=SEED,
    dataloader_pin_memory=False, dataloader_num_workers=0)
trainer = Trainer(model=g_model, args=args, train_dataset=train_ds,
                  eval_dataset=(None if SMOKE else val_ds), data_collator=collate)
t0=time.time(); torch.cuda.reset_peak_memory_stats()
trainer.train()
tl=[(h.get("step"),round(h["loss"],3)) for h in trainer.state.log_history if "loss" in h and "eval_loss" not in h]
el=[(round(h.get("epoch",0),2),round(h["eval_loss"],4)) for h in trainer.state.log_history if "eval_loss" in h]
print(f"\n   Sure {(time.time()-t0)/60:.1f}dk, peak {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
print(f"   Ilk loss {tl[0][1]} -> son {tl[-1][1]}")
if el: print(f"   Eval loss: {el}")

if SMOKE:
    print("\n>>> Loss HIZLA dusuyorsa (son <0.5) -> SMOKE=False yap, full.")
    print("    (G06 r8'de smoke son ~0.5'ti; r16+tum linear daha hizli dusebilir.)")
else:
    # 1) /content'e kaydet
    trainer.save_model(str(ADIR/"adapter"))
    json.dump({"variant":"OPTIMIZE","r":LORA_R,"alpha":LORA_ALPHA,"epochs":EPOCHS,
        "target_leaf":list(HEDEF_LEAF),"n_target":len(target_modules)},
        open(ADIR/"info.json","w"), indent=1)
    # 2) HEMEN DRIVE'A KOPYALA (KALICI — gecen sefer /content silindi, bir daha kaybetmeyelim)
    drive_adapter = DRIVE_OUT / "optimize_adapter"
    if drive_adapter.exists(): shutil.rmtree(drive_adapter)
    shutil.copytree(ADIR/"adapter", drive_adapter)
    shutil.copy(ADIR/"info.json", DRIVE_OUT/"optimize_info.json")
    print(f"\n   ✅ /content: {ADIR/'adapter'}")
    print(f"   ✅ DRIVE (KALICI): {drive_adapter}")
    print(f"   -> {os.listdir(drive_adapter)}")
    print("   Oturum kapansa bile DRIVE'da DURUR. Sonraki: eval.")