# =====================================================================
# BLOK G06b-v3 — Gemma QLoRA TRAIN (DOGRU TOKENIZE) — Colab A100
# =====================================================================
# KOK SEBEP: apply_chat_template(tokenize=FALSE)->string yeniden tokenize edilince
#   Gemma4 ozel token'lari (<|turn>=105, <turn|>=106) parcalaniyordu -> bozuk egitim.
# COZUM: apply_chat_template(tokenize=TRUE) DOGRU ID uretiyor (teshis6 dogruladi:
#   cevap = [GI,_,BLE,ED,ING,<turn|>,\n] tam dogru). Yeniden tokenize YOK.
#   + completion-only: prompt (tokenize=True, gen_prompt) uzunluguna kadar -100.
# SMOKE=True -> 30 adim (~2dk) loss kontrol; iyiyse SMOKE=False full.
# Onceki: train_recs, val_recs (G06a). Model bu blokta yuklenir.
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

# >>>>>>>>>>>>>>  AYAR  <<<<<<<<<<<<<<
SMOKE = True    # ONCE True (~2dk loss kontrol). Iyiyse False -> full train.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
GEMMA_ID = "google/gemma-4-E4B-it"
MAX_SEQ_LEN = 2048
SEED = 42
assert "train_recs" in dir() and "val_recs" in dir(), "Once G06a calistir!"

# ===== 0) Temizlik =====
print("[0/6] Temizlik...")
for v in ["trainer","g_model"]:
    if v in dir():
        try: del globals()[v]
        except Exception: pass
gc.collect(); torch.cuda.empty_cache()
_ADIR = Path("/content/gemma_out/gemma_qlora_s3")
if _ADIR.exists(): shutil.rmtree(_ADIR)
print(f"   GPU {torch.cuda.memory_allocated()/1e9:.2f}GB")

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

# ===== 2) target_modules =====
target_modules = [n for n,m in g_model.named_modules()
    if m.__class__.__name__ in ("Linear4bit","Linear","Linear8bitLt")
    and (n.endswith("q_proj") or n.endswith("v_proj"))
    and "language_model" in n and "vision" not in n and "audio" not in n]
print(f"\n[2/6] target_modules: {len(target_modules)}, ornek {target_modules[0]}")
assert target_modules

# ===== 3) Dataset — tokenize=True (DOGRU ID) + completion-only =====
print("\n[3/6] Dataset (tokenize=True, completion-only)...")
def to_ids(msgs, gen):
    x = g_proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=gen)
    if hasattr(x,"tolist"): x = x.tolist()
    if x and isinstance(x[0], list): x = x[0]
    return x

def build(r):
    user = f"{r['instruction']}\n\n{r['input']}"
    prompt_ids = to_ids([{"role":"user","content":user}], gen=True)         # prompt + model basligi
    full_ids   = to_ids([{"role":"user","content":user},
                         {"role":"assistant","content":r["output"]}], gen=False)  # + cevap
    # completion-only: prompt kismi -100, cevap loss'a katilir
    labels = [-100]*len(prompt_ids) + list(full_ids[len(prompt_ids):])
    ids = list(full_ids)
    # guvenlik: uzunluk uyumsuzlugu (nadir) -> labels'i ids'e hizala
    if len(labels) != len(ids):
        labels = ([-100]*len(prompt_ids) + list(ids[len(prompt_ids):]))[:len(ids)]
    # truncation: cevabi koru, prompt basini kirp
    if len(ids) > MAX_SEQ_LEN:
        kes = len(ids)-MAX_SEQ_LEN
        ids = ids[kes:]; labels = labels[kes:]
    return ids, labels

class DS(Dataset):
    def __init__(self, recs): self.recs=recs
    def __len__(self): return len(self.recs)
    def __getitem__(self, i):
        ids, labels = build(self.recs[i])
        pad = MAX_SEQ_LEN - len(ids)
        if pad>0:
            ids=ids+[g_tok.pad_token_id]*pad; labels=labels+[-100]*pad
            am=[1]*(MAX_SEQ_LEN-pad)+[0]*pad
        else:
            ids=ids[:MAX_SEQ_LEN]; labels=labels[:MAX_SEQ_LEN]; am=[1]*MAX_SEQ_LEN
        return {"input_ids":torch.tensor(ids),"attention_mask":torch.tensor(am),
                "labels":torch.tensor(labels)}
def collate(f): return {k: torch.stack([x[k] for x in f]) for k in f[0]}
train_ds, val_ds = DS(train_recs), DS(val_recs)
_s=train_ds[0]; _lt=(_s["labels"]!=-100).sum().item()
# loss token = cevap (GI_BLEEDING + turn-sonu) ~ 5-10 olmali
_loss_pos=(_s["labels"]!=-100).nonzero().flatten()
_dec = g_tok.decode([_s["input_ids"][p] for p in _loss_pos]) if len(_loss_pos)>0 else ""
print(f"   train={len(train_ds)} val={len(val_ds)} | loss token: {_lt}")
print(f"   loss'a giren metin: {repr(_dec)}  <- cevap (etiket) olmali!")
assert 2<=_lt<=40, f"loss token {_lt} beklenmedik!"

# ===== 4) LoRA =====
print("\n[4/6] LoRA r=8 alpha=16...")
g_model = prepare_model_for_kbit_training(g_model, use_gradient_checkpointing=True)
g_model.config.use_cache = False
if hasattr(g_model,"enable_input_require_grads"): g_model.enable_input_require_grads()
g_model = get_peft_model(g_model, LoraConfig(r=8, lora_alpha=16,
    target_modules=target_modules, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"))
g_model.print_trainable_parameters()

# ===== 5) TRAIN =====
ADIR = Path("/content/gemma_out/gemma_qlora_s3"); ADIR.mkdir(parents=True, exist_ok=True)
print(f"\n[5/6] {'SMOKE (30 adim)' if SMOKE else 'FULL (2 epoch)'} train...")
args = TrainingArguments(
    output_dir=str(ADIR/"checkpoints"),
    num_train_epochs=(1 if SMOKE else 2), max_steps=(30 if SMOKE else -1),
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
print(f"\n   Sure {(time.time()-t0)/60:.1f}dk, peak {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
print(f"   LOSS TRENDI: {tl}")
print(f"   Ilk {tl[0][1]} -> son {tl[-1][1]}")

if SMOKE:
    print("\n>>> Loss ~2-3'ten HIZLA dusuyorsa (son <1.0, hatta <0.5) -> DOGRU! SMOKE=False yap, full.")
    print("    1.5+'ta takiliyorsa haber ver.")
else:
    trainer.save_model(str(ADIR/"adapter"))
    print(f"\n   Adapter: {ADIR/'adapter'} — INDIR! Sonraki: G06c eval.")