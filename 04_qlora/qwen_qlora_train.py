"""Qwen3.5-4B QLoRA training: the configuration used for the reported Qwen QLoRA results.

Extracted verbatim from notebooks/experiment_notebook.ipynb (section "BLOK 15B").
Recorded run (training_curve.json): r=8, alpha=16, q_proj+v_proj, 2 epochs, lr 1e-4,
effective batch 8 (1 x 8), MAX_SEQ_LEN=512, INPUT_TOKEN_BUDGET=320 (training inputs are
truncated to their first 320 tokens). RTX 4060 Laptop 8 GB: 9.1 h, peak 6.89 GB.
"""

import json
import time
import random
import gc
from pathlib import Path
from collections import defaultdict, Counter

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# ============================================================
# CONFIG - V4 FULL S3 (4.400 ornek)
# ============================================================
OUTPUT_DIR = Path(r"./workdir/output")
JSON_DIR = OUTPUT_DIR / "qlora_json"
ADAPTER_DIR = OUTPUT_DIR / "qlora_adapters" / "v4_s3_full_2epoch_lr1e4"
ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "Qwen/Qwen3.5-4B"
TRAIN_JSON = JSON_DIR / "qlora_v4_train_S3.json"
VAL_JSON = JSON_DIR / "qlora_v4_val.json"

SEED = 42
MAX_SEQ_LEN = 512
INPUT_TOKEN_BUDGET = 320
PER_DEVICE_BATCH = 1
GRAD_ACCUM = 8
LR = 1e-4
LORA_R = 8
LORA_ALPHA = 16
EPOCHS = 2

ALLOWED = [
    "SEPSIS", "STROKE", "ATRIAL_FIBRILLATION", "AKI", "PNEUMONIA", "UTI",
    "HEART_FAILURE", "GI_BLEEDING", "DIABETES_COMPLICATION",
    "COPD_EXACERBATION", "OTHER"
]

print("=" * 80)
print("V4 QLoRA FULL S3 (4.400 ornek, 2 epoch, eff_batch=8 -> 1100 step)")
print("=" * 80)
print("ON KOSUL: 15A-SMOKE 'ACCEPT' yazdiysa devam et.")
print("Tahmini sure: ~7 saat (gece run icin)")

random.seed(SEED)
torch.manual_seed(SEED)

# Cleanup
for name in ["model", "tokenizer", "trainer"]:
    if name in globals():
        del globals()[name]
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

# ============================================================
# DATA
# ============================================================
print("\n[1/6] Veri yukleniyor...")
with open(TRAIN_JSON, "r", encoding="utf-8") as f:
    train_recs = json.load(f)
with open(VAL_JSON, "r", encoding="utf-8") as f:
    val_recs = json.load(f)
assert len(train_recs) == 4400, f"S3 4400 olmali, {len(train_recs)} bulundu"
print("Train:", len(train_recs), "Val:", len(val_recs))
print("Sinif dagilimi:", dict(Counter(r["output"] for r in train_recs)))

# ============================================================
# MODEL
# ============================================================
print("\n[2/6] Model/tokenizer yukleniyor (4-bit NF4)...")
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, local_files_only=True, trust_remote_code=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb, device_map="auto",
    torch_dtype=torch.float16, local_files_only=True, trust_remote_code=True,
)
model.config.use_cache = False

try:
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
except TypeError:
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()

lora_cfg = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()
print("Load sec:", round(time.time()-t0, 1))
print("GPU after load GB:", round(torch.cuda.memory_allocated()/1024**3, 3))

# ============================================================
# HELPERS + DATASET
# ============================================================
def trim_input_to_budget(text, budget=INPUT_TOKEN_BUDGET):
    ids = tokenizer(text or "", add_special_tokens=False)["input_ids"][:budget]
    return tokenizer.decode(ids, skip_special_tokens=True).strip()

def make_user_text(r):
    return f"{r['instruction']}\n\n{trim_input_to_budget(r['input'])}"

def apply_template(messages, add_generation_prompt):
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=add_generation_prompt, enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt,
        )

def build_texts(r):
    user_text = make_user_text(r)
    prompt_text = apply_template(
        [{"role":"user","content":user_text}], add_generation_prompt=True,
    )
    full_text = apply_template(
        [{"role":"user","content":user_text},
         {"role":"assistant","content":r["output"]}],
        add_generation_prompt=False,
    )
    return prompt_text, full_text

class S3Dataset(Dataset):
    def __init__(self, records):
        self.records = records
    def __len__(self):
        return len(self.records)
    def __getitem__(self, idx):
        prompt_text, full_text = build_texts(self.records[idx])
        prompt_ids = tokenizer(
            prompt_text, truncation=True, max_length=MAX_SEQ_LEN,
            add_special_tokens=False,
        )["input_ids"]
        enc = tokenizer(
            full_text, truncation=True, max_length=MAX_SEQ_LEN,
            padding="max_length", return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        labels[:min(len(prompt_ids), MAX_SEQ_LEN)] = -100
        if (labels != -100).sum().item() == 0:
            last_real = int(attention_mask.sum().item()) - 1
            if last_real >= 0:
                labels[last_real] = input_ids[last_real]
        return {"input_ids":input_ids, "attention_mask":attention_mask, "labels":labels}

def collate_fn(features):
    return {
        "input_ids": torch.stack([f["input_ids"] for f in features]),
        "attention_mask": torch.stack([f["attention_mask"] for f in features]),
        "labels": torch.stack([f["labels"] for f in features]),
    }

print("\n[3/6] Dataset hazirlaniyor...")
train_ds = S3Dataset(train_recs)
print("Train ds:", len(train_ds))

# ------------------------------------------------------------
# LABEL TRUNCATION AUDIT (500 ornek, train oncesi guvenlik)
# ------------------------------------------------------------
print("\nLabel truncation audit (500 ornek)...")
bad_prompt = 0
near_limit = 0
label_lens = []
for r in train_recs[:500]:
    prompt_text, full_text = build_texts(r)
    prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
    full_len = len(tokenizer(full_text, add_special_tokens=False)["input_ids"])
    if prompt_len >= MAX_SEQ_LEN - 8:
        bad_prompt += 1
    if full_len >= MAX_SEQ_LEN:
        near_limit += 1
    label_lens.append(max(0, min(full_len, MAX_SEQ_LEN) - min(prompt_len, MAX_SEQ_LEN)))

print("Audit sample n:", len(label_lens))
print("bad_prompt (prompt assistant alanini yiyor):", bad_prompt)
print("near_limit (full_len >= MAX_SEQ_LEN):", near_limit)
print("label token min/mean/max:", min(label_lens), round(sum(label_lens)/len(label_lens), 1), max(label_lens))

assert bad_prompt == 0, "Prompt assistant label alanini yiyor; INPUT_TOKEN_BUDGET dusur (300 dene)."
assert min(label_lens) > 0, "Bazi kayitlarda assistant label kayboluyor; MAX_SEQ_LEN/budget revize."

# ============================================================
# TRAIN
# ============================================================
print("\n[4/6] Train basliyor (2 epoch, ~1100 step, tahmini ~7 saat)...")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

ckpt_dir = ADAPTER_DIR / "checkpoints"

args = TrainingArguments(
    output_dir=str(ckpt_dir),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    optim="adamw_torch",
    fp16=True,
    logging_steps=10,
    save_strategy="steps",        # crash dayanikli
    save_steps=50,                 # her 50 optimizer step (~%4.5'te bir, 7 saatte crash olursa max ~20 dk kayip)
    save_total_limit=2,            # disk dostu
    report_to=[],
    seed=SEED,
    gradient_checkpointing=True,
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
)

trainer = Trainer(
    model=model, args=args,
    train_dataset=train_ds, data_collator=collate_fn,
)

# Resume kontrolu
last_ckpt = None
if ckpt_dir.exists():
    last_ckpt = get_last_checkpoint(str(ckpt_dir))
print("Resume checkpoint:", last_ckpt if last_ckpt else "(yok, sifirdan basliyor)")

t0 = time.time()
result = trainer.train(resume_from_checkpoint=last_ckpt if last_ckpt else None)
train_sec = time.time() - t0
peak_gb = torch.cuda.max_memory_allocated() / 1024**3

losses = [(h.get("step"), h["loss"]) for h in trainer.state.log_history if "loss" in h]
print("\nTrain bitti.")
print("Train sec:", round(train_sec, 0), "(", round(train_sec/60, 1), "dk )")
print("Peak GPU GB:", round(peak_gb, 3))
print("Final loss:", round(result.training_loss, 4))
print("\nLoss trend (her 10 step):")
for step, l in losses[:5]:
    print(f"   step {step}: {l:.4f}")
if len(losses) > 10:
    print(f"   ...({len(losses)-10} ara step atlandi)...")
for step, l in losses[-5:]:
    print(f"   step {step}: {l:.4f}")

# ============================================================
# SAVE
# ============================================================
print("\n[5/6] Adapter kaydediliyor...")
adapter_path = ADAPTER_DIR / "adapter"
trainer.save_model(str(adapter_path))
tokenizer.save_pretrained(str(adapter_path))
print("Adapter:", adapter_path)
for p in sorted(adapter_path.glob("*")):
    print(" ", p.name, round(p.stat().st_size / 1024, 1), "KB")

# Training curve kaydet (scaling analizi icin)
curve_yol = ADAPTER_DIR / "training_curve.json"
with open(curve_yol, "w", encoding="utf-8") as f:
    json.dump({
        "scenario": "S3_full_2epoch_lr1e4",
        "train_size": len(train_recs),
        "epochs": EPOCHS,
        "train_sec": train_sec,
        "peak_gb": peak_gb,
        "final_loss": result.training_loss,
        "losses": [(s, l) for s, l in losses],
        "hyperparams": {
            "model": MODEL_ID,
            "max_seq_len": MAX_SEQ_LEN,
            "input_token_budget": INPUT_TOKEN_BUDGET,
            "batch": PER_DEVICE_BATCH, "grad_accum": GRAD_ACCUM,
            "lr": LR, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
            "target_modules": ["q_proj","v_proj"],
            "optim": "adamw_torch", "scheduler": "cosine",
            "warmup_ratio": 0.03,
        },
    }, f, indent=2)
print("Training curve:", curve_yol)

# ============================================================
# QUICK VAL CHECK (5 ornek, sadece sanity)
# ============================================================
print("\n[6/6] 5 val ornegi quick check...")
model.eval()
tokenizer.padding_side = "left"

def parse_class(raw):
    t = (raw or "").strip().upper()
    for cls in sorted(ALLOWED, key=len, reverse=True):
        if cls in t[:80]:
            return cls
    return "INVALID"

val_by_label = defaultdict(list)
for r in val_recs:
    val_by_label[r["output"]].append(r)
quick = []
for cls in ALLOWED[:5]:
    if cls in val_by_label:
        quick.append(val_by_label[cls][0])

valid = 0
correct = 0
for i, r in enumerate(quick, 1):
    user_text = make_user_text(r)
    prompt = apply_template([{"role":"user","content":user_text}], add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    raw = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    pred = parse_class(raw)
    ok = pred in ALLOWED
    is_correct = pred == r["output"]
    valid += int(ok); correct += int(is_correct)
    mark = "OK" if is_correct else ("VALID" if ok else "BAD")
    print(f"[{i}] gold={r['output']:<22s} pred={pred:<22s} {mark}")

print("\n" + "=" * 80)
print("FULL S3 TAMAMLANDI.")
print("=" * 80)
print("Adapter:", adapter_path)
print("Train sure:", round(train_sec/60, 1), "dk")
print("Peak GPU:", round(peak_gb, 2), "GB")
print("Quick val: valid=" + str(valid) + "/5, correct=" + str(correct) + "/5")
print("\nSiradaki: Blok 15-EVAL (full val + test, accuracy/F1/per-class)")
