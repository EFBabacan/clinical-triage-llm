# =====================================================================
# BLOK 01 — Colab kurulum + Qwen3.5-4B yukleme (4-bit NF4)
# Tez: MIMIC-IV V4 time-aware triage, few-shot deneyleri
# Colab Pro / A100 veya L4 onerilir
# =====================================================================
# AYARLAR local koddan birebir alindi (__run_blok14c_v4_3shot_ultra_val.py):
#   - bnb_4bit_compute_dtype = float16  (bf16 DEGIL)
#   - torch_dtype = float16
#   - pad_token = eos_token
#   - trust_remote_code = True
# TEK FARK: local_files_only=True KALDIRILDI (Colab'da cache yok, HF'ten iner)
# =====================================================================
# KULLANIM: Bu blogu Colab'da TEK hucreye yapistir, calistir.
# Sure: paket ~3-5 dk + model indirme ~8-10 dk (ilk sefer)
# =====================================================================

# --- 1) GPU kontrol ---
import subprocess
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)

# --- 2) Paketler ---
# [Colab] !pip -q install -U "transformers>=4.44" "accelerate>=0.33" "bitsandbytes>=0.43" "sentencepiece"

# --- 3) Importlar ---
import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
print("torch:", torch.__version__, "| CUDA:", torch.cuda.is_available(),
      "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "YOK")

# --- 4) Model: Qwen3.5-4B 4-bit (local ile AYNI ayarlar) ---
MODEL_ID = "Qwen/Qwen3.5-4B"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,      # local ile ayni (fp16)
    bnb_4bit_use_double_quant=True,
)

t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token              # local ile ayni

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,                 # local ile ayni (fp16)
)
model.eval()
print(f"\nModel yuklendi: {time.time()-t0:.0f} sn")
print("GPU bellek:", f"{torch.cuda.memory_allocated()/1e9:.2f} GB / "
      f"{torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB")

# --- 5) Smoke test: chat template + enable_thinking=False (local ile ayni) ---
prompt = "Reply with one word: OK"
text = tok.apply_chat_template([{"role":"user","content":prompt}],
                               tokenize=False, add_generation_prompt=True,
                               enable_thinking=False)
inputs = tok(text, return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=10, do_sample=False,
                         pad_token_id=tok.eos_token_id)
raw = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
print("\nSMOKE TEST cikti:", repr(raw))
print("\n✅ BLOK 01 TAMAM — model + tokenizer hazir (degisken adlari: model, tok).")
print("   Sonraki blok: veriyi Colab'a getirme (val/test parquet + few-shot ornekleri).")
