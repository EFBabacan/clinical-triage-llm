# =====================================================================
# BLOK G01 — Gemma-4-E4B-it kurulum + model yukleme (Colab A100)
# =====================================================================
# Ikinci model karsilastirmasi: Qwen3.5-4B'ye karsi Gemma-4-E4B-it.
# Local'de E2B 8GB'a sigmamisti (PLE embedding ~4.7GB); A100 80GB'da E4B rahat sigar.
#
# Gemma'ya OZGU farklar (Qwen'den):
#   - AutoProcessor (tokenizer DEGIL — Gemma multimodal, processor kullanir)
#   - AutoModelForCausalLM ama TEXT-ONLY kullanim
#   - dtype = bfloat16 (Gemma standardi; Qwen fp16'ydi)
#   - enable_thinking=False destekleniyor (Qwen ile ayni)
# Bu blok: ONCE full bf16 (quantization yok). 4-bit varyanti ayri blokta (G01b).
# =====================================================================
import subprocess
print(subprocess.run(["nvidia-smi","--query-gpu=name,memory.used,memory.total","--format=csv"],
                     capture_output=True, text=True).stdout)

# Gemma 4 yeni mimari -> guncel transformers sart
# [Colab] !pip -q install -U "transformers" "accelerate" "sentencepiece"

import torch, time
from transformers import AutoProcessor, AutoModelForCausalLM

GEMMA_ID = "google/gemma-4-E4B-it"   # instruct (few-shot/zero-shot icin sart)

# HF token gerekebilir (Gemma gated model). Gerekirse:
# from huggingface_hub import login; login("hf_xxx")

print(f"\n[1/3] Processor yukleniyor: {GEMMA_ID}")
g_proc = AutoProcessor.from_pretrained(GEMMA_ID)
# text-only icin tokenizer'a erisim (chat template + decode):
g_tok = g_proc.tokenizer if hasattr(g_proc, "tokenizer") else g_proc

print(f"[2/3] Model yukleniyor (FULL bf16, quantization YOK)...")
t0 = time.time()
g_model = AutoModelForCausalLM.from_pretrained(
    GEMMA_ID,
    dtype=torch.bfloat16,        # Gemma standardi (yeni API: dtype, torch_dtype degil)
    device_map="auto",
)
g_model.eval()
print(f"   Yuklendi: {time.time()-t0:.0f} sn")
print(f"   GPU bellek: {torch.cuda.memory_allocated()/1e9:.2f} GB / "
      f"{torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB")
print(f"   Model class: {g_model.__class__.__name__}")

# ---------- Smoke test: chat template + enable_thinking=False ----------
print(f"\n[3/3] Smoke test...")
msgs = [{"role":"user","content":"Reply with one word: OK"}]
try:
    text = g_proc.apply_chat_template(msgs, tokenize=False,
                                      add_generation_prompt=True, enable_thinking=False)
except TypeError:
    # enable_thinking desteklenmezse parametresiz dene
    text = g_proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    print("   (not: enable_thinking parametresi yok, parametresiz cagrildi)")

inputs = g_tok(text, return_tensors="pt").to(g_model.device)
with torch.no_grad():
    out = g_model.generate(**inputs, max_new_tokens=10, do_sample=False,
                           pad_token_id=g_tok.eos_token_id)
raw = g_tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
print(f"   SMOKE TEST cikti: {repr(raw)}")
print(f"\n   Chat template ornegi (ilk 300 char):\n{text[:300]}")

print("\n✅ BLOK G01 TAMAM — Gemma hazir (degiskenler: g_model, g_proc, g_tok, GEMMA_ID).")
print("   Sonraki: BlokG02 — Gemma zero-shot val (Qwen ile birebir ayni prompt/parser/veri).")
