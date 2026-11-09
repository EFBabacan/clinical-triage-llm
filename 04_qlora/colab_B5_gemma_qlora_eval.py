# -*- coding: utf-8 -*-
"""
B5 — GEMMA ADIL QLoRA, TEST YENIDEN DEGERLENDIRME  (TIPTEKNO'26)
=================================================================

NEDEN
-----
Ana tablodaki Gemma adil QLoRA satiri (0.6285 / 0.5653) per-kayit CSV'si olmadigi
icin DOGRULANAMIYOR. Macro-F1'i "parser %100 direct -> INVALID yok -> payda zaten
11" cikarimiyla degismedi kabul ettik. Bu saglam bir akil yurutme ama DOGRULAMA
degil -- ve bu bildiride sistematik bir metrik hatasi bulundugu icin dogrulanmamis
satir birakmak kabul edilemez.

Ayrica: Qwen QLoRA (0.6424) vs Gemma adil QLoRA (0.6285) karsilastirmasi su an
bagimsiz CI ile "gurultu" diyor. Per-kayit veri gelince ESLESMIS McNemar
yapilabilecek -- ayni test setinde dogru test bu.

GEMMA'YA OZEL KRITIK AYRINTI
----------------------------
apply_chat_template TOKENIZE=TRUE ile cagrilmali. tokenize=False + yeniden
tokenize etmek Gemma4 ozel token'larini (<|turn>=105, <turn|>=106) parcaliyor;
bu hata daha once egitimi bozmustu (loss 1.55'te takilmisti, blokG06b-v3 notu).
Burada ayni tuzaga dusulmuyor.

ADAPTER
-------
gemma_qlora_adapter/  (r=8, alpha=16, q_proj+v_proj, 66 modul, yalniz
language_model — vision/audio HARIC). adapter_config.json ile dogrulandi.

KULLANIM (Colab A100)
---------------------
  !pip install -q -U "bitsandbytes>=0.46.1" peft
  from huggingface_hub import login; login()          # Gemma gated
  # gemma_qlora_adapter klasorunu yukleyin (veya Drive'a koyun)
  %run B5_colab_gemma_qlora_eval.py --adapter gemma_qlora_adapter
Sure ~40 dk.
"""
import argparse
import json
import os
import time

import pandas as pd
import torch

GEMMA_ID = "google/gemma-4-E4B-it"
MAX_SEQ = 2048
MAX_NEW_TOKENS = 20
V4COL = "triage_input_v4_timeaware_masked"

# egitimle birebir ayni instruction (qlora_v4_*.json)
INSTRUCTION = (
    "Classify the following admission note into exactly one of these 11 classes: "
    "SEPSIS, STROKE, ATRIAL_FIBRILLATION, AKI, PNEUMONIA, UTI, HEART_FAILURE, "
    "GI_BLEEDING, DIABETES_COMPLICATION, COPD_EXACERBATION, OTHER. "
    "Output only the class name."
)

ALLOWED_CLASSES = ["SEPSIS", "STROKE", "ATRIAL_FIBRILLATION", "AKI", "PNEUMONIA",
                   "UTI", "HEART_FAILURE", "GI_BLEEDING", "DIABETES_COMPLICATION",
                   "COPD_EXACERBATION", "OTHER"]
FUZZY_MAP = {
    "SEPS": "SEPSIS", "SEPT": "SEPSIS", "PNEUM": "PNEUMONIA",
    "STROK": "STROKE", "CVA": "STROKE",
    "ATRIAL": "ATRIAL_FIBRILLATION", "AFIB": "ATRIAL_FIBRILLATION", "FIBRIL": "ATRIAL_FIBRILLATION",
    "AKI": "AKI", "KIDNEY": "AKI", "RENAL": "AKI",
    "UTI": "UTI", "URIN": "UTI",
    "HEART": "HEART_FAILURE", "CHF": "HEART_FAILURE",
    "GI": "GI_BLEEDING", "BLEED": "GI_BLEEDING",
    "DIAB": "DIABETES_COMPLICATION", "DKA": "DIABETES_COMPLICATION",
    "COPD": "COPD_EXACERBATION", "OTHER": "OTHER", "NONE": "OTHER",
}


def parse_class(raw):
    if not raw:
        return "INVALID", "empty"
    t = raw.strip().upper()
    for c in sorted(ALLOWED_CLASSES, key=len, reverse=True):
        if c in t[:80]:
            return c, "direct"
    for k, v in FUZZY_MAP.items():
        if k in t[:80]:
            return v, "fuzzy"
    return "INVALID", "invalid"


def yol_bul(ad, klasor=False):
    for k in [os.getcwd(), "/content", "/content/b1_colab_upload",
              "/content/drive/MyDrive", "/content/drive/MyDrive/B5",
              os.path.dirname(os.path.abspath(__file__))]:
        p = os.path.join(k, os.path.basename(ad.rstrip("/\\")))
        if (os.path.isdir(p) if klasor else os.path.exists(p)):
            return p
    raise SystemExit(f"Bulunamadi: {ad}")


def paket_kontrol(gerekli=("torch", "transformers", "bitsandbytes", "accelerate", "peft")):
    """Model yuklemeden ONCE patla."""
    import importlib
    surum, eksik = {}, []
    for m in gerekli:
        try:
            surum[m] = getattr(importlib.import_module(m), "__version__", "?")
        except Exception:
            eksik.append(m)
    if eksik:
        print("\n" + "!" * 74)
        print(f"EKSIK PAKET: {', '.join(eksik)}")
        print("!" * 74)
        print('  !pip install -q -U "bitsandbytes>=0.46.1" accelerate peft\n')
        print("NOT: transformers'i GUNCELLEMEYIN.")
        raise SystemExit(1)
    return surum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="gemma_qlora_adapter")
    ap.add_argument("--veri", default="b1_test_1650.parquet")
    args = ap.parse_args()

    surum = paket_kontrol()      # importlardan ONCE — eksik paket net mesaj versin

    from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    OUTDIR = ("/content/drive/MyDrive/B5"
              if os.path.isdir("/content/drive/MyDrive") else "/content/b5_out")
    os.makedirs(OUTDIR, exist_ok=True)

    print("=" * 74)
    print("B5 — GEMMA ADIL QLoRA, TEST")
    print("=" * 74)
    p = torch.cuda.get_device_properties(0)
    print(f"  GPU     : {p.name}  {p.total_memory/1024**3:.1f} GiB")
    print("  Paket   : " + " · ".join(f"{k} {v}" for k, v in surum.items()))

    adapter = yol_bul(args.adapter, klasor=True)
    ac = json.load(open(os.path.join(adapter, "adapter_config.json"), encoding="utf-8"))
    print(f"  Adapter : {adapter}")
    print(f"            r={ac['r']} alpha={ac['lora_alpha']} "
          f"modul={len(ac['target_modules'])} dropout={ac['lora_dropout']}")
    if ac["r"] != 8 or ac["lora_alpha"] != 16:
        print("  !! UYARI: adil ayar r=8/alpha=16 bekleniyordu. Yanlis adapter olabilir.")

    df = pd.read_parquet(yol_bul(args.veri))
    df = df[df["split"] == "test"] if "split" in df.columns else df
    df = df.reset_index(drop=True)
    print(f"  Veri    : {len(df)} kayit")

    print("\n[1/3] Gemma + adapter yukleniyor (4-bit NF4, bf16 compute)...")
    t0 = time.time()
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    proc = AutoProcessor.from_pretrained(GEMMA_ID)
    tok = proc.tokenizer if hasattr(proc, "tokenizer") else proc
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        GEMMA_ID, quantization_config=bnb, device_map="auto", dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()
    print(f"   Yuklendi {time.time()-t0:.1f}s · GPU {torch.cuda.memory_allocated()/1024**2:,.0f} MiB")

    def ids_uret(user_msg):
        """KRITIK: tokenize=True. String uzerinden yeniden tokenize ETME —
        Gemma4 ozel token'lari parcalanir (blokG06b-v3 kok sebep notu)."""
        x = proc.apply_chat_template([{"role": "user", "content": user_msg}],
                                     tokenize=True, add_generation_prompt=True)
        if hasattr(x, "tolist"):
            x = x.tolist()
        if x and isinstance(x[0], list):
            x = x[0]
        return x

    cp = os.path.join(OUTDIR, "b5_gemma_qlora_test_checkpoint.csv")
    son = os.path.join(OUTDIR, "b5_gemma_qlora_test.csv")
    if os.path.exists(cp):
        e = pd.read_csv(cp)
        islenen = set(e["note_id"].astype(str))
        kayitlar = e.to_dict("records")
        print(f"   Resume: {len(islenen)} kayit zaten var.")
    else:
        islenen, kayitlar = set(), []

    print(f"\n[2/3] Cikarim (tek ornek, tokenize=True)")
    t_bas = time.time()
    yeni = 0
    for _, r in df.iterrows():
        nid = str(r["note_id"])
        if nid in islenen:
            continue
        v4 = r[V4COL] if isinstance(r[V4COL], str) else ""
        ids = ids_uret(f"{INSTRUCTION}\n\n{v4}")
        kesildi = len(ids) > MAX_SEQ
        if kesildi:
            ids = ids[:MAX_SEQ]
        inp = torch.tensor([ids], device=model.device)
        t1 = time.time()
        with torch.no_grad():
            out = model.generate(input_ids=inp, max_new_tokens=MAX_NEW_TOKENS,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        s = time.time() - t1
        raw = tok.decode(out[0][len(ids):], skip_special_tokens=True).strip()
        pred, kat = parse_class(raw)
        kayitlar.append({
            "note_id": nid, "subject_id": int(r["subject_id"]),
            "hadm_id": int(r["hadm_id"]), "label": r["label"],
            "raw_output": raw, "parsed_pred": pred, "parse_kategori": kat,
            "in_tokens": len(ids), "sure_s": round(s, 3), "kesildi": bool(kesildi),
        })
        yeni += 1
        if len(kayitlar) % 50 == 0:
            pd.DataFrame(kayitlar).to_csv(cp, index=False, encoding="utf-8-sig")
            gec = time.time() - t_bas
            ort = gec / yeni
            print(f"   [{len(kayitlar):>4}/{len(df)}] gecen {gec/60:5.1f} dk · "
                  f"{ort:.2f} s/ornek · kalan ~{(len(df)-len(kayitlar))*ort/60:.1f} dk")

    d = pd.DataFrame(kayitlar)
    d.to_csv(son, index=False, encoding="utf-8-sig")
    d.to_csv(cp, index=False, encoding="utf-8-sig")
    print(f"   BITTI. {len(d)} kayit, {(time.time()-t_bas)/60:.1f} dk")

    from sklearn.metrics import f1_score
    acc = float((d["label"] == d["parsed_pred"]).mean())
    mf1 = f1_score(d["label"], d["parsed_pred"], average="macro",
                   labels=ALLOWED_CLASSES, zero_division=0)
    inv = int((d["parsed_pred"] == "INVALID").sum())

    print("\n[3/3] SONUC")
    print("=" * 74)
    print(f"  n={len(d)}  acc={acc:.4f}  macroF1={mf1:.4f}  INVALID={inv}  "
          f"kesilen={int(d['kesildi'].sum())}")
    print(f"  gecikme: {d['sure_s'].mean():.2f} s/ornek · {d['sure_s'].sum()/60:.1f} dk")
    print("\n  --- ORIJINAL KOSUYLA KIYAS ---")
    print(f"  raporlanan : acc 0.6285  macroF1 0.5653")
    print(f"  bu kosu    : acc {acc:.4f}  macroF1 {mf1:.4f}")
    print(f"  fark       : {100*(acc-0.6285):+.1f} puan acc · {100*(mf1-0.5653):+.1f} puan MF1")
    if abs(acc - 0.6285) < 0.015:
        print("  ==> DOGRULANDI. Ana tablonun son dogrulanmamis satiri da kapandi.")
    else:
        print("  ==> FARK BUYUK. Sebebini bulmadan tabloya yazmayin.")
    print("\n  Qwen QLoRA (0.6424) ile eslesmis McNemar'i YERELDE yapin —")
    print("  Qwen'in per-kayit CSV'si orada.")

    ozet = dict(n=len(d), accuracy=round(acc, 4), macro_f1=round(mf1, 4),
                invalid=inv, kesilen=int(d["kesildi"].sum()),
                istem_token_ort=round(float(d["in_tokens"].mean()), 1),
                gecikme_s=round(float(d["sure_s"].mean()), 3),
                toplam_dk=round(float(d["sure_s"].sum()) / 60, 1),
                adapter_r=ac["r"], adapter_alpha=ac["lora_alpha"],
                gpu=torch.cuda.get_device_name(0))
    with open(os.path.join(OUTDIR, "b5_gemma_qlora_test_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(ozet, f, indent=2, ensure_ascii=False)
    print(f"\n  Kaydedildi -> {OUTDIR}  !! yerele indirin")
    print("=" * 74)


if __name__ == "__main__":
    main()
