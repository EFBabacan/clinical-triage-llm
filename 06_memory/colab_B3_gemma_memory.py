# -*- coding: utf-8 -*-
"""
B3 — GEMMA BELLEK PROFILI  (TIPTEKNO'26)   ***BU BIR SINAMA, OLCUM DEGIL***
===========================================================================

A1'de Qwen'de su model OLCUMLE dogrulandi (tahmin ile olcum arasi fark 0.02 MiB):

    4bit_toplam = gomme_tablosu + (toplam - gomme_tablosu) / 4

Yani 4-bit niceleme YALNIZ linear katmanlari ceyrekliyor; gomme tablosuna
hic dokunmuyor. Qwen'de gomme, bf16 modelin %15.1'i -> kazanc %63.6.

GEMMA TAHMINI (bildiride yazilacak iddia)
-----------------------------------------
Gemma-4-E4B'nin PLE (per-layer embedding) tablosu cok daha buyuk bir pay tutuyor,
dolayisiyla niceleme kazanci Qwen'den KUCUK olmali.
Notlardaki degerlerle: PLE 4.7 GB, toplam bf16 19 GB ->
    tahmin = 4.7 + (19 - 4.7)/4 = 8.3 GB   (yani -%56)

TEZDEKI DEGER ise 19 GB -> 12.4 GB (-%35). Bu mekanizmayla TUTMUYOR:
12.4 GB cikmasi icin nicelenmeyen kismin 10.2 GB olmasi gerekirdi.
Ustelik 19/12.4 degerleri make_final_graphs.py icinde ELLE YAZILMIS sabit,
arkasinda hicbir log yok.

  * Tahmin tutarsa  -> mekanizma kanitlanir, bildiride guclu bir teknik katki.
  * Tutmazsa        -> aciklama yeniden yazilir. OLCMEDEN YAZMA.

A1 ILE KIYASLANABILIRLIK
------------------------
  * batch = 1   (Gemma zero-shot kosusu batch=16 kullanmisti; bellek tepesi
                 batch'e baglidir, bu yuzden A1'deki Qwen ile ayni tutuldu)
  * ayni PROMPT_TEMPLATE, ayni test kayitlari, ayni uzunluk katmanlari
  * duyarlik bf16 (Gemma'nin kendi kosulariyla ayni; Qwen fp16'ydi — bellek
    acisindan ikisi de 2 bayt, fark yok)

KULLANIM (Colab, A100)
----------------------
  !pip install -q -U "bitsandbytes>=0.46.1"
  from huggingface_hub import login; login()      # Gemma GATED model
  %run B3_colab_gemma_memory.py --config bf16
  %run B3_colab_gemma_memory.py --config 4bit     # ayri hucrede
"""
import argparse
import json
import os
import subprocess
import time
from collections import defaultdict

import pandas as pd
import torch

MODEL_ID = "google/gemma-4-E4B-it"
MAX_LEN = 2048
MAX_NEW_TOKENS = 20

# A1'de Qwen icin olculen degerler (MiB) — kiyas icin
QWEN = dict(bf16_agirlik=8021.8, bit4_agirlik=2916.2, gomme=1212.5,
            aktivasyon=1144.8, kazanc_yuzde=63.6, gomme_pay_bf16=15.1)

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

MiB = 1024 ** 2


def smi(q="memory.used"):
    try:
        r = subprocess.run(["nvidia-smi", f"--query-gpu={q}",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=15)
        return float(r.stdout.strip().splitlines()[0])
    except Exception:
        return None


def dosya_bul(ad):
    for k in [os.getcwd(), "/content", "/content/drive/MyDrive",
              "/content/drive/MyDrive/B1",
              os.path.dirname(os.path.abspath(__file__))]:
        p = os.path.join(k, os.path.basename(ad))
        if os.path.exists(p):
            return p
    raise SystemExit(f"Bulunamadi: {ad} — yukleyin (files.upload()) veya tam yol verin.")


def sinifla(ad):
    """Parametreyi bilesene ayir. PLE tablosu AYRI izleniyor — bulgunun kalbi bu."""
    a = ad.lower()
    if "vision" in a or "visual" in a or "audio" in a:
        return "vision/audio"
    if "embed_tokens_per_layer" in a or "per_layer_embed" in a or "ple" in a:
        return "PLE (per-layer embedding)"
    if "embed" in a:
        return "embedding (standart)"
    return "linear/diger"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["bf16", "4bit"], required=True)
    ap.add_argument("--veri", default="b1_test_1650.parquet")
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()

    from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

    OUTDIR = ("/content/drive/MyDrive/B3"
              if os.path.isdir("/content/drive/MyDrive") else "/content/b3_out")
    os.makedirs(OUTDIR, exist_ok=True)

    props = torch.cuda.get_device_properties(0)
    smi_taban = smi()
    print("=" * 76)
    print(f"B3 — GEMMA BELLEK PROFILI   [config={args.config}]")
    print("=" * 76)
    print(f"  GPU          : {props.name}  {props.total_memory/MiB:,.0f} MiB")
    print(f"  Once dolu    : {smi_taban:,.0f} MiB")
    print(f"  Cikti        : {OUTDIR}")

    # ---------------------------------------------------------- tokenizer + veri
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    tok = proc.tokenizer if hasattr(proc, "tokenizer") else proc
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    df = pd.read_parquet(dosya_bul(args.veri))
    df = df[df["split"] == "test"] if "split" in df.columns else df
    df = df.reset_index(drop=True)

    def istem(satir):
        v = satir["triage_input_v4_timeaware_masked"]
        if not isinstance(v, str):
            v = ""
        msgs = [{"role": "user", "content": PROMPT_TEMPLATE.format(input=v)}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    metinler = [istem(r) for _, r in df.iterrows()]
    uz = pd.Series([min(len(tok(t)["input_ids"]), MAX_LEN) for t in metinler])
    print(f"  Veri         : {len(df)} kayit · token ort {uz.mean():.0f} maks {uz.max()}")

    sirali = uz.sort_values().index.tolist()
    k = max(1, args.n // 3)
    katmanlar = [("KISA", sirali[:k]),
                 ("ORTA", sirali[len(sirali)//2 - k//2: len(sirali)//2 - k//2 + k]),
                 ("UZUN", sirali[-k:])]

    # ---------------------------------------------------------- model
    print(f"\n[1/4] Gemma yukleniyor ({args.config})...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    if args.config == "4bit":
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, quantization_config=bnb, device_map="auto", dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, device_map="auto", dtype=torch.bfloat16)
    model.eval()
    print(f"   Yuklendi {time.time()-t0:.1f}s")

    # ---------------------------------------------------------- ayristirma
    print("\n[2/4] PARAMETRE AYRISTIRMASI  (PLE payi bulgunun kalbi)")
    grup = defaultdict(lambda: [0, 0])
    cihaz = defaultdict(int)
    for ad, p in model.named_parameters():
        g = sinifla(ad)
        grup[g][0] += p.numel()
        grup[g][1] += p.numel() * p.element_size()
        cihaz[str(p.device)] += p.numel() * p.element_size()

    toplam_b = sum(v[1] for v in grup.values())
    print(f"   {'bilesen':<28s} {'param':>14s} {'MiB':>10s} {'pay':>7s}")
    print("   " + "-" * 64)
    for g, (n, b) in sorted(grup.items(), key=lambda x: -x[1][1]):
        print(f"   {g:<28s} {n:>14,} {b/MiB:>10,.1f} {100*b/toplam_b:>6.1f}%")
    print("   " + "-" * 64)
    print(f"   {'TOPLAM':<28s} {'':>14s} {toplam_b/MiB:>10,.1f}")
    print(f"\n   Cihaz dagilimi: " + " · ".join(f"{d} {b/MiB:,.0f} MiB" for d, b in cihaz.items()))
    offload = any(not d.startswith("cuda") for d in cihaz)
    if offload:
        print("   !!! OFFLOAD VAR — sayilar 'sigdi' anlamina gelmez.")

    ple_b = grup["PLE (per-layer embedding)"][1]
    emb_b = grup["embedding (standart)"][1]
    vis_b = grup["vision/audio"][1]
    nicelenmeyen = ple_b + emb_b

    # ---------------------------------------------------------- warmup + cikarim
    print("\n[3/4] Warmup + cikarim olcumu (batch=1, A1 ile ayni)")
    w = tok(metinler[sirali[len(sirali)//2]], return_tensors="pt",
            truncation=True, max_length=MAX_LEN).to(model.device)
    with torch.no_grad():
        model.generate(**w, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                       pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    taban = torch.cuda.memory_allocated() / MiB
    print(f"   Taban (model yerlesik): {taban:,.0f} MiB")

    olc = []
    print(f"\n   {'katman':<7s} {'tok':>6s} {'sure_s':>7s} {'peak_alloc':>11s} {'peak_resv':>10s} {'aktiv':>8s}")
    print("   " + "-" * 60)
    for ad, idxs in katmanlar:
        for i in idxs:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            inp = tok(metinler[i], return_tensors="pt", truncation=True,
                      max_length=MAX_LEN).to(model.device)
            n_tok = int(inp["input_ids"].shape[1])
            t1 = time.time()
            with torch.no_grad():
                model.generate(**inp, max_new_tokens=MAX_NEW_TOKENS,
                               do_sample=False, pad_token_id=tok.eos_token_id)
            torch.cuda.synchronize()
            s = time.time() - t1
            pa = torch.cuda.max_memory_allocated() / MiB
            pr = torch.cuda.max_memory_reserved() / MiB
            olc.append(dict(katman=ad, in_tokens=n_tok, sure_s=round(s, 3),
                            peak_allocated_MiB=round(pa, 1),
                            peak_reserved_MiB=round(pr, 1),
                            aktivasyon_MiB=round(pa - taban, 1)))
            print(f"   {ad:<7s} {n_tok:>6d} {s:>7.2f} {pa:>10,.0f}M {pr:>9,.0f}M {pa-taban:>7,.0f}M")

    od = pd.DataFrame(olc)

    # ---------------------------------------------------------- SINAMA
    print("\n[4/4] MEKANIZMA SINAMASI")
    print("=" * 76)
    ozet = dict(config=args.config, model=MODEL_ID, gpu=props.name,
                toplam_agirlik_MiB=round(toplam_b / MiB, 1),
                PLE_MiB=round(ple_b / MiB, 1),
                embedding_MiB=round(emb_b / MiB, 1),
                vision_audio_MiB=round(vis_b / MiB, 1),
                nicelenmeyen_MiB=round(nicelenmeyen / MiB, 1),
                taban_allocated_MiB=round(taban, 1),
                tepe_allocated_MiB=round(float(od["peak_allocated_MiB"].max()), 1),
                tepe_reserved_MiB=round(float(od["peak_reserved_MiB"].max()), 1),
                aktivasyon_maks_MiB=round(float(od["aktivasyon_MiB"].max()), 1),
                gecikme_ort_s=round(float(od["sure_s"].mean()), 3),
                offload=offload)

    yol = os.path.join(OUTDIR, f"b3_gemma_{args.config}.json")
    with open(yol, "w", encoding="utf-8") as f:
        json.dump({"ozet": ozet, "olcumler": olc,
                   "gruplar": {g: {"param": n, "MiB": round(b / MiB, 1)}
                               for g, (n, b) in grup.items()}}, f, indent=2, ensure_ascii=False)
    od.to_csv(os.path.join(OUTDIR, f"b3_gemma_{args.config}_olcumler.csv"),
              index=False, encoding="utf-8-sig")

    print(f"  Agirliklar        : {toplam_b/MiB:,.1f} MiB")
    print(f"  PLE tablosu       : {ple_b/MiB:,.1f} MiB  ({100*ple_b/toplam_b:.1f}%)")
    print(f"  Standart gomme    : {emb_b/MiB:,.1f} MiB")
    print(f"  NICELENMEYEN top. : {nicelenmeyen/MiB:,.1f} MiB  ({100*nicelenmeyen/toplam_b:.1f}%)")
    print(f"  Cikarim tepesi    : {od['peak_reserved_MiB'].max():,.1f} MiB (reserved)")

    bf_yol = os.path.join(OUTDIR, "b3_gemma_bf16.json")
    if args.config == "4bit" and os.path.exists(bf_yol):
        bf = json.load(open(bf_yol, encoding="utf-8"))["ozet"]
        T = bf["toplam_agirlik_MiB"]
        E = bf["nicelenmeyen_MiB"]
        B4 = ozet["toplam_agirlik_MiB"]
        tahmin = E + (T - E) / 4
        gercek_kazanc = 100 * (T - B4) / T
        print("\n" + "=" * 76)
        print("  MEKANIZMA SINAMASI — A1'de Qwen'de dogrulanan model")
        print("=" * 76)
        print(f"    formul   : nicelenmeyen + (toplam - nicelenmeyen)/4")
        print(f"    bf16     : {T:,.1f} MiB  (nicelenmeyen {E:,.1f})")
        print(f"    TAHMIN   : {tahmin:,.1f} MiB")
        print(f"    OLCUM    : {B4:,.1f} MiB")
        print(f"    fark     : {tahmin - B4:+,.1f} MiB")
        if abs(tahmin - B4) < 0.02 * T:
            print("    ==> TAHMIN TUTTU. Mekanizma Gemma'da da gecerli, KANITLANDI.")
        else:
            print("    ==> TUTMADI. Aciklama yeniden yazilmali — OLCMEDEN YAZMA.")
        print(f"\n    Niceleme kazanci : %{gercek_kazanc:.1f}")
        print(f"    Qwen (A1)        : %{QWEN['kazanc_yuzde']:.1f}")
        print(f"    Tezdeki iddia    : 19 GB -> 12.4 GB (-%35), kaynagi YOK")
        print(f"\n    Nicelenmeyen pay : Gemma %{100*E/T:.1f}  ·  Qwen %{QWEN['gomme_pay_bf16']:.1f}")
        print("    => Pay ne kadar buyukse niceleme kazanci o kadar kucuk. Bildirinin iddiasi bu.")

    print(f"\n  Kaydedildi -> {yol}")
    print("  !! Drive'dan yerele indirin.")
    print("=" * 76)


if __name__ == "__main__":
    main()
