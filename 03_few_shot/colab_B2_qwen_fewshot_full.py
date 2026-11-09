# -*- coding: utf-8 -*-
"""
B2 — TAM 3-SHOT FEW-SHOT, TEST YENIDEN KOSUSU  (TIPTEKNO'26)
=============================================================

NEDEN
-----
Ana tablodaki 8 satirin 7'si ham per-kayit veriden dogrulandi. Bir satir haric:
Qwen tam 3-shot (0.5552 / 0.5621). Onun per-kayit CSV'si Colab oturumuyla kaybolmus,
elde yalniz ozet metrik var. Iki sonucu var:

  1. Bu satir DOGRULANAMIYOR. Denetimde sistematik bir metrik hatasi bulundugu
     icin (macro-F1 paydasi 11 yerine 12), dogrulanmamis satir birakmak kabul edilemez.
  2. B1'in mansset iddiasi -- sikistirma -5.9 puan -- su an ESLESMEMIS guven
     araligina dayaniyor. Oysa ayni test setinde dogru arac McNemar. Nitekim
     Gemma bf16-vs-4bit ornegi bunu gosterdi: bagimsiz CI "gurultu" dedi,
     eslesmis McNemar p=2.7e-5 verdi.

Bu kosu her ikisini de cozer.

ISTEM YENIDEN KURULMUYOR
------------------------
full_3shot_PROMPT_FROZEN.txt yerelde uretildi ve 60 val kaydinin in_tokens
degerleriyle KAYIT KAYIT dogrulandi (60/60 birebir, fark 0). Sabit istem 23.196 token.

ORIJINAL KOSUYLA AYNI
---------------------
  batch=4 (sol-padding) · MAX_LEN=32768 · max_new_tokens=20 · greedy
  ayni HEADER (4 kural) · ayni EX_TMPL · ayni parser · tam V4 sorgusu (sikistirilmamis)

KULLANIM (Colab A100)
---------------------
  !pip install -q -U "bitsandbytes>=0.46.1"
  %run B2_colab_full_fewshot.py
Sure ~140 dk. Kesinti olursa ayni komut kaldigi yerden devam eder.
"""
import argparse
import json
import os
import time

import pandas as pd
import torch

MODEL_ID = "Qwen/Qwen3.5-4B"
MAX_LEN = 32768
MAX_NEW_TOKENS = 20
TT = "\nNow classify:\n{input}\n=> "
V4COL = "triage_input_v4_timeaware_masked"
SABIT_BEKLENEN = 23196      # yerelde dogrulandi

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


def dosya_bul(ad):
    for k in [os.getcwd(), "/content", "/content/b1_colab_upload",
              "/content/drive/MyDrive", "/content/drive/MyDrive/B1",
              os.path.dirname(os.path.abspath(__file__))]:
        p = os.path.join(k, os.path.basename(ad))
        if os.path.exists(p):
            return p
    raise SystemExit(f"Bulunamadi: {ad} — yukleyin veya tam yol verin.")


def paket_kontrol(gerekli=("torch", "transformers", "bitsandbytes", "accelerate")):
    """Model yuklemeden ONCE patla — 2 dk agirlik indirdikten sonra degil."""
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
        print("Su hucreyi calistirip tekrar deneyin:\n")
        print('  !pip install -q -U "bitsandbytes>=0.46.1" accelerate peft\n')
        print("NOT: transformers'i GUNCELLEMEYIN — surum degisikligi davranisi")
        print("     degistirir ve onceki kosularla kiyasi bozar.")
        raise SystemExit(1)
    return surum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--veri", default="b1_test_1650.parquet")
    ap.add_argument("--istem", default="full_3shot_PROMPT_FROZEN.txt")
    ap.add_argument("--batch", type=int, default=4, help="orijinal kosu 4 kullandi")
    args = ap.parse_args()

    surum = paket_kontrol()      # importlardan ONCE

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    OUTDIR = ("/content/drive/MyDrive/B2"
              if os.path.isdir("/content/drive/MyDrive") else "/content/b2_out")
    os.makedirs(OUTDIR, exist_ok=True)

    print("=" * 74)
    print("B2 — TAM 3-SHOT FEW-SHOT, TEST")
    print("=" * 74)
    p = torch.cuda.get_device_properties(0)
    print(f"  GPU     : {p.name}  {p.total_memory/1024**3:.1f} GiB")
    print(f"  Cikti   : {OUTDIR}")
    print("  Paket   : " + " · ".join(f"{k} {v}" for k, v in surum.items()))

    SABIT = open(dosya_bul(args.istem), encoding="utf-8").read()
    print(f"  Istem   : {len(SABIT):,} karakter — YENIDEN KURULMADI")

    df = pd.read_parquet(dosya_bul(args.veri))
    df = df[df["split"] == "test"] if "split" in df.columns else df
    df = df.reset_index(drop=True)
    print(f"  Veri    : {len(df)} kayit · batch={args.batch}")

    print("\n[1/3] Model yukleniyor (4-bit NF4, fp16 compute)...")
    t0 = time.time()
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"        # batch generation icin sart
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto", dtype=torch.float16)
    model.eval()
    print(f"   Yuklendi {time.time()-t0:.1f}s")

    def build_text(q):
        return tok.apply_chat_template(
            [{"role": "user", "content": SABIT + TT.format(input=q)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)

    sabit_n = len(tok(build_text(""))["input_ids"])
    ok = abs(sabit_n - SABIT_BEKLENEN) <= 2
    print(f"   Sabit istem: {sabit_n:,} token (yerelde {SABIT_BEKLENEN:,}) "
          f"{'OK' if ok else '<-- FARKLI! dur ve incele'}")

    cp = os.path.join(OUTDIR, "b2_full3shot_test_checkpoint.csv")
    son = os.path.join(OUTDIR, "b2_full3shot_test.csv")
    if os.path.exists(cp):
        e = pd.read_csv(cp)
        islenen = set(e["note_id"].astype(str))
        kayitlar = e.to_dict("records")
        print(f"   Resume: {len(islenen)} kayit zaten var.")
    else:
        islenen, kayitlar = set(), []

    kalan = df[~df["note_id"].astype(str).isin(islenen)].reset_index(drop=True)
    print(f"\n[2/3] Cikarim — {len(kalan)} kayit kaldi (~{len(kalan)*5.1/60:.0f} dk)")
    t_bas = time.time()
    yeni = 0

    for s in range(0, len(kalan), args.batch):
        blok = kalan.iloc[s:s + args.batch]
        sorgular = [(r[V4COL] if isinstance(r[V4COL], str) else "")
                    for _, r in blok.iterrows()]
        texts = [build_text(q) for q in sorgular]
        enc = tok(texts, return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_LEN).to("cuda")
        t1 = time.time()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        dt = time.time() - t1
        gen = out[:, enc.input_ids.shape[1]:]
        for j, (_, r) in enumerate(blok.iterrows()):
            raw = tok.decode(gen[j], skip_special_tokens=True).strip()
            pred, kat = parse_class(raw)
            kayitlar.append({
                "note_id": str(r["note_id"]), "subject_id": int(r["subject_id"]),
                "hadm_id": int(r["hadm_id"]), "label": r["label"],
                "raw_output": raw, "parsed_pred": pred, "parse_kategori": kat,
                "in_tokens": int(enc.attention_mask[j].sum().item()),
                # batch kosusunda per-ornek sure yok; blok suresi / blok boyu
                "sure_s": round(dt / len(blok), 3),
                "has_think": ("<think>" in raw or "</think>" in raw),
            })
        yeni += len(blok)
        if (s // args.batch) % 10 == 0 or s + args.batch >= len(kalan):
            pd.DataFrame(kayitlar).to_csv(cp, index=False, encoding="utf-8-sig")
            gec = time.time() - t_bas
            ort = gec / max(yeni, 1)
            print(f"   [{len(kayitlar):>4}/{len(df)}] gecen {gec/60:5.1f} dk · "
                  f"{ort:.2f} s/ornek · kalan ~{(len(kalan)-yeni)*ort/60:.1f} dk")

    d = pd.DataFrame(kayitlar)
    d.to_csv(son, index=False, encoding="utf-8-sig")
    d.to_csv(cp, index=False, encoding="utf-8-sig")
    print(f"   BITTI. {len(d)} kayit, {(time.time()-t_bas)/60:.1f} dk")

    from sklearn.metrics import f1_score
    acc = float((d["label"] == d["parsed_pred"]).mean())
    mf1 = f1_score(d["label"], d["parsed_pred"], average="macro",
                   labels=ALLOWED_CLASSES, zero_division=0)
    inv = int((d["parsed_pred"] == "INVALID").sum())
    direct = int((d["parse_kategori"] == "direct").sum())

    print("\n[3/3] SONUC")
    print("=" * 74)
    print(f"  n={len(d)}  acc={acc:.4f}  macroF1={mf1:.4f}  INVALID={inv}  "
          f"direct=%{100*direct/len(d):.1f}")
    print(f"  istem token: ort={d['in_tokens'].mean():,.0f} maks={d['in_tokens'].max():,}")
    print(f"  gecikme    : {d['sure_s'].mean():.2f} s/ornek · toplam {d['sure_s'].sum()/60:.1f} dk")
    print("\n  --- ORIJINAL KOSUYLA KIYAS ---")
    print(f"  raporlanan : acc 0.5552  macroF1 0.5621 (wF1 sutunundan tureti)")
    print(f"  bu kosu    : acc {acc:.4f}  macroF1 {mf1:.4f}")
    print(f"  fark       : {100*(acc-0.5552):+.1f} puan acc · {100*(mf1-0.5621):+.1f} puan MF1")
    if abs(acc - 0.5552) < 0.015:
        print("  ==> DOGRULANDI. Son dogrulanmamis satir kapandi.")
    else:
        print("  ==> FARK BUYUK. Sebebini bulmadan tabloya yazmayin.")

    ozet = dict(n=len(d), accuracy=round(acc, 4), macro_f1=round(mf1, 4),
                invalid=inv, direct_oran=round(direct / len(d), 4),
                istem_token_ort=round(float(d["in_tokens"].mean()), 1),
                gecikme_s=round(float(d["sure_s"].mean()), 3),
                toplam_dk=round(float(d["sure_s"].sum()) / 60, 1),
                sabit_istem_token=sabit_n, batch=args.batch,
                gpu=torch.cuda.get_device_name(0))
    with open(os.path.join(OUTDIR, "b2_full3shot_test_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(ozet, f, indent=2, ensure_ascii=False)
    print(f"\n  Kaydedildi -> {OUTDIR}  !! yerele indirin")
    print("=" * 74)


if __name__ == "__main__":
    main()
