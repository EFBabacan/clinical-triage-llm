# -*- coding: utf-8 -*-
"""
A6 — QLoRA EVAL'INI TAM BAGLAMLA YENIDEN KOSTUR

NEDEN
-----
Raporlanan QLoRA sonucu (test acc 0.6345 / MF1 0.6220) test notlarinin %86'si
409 token'a KESILMISKEN uretilmis (76 talimat + 333 govde sert tavan).
Zero-shot ayni test setinde 2048 butcesiyle kosmus. Iki yontem farkli baglamla
olculmus -> karsilastirma esit kosullu degil.
Ayrica egitim TAM uzunlukta yapilmis (govde ort. 605, maks 2680 token)
-> egitim/degerlendirme uyumsuzlugu.

Bu script ayni adapter ile ayni test setini TAM baglamla (2048) yeniden degerlendirir.
Kesme disinda her sey birebir ayni: ayni INSTRUCTION, ayni parser, ayni greedy decode.

ESKI KOSUDAKI KESME HATASI TEKRARLANMIYOR
------------------------------------------
Eski akis prompt'u kurup sonundan kesiyordu; bu, sondaki uretim isaretcisini
(assistant turn) da kesiyor (FINAL_SONUCLAR "cozulmus tuzaklar" #2 ile ayni hata).
Burada NOT GOVDESI kesiliyor, talimat ve uretim isaretcisi korunuyor.

KULLANIM
--------
    python eval_qlora_fullcontext.py                 # test, 2048
    python eval_qlora_fullcontext.py --split val
    python eval_qlora_fullcontext.py --max-seq 4096  # tavani yukselt

Cikti: output/qlora_eval_fullctx/  (ESKI SONUCLARIN USTUNE YAZMAZ)
Kesinti olursa ayni komutla devam eder (checkpoint/resume).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE = Path(__file__).resolve().parent
OUT = BASE / "output"
SPLIT_YOL = OUT / "discharge_split.parquet"
ADAPTER = OUT / "qlora_adapters" / "v4_s3_full_2epoch_lr1e4" / "adapter"
ESKI_EVAL = OUT / "qlora_eval" / "v4_s3_full_2epoch_lr1e4"
YENI_EVAL = OUT / "qlora_eval_fullctx"
YENI_EVAL.mkdir(parents=True, exist_ok=True)

MODEL_ID = "Qwen/Qwen3.5-4B"
MAX_NEW_TOKENS = 20

ALLOWED_CLASSES = [
    "SEPSIS", "STROKE", "ATRIAL_FIBRILLATION", "AKI", "PNEUMONIA", "UTI",
    "HEART_FAILURE", "GI_BLEEDING", "DIABETES_COMPLICATION", "COPD_EXACERBATION", "OTHER",
]

# egitimle birebir ayni (qlora_v4_*.json icindeki instruction)
INSTRUCTION = (
    "Classify the following admission note into exactly one of these 11 classes: "
    "SEPSIS, STROKE, ATRIAL_FIBRILLATION, AKI, PNEUMONIA, UTI, HEART_FAILURE, "
    "GI_BLEEDING, DIABETES_COMPLICATION, COPD_EXACERBATION, OTHER. "
    "Output only the class name."
)

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
    """Tezdeki parser ile birebir ayni."""
    if not raw:
        return "INVALID", "empty"
    text = raw.strip().upper()
    for cls in sorted(ALLOWED_CLASSES, key=len, reverse=True):
        if cls in text[:80]:
            return cls, "direct"
    for k, v in FUZZY_MAP.items():
        if k in text[:80]:
            return v, "fuzzy"
    return "INVALID", "invalid"


def smi_used():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=15)
        return float(r.stdout.strip().splitlines()[0])
    except Exception:
        return None


def sablon(tok, govde):
    msgs = [{"role": "user", "content": f"{INSTRUCTION}\n\n{govde}"}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--max-taban", type=float, default=600,
                    help="olcum oncesi GPU'da izin verilen azami dolu MiB")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    taban = smi_used()
    print("=" * 78)
    print(f"A6 — QLoRA TAM BAGLAM EVAL   [split={args.split}  max_seq={args.max_seq}]")
    print("=" * 78)
    print(f"  Adapter        : {ADAPTER}")
    print(f"  GPU dolu (once): {taban:.0f} MiB" if taban else "  GPU: okunamadi")
    if taban is not None and taban > args.max_taban and not args.force:
        sys.exit(f"\nIPTAL — GPU bos degil ({taban:.0f} MiB). Kapatip tekrar deneyin "
                 f"veya --force kullanin.")
    if not ADAPTER.exists():
        sys.exit(f"Adapter bulunamadi: {ADAPTER}")

    # ---------------------------------------------------------- model
    print("\n[1/4] Model + adapter yukleniyor (4-bit NF4, fp16 compute)...")
    t0 = time.time()
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto",
        dtype=torch.float16, local_files_only=True)
    model = PeftModel.from_pretrained(base, str(ADAPTER))
    model.eval()
    print(f"   Yuklendi {time.time()-t0:.1f}s · GPU {torch.cuda.memory_allocated()/1024**2:.0f} MiB")

    # sabit maliyet: sablon + talimat (govde bos)
    sabit = len(tok(sablon(tok, ""))["input_ids"])
    govde_butce = args.max_seq - sabit
    print(f"   Sabit maliyet (sablon+talimat): {sabit} token")
    print(f"   Govde butcesi                 : {govde_butce} token")

    # ---------------------------------------------------------- veri
    print(f"\n[2/4] {args.split} seti...")
    df = pd.read_parquet(SPLIT_YOL)
    sub = df[df["split"] == args.split].reset_index(drop=True)
    print(f"   n = {len(sub)}")

    cp_yol = YENI_EVAL / f"{args.split}_predictions_checkpoint.csv"
    son_yol = YENI_EVAL / f"{args.split}_predictions.csv"
    if cp_yol.exists():
        cp = pd.read_csv(cp_yol)
        islenen = set(cp["note_id"].astype(str))
        kayitlar = cp.to_dict("records")
        print(f"   Resume: {len(islenen)} kayit zaten var.")
    else:
        islenen, kayitlar = set(), []

    # ---------------------------------------------------------- cikarim
    print(f"\n[3/4] Cikarim (checkpoint/resume her 50 kayitta)")
    t_bas = time.time()
    yeni = 0
    n_kesilen = 0

    for _, row in sub.iterrows():
        nid = str(row["note_id"])
        if nid in islenen:
            continue
        govde = row["triage_input_v4_timeaware_masked"]
        if not isinstance(govde, str):
            govde = ""

        # --- GOVDEYI kes, sablonu DEGIL (eski kosudaki hata buydu) ---
        gid = tok(govde, add_special_tokens=False)["input_ids"]
        kesildi = len(gid) > govde_butce
        if kesildi:
            govde = tok.decode(gid[:govde_butce], skip_special_tokens=True)
            n_kesilen += 1

        text = sablon(tok, govde)
        inputs = tok(text, return_tensors="pt").to(model.device)
        in_n = int(inputs["input_ids"].shape[1])

        t1 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        sure = time.time() - t1
        raw = tok.decode(out[0][in_n:], skip_special_tokens=True).strip()
        pred, kat = parse_class(raw)

        kayitlar.append({
            "note_id": nid, "gold": row["label"], "pred": pred,
            "parse_kategori": kat, "raw": raw,
            "input_tokens": in_n, "sure_s": round(sure, 3),
            "kesildi": bool(kesildi),
            "has_think": ("<think>" in raw or "</think>" in raw),
        })
        yeni += 1
        if len(kayitlar) % 50 == 0:
            pd.DataFrame(kayitlar).to_csv(cp_yol, index=False, encoding="utf-8-sig")
            gecen = time.time() - t_bas
            ort = gecen / yeni
            kalan = (len(sub) - len(kayitlar)) * ort
            print(f"   [{len(kayitlar):>4}/{len(sub)}] gecen {gecen/60:5.1f} dk · "
                  f"ort {ort:.2f} s/ornek · kalan ~{kalan/60:.1f} dk")

    d = pd.DataFrame(kayitlar)
    d.to_csv(son_yol, index=False, encoding="utf-8-sig")
    d.to_csv(cp_yol, index=False, encoding="utf-8-sig")
    toplam_dk = (time.time() - t_bas) / 60
    print(f"   BITTI. {len(d)} kayit, {toplam_dk:.1f} dk")

    # ---------------------------------------------------------- metrikler
    print(f"\n[4/4] METRIKLER  (macro-F1 labels=CLASSES ile, payda 11)")
    print("=" * 78)
    acc = float((d["gold"] == d["pred"]).mean())
    mf1 = f1_score(d["gold"], d["pred"], average="macro",
                   labels=ALLOWED_CLASSES, zero_division=0)
    gecerli = int((d["pred"] != "INVALID").sum())
    print(f"  n={len(d)}  acc={acc:.4f}  macroF1={mf1:.4f}  "
          f"gecerli={gecerli}/{len(d)}  <think>={int(d['has_think'].sum())}")
    print(f"  input_tokens: ort={d['input_tokens'].mean():.0f} "
          f"medyan={d['input_tokens'].median():.0f} maks={d['input_tokens'].max():.0f}")
    print(f"  {args.max_seq}'e dayanip kesilen kayit: {int(d['kesildi'].sum())}/{len(d)} "
          f"(%{100*d['kesildi'].mean():.1f})")
    print(f"  gecikme: ort {d['sure_s'].mean():.2f} s/ornek · toplam {d['sure_s'].sum()/60:.1f} dk")

    ozet = dict(split=args.split, n=len(d), max_seq=args.max_seq,
                accuracy=round(acc, 4), macro_f1=round(mf1, 4),
                gecerli_oran=round(gecerli / len(d), 4),
                kesilen=int(d["kesildi"].sum()),
                istem_token_ort=round(float(d["input_tokens"].mean()), 1),
                gecikme_s_ornek=round(float(d["sure_s"].mean()), 3),
                toplam_dk=round(float(d["sure_s"].sum()) / 60, 1))

    # ---------------------------------------------------------- eski ile kiyas
    eski_yol = ESKI_EVAL / f"{args.split}_predictions.csv"
    if eski_yol.exists():
        e = pd.read_csv(eski_yol)
        e_acc = float((e["gold"] == e["pred"]).mean())
        e_mf1 = f1_score(e["gold"], e["pred"], average="macro",
                         labels=ALLOWED_CLASSES, zero_division=0)
        print("\n" + "=" * 78)
        print("ESKI (kesik, 409 token) vs YENI (tam baglam)")
        print("=" * 78)
        print(f"  {'':<22s} {'ESKI':>10s} {'YENI':>10s} {'fark':>9s}")
        print(f"  {'accuracy':<22s} {e_acc:>10.4f} {acc:>10.4f} {acc-e_acc:>+9.4f}")
        print(f"  {'macro-F1':<22s} {e_mf1:>10.4f} {mf1:>10.4f} {mf1-e_mf1:>+9.4f}")
        print(f"  {'istem token (ort)':<22s} {e['input_tokens'].mean():>10.0f} "
              f"{d['input_tokens'].mean():>10.0f} "
              f"{d['input_tokens'].mean()-e['input_tokens'].mean():>+9.0f}")

        # eslesmis McNemar — ayni test seti
        m = (e[["note_id", "gold", "pred"]]
             .merge(d[["note_id", "pred"]], on="note_id", suffixes=("_eski", "_yeni")))
        ce = m["pred_eski"] == m["gold"]
        cy = m["pred_yeni"] == m["gold"]
        b = int((ce & ~cy).sum())
        c = int((~ce & cy).sum())
        try:
            from scipy.stats import binomtest
            p = binomtest(b, b + c, 0.5).pvalue if (b + c) else 1.0
            pstr = f"{p:.4g}  -> {'ANLAMLI' if p < 0.05 else 'anlamli degil'}"
        except Exception:
            pstr = "scipy yok"
        print(f"\n  McNemar (eslesmis, n={len(m)}):")
        print(f"    yalniz ESKI dogru : {b}")
        print(f"    yalniz YENI dogru : {c}")
        print(f"    p = {pstr}")
        ozet.update(eski_accuracy=round(e_acc, 4), eski_macro_f1=round(e_mf1, 4),
                    delta_acc=round(acc - e_acc, 4), delta_mf1=round(mf1 - e_mf1, 4),
                    mcnemar_yalniz_eski=b, mcnemar_yalniz_yeni=c)

    with open(YENI_EVAL / f"{args.split}_summary.json", "w", encoding="utf-8") as f:
        json.dump(ozet, f, indent=2, ensure_ascii=False)
    pd.DataFrame([ozet]).to_csv(YENI_EVAL / f"{args.split}_summary.csv",
                                index=False, encoding="utf-8-sig")
    print(f"\n  Kaydedildi -> {YENI_EVAL}")
    print(f"    {args.split}_predictions.csv · {args.split}_summary.csv/json")
    print("=" * 78)


if __name__ == "__main__":
    main()
