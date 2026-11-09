# -*- coding: utf-8 -*-
"""
B1 — SIKISTIRILMIS FEW-SHOT, A100 KOSUSU  (TIPTEKNO'26)
=======================================================

SORUN
-----
Bildirinin 2. iddiasi su an karistiricili: sikistirilmis few-shot RTX 4060'ta,
tam few-shot A100'de kosmus. Iki degisken birden degisiyor (istem uzunlugu VE
donanim), hakem "sebep hangisi?" diye sorar.

COZUM — IKI ASAMA
-----------------
  ASAMA 1 (val, ~10 dk)  : DONANIM KONTROLU.
      Ayni istem, ayni 550 kayit, tek fark A100. Yerelde 0.4709 acc alinmisti.
      Buna yakin cikarsa -> donanimin DOGRULUGA etkisi yok, karistirici cozulur.
      (Greedy decoding deterministik; fark cikarsa sebebi arastirilmali.)

  ASAMA 2 (test, ~30 dk) : ASIL OLCUM.
      Sikistirilmis few-shot, 1650 kayit, A100. Artik tam few-shot (0.5552)
      ile AYNI donanimda kiyaslanabilir -> tek fark istem uzunlugu.
      Ayrica n 550'den 1650'ye cikar, CI +-4.2 puandan +-2.4 puana iner.

ISTEM YENIDEN KURULMUYOR
------------------------
compressed_3shot_PROMPT_FROZEN.txt yerelde uretildi ve log'daki token sayisiyla
BIREBIR dogrulandi (1805 token, 33 ornek, 959 ornek-token). Bu script onu
oldugu gibi okur — tek karakter kaymasin diye.

KULLANIM (Colab)
----------------
  1) Runtime > Change runtime type > A100
  2) Bu klasordeki 2 dosyayi yukleyin:
       b1_test_1650.parquet
       compressed_3shot_PROMPT_FROZEN.txt
  3) %run B1_colab_compressed_fewshot.py --split val      <- ONCE BU
  4) Sonuc ~0.47 civariysa:
     %run B1_colab_compressed_fewshot.py --split test
  Kesinti olursa ayni komut kaldigi yerden devam eder (Drive'a checkpoint).
"""
import argparse
import json
import os
import re
import time

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen3.5-4B"
MAX_LEN = 4096          # ultracompact kosusuyla ayni
MAX_NEW_TOKENS = 20     # ayni
TT = "\nNow classify:\n{input}\n=> "

ALLOWED_CLASSES = ["SEPSIS", "STROKE", "ATRIAL_FIBRILLATION", "AKI", "PNEUMONIA",
                   "UTI", "HEART_FAILURE", "GI_BLEEDING", "DIABETES_COMPLICATION",
                   "COPD_EXACERBATION", "OTHER"]

# yerelde alinan referans degerler (val, RTX 4060)
YEREL_VAL_ACC, YEREL_VAL_MF1 = 0.4709, 0.4699

# ---------------------------------------------------------------------------
# compact_v4_query — __run_blok14c_v4_3shot_ultra_val.py satir 35-98'den BIREBIR
# ---------------------------------------------------------------------------
SECTION_RE = re.compile(
    r"(Chief Complaint|Symptom History|Past Medical History|Allergies|"
    r"Medications on Admission|Vital Signs):\s*\n?", re.IGNORECASE)
SYMPTOM_KEYWORDS = re.compile(
    r"\b(presents?|reports?|complains?|denies|developed|noted|noticed|started|began|"
    r"pain|cough|fever|dyspnea|SOB|fatigue|weakness|nausea|vomit|diarrhea|bleeding|"
    r"dizz|confusion|swelling|edema|chest|abdomen|headache|hypoxic|hypotens|tachycard|"
    r"palpitations|orthopnea|wheez|melena|hemato|dysuria|frequen|urgency)\b", re.IGNORECASE)


def parse_sections(text):
    if not text:
        return {}
    parts = SECTION_RE.split(text)
    s = {}
    for i in range(1, len(parts) - 1, 2):
        s[parts[i].strip()] = parts[i + 1].strip()
    return s


def cumlelere_bol(t):
    if not t:
        return []
    return [c.strip() for c in re.split(r"(?<=[.!?])\s+(?=[A-Z\d])", t) if c.strip()]


def hasta_skor(c):
    return len(SYMPTOM_KEYWORDS.findall(c))


def compact_v4_query(full_text):
    sections = parse_sections(full_text)
    parts = []
    cc = sections.get("Chief Complaint", "").strip()
    if cc:
        parts.append(f"Chief Complaint: {cc}")
    sh = sections.get("Symptom History", "").strip()
    if sh:
        ros = re.search(r"(?im)^\s*(Review of Systems?|ROS)\s*:", sh)
        if ros:
            sh = sh[:ros.start()]
        cumleler = cumlelere_bol(sh)
        skorli = [(i, c, hasta_skor(c)) for i, c in enumerate(cumleler[:5])]
        secilen = [c for i, c, s in skorli if s > 0][:3]
        if not secilen:
            secilen = [c for i, c, s in skorli][:2]
        else:
            secilen = secilen[:3]
        if secilen:
            parts.append("Key Symptoms: " + " ".join(secilen))
    pmh = sections.get("Past Medical History", "").strip()
    if pmh:
        satirlar = [x.strip("-• \t") for x in pmh.split("\n") if x.strip()]
        if satirlar:
            parts.append("History: " + "; ".join(satirlar[:6]))
    vs = sections.get("Vital Signs", "").strip()
    if vs:
        parts.append(f"Vitals: {vs.splitlines()[0][:150]}")
    al = sections.get("Allergies", "").strip()
    if al:
        al_kisa = al.split("\n")[0].strip()[:100]
        if al_kisa and al_kisa.upper() not in ("NKDA", "NONE", "NKA"):
            parts.append(f"Allergies: {al_kisa}")
    meds = sections.get("Medications on Admission", "").strip()
    if meds:
        ilac_satirlar = re.findall(r"\d+\.\s*([A-Z][A-Za-z][^.\n]{2,80})", meds)
        if ilac_satirlar:
            kisa = [" ".join(s.split()[:3]) for s in ilac_satirlar[:5]]
            parts.append("Medications: " + ", ".join(kisa))
    return "\n".join(parts)


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
    text = raw.strip().upper()
    for cls in sorted(ALLOWED_CLASSES, key=len, reverse=True):
        if cls in text[:80]:
            return cls, "direct"
    for k, v in FUZZY_MAP.items():
        if k in text[:80]:
            return v, "fuzzy"
    return "INVALID", "invalid"


# ---------------------------------------------------------------------------
def dosya_bul(ad, tur):
    """Colab'da dosya nerede olursa olsun bul. Bulamazsa NE VAR onu goster."""
    if os.path.isabs(ad) and os.path.exists(ad):
        return ad
    aranan = [
        os.getcwd(),
        os.path.dirname(os.path.abspath(__file__)),
        "/content",
        "/content/b1_colab_upload",
        "/content/drive/MyDrive",
        "/content/drive/MyDrive/B1",
        "/content/drive/MyDrive/b1_colab_upload",
    ]
    gorulen = []
    for k in aranan:
        if k in gorulen:
            continue
        gorulen.append(k)
        aday = os.path.join(k, os.path.basename(ad))
        if os.path.exists(aday):
            print(f"  bulundu: {aday}")
            return aday

    print("\n" + "!" * 74)
    print(f"BULUNAMADI: {ad}   ({tur})")
    print("!" * 74)
    print("Aranan klasorler ve icindeki ilgili dosyalar:\n")
    for k in gorulen:
        if not os.path.isdir(k):
            print(f"  {k}  -> klasor yok")
            continue
        try:
            hepsi = os.listdir(k)
        except Exception as e:
            print(f"  {k}  -> okunamadi ({e})")
            continue
        ilgili = [x for x in hepsi if x.endswith((".txt", ".parquet", ".py"))]
        print(f"  {k}  ({len(hepsi)} oge)")
        for x in sorted(ilgili)[:12]:
            print(f"      {x}")
        if not ilgili:
            print("      (.txt/.parquet/.py yok)")
    print("\nCozum: eksik dosyayi yukleyin —")
    print("  from google.colab import files; files.upload()")
    print("veya tam yolu verin:  --istem /tam/yol/dosya.txt")
    raise SystemExit(1)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--veri", default="b1_test_1650.parquet")
    ap.add_argument("--istem", default="compressed_3shot_PROMPT_FROZEN.txt")
    ap.add_argument("--cikti", default=None, help="varsayilan: Drive varsa Drive, yoksa /content")
    args = ap.parse_args()

    # --- cikti yeri: Drive tercih edilir (/content oturum yenilenince SILINIR) ---
    if args.cikti:
        OUTDIR = args.cikti
    elif os.path.isdir("/content/drive/MyDrive"):
        OUTDIR = "/content/drive/MyDrive/B1"
    else:
        OUTDIR = "/content/b1_out"
        print("!! UYARI: Drive bagli degil. /content oturum yenilenince SILINIR.")
        print("!! Onerilen: from google.colab import drive; drive.mount('/content/drive')")
    os.makedirs(OUTDIR, exist_ok=True)

    print("=" * 74)
    print(f"B1 — SIKISTIRILMIS FEW-SHOT   [split={args.split}]")
    print("=" * 74)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"  GPU    : {p.name}  {p.total_memory/1024**3:.1f} GiB")
    else:
        raise SystemExit("GPU yok — Runtime > Change runtime type > A100")
    print(f"  Cikti  : {OUTDIR}")

    # --- ON KONTROL: paketler (model yuklemeden ONCE patlasin) ---
    import importlib
    surumler = {}
    eksik = []
    for m in ["torch", "transformers", "bitsandbytes", "accelerate"]:
        try:
            surumler[m] = getattr(importlib.import_module(m), "__version__", "?")
        except Exception:
            eksik.append(m)
    if eksik:
        print("\n" + "!" * 74)
        print(f"EKSIK PAKET: {', '.join(eksik)}")
        print("!" * 74)
        print("Colab'da su hucreyi calistirip tekrar deneyin:\n")
        print('  !pip install -q -U "bitsandbytes>=0.46.1" accelerate\n')
        print("NOT: transformers'i GUNCELLEMEYIN — surum degisikligi davranisi")
        print("     degistirebilir ve yerel kosuyla kiyasi bozar.")
        raise SystemExit(1)
    print("  Paket  : " + " · ".join(f"{k} {v}" for k, v in surumler.items()))
    # surumler ozete yazilir -> bildirinin yeniden uretilebilirlik bolumu icin
    globals()["_SURUMLER"] = surumler

    # --- dondurulmus istem ---
    istem_yol = dosya_bul(args.istem, "dondurulmus istem")
    with open(istem_yol, encoding="utf-8") as f:
        SABIT = f.read()
    print(f"  Istem  : {os.path.basename(istem_yol)} ({len(SABIT):,} karakter) — YENIDEN KURULMADI")

    # --- veri ---
    veri_yol = dosya_bul(args.veri, "veri dosyasi")
    df = pd.read_parquet(veri_yol)
    if args.split == "val":
        # val icin ayri dosya yoksa test'ten ILK 550'yi ALMA — dogru val gerekir.
        if "split" in df.columns:
            df = df[df["split"] == "val"]
        else:
            raise SystemExit(
                "val kosusu icin val bolumu gerekli.\n"
                "Yerelde su komutla uretin ve yukleyin:\n"
                "  python -c \"import pandas as pd;"
                "d=pd.read_parquet('output/discharge_split.parquet');"
                "d[d.split=='val'][['note_id','subject_id','hadm_id','label',"
                "'triage_input_v4_timeaware_masked']].to_parquet('b1_val_550.parquet')\"\n"
                "sonra: --split val --veri b1_val_550.parquet")
    df = df.reset_index(drop=True)
    print(f"  Veri   : {len(df)} kayit")

    # --- model: yerel kosuyla BIREBIR ayni yukleme ---
    print("\n[1/3] Model yukleniyor (4-bit NF4, fp16 compute — yerelle ayni)...")
    t0 = time.time()
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto", dtype=torch.float16)
    model.eval()
    print(f"   Yuklendi {time.time()-t0:.1f}s · GPU {torch.cuda.memory_allocated()/1024**2:.0f} MiB")

    # sabit istem token dogrulamasi (yerelde 1805 idi)
    msgs = [{"role": "user", "content": SABIT + TT.format(input="")}]
    txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                  enable_thinking=False)
    sabit_n = len(tok.encode(txt, add_special_tokens=False))
    print(f"   Sabit istem: {sabit_n} token (yerelde 1805) "
          f"{'OK' if sabit_n == 1805 else '<-- FARKLI! dur ve incele'}")

    # --- checkpoint / resume ---
    cp = os.path.join(OUTDIR, f"b1_compressed_{args.split}_checkpoint.csv")
    son = os.path.join(OUTDIR, f"b1_compressed_{args.split}.csv")
    if os.path.exists(cp):
        eski = pd.read_csv(cp)
        islenen = set(eski["note_id"].astype(str))
        kayitlar = eski.to_dict("records")
        print(f"   Resume: {len(islenen)} kayit zaten var.")
    else:
        islenen, kayitlar = set(), []

    print(f"\n[2/3] Cikarim ({len(df)} kayit, tek ornek dongusu — yerelle ayni)")
    t_bas = time.time()
    yeni = 0
    for _, row in df.iterrows():
        nid = str(row["note_id"])
        if nid in islenen:
            continue
        v4 = row["triage_input_v4_timeaware_masked"]
        if not isinstance(v4, str):
            v4 = ""
        inp = compact_v4_query(v4)
        prompt = SABIT + TT.format(input=inp)
        msgs = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        inputs = tok(text, return_tensors="pt", truncation=True,
                     max_length=MAX_LEN).to("cuda")
        in_n = int(inputs["input_ids"].shape[1])

        t1 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        sure = time.time() - t1
        raw = tok.decode(out[0][in_n:], skip_special_tokens=True).strip()
        pred, kat = parse_class(raw)

        kayitlar.append({
            "note_id": nid, "subject_id": int(row["subject_id"]),
            "hadm_id": int(row["hadm_id"]), "label": row["label"],
            "raw_output": raw, "parsed_pred": pred, "parse_kategori": kat,
            "in_tokens": in_n, "sure_s": round(sure, 3),
            "has_think": ("<think>" in raw or "</think>" in raw),
        })
        yeni += 1
        if len(kayitlar) % 50 == 0:
            pd.DataFrame(kayitlar).to_csv(cp, index=False, encoding="utf-8-sig")
            gec = time.time() - t_bas
            ort = gec / yeni
            print(f"   [{len(kayitlar):>4}/{len(df)}] gecen {gec/60:5.1f} dk · "
                  f"ort {ort:.2f} s/ornek · kalan ~{(len(df)-len(kayitlar))*ort/60:.1f} dk")

    d = pd.DataFrame(kayitlar)
    d.to_csv(son, index=False, encoding="utf-8-sig")
    d.to_csv(cp, index=False, encoding="utf-8-sig")
    print(f"   BITTI. {len(d)} kayit, {(time.time()-t_bas)/60:.1f} dk")

    # --- metrikler ---
    from sklearn.metrics import f1_score
    acc = float((d["label"] == d["parsed_pred"]).mean())
    mf1 = f1_score(d["label"], d["parsed_pred"], average="macro",
                   labels=ALLOWED_CLASSES, zero_division=0)   # payda 11
    inv = int((d["parsed_pred"] == "INVALID").sum())

    print(f"\n[3/3] SONUC")
    print("=" * 74)
    print(f"  n={len(d)}  acc={acc:.4f}  macroF1={mf1:.4f}  INVALID={inv}  "
          f"<think>={int(d['has_think'].sum())}")
    print(f"  istem token: ort={d['in_tokens'].mean():.0f} maks={d['in_tokens'].max()}")
    print(f"  gecikme    : ort {d['sure_s'].mean():.2f} s/ornek · toplam {d['sure_s'].sum()/60:.1f} dk")

    if args.split == "val":
        print("\n  --- DONANIM KONTROLU (asil amac) ---")
        print(f"  RTX 4060 (yerel) : acc {YEREL_VAL_ACC:.4f}  macroF1 {YEREL_VAL_MF1:.4f}")
        print(f"  A100 (bu kosu)   : acc {acc:.4f}  macroF1 {mf1:.4f}")
        fark = abs(acc - YEREL_VAL_ACC)
        if fark < 0.02:
            print(f"  fark {100*fark:.1f} puan -> DONANIMIN DOGRULUGA ETKISI YOK.")
            print("  => Karistirici cozuldu. Sikistirmanin etkisi temiz olarak atfedilebilir.")
            print("  => Simdi test kosusuna gecin: --split test")
        else:
            print(f"  fark {100*fark:.1f} puan -> BEKLENENDEN BUYUK. Once sebebini bulun")
            print("     (istem farki? tokenizer surumu? decoding?). Test'e gecmeyin.")
    else:
        print("\n  --- ASIL KIYAS (hepsi A100, tek fark istem uzunlugu) ---")
        print(f"  Sikistirilmis 3-shot : acc {acc:.4f}   (bu kosu, ort {d['in_tokens'].mean():.0f} token)")
        print(f"  Tam 3-shot           : acc 0.5552   (~23.8k token)")
        print(f"  Zero-shot (4060)     : acc 0.5152   (~865 token)")
        print(f"\n  Sikistirilmis - tam  : {100*(acc-0.5552):+.1f} puan")
        print(f"  Sikistirilmis - zero : {100*(acc-0.5152):+.1f} puan")
        print("  McNemar testini yerelde yapin (per-kayit CSV'ler elde).")

    ozet = dict(split=args.split, n=len(d), accuracy=round(acc, 4),
                macro_f1=round(mf1, 4), invalid=inv,
                istem_token_ort=round(float(d["in_tokens"].mean()), 1),
                gecikme_s=round(float(d["sure_s"].mean()), 3),
                toplam_dk=round(float(d["sure_s"].sum()) / 60, 1),
                sabit_istem_token=sabit_n, gpu=torch.cuda.get_device_name(0),
                surumler=globals().get("_SURUMLER", {}))
    with open(os.path.join(OUTDIR, f"b1_compressed_{args.split}_summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(ozet, f, indent=2, ensure_ascii=False)
    print(f"\n  Kaydedildi -> {OUTDIR}")
    print(f"    b1_compressed_{args.split}.csv + _summary.json")
    print("  !! Bu dosyalari yerele indirin (Colab oturumu silinebilir).")
    print("=" * 74)


if __name__ == "__main__":
    main()
