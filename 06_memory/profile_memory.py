# -*- coding: utf-8 -*-
"""
A1 / K1 — QWEN BELLEK PROFILI  (TIPTEKNO'26 bildirisi icin)

AMAC
----
Bildirinin ana iddiasinin ("kisitli bellekte yontem secimi donanimin dayattigi
odunlerle belirlenir") arkasinda su an OLCULMUS tek bir bellek sayisi yok.
Elde olan "19 GB -> 12.4 GB" degeri make_final_graphs.py:214'te elle yazilmis
bir sabit; log'u yok. Bu script o boslugu doldurur.

TEZDEKI KOSUYLA BIREBIR AYNI AYARLAR (__run_blok13b_v4_test.py'den alindi)
--------------------------------------------------------------------------
  * 4-bit NF4 + double quant, compute dtype = float16   (Qwen fp16'ydi, Gemma bf16)
  * ayni PROMPT_TEMPLATE
  * apply_chat_template(..., enable_thinking=False)
  * truncation max_length=2048
  * max_new_tokens=20, do_sample=False (greedy), BATCH YOK (tek ornek)
  * gercek test istemleri (triage_input_v4_timeaware_masked)

OLCULEN
-------
  allocated / reserved / nvidia-smi ucu AYRI AYRI   (hakem farki sorar)
  yukleme sonrasi taban  vs  cikarim tepesi         (ayri satirlar)
  vision tower VRAM'de mi                           (cok kipli model, tek kipli kullanim)
  CPU/disk offload oldu mu                          (device_map="auto" sessizce offload edebilir)
  istem uzunluguna gore tepe egrisi                 (tepe bellek dizi uzunluguna bagli)

KULLANIM
--------
  Once 4-bit (tezdeki yapilandirma, 8 GB'a sigmasi beklenir):
      python profile_memory.py --config 4bit

  Sonra fp16 (sigmamasi beklenir — "sigmadi" satirini OLCUMLE belgelemek icin):
      python profile_memory.py --config fp16

  Her config AYRI SUREC olmali; tek surecte ikisini yuklemek torch'un
  cache'leyen ayiricisi yuzunden tepe degerini kirletir. Bu yuzden script
  bilerek tek config koşturur.

ONEMLI
------
  Olcum sirasinda GPU'yu bos tutun (oyun / LM Studio / Ollama / tarayici
  donanim hizlandirma). 8 GB'lik kartta birkac yuz MB sonucu degistirir.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ---------------------------------------------------------------- yollar
BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "output"
SPLIT_YOL = OUT_DIR / "discharge_split.parquet"
PROF_DIR = OUT_DIR / "memory_profile"
PROF_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "Qwen/Qwen3.5-4B"

# ---------------------------------------------------------------- tezdeki istem (birebir)
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

MAX_LEN = 2048          # zero-shot kosusuyla ayni
MAX_NEW_TOKENS = 20     # zero-shot kosusuyla ayni


# ================================================================ yardimcilar
def smi(query):
    """nvidia-smi'den tek deger oku. Hata olursa None."""
    try:
        r = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        if r.returncode != 0:
            return None
        return float(r.stdout.strip().splitlines()[0])
    except Exception:
        return None


def smi_used_mb():
    """GPU'da TOPLAM kullanilan MB (tum surecler).
    Not: Windows WDDM'de nvidia-smi surec-basi bellegi [N/A] dondurur,
    bu yuzden global deger - taban seklinde raporluyoruz."""
    return smi("memory.used")


def torch_mb(fn):
    return fn() / 1024 ** 2


def snapshot(etiket, smi_taban):
    """O andaki bellek durumunu al."""
    torch.cuda.synchronize()
    used = smi_used_mb()
    return {
        "asama": etiket,
        "torch_allocated_MB": round(torch_mb(torch.cuda.memory_allocated), 1),
        "torch_reserved_MB": round(torch_mb(torch.cuda.memory_reserved), 1),
        "torch_peak_allocated_MB": round(torch_mb(torch.cuda.max_memory_allocated), 1),
        "torch_peak_reserved_MB": round(torch_mb(torch.cuda.max_memory_reserved), 1),
        "nvidia_smi_used_MB": round(used, 1) if used is not None else None,
        "nvidia_smi_delta_MB": round(used - smi_taban, 1) if (used is not None and smi_taban is not None) else None,
    }


def modul_dagilimi(model):
    """Parametreleri ust-seviye bilesene ve cihaza gore grupla.
    Vision tower VRAM'e geliyor mu, offload var mi -> burada gorunur."""
    grup = {}
    for ad, p in model.named_parameters():
        kok = ad.split(".")[0]
        # ic ice sarmalayicilari (model.model.language_model...) biraz ac
        parcalar = ad.split(".")
        for cand in ("vision_tower", "visual", "language_model", "model", "lm_head"):
            if cand in parcalar:
                kok = cand
                break
        d = str(p.device)
        anahtar = (kok, d)
        n = p.numel()
        # 4-bit paketli parametreler uint8 olarak tutulur -> element_size dogru bayti verir
        b = n * p.element_size()
        if anahtar not in grup:
            grup[anahtar] = [0, 0]
        grup[anahtar][0] += n
        grup[anahtar][1] += b
    return grup


# ================================================================ ana akis
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["4bit", "fp16"], required=True,
                    help="4bit = tezdeki yapilandirma · fp16 = tam duyarlik (sigmasi beklenmiyor)")
    ap.add_argument("--device-map", default=None,
                    help='varsayilan: 4bit->auto (tezdeki gibi), fp16->cuda:0 (temiz OOM icin)')
    ap.add_argument("--n", type=int, default=30, help="kac ornek uzerinde cikarim (varsayilan 30)")
    ap.add_argument("--max-taban", type=float, default=600,
                    help="olcum oncesi GPU'da izin verilen azami dolu MiB (varsayilan 600)")
    ap.add_argument("--emniyet", type=float, default=384,
                    help="torch tavani icin bos VRAM'den dusulecek emniyet payi MiB")
    ap.add_argument("--force", action="store_true",
                    help="GPU dolu olsa da kostur (sonuc RAPORLANABILIR DEGIL)")
    args = ap.parse_args()

    if args.device_map is None:
        args.device_map = "auto" if args.config == "4bit" else "cuda:0"

    if not torch.cuda.is_available():
        sys.exit("CUDA yok — olcum anlamsiz. Cikiliyor.")

    dev_props = torch.cuda.get_device_properties(0)
    smi_toplam = smi("memory.total")
    smi_taban = smi_used_mb()

    print("=" * 78)
    print(f"A1 / K1 — QWEN BELLEK PROFILI   [config={args.config}]")
    print("=" * 78)
    print(f"  GPU              : {dev_props.name}")
    print(f"  VRAM (torch)     : {dev_props.total_memory / 1024**2:.0f} MiB")
    print(f"  VRAM (nvidia-smi): {smi_toplam:.0f} MiB" if smi_toplam else "  VRAM (nvidia-smi): okunamadi")
    print(f"  Olcum oncesi dolu: {smi_taban:.0f} MiB  <-- bu deger yuksekse GPU'yu bosaltin")
    print(f"  device_map       : {args.device_map}")
    print(f"  ornek sayisi     : {args.n}")
    print()

    # ------------------------------------------------------------------
    # GUARD 1 — GPU temiz mi?
    # 8 GB'lik kartta bir kac yuz MB sonucu degistirir. Kart doluysa surucu
    # sistem RAM'ine tasar ve olcum anlamsizlasir (bkz. GUARD 2).
    # ------------------------------------------------------------------
    if smi_taban is not None and smi_taban > args.max_taban and not args.force:
        print("!" * 78)
        print(f"IPTAL — GPU bos degil: {smi_taban:.0f} MiB dolu (esik {args.max_taban} MiB).")
        print("Bu haliyle olcum GECERSIZ olur: model VRAM'e sigmaz, surucu sistem RAM'ine")
        print("tasar, torch sayaclari sisik deger gosterir ve gecikme 100 kat artar.")
        print("")
        print("Yapilacak: GPU kullanan uygulamalari kapatin (oyun, LM Studio, Ollama,")
        print("tarayici donanim hizlandirma, onceki python surecleri), sonra tekrar deneyin.")
        print("Kontrol:  nvidia-smi --query-compute-apps=pid,process_name --format=csv")
        print("")
        print("Yine de kosturmak icin: --force  (sonuc raporlanabilir DEGILDIR)")
        print("!" * 78)
        sys.exit(2)

    # ------------------------------------------------------------------
    # GUARD 2 — Windows sysmem fallback'i kapat.
    # WDDM surucusu VRAM dolunca sessizce sistem RAM'ine tasar; cihaz hala
    # "cuda:0" gorunur, parametre yerlesim denetimi bunu YAKALAYAMAZ.
    # torch'un ayiricisini gercek bos VRAM'in altina sabitleyerek surucu
    # tasmadan ONCE temiz bir OutOfMemoryError almasini sagliyoruz.
    # ------------------------------------------------------------------
    toplam_mib = dev_props.total_memory / 1024 ** 2
    bos_mib = (smi_toplam - smi_taban) if (smi_toplam and smi_taban is not None) else toplam_mib
    izin_mib = max(512.0, bos_mib - args.emniyet)
    frac = min(0.99, izin_mib / toplam_mib)
    torch.cuda.set_per_process_memory_fraction(frac, 0)
    print(f"  Bos VRAM         : {bos_mib:.0f} MiB")
    print(f"  torch tavani     : {izin_mib:.0f} MiB (fraction={frac:.3f}) "
          f"-> tasma yerine TEMIZ OOM")
    print()

    kayitlar = []
    olay = {"config": args.config, "device_map": args.device_map,
            "tarih": datetime.now().isoformat(timespec="seconds"),
            "gpu": dev_props.name,
            "vram_toplam_MiB": round(dev_props.total_memory / 1024 ** 2, 1),
            "smi_taban_MiB": smi_taban, "n_ornek": args.n,
            "max_len": MAX_LEN, "max_new_tokens": MAX_NEW_TOKENS}

    # ---------------------------------------------------------- 1) tokenizer
    print("[1/5] Tokenizer...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---------------------------------------------------------- 2) veri + uzunluk katmanlari
    print("[2/5] Test seti ve istem uzunluklari...")
    if not SPLIT_YOL.exists():
        sys.exit(f"Veri bulunamadi: {SPLIT_YOL}")
    df = pd.read_parquet(SPLIT_YOL)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    def istem_metni(satir):
        inp = satir["triage_input_v4_timeaware_masked"]
        if not isinstance(inp, str):
            inp = ""
        prompt = PROMPT_TEMPLATE.format(input=inp)
        msgs = [{"role": "user", "content": prompt}]
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)

    metinler = [istem_metni(r) for _, r in test_df.iterrows()]
    uzunluklar = [min(len(tok(t)["input_ids"]), MAX_LEN) for t in metinler]
    ser = pd.Series(uzunluklar)
    print(f"   Test n={len(test_df)}  token: ort={ser.mean():.0f} "
          f"medyan={ser.median():.0f} p95={ser.quantile(.95):.0f} maks={ser.max()}")

    # uzunluk katmanlari: kisa / ortalama / uzun — tepe egrisini gormek icin
    sirali = ser.sort_values().index.tolist()
    k = max(1, args.n // 3)
    sec_kisa = sirali[:k]
    orta_bas = len(sirali) // 2 - k // 2
    sec_orta = sirali[orta_bas:orta_bas + k]
    sec_uzun = sirali[-k:]
    secim = [("KISA", sec_kisa), ("ORTA", sec_orta), ("UZUN", sec_uzun)]
    print(f"   Katmanlar: KISA~{ser[sec_kisa].mean():.0f} tok · "
          f"ORTA~{ser[sec_orta].mean():.0f} tok · UZUN~{ser[sec_uzun].mean():.0f} tok")

    # ---------------------------------------------------------- 3) model yukleme
    print(f"\n[3/5] Model yukleniyor ({args.config})...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    kayitlar.append(snapshot("00_yukleme_oncesi", smi_taban))

    yuk = dict(local_files_only=True, device_map=args.device_map)
    if args.config == "4bit":
        yuk["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        yuk["dtype"] = torch.float16      # tezdeki torch_dtype=float16 ile ayni
    else:
        yuk["dtype"] = torch.float16

    t0 = time.time()
    oom_asamasi = None
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **yuk)
        model.eval()
    except torch.cuda.OutOfMemoryError as e:
        oom_asamasi = "yukleme"
        print(f"\n   >>> OOM: model YUKLENEMEDI ({args.config})")
        print(f"   >>> {str(e)[:200]}")
    except Exception as e:
        oom_asamasi = f"yukleme_hata_{type(e).__name__}"
        print(f"\n   >>> HATA: {type(e).__name__}: {str(e)[:300]}")

    if oom_asamasi:
        olay["sonuc"] = "SIGMADI"
        olay["oom_asamasi"] = oom_asamasi
        olay["yukleme_s"] = round(time.time() - t0, 1)
        kayitlar.append(snapshot("01_OOM", smi_taban))
        _kaydet(olay, kayitlar, [], None)
        print("\n" + "=" * 78)
        print(f"SONUC: {args.config} bu kartta ({dev_props.total_memory/1024**2:.0f} MiB) SIGMADI.")
        print("Bu bir basarisizlik degil — kaynak tablosunun '8 GB'a sigdi mi' sutununun VERISI.")
        print("=" * 78)
        return

    yukleme_s = time.time() - t0
    olay["yukleme_s"] = round(yukleme_s, 1)
    kayitlar.append(snapshot("02_yukleme_sonrasi", smi_taban))
    print(f"   Yuklendi {yukleme_s:.1f}s")

    # ---------------------------------------------------------- 3b) yerlesim denetimi
    print("\n   --- Parametre yerlesimi (vision tower / offload denetimi) ---")
    grup = modul_dagilimi(model)
    toplam_b = sum(v[1] for v in grup.values())
    cpu_b = sum(b for (kok, d), (n, b) in grup.items() if d.startswith("cpu"))
    meta_b = sum(b for (kok, d), (n, b) in grup.items() if d.startswith("meta"))
    for (kok, d), (n, b) in sorted(grup.items(), key=lambda x: -x[1][1]):
        print(f"      {kok:<18s} {d:<10s} {n/1e6:8.1f} M param  {b/1024**2:9.1f} MiB")
    print(f"      {'TOPLAM':<18s} {'':<10s} {'':>8s}           {toplam_b/1024**2:9.1f} MiB")

    hf_map = getattr(model, "hf_device_map", None)
    offload = bool(cpu_b or meta_b)
    if offload:
        print(f"   !!! OFFLOAD VAR — CPU {cpu_b/1024**2:.0f} MiB, meta/disk {meta_b/1024**2:.0f} MiB")
        print("   !!! Bu durumda VRAM sayisi 'sigdi' anlamina GELMEZ. Raporda belirt.")
    else:
        print("   Offload yok — tum parametreler GPU'da.")

    vision_b = sum(b for (kok, d), (n, b) in grup.items()
                   if kok in ("vision_tower", "visual") and d.startswith("cuda"))
    if vision_b:
        print(f"   NOT: vision tower VRAM'de, {vision_b/1024**2:.1f} MiB. "
              f"Tezde yalniz metin cozucusu kullanildi -> bu yuk raporda ayrilmali.")
    else:
        print("   NOT: vision tower VRAM'de gorunmuyor (ya yuklenmedi ya baska cihazda).")

    olay.update(param_toplam_MiB=round(toplam_b / 1024 ** 2, 1),
                param_cpu_MiB=round(cpu_b / 1024 ** 2, 1),
                param_meta_MiB=round(meta_b / 1024 ** 2, 1),
                vision_tower_MiB=round(vision_b / 1024 ** 2, 1),
                offload_var=offload,
                hf_device_map=json.dumps(hf_map) if isinstance(hf_map, dict) else str(hf_map))

    # ---------------------------------------------------------- 4) warmup
    print("\n[4/5] Warmup (ilk generate cekirdek derlemesi iceriri — olcume katilmaz)...")
    try:
        w = tok(metinler[sirali[len(sirali) // 2]], return_tensors="pt",
                truncation=True, max_length=MAX_LEN).to(model.device)
        with torch.no_grad():
            model.generate(**w, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                           pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        print("   Warmup tamam.")
    except Exception as e:
        print(f"   Warmup hatasi: {type(e).__name__}: {str(e)[:150]}")

    # warmup sonrasi taban: model + cekirdekler yerlesti
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    kayitlar.append(snapshot("03_warmup_sonrasi_taban", smi_taban))
    taban_alloc = torch_mb(torch.cuda.memory_allocated)
    print(f"   Taban (model yerlesik): {taban_alloc:.0f} MiB allocated")

    # ---------------------------------------------------------- 5) cikarim olcumu
    print(f"\n[5/5] Cikarim olcumu — {args.n} ornek, 3 uzunluk katmani")
    print(f"\n   {'katman':<7s} {'idx':>5s} {'tok':>6s} {'sure_s':>7s} "
          f"{'peak_alloc':>11s} {'peak_resv':>10s} {'smi_delta':>10s} {'aktivasyon':>11s}")
    print("   " + "-" * 78)

    olcumler = []
    for katman, idx_list in secim:
        for idx in idx_list:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            metin = metinler[idx]
            inputs = tok(metin, return_tensors="pt", truncation=True,
                         max_length=MAX_LEN).to(model.device)
            n_tok = int(inputs["input_ids"].shape[1])
            try:
                t1 = time.time()
                with torch.no_grad():
                    model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                   do_sample=False, pad_token_id=tok.eos_token_id)
                torch.cuda.synchronize()
                sure = time.time() - t1
                pa = torch_mb(torch.cuda.max_memory_allocated)
                pr = torch_mb(torch.cuda.max_memory_reserved)
                used = smi_used_mb()
                sd = (used - smi_taban) if (used is not None and smi_taban is not None) else None
                akt = pa - taban_alloc          # modelin ustune eklenen: aktivasyon + KV cache

                # --- sysmem fallback denetimi ---
                # Saglikli durumda nvidia-smi deltasi torch'un allocated'indan
                # BUYUK olur (CUDA context ~300-600 MiB torch'a gorunmez).
                # Tersi ise surucunun sistem RAM'ine tastigini gosterir.
                tasma = (sd is not None and pa > sd)
                if pr > toplam_mib:
                    tasma = True

                bayrak = " <<TASMA" if tasma else ""
                print(f"   {katman:<7s} {idx:>5d} {n_tok:>6d} {sure:>7.2f} "
                      f"{pa:>10.0f}M {pr:>9.0f}M "
                      f"{(f'{sd:.0f}M' if sd is not None else '   n/a'):>10s} {akt:>10.0f}M{bayrak}")
                olcumler.append(dict(katman=katman, idx=int(idx), in_tokens=n_tok,
                                     sure_s=round(sure, 3),
                                     peak_allocated_MiB=round(pa, 1),
                                     peak_reserved_MiB=round(pr, 1),
                                     smi_delta_MiB=round(sd, 1) if sd is not None else None,
                                     aktivasyon_MiB=round(akt, 1),
                                     sysmem_tasma=bool(tasma)))
            except torch.cuda.OutOfMemoryError:
                print(f"   {katman:<7s} {idx:>5d} {n_tok:>6d}   OOM")
                olcumler.append(dict(katman=katman, idx=int(idx), in_tokens=n_tok,
                                     sure_s=None, peak_allocated_MiB=None,
                                     peak_reserved_MiB=None, smi_delta_MiB=None,
                                     aktivasyon_MiB=None))
                torch.cuda.empty_cache()

    kayitlar.append(snapshot("04_cikarim_sonrasi", smi_taban))

    # ---------------------------------------------------------- ozet
    od = pd.DataFrame(olcumler)
    basarili = od.dropna(subset=["peak_allocated_MiB"])
    n_tasma = int(od["sysmem_tasma"].sum()) if "sysmem_tasma" in od.columns else 0
    olay["sysmem_tasma_sayisi"] = n_tasma

    if n_tasma:
        olay["sonuc"] = "GECERSIZ_SYSMEM_TASMASI"
    elif offload:
        olay["sonuc"] = "SIGMADI_OFFLOAD_EDILDI"
    elif len(basarili) == 0:
        olay["sonuc"] = "SIGMADI_OOM_CIKARIMDA"
    else:
        olay["sonuc"] = "SIGDI"
    if len(basarili):
        olay.update(
            taban_allocated_MiB=round(taban_alloc, 1),
            tepe_allocated_MiB=round(basarili["peak_allocated_MiB"].max(), 1),
            tepe_reserved_MiB=round(basarili["peak_reserved_MiB"].max(), 1),
            tepe_smi_delta_MiB=(round(basarili["smi_delta_MiB"].max(), 1)
                                if basarili["smi_delta_MiB"].notna().any() else None),
            aktivasyon_maks_MiB=round(basarili["aktivasyon_MiB"].max(), 1),
            gecikme_ort_s=round(basarili["sure_s"].mean(), 3),
        )

    _kaydet(olay, kayitlar, olcumler, od)

    print("\n" + "=" * 78)
    print(f"OZET — {args.config}")
    print("=" * 78)
    print(f"  Model tabani (allocated)     : {taban_alloc:>8.0f} MiB")
    if len(basarili):
        print(f"  Cikarim TEPE allocated       : {basarili['peak_allocated_MiB'].max():>8.0f} MiB")
        print(f"  Cikarim TEPE reserved        : {basarili['peak_reserved_MiB'].max():>8.0f} MiB")
        if basarili["smi_delta_MiB"].notna().any():
            print(f"  nvidia-smi TEPE delta        : {basarili['smi_delta_MiB'].max():>8.0f} MiB")
        print(f"  Aktivasyon+KV (tepe - taban) : {basarili['aktivasyon_MiB'].max():>8.0f} MiB")
        print(f"  VRAM toplam                  : {dev_props.total_memory/1024**2:>8.0f} MiB")
        pay = 100 * basarili["peak_reserved_MiB"].max() / (dev_props.total_memory / 1024 ** 2)
        print(f"  Kartin ne kadari kullanildi  : {pay:>8.1f} %")
        print("\n  Uzunluk katmanina gore tepe (allocated):")
        for kt, g in basarili.groupby("katman"):
            print(f"    {kt:<6s} ort {g['in_tokens'].mean():>6.0f} tok  ->  "
                  f"{g['peak_allocated_MiB'].mean():>7.0f} MiB")
    print(f"\n  Offload : {'VAR (sayilar sigdi anlamina gelmez)' if offload else 'yok'}")
    print(f"  Sonuc   : {olay['sonuc']}")

    if n_tasma:
        print("\n" + "!" * 78)
        print(f"UYARI — {n_tasma}/{len(od)} olcumde SYSMEM TASMASI tespit edildi.")
        print("Surucu VRAM yerine sistem RAM'i kullanmis; torch sayaclari sisik.")
        print("BU SAYILAR RAPORLANAMAZ. Belirtiler: allocated > VRAM toplami,")
        print("nvidia-smi deltasinin torch allocated'in altinda kalmasi, gecikmenin")
        print("10-100 kat artmasi.")
        print("Cozum: GPU'yu tamamen bosaltip tekrar kosturun.")
        print("!" * 78)
    print("=" * 78)


def _kaydet(olay, kayitlar, olcumler, od):
    etiket = f"{olay['config']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with open(PROF_DIR / f"profil_{etiket}.json", "w", encoding="utf-8") as f:
        json.dump({"olay": olay, "asamalar": kayitlar, "olcumler": olcumler},
                  f, indent=2, ensure_ascii=False)
    pd.DataFrame(kayitlar).to_csv(PROF_DIR / f"asamalar_{etiket}.csv",
                                  index=False, encoding="utf-8-sig")
    if od is not None and len(od):
        od.to_csv(PROF_DIR / f"olcumler_{etiket}.csv", index=False, encoding="utf-8-sig")

    # ana ozet tablosuna EKLE (her kosu bir satir)
    ozet_yol = PROF_DIR / "memory_profile_ozet.csv"
    satir = pd.DataFrame([olay])
    if ozet_yol.exists():
        satir = pd.concat([pd.read_csv(ozet_yol), satir], ignore_index=True)
    satir.to_csv(ozet_yol, index=False, encoding="utf-8-sig")
    print(f"\n  Kaydedildi -> {PROF_DIR}")
    print(f"    profil_{etiket}.json · asamalar_{etiket}.csv · memory_profile_ozet.csv")


if __name__ == "__main__":
    main()
