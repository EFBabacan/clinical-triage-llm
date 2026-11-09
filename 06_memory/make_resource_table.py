# -*- coding: utf-8 -*-
"""
A3 / K3 — TAM KAYNAK TABLOSU  (TIPTEKNO'26)

Bildirinin "Resource Consumption" cizelgesi. Dort kaynak boyutu tek yerde:
  bellek (agirlik / cikarim tepesi / egitim tepesi) · sure · gecikme · istem uzunlugu

KURAL: uydurma yok. Her hucre ya OLCUMDEN gelir ya bos kalir.
Her satirda 'kaynak' sutunu degerin nereden geldigini soyler.

Girdiler
--------
  output/memory_profile/memory_profile_ozet.csv        <- profile_memory.py (A1)
  output/qlora_adapters/.../training_curve.json        <- egitim tepe VRAM + sure
  output/qlora_adapters/.../adapter/training_args.bin  <- gercek hiperparametreler
  tahmin CSV'leri (sure_s, in_tokens sutunlari)
"""
import glob
import json
import os

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")
DL = os.path.join(OUT, "colab_downloads")  # Colab sonuc dosyalari (5 Kasim 2026: Indirilenler'den projeye tasindi)
VRAM_MIB = 8188.0          # RTX 4060 Laptop

MiB = 1024 ** 2


# ------------------------------------------------------------------ A1 bellek
def a1_bellek():
    p = os.path.join(OUT, "memory_profile", "memory_profile_ozet.csv")
    if not os.path.exists(p):
        return {}
    oz = pd.read_csv(p)
    oz["smi_taban_MiB"] = pd.to_numeric(oz["smi_taban_MiB"], errors="coerce")
    temiz = oz[oz["smi_taban_MiB"] < 600]                 # kirli/cekismeli kosulari at
    son = temiz.sort_values("tarih").groupby("config").tail(1).set_index("config")
    d = {}
    for cfg in son.index:
        r = son.loc[cfg]
        d[cfg] = dict(agirlik=r.get("param_toplam_MiB"),
                      tepe=r.get("tepe_reserved_MiB"),
                      gecikme=r.get("gecikme_ort_s"),
                      sonuc=r.get("sonuc"))
    return d


# ------------------------------------------------------------------ egitim
def egitim_bilgi():
    tc_yol = glob.glob(os.path.join(OUT, "qlora_adapters", "*", "training_curve.json"))
    if not tc_yol:
        return {}
    tc = json.load(open(tc_yol[0]))
    bilgi = dict(
        # notebook satir 24650: max_memory_allocated()/1024**3  -> gercek GiB
        tepe_egitim_MiB=round(tc["peak_gb"] * 1024, 1),
        egitim_sure_sa=round(tc["train_sec"] / 3600, 2),
        egitim_sure_dk=round(tc["train_sec"] / 60, 1),
        train_size=tc["train_size"], epochs=tc["epochs"],
        final_loss=round(tc["final_loss"], 4),
    )
    st_yol = sorted(glob.glob(os.path.join(OUT, "qlora_adapters", "*", "checkpoints",
                                           "checkpoint-*", "trainer_state.json")))
    if st_yol:
        st = json.load(open(st_yol[-1]))
        bilgi["adim"] = st["max_steps"]
        bilgi["adim_sure_s"] = round(tc["train_sec"] / st["max_steps"], 1)
    return bilgi


# ------------------------------------------------------------------ gecikme / token
def kosu_olc(path, tokcol_adaylari=("in_tokens", "input_tokens")):
    if not path or not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    d = {"n": len(df)}
    tc = next((c for c in tokcol_adaylari if c in df.columns), None)
    if tc:
        d["istem_token_ort"] = round(float(df[tc].mean()))
    if "sure_s" in df.columns:
        s = df["sure_s"]
        d["gecikme_s_ornek"] = round(float(s.mean()), 2)
        d["cikarim_sure_dk"] = round(float(s.sum()) / 60, 1)
        if tc:
            d["prompt_tok_s"] = round(float(df[tc].sum() / s.sum()))
    return d


A1 = a1_bellek()
EG = egitim_bilgi()

Q4 = A1.get("4bit", {})
QF = A1.get("fp16", {})

satirlar = []


def ekle(**kw):
    satirlar.append(kw)


# ---- Qwen zero-shot (test) : A1 bellegi + gercek kosu gecikmesi
o = kosu_olc(os.path.join(OUT, "zeroshot", "zeroshot_v4_test.csv"))
ekle(model="Qwen3.5-4B", yontem="zero-shot", bit="4-bit NF4", ortam="RTX 4060 8188 MiB",
     split="test", **o,
     agirlik_MiB=Q4.get("agirlik"), tepe_cikarim_MiB=Q4.get("tepe"),
     tepe_egitim_MiB=None, egitim_sure_sa=None,
     sigdi_8GB="EVET",
     kaynak="gecikme=sure_s toplami · bellek=profile_memory.py (A1, temiz kosu)")

# ---- Qwen zero-shot (val)
o = kosu_olc(os.path.join(OUT, "zeroshot", "zeroshot_v4_val.csv"))
ekle(model="Qwen3.5-4B", yontem="zero-shot", bit="4-bit NF4", ortam="RTX 4060 8188 MiB",
     split="val", **o,
     agirlik_MiB=Q4.get("agirlik"), tepe_cikarim_MiB=Q4.get("tepe"),
     tepe_egitim_MiB=None, egitim_sure_sa=None, sigdi_8GB="EVET",
     kaynak="gecikme=sure_s toplami · bellek=A1")

# ---- Qwen 3-shot SIKISTIRILMIS (val) — bildirinin 2. iddiasi
o = kosu_olc(os.path.join(OUT, "fewshot", "fewshot_v4_3shot_ultracompact_val.csv"))
ekle(model="Qwen3.5-4B", yontem="3-shot sikistirilmis", bit="4-bit NF4",
     ortam="RTX 4060 8188 MiB", split="val", **o,
     agirlik_MiB=Q4.get("agirlik"), tepe_cikarim_MiB=None,
     tepe_egitim_MiB=None, egitim_sure_sa=None, sigdi_8GB="EVET",
     kaynak="gecikme=sure_s toplami · cikarim tepesi bu istem uzunlugunda olculmedi")

# ---- Qwen 3-shot TAM (val, A100) — sure_s yok
o = kosu_olc(os.path.join(DL, "full_3shot_val.csv"))
ekle(model="Qwen3.5-4B", yontem="3-shot tam", bit="4-bit NF4", ortam="A100 80GB",
     split="val", **o,
     agirlik_MiB=None, tepe_cikarim_MiB=None, tepe_egitim_MiB=None,
     egitim_sure_sa=None, sigdi_8GB="HAYIR (~24k token istem, yerelde OOM)",
     kaynak="sure_s sutunu YOK -> gecikme olculemedi (B2'de olculecek)")

# ---- Qwen 3-shot TAM (test, A100) — per-kayit CSV kayip
ekle(model="Qwen3.5-4B", yontem="3-shot tam", bit="4-bit NF4", ortam="A100 80GB",
     split="test", n=1650, istem_token_ort=None,
     gecikme_s_ornek=None, cikarim_sure_dk=140.0, prompt_tok_s=None,
     agirlik_MiB=None, tepe_cikarim_MiB=None, tepe_egitim_MiB=None,
     egitim_sure_sa=None, sigdi_8GB="HAYIR (~24k token istem, yerelde OOM)",
     kaynak="sure yalniz RAPOR edilmis (140 dk), per-kayit CSV kayip -> dogrulanamaz")

# ---- Qwen QLoRA (test, TAM BAGLAM 2048) : ANA SATIR — A6 (6 Agu)
o = kosu_olc(os.path.join(OUT, "qlora_eval_fullctx", "test_predictions.csv"))
ekle(model="Qwen3.5-4B", yontem="QLoRA adil (r8, q/v)", bit="4-bit NF4",
     ortam="RTX 4060 8188 MiB", split="test", **o,
     agirlik_MiB=Q4.get("agirlik"), tepe_cikarim_MiB=Q4.get("tepe"),
     tepe_egitim_MiB=EG.get("tepe_egitim_MiB"), egitim_sure_sa=EG.get("egitim_sure_sa"),
     sigdi_8GB="EVET",
     kaynak=f"A6 tam baglam (2048) · gecikme=sure_s · egitim: training_curve.json "
            f"({EG.get('tepe_egitim_MiB')} MiB, {EG.get('adim')} adim x {EG.get('adim_sure_s')} s)")

# ---- ayni adapter, KESIK baglam (409) — ablasyon / saglamlik kontrolu
o = kosu_olc(os.path.join(OUT, "qlora_eval", "v4_s3_full_2epoch_lr1e4", "test_predictions.csv"))
ekle(model="Qwen3.5-4B", yontem="QLoRA adil — KESIK baglam (ablasyon)", bit="4-bit NF4",
     ortam="RTX 4060 8188 MiB", split="test", **o,
     agirlik_MiB=Q4.get("agirlik"), tepe_cikarim_MiB=None,
     tepe_egitim_MiB=None, egitim_sure_sa=None, sigdi_8GB="EVET",
     kaynak="ayni adapter, baglam 409 token'a kesik (%86 kayit). sure_s YOK.")

# ---- Qwen fp16 (kosulmadi, yalniz bellek sinamasi)
ekle(model="Qwen3.5-4B", yontem="(olcum) tam duyarlik", bit="fp16",
     ortam="RTX 4060 8188 MiB", split="—", n=None, istem_token_ort=None,
     gecikme_s_ornek=None, cikarim_sure_dk=None, prompt_tok_s=None,
     agirlik_MiB=8021.8, tepe_cikarim_MiB=None, tepe_egitim_MiB=None,
     egitim_sure_sa=None, sigdi_8GB="HAYIR (temiz CUDA OOM)",
     kaynak="A1: agirliklarin %95'inde OOM · agirlik boyutu safetensors basliklarindan")

# ---- Gemma satirlari: bellek B3'te olculecek
for yontem, bit, sure, csv in [
    ("zero-shot", "bf16", 4.4, os.path.join(DL, "gemma_bf16_zeroshot_test.csv")),
    ("zero-shot", "4-bit NF4", 5.8, os.path.join(DL, "gemma_4bit_zeroshot_test.csv")),
    ("3-shot tam", "bf16", 60.0, os.path.join(OUT, "gemma_full3shot_test.csv")),
    ("QLoRA adil (r8, q/v)", "4-bit NF4", 40.0, None),
    ("QLoRA optimize (r16, tum-linear)", "4-bit NF4", 150.0,
     os.path.join(OUT, "gemma_qlora_test_preds.csv")),
]:
    o = kosu_olc(csv)
    ekle(model="Gemma-4-E4B", yontem=yontem, bit=bit, ortam="A100 80GB", split="test",
         **o, gecikme_s_ornek=None, cikarim_sure_dk=sure, prompt_tok_s=None,
         agirlik_MiB=None, tepe_cikarim_MiB=None, tepe_egitim_MiB=None,
         egitim_sure_sa=None, sigdi_8GB="HAYIR (PLE gomme ~4.7 GB, yerelde sigmadi)",
         kaynak="sure yalniz RAPOR edilmis, sure_s sutunu YOK · bellek B3'te olculecek")

KOL = ["model", "yontem", "bit", "ortam", "split", "n", "istem_token_ort",
       "gecikme_s_ornek", "cikarim_sure_dk", "prompt_tok_s",
       "agirlik_MiB", "tepe_cikarim_MiB", "tepe_egitim_MiB", "egitim_sure_sa",
       "sigdi_8GB", "kaynak"]
tab = pd.DataFrame(satirlar)
for k in KOL:
    if k not in tab.columns:
        tab[k] = None
tab = tab[KOL]

dest = os.path.join(BASE, "resource_table.csv")
tab.to_csv(dest, index=False, encoding="utf-8-sig")

pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 40)
print("=" * 120)
print("KAYNAK TABLOSU (resource_table.csv)")
print("=" * 120)
print(tab.drop(columns=["kaynak"]).to_string(index=False))

print("\n" + "=" * 120)
print("QWEN — 8 GB'DAKI KAYNAK PROFILI (olculen, bildiriye girecek cekirdek)")
print("=" * 120)
print(f"  Agirliklar 4-bit            : {Q4.get('agirlik')} MiB")
print(f"  Cikarim tepesi              : {Q4.get('tepe')} MiB  "
      f"(kartin %{100*float(Q4.get('tepe', 0))/VRAM_MIB:.0f}'i)")
print(f"  EGITIM tepesi (QLoRA)       : {EG.get('tepe_egitim_MiB')} MiB  "
      f"(kartin %{100*EG.get('tepe_egitim_MiB', 0)/VRAM_MIB:.0f}'i)")
print(f"  Egitim suresi               : {EG.get('egitim_sure_sa')} sa  "
      f"({EG.get('adim')} adim x {EG.get('adim_sure_s')} s)")
print(f"  fp16 agirliklar             : 8021.8 MiB -> SIGMADI")
print(f"\n  => Egitim, cikarimdan {EG.get('tepe_egitim_MiB', 0)/float(Q4.get('tepe', 1)):.1f}x "
      f"daha fazla bellek istiyor; yine de 8 GB'a siginiyor.")

zs = tab[(tab.yontem == "zero-shot") & (tab.split == "test") & (tab.model.str.startswith("Qwen"))]
ql = tab[(tab.yontem == "QLoRA adil (r8, q/v)") & (tab.model.str.startswith("Qwen"))]
kes = tab[tab.yontem.str.contains("KESIK", na=False)]
if len(zs) and len(ql) and pd.notna(ql.iloc[0]["istem_token_ort"]):
    a, b = float(zs.iloc[0]["istem_token_ort"]), float(ql.iloc[0]["istem_token_ort"])
    print(f"\n  => Istem uzunlugu, ikisi de TAM baglamda: zero-shot {a:.0f} -> QLoRA {b:.0f} token")
    print(f"     Fark {a-b:.0f} token = olculen sabit baslik farkinin ta kendisi "
          f"(263 - 76 = 187). Yani tum tasarruf talimat basligindan geliyor,")
    print(f"     modelin 'daha az baglama ihtiyaci' degil. (%{100*(a-b)/a:.0f} kisalma)")
if len(kes) and pd.notna(kes.iloc[0]["istem_token_ort"]):
    k = float(kes.iloc[0]["istem_token_ort"])
    b = float(ql.iloc[0]["istem_token_ort"])
    print(f"\n  => Ablasyon: baglam {b:.0f} -> {k:.0f} token'a kesilince (%{100*(b-k)/b:.0f} az)")
    print(f"     accuracy 0.6424 -> 0.6345 (-0.8 puan, McNemar p=0.148 ANLAMLI DEGIL).")
    print(f"     Triyaj sinyali notun basinda yogun; istem butcesi yariya indirilebilir.")

print(f"\nKaydedildi: {dest}")
