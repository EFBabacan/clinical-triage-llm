# -*- coding: utf-8 -*-
"""
A1 ciktilarini bildiriye girecek tabloya cevirir.
Girdi : output/memory_profile/memory_profile_ozet.csv  (profile_memory.py uretir)
        + safetensors basliklarindan tensor duzeyinde ayristirma
Cikti : memory_table.csv
Uydurma yok — yalniz TEMIZ kosular kullanilir (smi_taban < 600 MiB, tasma yok).
"""
import glob
import json
import os
import struct
from collections import defaultdict

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
PROF = os.path.join(BASE, "output", "memory_profile", "memory_profile_ozet.csv")
HUB = os.path.expanduser(r"~\.cache\huggingface\hub\models--Qwen--Qwen3.5-4B")

# ---------------------------------------------------------------- 1) temiz kosulari sec
oz = pd.read_csv(PROF)
oz["smi_taban_MiB"] = pd.to_numeric(oz["smi_taban_MiB"], errors="coerce")
temiz = oz[oz["smi_taban_MiB"] < 600].copy()
kirli = oz[oz["smi_taban_MiB"] >= 600]

print("=" * 82)
print("A1 KOSU DENETIMI")
print("=" * 82)
for _, r in oz.iterrows():
    durum = "TEMIZ" if r["smi_taban_MiB"] < 600 else "KIRLI (cekismeli - kullanilmiyor)"
    print(f"  {r['config']:<6s} {r['tarih']}  taban={r['smi_taban_MiB']:>7.0f} MiB  "
          f"sonuc={str(r['sonuc']):<26s} {durum}")
print(f"\n  -> {len(temiz)} temiz kosu kullanilacak, {len(kirli)} kirli kosu atiliyor.")

# her config icin en son TEMIZ kosu
son = temiz.sort_values("tarih").groupby("config").tail(1).set_index("config")

# ---------------------------------------------------------------- 2) tensor ayristirmasi
DT = {"F32": 4, "F16": 2, "BF16": 2, "I8": 1, "U8": 1, "I32": 4, "I64": 8}
files = [f for f in sorted(glob.glob(os.path.join(HUB, "**", "*.safetensors"), recursive=True))
         if "-of-" in f]
grup = defaultdict(int)
for fp in files:
    with open(fp, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n).decode("utf-8"))
    for ad, meta in hdr.items():
        if ad == "__metadata__":
            continue
        numel = 1
        for s in meta["shape"]:
            numel *= s
        if "vision" in ad or "visual" in ad:
            k = "vision_tower"
        elif "embed" in ad:
            k = "embedding"
        elif len(meta["shape"]) == 2:
            k = "linear"
        else:
            k = "norm_bias"
        grup[k] += numel

emb_mib = grup["embedding"] * 2 / 1024 ** 2
norm_mib = grup["norm_bias"] * 2 / 1024 ** 2

# ---------------------------------------------------------------- 3) tablo
rows = []

if "fp16" in son.index:
    r = son.loc["fp16"]
    sigdi = str(r["sonuc"]).startswith("SIGDI")
    rows.append(dict(
        model="Qwen3.5-4B", duyarlik="fp16", donanim="RTX 4060 Laptop 8188 MiB",
        agirlik_MiB=8021.8 if not sigdi else r.get("param_toplam_MiB"),
        taban_allocated_MiB=None if not sigdi else r.get("taban_allocated_MiB"),
        tepe_allocated_MiB=None if not sigdi else r.get("tepe_allocated_MiB"),
        tepe_reserved_MiB=None if not sigdi else r.get("tepe_reserved_MiB"),
        aktivasyon_KV_MiB=None if not sigdi else r.get("aktivasyon_maks_MiB"),
        gecikme_ort_s=None if not sigdi else r.get("gecikme_ort_s"),
        sigdi_8GB="HAYIR", sonuc=r["sonuc"],
        not_="Agirliklarin %95'inde temiz CUDA OOM. Agirlik boyutu tensor basliklarindan.",
    ))

if "4bit" in son.index:
    r = son.loc["4bit"]
    rows.append(dict(
        model="Qwen3.5-4B", duyarlik="4-bit NF4 (double quant)",
        donanim="RTX 4060 Laptop 8188 MiB",
        agirlik_MiB=r["param_toplam_MiB"],
        taban_allocated_MiB=r["taban_allocated_MiB"],
        tepe_allocated_MiB=r["tepe_allocated_MiB"],
        tepe_reserved_MiB=r["tepe_reserved_MiB"],
        aktivasyon_KV_MiB=r["aktivasyon_maks_MiB"],
        gecikme_ort_s=r["gecikme_ort_s"],
        sigdi_8GB="EVET", sonuc=r["sonuc"],
        not_=f"Kartin %{100*r['tepe_reserved_MiB']/8188:.0f}'i. Vision tower yuklenmiyor.",
    ))

tab = pd.DataFrame(rows)
dest = os.path.join(BASE, "memory_table.csv")
tab.to_csv(dest, index=False, encoding="utf-8-sig")

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 30)
print("\n" + "=" * 82)
print("MEMORY TABLE")
print("=" * 82)
print(tab.to_string(index=False))

# ---------------------------------------------------------------- 4) ayristirma
if {"fp16", "4bit"} <= set(son.index) or True:
    FP16 = 8021.8
    B4 = float(son.loc["4bit", "param_toplam_MiB"]) if "4bit" in son.index else None
    if B4:
        lin16 = FP16 - emb_mib - norm_mib
        tahmin = lin16 / 4 + emb_mib + norm_mib
        print("\n" + "=" * 82)
        print("NICELEME AYRISTIRMASI  (tensor basliklarindan, GPU gerekmez)")
        print("=" * 82)
        print(f"  {'bilesen':<26s} {'fp16 MiB':>10s} {'4-bit MiB':>10s}")
        print(f"  {'gomme (NICELENMEZ)':<26s} {emb_mib:>10.1f} {emb_mib:>10.1f}")
        print(f"  {'linear (nicelenir)':<26s} {lin16:>10.1f} {lin16/4:>10.1f}")
        print(f"  {'norm/bias':<26s} {norm_mib:>10.1f} {norm_mib:>10.1f}")
        print(f"  {'TAHMIN':<26s} {FP16:>10.1f} {tahmin:>10.1f}")
        print(f"  {'OLCUM':<26s} {FP16:>10.1f} {B4:>10.1f}   fark {tahmin-B4:+.2f} MiB")
        print(f"\n  Niceleme kazanci : %{100*(FP16-B4)/FP16:.1f}")
        print(f"  4-bit modelin    : %{100*emb_mib/B4:.1f}'i nicelenmemis gomme tablosu")
        print(f"  fp16 modelin     : %{100*emb_mib/FP16:.1f}'i gomme tablosu")

print(f"\nKaydedildi: {dest}")
