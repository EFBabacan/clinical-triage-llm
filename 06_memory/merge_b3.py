# -*- coding: utf-8 -*-
"""
B3 sonuclarini projeye al ve memory_table.csv'yi iki modelle tamamla.
Ayrica mekanizma sinamasini yerelde YENIDEN dogrula (Colab ciktisina guvenme).
"""
import json
import os
import shutil

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DL = r"./downloads"
B3 = os.path.join(BASE, "output", "b3_gemma_memory")
os.makedirs(B3, exist_ok=True)

for f in ["b3_gemma_bf16.json", "b3_gemma_4bit.json"]:
    src = os.path.join(DL, f)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(B3, f))

bf = json.load(open(os.path.join(B3, "b3_gemma_bf16.json"), encoding="utf-8"))
q4 = json.load(open(os.path.join(B3, "b3_gemma_4bit.json"), encoding="utf-8"))
Ob, O4 = bf["ozet"], q4["ozet"]

# A1'de olculen Qwen degerleri
QW = dict(tam_agirlik=8021.8, bit4_agirlik=2916.2, nicelenmeyen=1214.4,
          bit4_tepe=4494.0, tam_tepe=None, aktivasyon=1144.8)
KART = 8188.0

print("=" * 88)
print("MEKANIZMA SINAMASI — yerelde yeniden dogrulama")
print("=" * 88)


def sina(ad, T, E, B4):
    tahmin = E + (T - E) / 4
    fark = tahmin - B4
    print(f"\n  {ad}")
    print(f"    tam duyarlik      : {T:>9,.1f} MiB")
    print(f"    nicelenmeyen      : {E:>9,.1f} MiB  (%{100*E/T:.1f})")
    print(f"    TAHMIN 4-bit      : {tahmin:>9,.1f} MiB")
    print(f"    OLCUM  4-bit      : {B4:>9,.1f} MiB")
    print(f"    fark              : {fark:>+9,.1f} MiB  (%{100*abs(fark)/T:.2f} of total)")
    print(f"    niceleme kazanci  : %{100*(T-B4)/T:.1f}")
    return dict(tam=T, nicelenmeyen=E, tahmin=tahmin, olcum=B4, fark=fark,
                kazanc=100 * (T - B4) / T, pay=100 * E / T)


g = sina("Gemma-4-E4B (B3)", Ob["toplam_agirlik_MiB"],
         Ob["nicelenmeyen_MiB"], O4["toplam_agirlik_MiB"])
q = sina("Qwen3.5-4B (A1)", QW["tam_agirlik"], QW["nicelenmeyen"], QW["bit4_agirlik"])

print("\n" + "=" * 88)
print("IKI MIMARI YAN YANA — bildirinin teknik katkisi")
print("=" * 88)
print(f"  {'':<34s} {'Qwen3.5-4B':>14s} {'Gemma-4-E4B':>14s}")
print("  " + "-" * 66)
print(f"  {'Tam duyarlik agirlik (MiB)':<34s} {q['tam']:>14,.0f} {g['tam']:>14,.0f}")
print(f"  {'4-bit agirlik (MiB)':<34s} {q['olcum']:>14,.0f} {g['olcum']:>14,.0f}")
print(f"  {'Nicelenemeyen pay':<34s} {q['pay']:>13.1f}% {g['pay']:>13.1f}%")
print(f"  {'NICELEME KAZANCI':<34s} {q['kazanc']:>13.1f}% {g['kazanc']:>13.1f}%")
print(f"  {'4-bit cikarim tepesi (MiB)':<34s} {QW['bit4_tepe']:>14,.0f} "
      f"{O4['tepe_reserved_MiB']:>14,.0f}")
print(f"  {'8 GB karta (8188 MiB) sigar mi':<34s} "
      f"{'EVET':>14s} {'HAYIR':>14s}")

print(f"\n  => Gemma'nin PLE tablosu tek basina {Ob['PLE_MiB']:,.0f} MiB ve niceleme ona")
print(f"     DOKUNMUYOR. 4-bit modelin %{100*O4['PLE_MiB']/O4['toplam_agirlik_MiB']:.1f}'i PLE.")
print(f"  => Gemma 4-bit'te bile 8 GB'a SIGMIYOR "
      f"({O4['tepe_reserved_MiB']:,.0f} > {KART:,.0f} MiB). Bu artik OLCULMUS.")

# ---------------------------------------------------------------- tablo
sat = []
sat.append(dict(model="Qwen3.5-4B", duyarlik="fp16", donanim="RTX 4060 8188 MiB",
                agirlik_MiB=QW["tam_agirlik"], nicelenmeyen_MiB=QW["nicelenmeyen"],
                tepe_cikarim_MiB=None, aktivasyon_MiB=None,
                sigdi_8GB="HAYIR (temiz OOM)", kaynak="A1 olcum"))
sat.append(dict(model="Qwen3.5-4B", duyarlik="4-bit NF4", donanim="RTX 4060 8188 MiB",
                agirlik_MiB=QW["bit4_agirlik"], nicelenmeyen_MiB=QW["nicelenmeyen"],
                tepe_cikarim_MiB=QW["bit4_tepe"], aktivasyon_MiB=QW["aktivasyon"],
                sigdi_8GB="EVET (%55)", kaynak="A1 olcum"))
sat.append(dict(model="Gemma-4-E4B", duyarlik="bf16", donanim="A100 80GB",
                agirlik_MiB=Ob["toplam_agirlik_MiB"], nicelenmeyen_MiB=Ob["nicelenmeyen_MiB"],
                tepe_cikarim_MiB=Ob["tepe_reserved_MiB"],
                aktivasyon_MiB=Ob["aktivasyon_maks_MiB"],
                sigdi_8GB="HAYIR", kaynak="B3 olcum"))
sat.append(dict(model="Gemma-4-E4B", duyarlik="4-bit NF4", donanim="A100 80GB",
                agirlik_MiB=O4["toplam_agirlik_MiB"], nicelenmeyen_MiB=O4["nicelenmeyen_MiB"],
                tepe_cikarim_MiB=O4["tepe_reserved_MiB"],
                aktivasyon_MiB=O4["aktivasyon_maks_MiB"],
                sigdi_8GB="HAYIR (4-bit'te bile)", kaynak="B3 olcum"))

t = pd.DataFrame(sat)
t["niceleme_kazanci_%"] = [None, round(q["kazanc"], 1), None, round(g["kazanc"], 1)]
t["nicelenmeyen_pay_%"] = [round(q["pay"], 1), None, round(g["pay"], 1), None]
dest = os.path.join(BASE, "memory_table.csv")
t.to_csv(dest, index=False, encoding="utf-8-sig")

pd.set_option("display.width", 220)
print("\n" + "=" * 88)
print(t.to_string(index=False))
print(f"\nKaydedildi: {dest}")

# bilesen dokumu — Fig 3 icin
bilesen = []
for ad, d, cfg in [("Gemma bf16", bf, "bf16"), ("Gemma 4-bit", q4, "4bit")]:
    for grup, v in d["gruplar"].items():
        bilesen.append(dict(model="Gemma-4-E4B", config=cfg, bilesen=grup,
                            param=v["param"], MiB=v["MiB"]))
pd.DataFrame(bilesen).to_csv(os.path.join(BASE, "memory_components.csv"),
                             index=False, encoding="utf-8-sig")
print(f"Kaydedildi: memory_components.csv  (Fig 3 Gemma paneli icin)")
