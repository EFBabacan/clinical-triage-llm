# -*- coding: utf-8 -*-
"""
METIN DENETIMI — main.tex'e ELLE yazilmis her sayiyi ham veriden dogrula.

NEDEN: verify_all.py yalniz numbers.tex makrolarini kontrol ediyordu.
"%61" hatasi oradan gecti cunku metne dogrudan yazilmisti ve hicbir makroya
bagli degildi. Bu script o bosugu kapatir:

  A) TURETILMIS ORANLAR — metindeki her oran/kat iddiasini yeniden hesapla
  B) SERBEST SAYI TARAMASI — makroya baglanmamis her sayiyi listele ki
     gozden gecirilsin (hepsi hata degil; hiperparametreler mesru)
"""
import os
import re
import sys

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(BASE)
OUT = os.path.join(PROJ, "output")

tex = open(os.path.join(BASE, "main.tex"), encoding="utf-8").read()
tex = re.sub(r"(?m)^\s*%.*$", "", tex)          # yorumlari at
# LaTeX satir sonlari kontrolleri sessizce atlatiyordu: tek satira indir.
duz = re.sub(r"\s+", " ", tex)

rm = pd.read_csv(os.path.join(PROJ, "results_master.csv")).set_index("yapilandirma")
mc = pd.read_csv(os.path.join(PROJ, "memory_components.csv"))
mt = pd.read_csv(os.path.join(PROJ, "memory_table.csv"))
mt["duyarlik"] = mt["duyarlik"].str.replace(" NF4", "", regex=False)

# numbers.tex makrolari — metin artik bunlari kullaniyor, degerleri burada dogrula
mak = {}
_h = re.sub(r"\\ensuremath\{([^}]*)\}", r"\1",
            open(os.path.join(BASE, "numbers.tex"), encoding="utf-8").read())
for _m in re.finditer(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", _h):
    mak[_m.group(1)] = _m.group(2).replace("\\%", "")

hata = 0


def kontrol(ad, metinde, hesap, tol, birim=""):
    global hata
    ok = abs(metinde - hesap) <= tol
    if not ok:
        hata += 1
    print(f"  {'OK ' if ok else '!! '}{ad:<40s} metin={metinde:>9.2f}{birim}"
          f"  hesap={hesap:>9.2f}{birim}")


def yakala(ad, kalip):
    """Duzlestirilmis metinde ara. BULAMAZSA HATA — sessiz atlama yok.

    Ilk surumde eslesmeyen regex sessizce atlaniyordu; uc kontrol hic
    kosmadan 'OK' izlenimi veriyordu. Artik eslesmeme de bir bulgudur.
    """
    global hata
    m = re.search(kalip, duz)
    if m is None:
        hata += 1
        print(f"  !! {ad:<40s} METINDE BULUNAMADI — cumle degismis olabilir")
    return m


print("=" * 88)
print("A) TURETILMIS ORAN VE KAT IDDIALARI")
print("=" * 88)

# --- gomme paylari (61% hatasinin ciktigi yer) ---
for cfg, ad in [("bf16", "bf16"), ("4bit", "4-bit")]:
    s = mc[mc.config == cfg]
    tot = s["MiB"].sum()
    emb = s[s.bilesen.isin(["PLE (per-layer embedding)", "embedding (standart)"])]["MiB"].sum()
    ple = s[s.bilesen == "PLE (per-layer embedding)"]["MiB"].sum()
    print(f"  Gemma {ad:<6s}: toplam {tot:9,.1f}  gomme {emb:8,.1f} (%{100*emb/tot:.1f})"
          f"  |  yalniz PLE %{100*ple/tot:.1f}")

GE_T = mc[mc.config == "bf16"]["MiB"].sum()
GE_4 = mc[mc.config == "4bit"]["MiB"].sum()
GE_E = mc[(mc.config == "bf16") & mc.bilesen.isin(
    ["PLE (per-layer embedding)", "embedding (standart)"])]["MiB"].sum()

yakala("GemShareAfter makro olarak kullanilmis",
       r"to \\GemShareAfter\\% of\s*Gemma's quantized weights")

# ---------------------------------------------------------------------------
# OKURUN ARITMETIGI. Bildirideki turev sayilar, TABLODA BASILI yuvarlanmis
# degerlerden cikarilinca tutmali. Ham degerlerden hesaplamak 1 MiB'lik
# sapmalar uretiyordu (6,807 vs 6,808 ; 8,552 vs 8,551) ve hakem tabloyu
# alip cikarma yapinca tutturamiyordu.
# Tabloda BASILAN 3 haneli degerlerin farki, metindeki puan farkiyla tutmali.
# results_master.csv bir degeri 4 haneye yuvarlanmis sakladiginda cift
# yuvarlama olusuyordu: 0.551515 -> (CSV) 0.5515 -> (tablo) 0.551, oysa
# dogrusu 0.552. Tablo 5.5 gosterirken metin 5.6 diyordu.
# YON CELISKISI: "fall/rise/drop" gibi bir fiil, hemen ardindan gelen sayi
# dizisiyle ayni yone bakmali. "rare and fall ... 0.7%, 4.2%, 0.0%" yazilmisti;
# 0.7 -> 4.2 bir ARTIS oldugu icin fiil sayilari yalanliyordu.
print("\n  -- yon bildiren fiil, ardindan gelen sayi dizisiyle tutuyor mu --")
_YON = {"fall": -1, "falls": -1, "drop": -1, "drops": -1, "decrease": -1,
        "rise": +1, "rises": +1, "grow": +1, "grows": +1, "increase": +1}
_bulundu = 0
for m in re.finditer(r"\b(" + "|".join(_YON) + r")\b([^.]{0,160})", duz):
    fiil, kuyruk = m.group(1), m.group(2)
    # KARSITLIK belirtecinde kes: "76% against 42% for Qwen" bir DIZI degil,
    # iki modelin kiyasi. Kesmezsek 76->42'yi dusus sanip yanlis alarm verir.
    kuyruk = re.split(r"\b(?:against|versus|compared with|while|whereas|"
                      r"than|rather than)\b", kuyruk)[0]
    sayilar = [float(x) for x in re.findall(r"\\[A-Za-z]*?(\d+\.\d+)", kuyruk)]
    if len(sayilar) < 2:
        # makro adlariyla yazilmis olabilir: degerlerini numbers.tex'ten coz
        adlar = re.findall(r"\\([A-Za-z]+)\\?%", kuyruk)
        sayilar = [float(mak[a]) for a in adlar if a in mak and
                   re.fullmatch(r"[\d.]+", mak[a])]
    if len(sayilar) >= 2:
        _bulundu += 1
        gercek = 1 if sayilar[-1] > sayilar[0] else -1
        ok = gercek == _YON[fiil] or len(set(sayilar)) == 1
        if not ok:
            hata += 1
        print(f"  {'OK ' if ok else '!! '}'{fiil}' + {sayilar} "
              f"-> dizi {'artiyor' if gercek > 0 else 'azaliyor'}")
if not _bulundu:
    print("  OK  yon bildiren fiil + sayi dizisi kalibi yok")

print("\n  -- tablodaki 3 haneli degerlerin farki metinle tutuyor mu --")
_tab = open(os.path.join(BASE, "tab_results.tex"), encoding="utf-8").read()
_tab = re.sub(r"\\textbf\{([^}]*)\}", r"\1", _tab)


def _satir(etiket):
    m = re.search(re.escape(etiket) + r"[^\\]*?&\s*([\d.]+)\s*&", _tab)
    return float(m.group(1)) if m else None


for ad, e1, e2, makro in [
    ("sikistirma bedeli", "Few-shot, compressed", "Few-shot, full", "CompCost"),
    ("QLoRA kazanci (Qwen)", "QLoRA (fair)", "Zero-shot", "QloraGain"),
]:
    a, b = _satir(e1), _satir(e2)
    if a is not None and b is not None:
        kontrol(ad + " (tablo farki)", abs(float(mak[makro])), abs(100 * (a - b)), 0.051, "p")

print("\n  -- okurun tablodan yapacagi cikarma --")


def yuv(model, duyarlik, sutun):
    return round(float(mt[(mt.model == model) & (mt.duyarlik == duyarlik)][sutun].iloc[0]))


for ad, makro, a, b in [
    ("Qwen gomme-haric", "QwNonEmb",
     yuv("Qwen3.5-4B", "fp16", "agirlik_MiB"), yuv("Qwen3.5-4B", "fp16", "nicelenmeyen_MiB")),
    ("Gemma gomme-haric", "GemNonEmb",
     yuv("Gemma-4-E4B", "bf16", "agirlik_MiB"), yuv("Gemma-4-E4B", "bf16", "nicelenmeyen_MiB")),
    ("Gemma kulesiz 4-bit", "GemNoVA",
     yuv("Gemma-4-E4B", "4-bit", "agirlik_MiB"), int(mak["GemVA"])),
]:
    kontrol(ad, float(mak[makro].replace("\\,", "")), a - b, 0.5, " MiB")
kontrol("Gemma kulesiz butce asimi", float(mak["GemOverVA"]),
        float(mak["GemNoVA"].replace("\\,", "")) - 8 * 1024, 0.5, " MiB")

# Giris'te denklem (1)'e ILERI ATIF olmamali — denklem Bolum IV-C'de tanimli.
_giris = duz[duz.find("Introduction"):duz.find("Related Work")]
if "\\ref{eq:mem}" in _giris:
    hata += 1
    print("  !! Giris'te tanimlanmamis denkleme atif var — silin")
else:
    print("  OK  Giris'te ileri denklem atifi yok")
kontrol("Gemma gomme payi, niceleme SONRASI", 76.0, 100 * GE_E / GE_4, 0.6, "%")
kontrol("Gemma gomme payi, TAM duyarlik", 44.0, 100 * GE_E / GE_T, 0.6, "%")

# --- denklem (1) artiklari ---
tahmin = GE_E + (GE_T - GE_E) / 4
artik = abs(tahmin - GE_4)
yakala("denklem (1) artiklari makro",
       r"differs from the measured 4-bit footprint by \\QwResid\\,MiB for\s*"
       r"Qwen and \\GemResid\\,MiB for Gemma")
yakala("denklem (1) girdileri MiB olarak veriliyor",
       r"Taking \$E\$ from\s*Table~\\ref\{tab:memory\} \(\\QwEmbMiB\\,MiB")

# Cihaz kontrolu: discordant ciftler ve p METINDE, makro olarak.
# Eski metin "3 of those 550" diyordu — net fark 3, ama DEGISEN tahmin 33.
yakala("cihaz kontrolu discordant ciftleri veriyor",
       r"changed \\DevDisc\{\} individual predictions.*?\\DevA\{\} against "
       r"\\DevB\{\} discordant,\s*\$p=\\Pdevice\$")
kontrol("cihaz kontrolu: discordant toplami", float(mak["DevDisc"]),
        float(mak["DevA"]) + float(mak["DevB"]), 0.5)
print(f"     (memory_components toplamindan {artik:.1f} MiB; makro memory_table'dan "
      f"turetiliyor — bildiride BASILAN tablo o)")

# --- istem uzunluk oranlari ---
def tok(yol, kol="in_tokens"):
    return float(pd.read_csv(yol)[kol].mean())


z = tok(os.path.join(OUT, "zeroshot", "zeroshot_v4_test.csv"))
c = tok(os.path.join(OUT, "b1_compressed_a100", "b1_compressed_test.csv"))
f = tok(os.path.join(OUT, "b2_b5_verify", "b2_full3shot_test.csv"))

SAYI = {"three": 3, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13}

yakala("sikistirilmis/zero-shot kat makro", r"remains \\CompVsZeroTok\{\} times longer")
kontrol("sikistirilmis / zero-shot kat", float(mak["CompVsZeroTok"]), c / z, 0.06, "x")

m = yakala("tam/sikistirilmis kat cumlesi", r"factor of about (\w+) to")
if m:
    if m.group(1) not in SAYI:
        hata += 1
        print(f"  !! yazi ile sayi cozulemedi: '{m.group(1)}'")
    else:
        kontrol("tam / sikistirilmis kat", SAYI[m.group(1)], f / c, 0.6, "x")

# --- gomme payi farki "factor of three" ---
QW_PAY = float(mt[(mt.model == "Qwen3.5-4B") & (mt.duyarlik == "fp16")]["nicelenmeyen_pay_%"].iloc[0])
m = yakala("gomme payi orani cumlesi", r"by a factor of (\w+) between")
if m and m.group(1) in SAYI:
    kontrol("gomme payi orani (Gemma/Qwen)", SAYI[m.group(1)], (100 * GE_E / GE_T) / QW_PAY,
            0.35, "x")

# --- marjinal %95 araligi (Wilson, p=0.5'te en genis) ---
yakala("marjinal %95 aralik makro", r"\$\\pm\\MarginCI\$ percentage points")
import math
zc, n = 1.959964, 1650
yari = zc * math.sqrt(0.25 / n + zc * zc / (4 * n * n)) / (1 + zc * zc / n)
kontrol("Wilson yari-genislik (p=0.5)", float(mak["MarginCI"]), 100 * yari, 0.06, "p")

# --- kesme ablasyonu ---
qw_full = tok(os.path.join(OUT, "qlora_eval_fullctx", "test_predictions.csv"), "input_tokens")
qw_kes = tok(os.path.join(OUT, "qlora_eval", "v4_s3_full_2epoch_lr1e4", "test_predictions.csv"),
             "input_tokens")
# KESME ABLASYONU bildiriden CIKARILDI (4 sayfa siniri, 14 Agu 2026) — ikincil
# bulguydu, tam metin surumune birakildi. Makrolar uretilmeye devam ediyor ve
# sayilari burada dogruluyoruz ki geri konuldugunda hazir olsun; metinde
# GECMESI artik sart degil.
kontrol("[tam metin icin] baglam kisaltma orani", float(mak["TruncPct"]),
        100 * (qw_full - qw_kes) / qw_full, 1.0, "%")
a_full = float(rm.loc["Qwen QLoRA adil", "accuracy"])
a_kes = float(rm.loc["Qwen QLoRA adil [kesik 409 tok]", "accuracy"])
kontrol("[tam metin icin] kesme maliyeti (puan)", float(mak["TruncCost"]),
        100 * (a_full - a_kes), 0.06, "p")

# --- cikarim ve egitim tepeleri: metin makro kullaniyor mu ---
yakala("cikarim/egitim tepesi makro",
       r"\\QwPeak\\,MiB reserved during inference and QLoRA training at\s*\\TrainPeak\\,MiB")
qr = mt[mt.model == "Qwen3.5-4B"].set_index("duyarlik")
kontrol("QwPeak makro <-> memory_table", float(mak["QwPeak"].replace("\\,", "")),
        float(qr.loc["4-bit", "tepe_cikarim_MiB"]), 1.0, " MiB")

print("\n" + "=" * 88)
print("B) SERBEST SAYI TARAMASI — makroya baglanmamis sayilar")
print("=" * 88)
print("  (hiperparametreler mesru; ORAN/YUZDE olanlar gozden gecirilmeli)")

# makro cagrilarini ve matematik ortamini maskele
mask = re.sub(r"\\[A-Z][a-zA-Z]+\{?\}?", " MACRO ", tex)
mask = re.sub(r"\$[^$]*\$", " MATH ", mask)
mask = re.sub(r"\\cite\{[^}]*\}|\\ref\{[^}]*\}|\\label\{[^}]*\}", " ", mask)

MESRU = {"8", "4", "1", "2", "3", "16", "8", "2048", "20", "33", "11", "150", "550",
         "1650", "4400", "0.05", "0.03", "4060", "80", "100", "0.73", "0.064", "300"}
bulunan = {}
for m in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)\\?%?", mask):
    s = m.group(1)
    if s in MESRU:
        continue
    ctx = mask[max(0, m.start() - 55):m.end() + 20].replace("\n", " ")
    ctx = re.sub(r"\s+", " ", ctx).strip()
    bulunan.setdefault(s, ctx)

for s, ctx in sorted(bulunan.items(), key=lambda x: -len(x[0])):
    print(f"    {s:>8s}  ...{ctx}")

print("\n" + "=" * 88)
print("SONUC: " + (f"{hata} TUTARSIZLIK" if hata else "TURETILMIS IDDIALARIN HEPSI TUTUYOR"))
print("=" * 88)
sys.exit(1 if hata else 0)
