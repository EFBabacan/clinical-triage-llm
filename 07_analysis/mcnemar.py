# -*- coding: utf-8 -*-
"""IDDIA 2: sikistirilmis few-shot vs zero-shot, AYNI 550 val kaydi (eslesmis)."""
import os
import pandas as pd
from scipy.stats import binomtest

OUT = "./workdir/output"  # workspace/workdir (see README)
DL = os.path.join(OUT, "colab_downloads")  # Colab sonuc dosyalari (5 Kasim 2026: Indirilenler'den projeye tasindi)

zero = pd.read_csv(os.path.join(OUT, "zeroshot", "zeroshot_v4_val.csv"))
comp = pd.read_csv(os.path.join(OUT, "fewshot", "fewshot_v4_3shot_ultracompact_val.csv"))
full = pd.read_csv(os.path.join(DL, "full_3shot_val.csv"))


def mcnemar(dfa, dfb, na, nb):
    m = dfa[["note_id", "label", "parsed_pred"]].merge(
        dfb[["note_id", "parsed_pred"]], on="note_id", suffixes=("_a", "_b"))
    ca = (m["parsed_pred_a"] == m["label"])
    cb = (m["parsed_pred_b"] == m["label"])
    b = int((ca & ~cb).sum())
    c = int((~ca & cb).sum())
    p = binomtest(b, b + c, 0.5).pvalue if (b + c) > 0 else 1.0
    print(f"\n{na}  vs  {nb}   (eslesmis n={len(m)})")
    print(f"   {na} dogru & {nb} yanlis : {b}")
    print(f"   {nb} dogru & {na} yanlis : {c}")
    print(f"   p = {p:.4g}  -> {'ANLAMLI (p<0.05)' if p < 0.05 else 'ANLAMLI DEGIL'}")
    return p


print("=" * 92)
print("McNEMAR — VAL seti (n=550), IDDIA 2'nin gercek testi")
print("=" * 92)
mcnemar(zero, comp, "zero-shot", "SIKISTIRILMIS 3-shot")
mcnemar(full, zero, "TAM 3-shot", "zero-shot")
mcnemar(full, comp, "TAM 3-shot", "SIKISTIRILMIS 3-shot")

print("\n" + "=" * 92)
print("PROMPT UZUNLUGU (K3 icin hazir veri)")
print("=" * 92)
for nm, df in [("zero-shot val", zero), ("sikistirilmis 3-shot val", comp), ("TAM 3-shot val", full)]:
    tok = df["in_tokens"]
    line = f"{nm:26s} girdi token: ort={tok.mean():8.0f}  medyan={tok.median():8.0f}  maks={tok.max():8.0f}"
    if "sure_s" in df.columns:
        s = df["sure_s"]
        line += f" | gecikme ort={s.mean():5.2f}s  toplam={s.sum()/60:6.1f} dk"
    else:
        line += " | gecikme: KAYIT YOK"
    print(line)
