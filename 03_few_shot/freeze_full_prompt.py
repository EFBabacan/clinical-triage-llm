# -*- coding: utf-8 -*-
"""
B2 hazirligi — TAM (sikistirilmamis) 3-shot ISTEMINI DONDUR

B1'de sikistirilmis istemi dondurup log'daki 1805 token ile dogrulamistik.
Burada ayni sey tam istem icin, ama DAHA GUCLU bir kontrolle:
istem yeniden kurulup val kayitlarinin in_tokens degerleri hesaplanir ve
full_3shot_val.csv'deki GERCEK kayitlarla KAYIT KAYIT karsilastirilir.
Hepsi tutarsa yeniden uretim kanitlanmis olur.

Kaynak: colab_fewshot/blok04_full_3shot_TEST.py satir 22-55
"""
import json
import os

import pandas as pd
from transformers import AutoTokenizer

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")
DL = os.path.join(OUT, "colab_downloads")  # Colab sonuc dosyalari (5 Kasim 2026: Indirilenler'den projeye tasindi)
CIKTI = os.path.join(OUT, "fewshot", "full_3shot_PROMPT_FROZEN.txt")

ALLOWED_CLASSES = ["SEPSIS", "STROKE", "ATRIAL_FIBRILLATION", "AKI", "PNEUMONIA",
                   "UTI", "HEART_FAILURE", "GI_BLEEDING", "DIABETES_COMPLICATION",
                   "COPD_EXACERBATION", "OTHER"]
V4COL = "triage_input_v4_timeaware_masked"

HEADER = """You are a clinical NLP system. Given a patient's triage information collected at first contact (chief complaint, symptoms, brief history, medications, allergies, vital signs), classify the most likely PRIMARY reason for admission into ONE of these 11 categories:

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
- OTHER is NOT an uncertainty label. Choose OTHER ONLY when none of the 10 named diagnoses is clinically supported.
- If one of the 10 named classes is plausible, choose that named class instead of OTHER.

Below are 33 example triage notes (3 per class):
"""
EX_TMPL = "\nExample {i} ({label}):\n{input}\n=> {label}\n"
TT = "\nNow classify:\n{input}\n=> "

ex3 = json.load(open(os.path.join(OUT, "fewshot", "fewshot_v4_3shot_examples.json"),
                     encoding="utf-8"))

SABIT = HEADER
i = 0
for cls in ALLOWED_CLASSES:
    for ex in ex3[cls]:
        t = str(ex.get(V4COL) or "")
        i += 1
        SABIT += EX_TMPL.format(i=i, label=cls, input=t.strip())

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", local_files_only=True)


def toplam_token(query):
    prompt = SABIT + TT.format(input=query)
    txt = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  tokenize=False, add_generation_prompt=True,
                                  enable_thinking=False)
    return len(tok(txt)["input_ids"])


print("=" * 74)
print("TAM 3-SHOT ISTEMI — YENIDEN URETIM DOGRULAMASI")
print("=" * 74)
print(f"  Ornek sayisi : {i}  (beklenen 33)")
print(f"  Sabit karakter: {len(SABIT):,}")
print(f"  Sabit token   : {toplam_token(''):,}")

# --- KAYIT KAYIT dogrulama: val kosusunun gercek in_tokens'lariyla kiyas ---
val_csv = os.path.join(DL, "full_3shot_val.csv")
if not os.path.exists(val_csv):
    print(f"\n  [ATLA] {val_csv} yok — kayit kayit dogrulama yapilamadi.")
else:
    v = pd.read_csv(val_csv)
    df = pd.read_parquet(os.path.join(OUT, "discharge_split.parquet"))
    dv = df[df["split"] == "val"][["note_id", V4COL]]
    m = v[["note_id", "in_tokens"]].merge(dv, on="note_id").head(60)

    farklar = []
    for _, r in m.iterrows():
        q = r[V4COL] if isinstance(r[V4COL], str) else ""
        farklar.append(toplam_token(q) - int(r["in_tokens"]))
    fs = pd.Series(farklar)
    tam = int((fs == 0).sum())
    print(f"\n  Kayit kayit kontrol (ilk {len(m)} val kaydi):")
    print(f"     birebir tutan : {tam}/{len(m)}")
    print(f"     fark ort/maks : {fs.mean():+.2f} / {fs.abs().max()}")

    if tam == len(m):
        print("\n  ==> BIREBIR TUTUYOR. Istem yeniden uretildi.")
        with open(CIKTI, "w", encoding="utf-8") as f:
            f.write(SABIT)
        print(f"  Kaydedildi: {CIKTI}")
    elif fs.abs().max() <= 2:
        print("\n  ==> +-2 token icinde (tokenizer surum farki). Kabul edilebilir,")
        print("      istem donduruluyor — ama B2 ciktisinda in_tokens'i kontrol edin.")
        with open(CIKTI, "w", encoding="utf-8") as f:
            f.write(SABIT)
        print(f"  Kaydedildi: {CIKTI}")
    else:
        print("\n  ==> TUTMUYOR. Istem birebir yeniden uretilemedi.")
        print("      B2'ye gecmeden once bu cozulmeli; yoksa 0.5552'yi dogrulamis olmayiz.")
