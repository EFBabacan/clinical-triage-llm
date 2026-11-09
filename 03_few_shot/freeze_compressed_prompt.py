# -*- coding: utf-8 -*-
"""
B1 hazirligi — SIKISTIRILMIS FEW-SHOT ISTEMINI DONDUR

Neden: B1'de sikistirilmis istemi A100'de yeniden kosturacagiz. Istemi Colab'da
sifirdan kurmak riskli — tek karakter kayarsa "istem uzunlugu mu, donanim mi"
sorusunu cozmek yerine ucuncu bir degisken eklemis oluruz.

Bu script istemi yerelde yeniden uretir ve LOG'DAKI TOKEN SAYISIYLA DOGRULAR:
  __blok14v4_3shot_ultracompact_output.txt ->
     "Sabit prompt (header + 33 ultra example + trailer wrap): 1805 token"
Tutarsa istem birebir yeniden uretilmis demektir; dosyaya donduruluyor.
"""
import json
import os

from transformers import AutoTokenizer

BASE = os.path.dirname(os.path.abspath(__file__))
JSON_YOL = os.path.join(BASE, "output", "fewshot",
                        "fewshot_v4_3shot_ultracompact_examples.json")
CIKTI = os.path.join(BASE, "output", "fewshot", "compressed_3shot_PROMPT_FROZEN.txt")

BEKLENEN_TOKEN = 1805      # __blok14v4_3shot_ultracompact_output.txt satir 51

ALLOWED_CLASSES = ["SEPSIS", "STROKE", "ATRIAL_FIBRILLATION", "AKI", "PNEUMONIA",
                   "UTI", "HEART_FAILURE", "GI_BLEEDING", "DIABETES_COMPLICATION",
                   "COPD_EXACERBATION", "OTHER"]

# __run_blok14c_v4_3shot_ultra_val.py satir 103-126'dan BIREBIR
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

Below are 33 ultra-compact example triage notes (3 per class):
"""
EX_TMPL = "\nEx{i} ({label}): {input}\n=> {label}\n"
TT = "\nNow classify:\n{input}\n=> "

ultra = json.load(open(JSON_YOL, encoding="utf-8"))

SABIT = HEADER
i = 1
n_ornek = 0
for cls in ALLOWED_CLASSES:
    for ex in ultra[cls]:
        SABIT += EX_TMPL.format(i=i, label=cls,
                                input=ex["ultra_text"].replace("\n", " | "))
        i += 1
        n_ornek += 1

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", local_files_only=True)


def chat_token_count(prompt):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    return len(tok.encode(text, add_special_tokens=False))


sabit_n = chat_token_count(SABIT + TT.format(input=""))

print("=" * 72)
print("SIKISTIRILMIS ISTEM — YENIDEN URETIM DOGRULAMASI")
print("=" * 72)
print(f"  Ornek sayisi        : {n_ornek}  (beklenen 33)")
print(f"  Ornek token toplami : {sum(e['ultra_token'] for c in ALLOWED_CLASSES for e in ultra[c])}")
print(f"  Sabit istem token   : {sabit_n}")
print(f"  Log'daki deger      : {BEKLENEN_TOKEN}")

if sabit_n == BEKLENEN_TOKEN and n_ornek == 33:
    print(f"\n  ==> BIREBIR TUTUYOR. Istem yeniden uretildi, donduruluyor.")
    with open(CIKTI, "w", encoding="utf-8") as f:
        f.write(SABIT)
    print(f"  Kaydedildi: {CIKTI}")
    print(f"  ({len(SABIT):,} karakter)")
    print("\n  B1 Colab script'i bu dosyayi AYNEN kullanacak — yeniden kurmayacak.")
else:
    print(f"\n  ==> TUTMUYOR! Fark {sabit_n - BEKLENEN_TOKEN:+d} token.")
    print("  Istem birebir yeniden uretilemedi. B1'e gecmeden once bu cozulmeli,")
    print("  yoksa 'istem uzunlugu mu donanim mi' sorusuna ucuncu degisken eklenir.")
