# =====================================================================
# BLOK G06a — Gemma QLoRA veri yukle (files.upload) — Colab
# =====================================================================
# YUKLENECEK 3 JSON (local'den, surukle-birak):
#   1) qlora_v4_train_S3.json  (4400)  -> egitim (Qwen S3 ile AYNI dosya)
#   2) qlora_v4_val.json       (550)   -> val
#   3) qlora_v4_test.json      (1650)  -> test
#   KONUM: workdir/output/qlora_json/qlora_v4_*.json
# NOT: Bu 3 dosya Qwen QLoRA'nin egitildigi DOSYALARIN AYNISI -> adil karsilastirma.
#      Icerik: instruction + input(triage_input_v4) + output(sinif). Train ile eval ayni instruction.
# =====================================================================
from google.colab import files
import os, json

# Dosyalar zaten /content'te varsa upload'i atla:
gerekli = ["qlora_v4_train_S3.json", "qlora_v4_val.json", "qlora_v4_test.json"]
eksik = [f for f in gerekli if not os.path.exists(f)]

if eksik:
    print(f">>> Su dosyalari sec ve yukle: {eksik}")
    files.upload()
else:
    print("Tum dosyalar zaten /content'te var, upload atlandi.")

# ---- Dogrulama ----
def load_json(p):
    with open(p, encoding="utf-8") as f: return json.load(f)

for f in gerekli:
    assert os.path.exists(f), f"EKSIK: {f} yuklenmedi!"

train_recs = load_json("qlora_v4_train_S3.json")
val_recs   = load_json("qlora_v4_val.json")
test_recs  = load_json("qlora_v4_test.json")

print(f"\ntrain_S3 = {len(train_recs)} (hedef 4400)")
print(f"val      = {len(val_recs)} (hedef 550)")
print(f"test     = {len(test_recs)} (hedef 1650)")
assert len(train_recs)==4400 and len(val_recs)==550 and len(test_recs)==1650, "BOYUT UYUSMUYOR!"

# Yapi kontrolu (instruction/input/output)
ornek = train_recs[0]
print(f"\nKayit anahtarlari: {list(ornek.keys())}")
assert all(k in ornek for k in ["instruction","input","output"]), "instruction/input/output eksik!"
print(f"\n--- Ornek instruction ---\n{ornek['instruction']}")
print(f"\n--- Ornek input (ilk 300 char) ---\n{ornek['input'][:300]}")
print(f"\n--- Ornek output ---\n{ornek['output']}")

# Sinif dagilimi (dengeli mi?)
from collections import Counter
dag = Counter(r["output"] for r in train_recs)
print(f"\nTrain sinif dagilimi (11x400 olmali): {dict(dag)}")

print("\n✅ BLOK G06a TAMAM — veri hazir (train_recs, val_recs, test_recs).")
print("   Sonraki: BlokG06b — Gemma QLoRA train (Qwen S3 ile ayni ayar).")
