# =====================================================================
# KLASIK ML BASELINE — V4 Triage (11 sinif, TF-IDF + LR/SVM/RF)
# =====================================================================
# Amac: Tezdeki LLM sonuclarina (zero-shot / few-shot / QLoRA) ADIL bir
#       klasik makine ogrenmesi karsilastirmasi eklemek.
#
# ADIL KARSILASTIRMA:
#   - Egitim: qlora_v4_train_S3.json (4400) -> QLoRA ile AYNI dosya
#   - Test  : qlora_v4_test.json     (1650) -> LLM tablosuyla AYNI test seti
#   - Girdi : kayitlarin "input" alani (V4 time-aware, maskeli)
#   - Etiket: "output" alani (11 siniftan biri)
#   - Sabit random_state=42 -> tekrar uretilebilir (hoca calistirinca ayni sonuc)
#
# GEREKEN: scikit-learn (GPU YOK). CPU'da ~1-2 dk.
#   pip install scikit-learn
#
# CALISTIRMA (proje kokunden):
#   python klasik_ml_v4_baseline.py
# =====================================================================
import os, json
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)

RANDOM_STATE = 42
BASE = os.path.join("output", "qlora_json")
TRAIN_JSON = os.path.join(BASE, "qlora_v4_train_S3.json")  # 4400
TEST_JSON  = os.path.join(BASE, "qlora_v4_test.json")      # 1650
OUT_DIR    = os.path.join("output", "classical_ml_v4")     # YENI klasor (eskiye dokunmuyoruz)
os.makedirs(OUT_DIR, exist_ok=True)

# 11 sinif (sabit sira -> rapor/confusion tutarli olsun)
CLASSES = ["SEPSIS", "STROKE", "ATRIAL_FIBRILLATION", "AKI", "PNEUMONIA",
           "UTI", "HEART_FAILURE", "GI_BLEEDING", "DIABETES_COMPLICATION",
           "COPD_EXACERBATION", "OTHER"]

# ---------------------------------------------------------------------
# 1) Veri yukle
# ---------------------------------------------------------------------
def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)

train = load(TRAIN_JSON)
test  = load(TEST_JSON)
print(f"train = {len(train)} (hedef 4400) | test = {len(test)} (hedef 1650)")

X_train = [r["input"] for r in train]
y_train = [r["output"].strip() for r in train]
X_test  = [r["input"] for r in test]
y_test  = [r["output"].strip() for r in test]

# Etiketler beklenen 11 sinifta mi?
bilinmeyen = set(y_train) | set(y_test)
bilinmeyen -= set(CLASSES)
assert not bilinmeyen, f"Beklenmeyen etiket(ler): {bilinmeyen}"

print("Test sinif dagilimi:", dict(pd.Series(y_test).value_counts()))

# ---------------------------------------------------------------------
# 2) TF-IDF (1-2 gram). LLM ne gordu ise klasik ML de ayni metni gorsun.
# ---------------------------------------------------------------------
vec = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=20000,
    sublinear_tf=True,
    strip_accents="unicode",
    lowercase=True,
)
Xtr = vec.fit_transform(X_train)
Xte = vec.transform(X_test)
print(f"TF-IDF ozellik sayisi: {Xtr.shape[1]}")

# ---------------------------------------------------------------------
# 3) Modeller: LR, Linear SVM, Random Forest
# ---------------------------------------------------------------------
models = {
    "TF-IDF + LR":  LogisticRegression(max_iter=2000, C=10.0,
                                       class_weight="balanced",
                                       random_state=RANDOM_STATE),
    "TF-IDF + SVM": LinearSVC(C=1.0, class_weight="balanced",
                              random_state=RANDOM_STATE),
    "TF-IDF + RF":  RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                           class_weight="balanced",
                                           random_state=RANDOM_STATE),
}

rows = []
per_class_frames = {}
best_name, best_mf1, best_pred = None, -1.0, None

for name, clf in models.items():
    clf.fit(Xtr, y_train)
    pred = clf.predict(Xte)

    acc = accuracy_score(y_test, pred)
    mf1 = f1_score(y_test, pred, average="macro", labels=CLASSES, zero_division=0)
    wf1 = f1_score(y_test, pred, average="weighted", labels=CLASSES, zero_division=0)

    rows.append({"Yontem": name, "Test_Acc": round(acc, 4),
                 "Test_MacroF1": round(mf1, 4), "Test_wF1": round(wf1, 4)})
    print(f"  {name:14s} Acc={acc:.4f}  MacroF1={mf1:.4f}  wF1={wf1:.4f}")

    # per-class F1
    rep = classification_report(y_test, pred, labels=CLASSES,
                                output_dict=True, zero_division=0)
    per_class_frames[name] = {c: round(rep[c]["f1-score"], 3) for c in CLASSES}

    if mf1 > best_mf1:
        best_name, best_mf1, best_pred = name, mf1, pred

# ---------------------------------------------------------------------
# 4) Sonuc tablolari yaz
# ---------------------------------------------------------------------
res = pd.DataFrame(rows)

# LLM referans satirlari (tezden — SUNUM_OZET_V4.md Bolum 5)
llm_ref = pd.DataFrame([
    {"Yontem": "LLM zero-shot", "Test_Acc": 0.5152, "Test_MacroF1": 0.4681, "Test_wF1": 0.5107},
    {"Yontem": "LLM QLoRA-S3",  "Test_Acc": 0.6345, "Test_MacroF1": 0.6220, "Test_wF1": 0.6220},
])
res_full = pd.concat([res, llm_ref], ignore_index=True)
res_full.to_csv(os.path.join(OUT_DIR, "classical_ml_v4_results.csv"), index=False)

# per-class F1 (tum modeller + en iyi vurgulu)
pc = pd.DataFrame(per_class_frames)
pc.index.name = "Sinif"
pc.to_csv(os.path.join(OUT_DIR, "classical_ml_v4_per_class_f1.csv"))

# en iyi modelin confusion matrix'i
cm = confusion_matrix(y_test, best_pred, labels=CLASSES)
pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(
    os.path.join(OUT_DIR, "classical_ml_v4_confusion.csv"))

# tahminler
pd.DataFrame({"gercek": y_test, "tahmin": best_pred}).to_csv(
    os.path.join(OUT_DIR, "classical_ml_v4_predictions.csv"), index=False)

# ---------------------------------------------------------------------
# 5) Makaleye yapistirilacak ozet
# ---------------------------------------------------------------------
print("\n" + "=" * 60)
print("MAKALE TABLOSU (Test seti, n=1650)")
print("=" * 60)
print(res_full.to_string(index=False))
print(f"\nEn iyi klasik ML: {best_name} (MacroF1={best_mf1:.4f})")
print(f"\nCiktilar -> {OUT_DIR}\\")
print("  - classical_ml_v4_results.csv      (ana tablo + LLM referans)")
print("  - classical_ml_v4_per_class_f1.csv (11 sinif F1)")
print("  - classical_ml_v4_confusion.csv    (11x11)")
print("  - classical_ml_v4_predictions.csv  (gercek vs tahmin)")
