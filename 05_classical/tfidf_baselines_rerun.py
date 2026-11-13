# -*- coding: utf-8 -*-
"""Classical baselines: per-record test predictions for all three TF-IDF models.

tfidf_baselines.py kept per-record predictions only for its best model (the SVM).
This script re-runs it with the identical vectorizer, models and seed and keeps the
predictions of logistic regression and the random forest as well. It checks the re-run
against the stored aggregate scores and the stored SVM predictions, and writes the SVM's
largest feature weights per class (aggregate; only n-grams found in >= 20 training notes)
for the leakage sanity check. No note text or record ID is written.

Outputs (output/classical_ml_v4/):
  classical_ml_v4_predictions_all_models.csv  gold + lr/svm/rf, order of qlora_v4_test.json
  classical_ml_v4_svm_top_features.csv        class, rank, feature, weight, train_doc_freq
  classical_ml_v4_rerun_meta.json             versions and reproduction checks
Run: from the workspace root, python /path/to/repo/05_classical/tfidf_baselines_rerun.py
     (paths above are relative to workdir/)
"""
import json, os, platform
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC

RS = 42
BASE = "./workdir"  # workspace/workdir: run from the workspace root (see README)
QJ = os.path.join(BASE, "output", "qlora_json")
OUT = os.path.join(BASE, "output", "classical_ml_v4")
C = ["SEPSIS", "STROKE", "ATRIAL_FIBRILLATION", "AKI", "PNEUMONIA", "UTI", "HEART_FAILURE",
     "GI_BLEEDING", "DIABETES_COMPLICATION", "COPD_EXACERBATION", "OTHER"]
MIN_DF_SHOWN, TOP_K = 20, 8

load = lambda n: json.load(open(os.path.join(QJ, n), encoding="utf-8"))
train, test = load("qlora_v4_train_S3.json"), load("qlora_v4_test.json")
Xtr_txt, ytr = [r["input"] for r in train], [r["output"].strip() for r in train]
Xte_txt, yte = [r["input"] for r in test], [r["output"].strip() for r in test]

# identical to tfidf_baselines.py
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True,
                      strip_accents="unicode", lowercase=True)
Xtr, Xte = vec.fit_transform(Xtr_txt), vec.transform(Xte_txt)
models = {
    "lr": LogisticRegression(max_iter=2000, C=10.0, class_weight="balanced", random_state=RS),
    "svm": LinearSVC(C=1.0, class_weight="balanced", random_state=RS),
    "rf": RandomForestClassifier(n_estimators=300, n_jobs=-1, class_weight="balanced", random_state=RS),
}
pred, fitted = {}, {}
for k, m in models.items():
    fitted[k] = m.fit(Xtr, ytr)
    pred[k] = m.predict(Xte)

# reproduction checks against the original run
stored = pd.read_csv(os.path.join(OUT, "classical_ml_v4_results.csv")).set_index("Yontem")
svm_old = pd.read_csv(os.path.join(OUT, "classical_ml_v4_predictions.csv"))
assert (svm_old["gercek"].values == np.array(yte)).all(), "test order differs from the stored SVM file"
checks = {"svm_predictions_identical": int((svm_old["tahmin"].values == pred["svm"]).sum())}
for k, name in [("lr", "TF-IDF + LR"), ("svm", "TF-IDF + SVM"), ("rf", "TF-IDF + RF")]:
    acc = accuracy_score(yte, pred[k]); mf = f1_score(yte, pred[k], labels=C, average="macro", zero_division=0)
    checks[k] = dict(accuracy=round(acc, 4), macro_f1=round(mf, 4),
                     stored_accuracy=float(stored.loc[name, "Test_Acc"]), stored_macro_f1=float(stored.loc[name, "Test_MacroF1"]))
    checks[k]["reproduces_stored"] = bool(round(acc, 4) == checks[k]["stored_accuracy"] and
                                          round(mf, 4) == checks[k]["stored_macro_f1"])

pd.DataFrame({"gercek": yte, "lr": pred["lr"], "svm": pred["svm"], "rf": pred["rf"]}).to_csv(
    os.path.join(OUT, "classical_ml_v4_predictions_all_models.csv"), index=False)

# SVM feature weights (one-vs-rest), aggregate only
svm, vocab = fitted["svm"], np.array(vec.get_feature_names_out())
df_train = np.asarray((Xtr > 0).sum(axis=0)).ravel()
rows = []
for ci, cls in enumerate(svm.classes_):
    w = svm.coef_[ci]
    idx = [i for i in np.argsort(-w) if df_train[i] >= MIN_DF_SHOWN][:TOP_K]
    rows += [dict(cls=cls, rank=r + 1, feature=vocab[i], weight=round(float(w[i]), 3),
                  train_doc_freq=int(df_train[i])) for r, i in enumerate(idx)]
pd.DataFrame(rows).to_csv(os.path.join(OUT, "classical_ml_v4_svm_top_features.csv"), index=False)

meta = dict(python=platform.python_version(), sklearn=sklearn.__version__, numpy=np.__version__,
            n_features=int(Xtr.shape[1]), random_state=RS, checks=checks)
json.dump(meta, open(os.path.join(OUT, "classical_ml_v4_rerun_meta.json"), "w"), indent=2)
print(json.dumps(meta, indent=2))
