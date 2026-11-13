# -*- coding: utf-8 -*-
"""Additional checks requested in review: class prevalence, residual-term leakage, SVM weights.

1) Class distribution of the eligible pool (output/discharge_parsed.parquet, label column only)
   and the test confusion matrices (T06) reweighted from the balanced test set to that
   distribution (prior shift; class-conditional predictions assumed unchanged).
2) Two leakage checks that do not use the rules of the leakage screen:
   a) target abbreviations or synonyms left unmasked (e.g. GIB, IPH, urosepsis): share of
      test inputs with a match in the chief complaint (CC) or symptom history (HPI);
      SVM retrained with every match deleted from every section;
   b) mentions of laboratory tests in CC or HPI; SVM retrained with them deleted there.
   The SVM is the paper's TF-IDF linear SVM (identical settings; its predictions are checked).
3) Largest SVM weights per class (n-grams in >= 20 training notes, no placeholder tokens).
4) Cohort size, primary codes that the prefix mapping sends to "other" although they describe a
   target condition, an SVM learning curve, the SVM on truncated inputs, and the SVM's run time.
Outputs (output/results_tables/, aggregate only: no note text, no record IDs):
  T15_pool_class_distribution.csv, T16_prevalence_reweighted.csv,
  T17_leakage_term_checks.csv, T18_svm_top_features.csv, T19_label_gaps.csv,
  T20_svm_learning_curve.csv, T21_svm_truncation.csv, T22_cohort_and_time.csv,
  T24_term_stratified_recall.csv, T25_unparseable_outputs.csv
5) For every configuration: recall on the target-class test records whose CC or HPI contains an
   unmasked term of their own class, against records of the same classes without one, and AKI
   recall with and without a laboratory mention in CC or HPI (the language models were not
   re-run on cleaned inputs, so this is observational). And what the
   unparseable outputs are: counts only (prefix, length in tokens, whether a class is named).
   The token counts need the `tokenizers` package and the Qwen tokenizer file (TOKENIZER below).
Run: from the workspace root, python /path/to/repo/07_analysis/build_reviewer_checks.py
     (after build_results_tables.py; paths below are relative to workdir/)
"""
import json, os, re
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

BASE = "./workdir"  # workspace/workdir: run from the workspace root (see README)
OUT = os.path.join(BASE, "output")
DST = os.path.join(OUT, "results_tables")
C = ["SEPSIS", "STROKE", "ATRIAL_FIBRILLATION", "AKI", "PNEUMONIA", "UTI", "HEART_FAILURE",
     "GI_BLEEDING", "DIABETES_COMPLICATION", "COPD_EXACERBATION", "OTHER"]
TARGETS = C[:-1]
N_PER_CLASS = 150

# ---- 1) pool distribution and prevalence reweighting ---------------------------------
pool = pd.read_parquet(os.path.join(OUT, "discharge_parsed.parquet"), columns=["label"])["label"]
cnt = pool.value_counts().reindex(C).fillna(0).astype(int)
T15 = pd.DataFrame({"class": C, "n": cnt.values, "pct": (100 * cnt / cnt.sum()).round(2).values})
T15.to_csv(os.path.join(DST, "T15_pool_class_distribution.csv"), index=False)
pi = (cnt / cnt.sum()).values

T06 = pd.read_csv(os.path.join(DST, "T06_confusion_matrices_long.csv"))
rows = []
for k, g in T06.groupby("config", sort=False):
    M = g.pivot(index="true_class", columns="predicted_class", values="count").reindex(index=C, columns=C).fillna(0).values
    assert (M.sum(1) <= N_PER_CLASS).all()
    R = M / N_PER_CLASS                       # P(pred = j | true = i); unparseable outputs are in no column

    def summary(w):
        J = w[:, None] * R
        prec = np.divide(np.diag(J), J.sum(0), out=np.zeros(len(C)), where=J.sum(0) > 0)
        rec = np.diag(R)
        f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(len(C)), where=(prec + rec) > 0)
        return np.diag(J).sum(), f1.mean(), prec[:-1].mean()
    acc_b, mf1_b, ppv_b = summary(np.full(len(C), 1 / len(C)))
    acc_p, mf1_p, ppv_p = summary(pi)
    rows.append(dict(config=k, accuracy_balanced=round(acc_b, 6), accuracy_pool=round(acc_p, 6),
                     macro_f1_balanced=round(mf1_b, 6), macro_f1_pool=round(mf1_p, 6),
                     target_precision_balanced=round(ppv_b, 6), target_precision_pool=round(ppv_p, 6)))
pd.DataFrame(rows).to_csv(os.path.join(DST, "T16_prevalence_reweighted.csv"), index=False)

# ---- 2) leakage checks independent of the screen's rules ------------------------------
J_ = lambda n: json.load(open(os.path.join(OUT, "qlora_json", n), encoding="utf-8"))
tr, te = J_("qlora_v4_train_S3.json"), J_("qlora_v4_test.json")
ytr = [r["output"].strip() for r in tr]
yte = np.array([r["output"].strip() for r in te])
TERMS = {  # target abbreviations and synonyms (the placeholder replaced the main names)
    "SEPSIS": r"urosepsis|\bsepsis\b|\bseptic\b",
    "STROKE": r"\b(?:stroke|cva|iph|ich|sah)\b|intraparenchymal hem|subarachnoid hem",
    "ATRIAL_FIBRILLATION": r"\b(?:afib|a-fib|a\. ?fib|atrial fibrillation|af)\b",
    "AKI": r"\b(?:aki|arf)\b|acute kidney injury|renal failure|kidney failure",
    "PNEUMONIA": r"\b(?:pna|hcap)\b|pneumonia",
    "UTI": r"\buti\b|urinary tract infection",
    "HEART_FAILURE": r"\b(?:chf|hfref|hfpef)\b|heart failure",
    "GI_BLEEDING": r"\b(?:gib|ugib|lgib)\b|gi bleed|gastrointestinal bleed",
    "DIABETES_COMPLICATION": r"\b(?:dka|hhs|hhnk)\b|ketoacidosis|hyperosmolar",
    "COPD_EXACERBATION": r"\baecopd\b|copd exacerbation|copd flare",
}
ALL_TERMS = re.compile("|".join(f"(?:{p})" for p in TERMS.values()), re.I)
LAB = re.compile(r"\b(?:creatinine|cr|labs?|lactate|lactic|wbc|leukocytosis|troponins?|trop|bnp|inr|hgb|"
                 r"hemoglobin|hct|platelets?|cultures?|bicarb|anion gap|ua|urinalysis|"
                 r"hyperkalemi[ac]|hypokalemi[ac]|hyponatremi[ac]|hypernatremi[ac])\b", re.I)
HEAD = re.compile(r"(?m)^\s*(Chief Complaint|Symptom History|Past Medical History|"
                  r"Medications on Admission|Allergies|Vital Signs):")
CUR = ("Chief Complaint", "Symptom History")

def segments(t):
    pos = [(m.start(), m.group(1)) for m in HEAD.finditer(t)]
    first = pos[0][0] if pos else len(t)
    return t[:first], [(t[s:(pos[i + 1][0] if i + 1 < len(pos) else len(t))], h) for i, (s, h) in enumerate(pos)]

def current_text(t):
    return " ".join(s for s, h in segments(t)[1] if h in CUR)

def delete(t, rx, only_current):
    head, segs = segments(t)
    return head + "".join(rx.sub(" ", s) if (h in CUR or not only_current) else s for s, h in segs)

def svm_predict(Xa, Xb):
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True,
                        strip_accents="unicode", lowercase=True)
    m = LinearSVC(C=1.0, class_weight="balanced", random_state=42).fit(v.fit_transform(Xa), ytr)
    return m.predict(v.transform(Xb)), m, v

def paired(base, new):
    a, b = base == yte, new == yte
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    return round(100 * (b.mean() - a.mean()), 4), binomtest(n10, n10 + n01, 0.5).pvalue

base, svm, vec = svm_predict([r["input"] for r in tr], [r["input"] for r in te])
stored = pd.read_csv(os.path.join(OUT, "classical_ml_v4", "classical_ml_v4_predictions.csv"))["tahmin"].values
assert (base == stored).all(), "re-trained SVM differs from the paper's SVM"

cur_te = [current_text(r["input"]) for r in te]
own_share = {c: 100 * np.mean([bool(re.search(TERMS[c], cur_te[i], re.I)) for i in np.where(yte == c)[0]]) for c in TARGETS}
p_term, _, _ = svm_predict([delete(r["input"], ALL_TERMS, False) for r in tr], [delete(r["input"], ALL_TERMS, False) for r in te])
d_term, pv_term = paired(base, p_term)
lab_hit = np.array([bool(LAB.search(t)) for t in cur_te])
p_lab, _, _ = svm_predict([delete(r["input"], LAB, True) for r in tr], [delete(r["input"], LAB, True) for r in te])
d_lab, pv_lab = paired(base, p_lab)
aki = yte == "AKI"
T17 = pd.DataFrame([
    dict(metric="target term in CC or HPI, own class, max over classes, %", value=round(max(own_share.values()), 2)),
    dict(metric="class with the max share", value=max(own_share, key=own_share.get)),
    *[dict(metric=f"target term in CC or HPI, own class, {c}, %", value=round(v, 2)) for c, v in own_share.items()],
    dict(metric="SVM accuracy change, target terms deleted everywhere, points", value=d_term),
    dict(metric="p (exact McNemar, unadjusted), target terms", value=float(f"{pv_term:.2g}")),
    dict(metric="lab mention in CC or HPI, all test inputs, %", value=round(100 * lab_hit.mean(), 2)),
    dict(metric="lab mention in CC or HPI, AKI inputs, %", value=round(100 * lab_hit[aki].mean(), 2)),
    dict(metric="SVM accuracy change, lab mentions deleted from CC and HPI, points", value=d_lab),
    dict(metric="p (exact McNemar, unadjusted), lab mentions", value=float(f"{pv_lab:.2g}")),
    dict(metric="SVM recall for AKI, baseline", value=round((base == yte)[aki].mean(), 4)),
    dict(metric="SVM recall for AKI, lab mentions deleted", value=round((p_lab == yte)[aki].mean(), 4)),
])
T17.to_csv(os.path.join(DST, "T17_leakage_term_checks.csv"), index=False)

# ---- 3) largest SVM weights -------------------------------------------------------------
Xtr = vec.transform([r["input"] for r in tr])
dfreq = np.asarray((Xtr > 0).sum(axis=0)).ravel()
vocab = np.array(vec.get_feature_names_out())
rows = []
for ci, cls in enumerate(svm.classes_):
    order = [i for i in np.argsort(-svm.coef_[ci])
             if dfreq[i] >= 20 and "___" not in vocab[i] and "diag_masked" not in vocab[i]][:5]
    rows += [dict(cls=cls, rank=r + 1, feature=vocab[i], weight=round(float(svm.coef_[ci][i]), 2)) for r, i in enumerate(order)]
pd.DataFrame(rows).to_csv(os.path.join(DST, "T18_svm_top_features.csv"), index=False)
print(T15.to_string(index=False)); print(pd.DataFrame(rows).groupby("cls")["feature"].apply(", ".join).to_string())
print(pd.read_csv(os.path.join(DST, "T16_prevalence_reweighted.csv")).to_string(index=False))
print(T17.to_string(index=False))

# ---- 4) cohort, label gaps, learning curve, truncation, run time ------------------------
import time
labeled = pd.read_parquet(os.path.join(OUT, "discharge_etiketli_full.parquet"), columns=["label"])
codes = pd.read_parquet(os.path.join(OUT, "discharge_parsed.parquet"), columns=["label", "icd_code"])
GAPS = [  # (group, target class it belongs to clinically, regex on the primary ICD-10 code)
    ("hypertensive heart or kidney disease with heart failure (I11.0, I13.0, I13.2)", "HEART_FAILURE", r"^(I110|I130|I132)"),
    ("bleeding ulcer (K25-K28 with hemorrhage)", "GI_BLEEDING", r"^K2[5-8][0246]"),
    ("diverticular bleeding (K57 with bleeding)", "GI_BLEEDING", r"^K57[0-9][13]"),
    ("angiodysplasia with bleeding (K55.21, K31.811)", "GI_BLEEDING", r"^(K5521|K31811)"),
    ("bleeding varices (I85.01, I85.11)", "GI_BLEEDING", r"^(I8501|I8511)"),
    ("pyelonephritis or cystitis (N10, N30)", "UTI", r"^(N10|N30)"),
    ("aspiration pneumonitis (J69.0)", "PNEUMONIA", r"^J690"),
    ("influenza with pneumonia (J09-J11 .x0)", "PNEUMONIA", r"^J(09|10|11)[0-9]?0"),
]
rows = []
for grp, cls, rx in GAPS:
    m = codes["icd_code"].str.match(rx)
    assert (codes.loc[m, "label"] == "OTHER").all(), grp
    rows.append(dict(group=grp, target_class=cls, n_in_other=int(m.sum()),
                     n_in_target_class=int((codes["label"] == cls).sum())))
pd.DataFrame(rows).to_csv(os.path.join(DST, "T19_label_gaps.csv"), index=False)

def svm_fit_predict(Xa, ya, Xb):
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2 if len(Xa) > 200 else 1, max_features=20000,
                        sublinear_tf=True, strip_accents="unicode", lowercase=True)
    return LinearSVC(C=1.0, class_weight="balanced", random_state=42).fit(v.fit_transform(Xa), ya).predict(v.transform(Xb))

Xa_all, ya_all = [r["input"] for r in tr], np.array(ytr)
Xt_all = [r["input"] for r in te]
rng = np.random.default_rng(0)
lc = []
for k in [3, 10, 50, 100, 200, 400]:
    reps = 20 if k < 100 else 5 if k < 400 else 1
    accs = []
    for _ in range(reps):
        idx = np.concatenate([rng.choice(np.where(ya_all == c)[0], k, replace=False) for c in C])
        accs.append(float((svm_fit_predict([Xa_all[i] for i in idx], ya_all[idx], Xt_all) == yte).mean()))
    lc.append(dict(setting=f"{k} per class, random", n_labels=11 * k, repeats=reps,
                   accuracy_mean=round(np.mean(accs), 6), accuracy_sd=round(np.std(accs), 6)))
ux = json.load(open(os.path.join(OUT, "fewshot", "fewshot_v4_3shot_ultracompact_examples.json"), encoding="utf-8"))
fs_ids = {str(e["note_id"]) for c in ux.values() for e in (c if isinstance(c, list) else [c])}
fs_idx = [i for i, r in enumerate(tr) if str(r["note_id"]) in fs_ids]
assert len(fs_idx) == 33
p_fs = svm_fit_predict([Xa_all[i] for i in fs_idx], ya_all[fs_idx], Xt_all)
lc.append(dict(setting="the 33 few-shot examples (full text)", n_labels=33, repeats=1,
               accuracy_mean=round(float((p_fs == yte).mean()), 6), accuracy_sd=0.0))
pd.DataFrame(lc).to_csv(os.path.join(DST, "T20_svm_learning_curve.csv"), index=False)

first = lambda t, n: " ".join(t.split()[:n])
words = np.array([len(t.split()) for t in Xt_all])
tc = []
for n in [60, 120, 180, 240, 360]:
    tc.append(dict(first_words=n,
                   acc_full_train_truncated_test=round(float((svm_fit_predict(Xa_all, ya_all, [first(t, n) for t in Xt_all]) == yte).mean()), 6),
                   acc_truncated_train_and_test=round(float((svm_fit_predict([first(t, n) for t in Xa_all], ya_all, [first(t, n) for t in Xt_all]) == yte).mean()), 6)))
pd.DataFrame(tc).to_csv(os.path.join(DST, "T21_svm_truncation.csv"), index=False)

t0 = time.perf_counter()
p_full = svm_fit_predict(Xa_all, ya_all, Xt_all)
svm_sec = time.perf_counter() - t0
assert (p_full == stored).all()
pd.DataFrame([
    dict(metric="admissions with a discharge summary and a primary ICD-10 code", value=int(len(labeled))),
    dict(metric="of these, with parsable sections (eligible pool)", value=int(len(codes))),
    dict(metric="test input length, words, mean", value=round(float(words.mean()), 1)),
    dict(metric="SVM: TF-IDF fit, training and labeling of the test set, seconds (2-core CPU)", value=round(svm_sec, 2)),
    dict(metric="CPU cores", value=os.cpu_count()),
    dict(metric="COPD class in the eligible pool, n", value=int((codes["label"] == "COPD_EXACERBATION").sum())),
    dict(metric="of these, J44.9 (no exacerbation coded), n",
         value=int(((codes["label"] == "COPD_EXACERBATION") & (codes["icd_code"] == "J449")).sum())),
]).to_csv(os.path.join(DST, "T22_cohort_and_time.csv"), index=False)
print(pd.DataFrame(rows).to_string(index=False)); print(pd.DataFrame(lc).to_string(index=False))
print(pd.DataFrame(tc).to_string(index=False)); print(pd.read_csv(os.path.join(DST, "T22_cohort_and_time.csv")).to_string(index=False))


# ---- 5) records with an unmasked own-class term; unparseable outputs ---------------------
FILES = {  # per-record predictions: the same files as TEST in build_results_tables.py
    "qwen_zero": "zeroshot/zeroshot_v4_test.csv",
    "qwen_fs_comp": "b1_compressed_a100/b1_compressed_test.csv",
    "qwen_fs_full": "b2_b5_verify/b2_full3shot_test.csv",
    "qwen_qlora": "qlora_eval_fullctx/test_predictions.csv",
    "gemma_zero_bf16": "colab_downloads/gemma_bf16_zeroshot_test.csv",
    "gemma_zero_4bit": "colab_downloads/gemma_4bit_zeroshot_test.csv",
    "gemma_fs_full": "gemma_full3shot_test.csv",
    "gemma_qlora_fair": "b2_b5_verify/b5_gemma_qlora_test.csv",
}
TOKENIZER = os.path.join(OUT, "qlora_adapters", "v4_s3_full_2epoch_lr1e4", "adapter", "tokenizer.json")
te_ids = [str(r["note_id"]) for r in te]

def preds(path):
    d = pd.read_csv(os.path.join(OUT, path))
    g = next(c for c in ("gold", "label", "output", "gercek") if c in d)
    p = next(c for c in ("pred", "parsed_pred", "tahmin") if c in d)
    d["note_id"] = d["note_id"].astype(str)
    d = d.set_index("note_id").loc[te_ids]
    assert (d[g].values == yte).all(), f"label order mismatch: {path}"
    return d[p].fillna("INVALID").astype(str).values, d

PR = {k: preds(f)[0] for k, f in FILES.items()}
_q = pd.read_csv(os.path.join(OUT, "qlora_eval_fullctx/test_predictions.csv"))
_g = pd.read_csv(os.path.join(OUT, "gemma_qlora_test_preds.csv"))
assert (_q["gold"].values == _g["output"].values).all()
_m = dict(zip(_q["note_id"].astype(str), _g["pred"].astype(str)))
PR["gemma_qlora_opt"] = np.array([_m[i] for i in te_ids])
PR["svm"] = base
T01 = pd.read_csv(os.path.join(DST, "T01_main_results_test.csv")).set_index("config")
has = np.array([c in TERMS and bool(re.search(TERMS[c], cur_te[i], re.I)) for i, c in enumerate(yte)])
rows = []
for k, p in PR.items():
    ok = p == yte
    assert abs(ok.mean() - T01.at[k, "accuracy"]) < 5e-5, f"{k}: predictions differ from T01"
    exp = sum(has[yte == c].sum() * ok[(yte == c) & ~has].mean() for c in TARGETS if has[yte == c].any())
    rows.append(dict(config=k, n_target_records_with_term=int(has.sum()), correct_with_term=int(ok[has].sum()),
                     recall_with_term=round(float(ok[has].mean()), 6),
                     recall_without_term_same_class_mix=round(float(exp / has.sum()), 6),
                     accuracy_points_attributable=round(float(100 * (ok[has].sum() - exp) / len(yte)), 4),
                     aki_records_with_lab_mention=int((aki & lab_hit).sum()),
                     aki_recall_with_lab_mention=round(float(ok[aki & lab_hit].mean()), 6),
                     aki_recall_without_lab_mention=round(float(ok[aki & ~lab_hit].mean()), 6)))
T24 = pd.DataFrame(rows)
T24.to_csv(os.path.join(DST, "T24_term_stratified_recall.csv"), index=False)

from tokenizers import Tokenizer
tk = Tokenizer.from_file(TOKENIZER); tk.no_padding(); tk.no_truncation()
NAMES = re.compile(r"(?<![A-Z_])(?:" + "|".join(C) + r")(?![A-Z_])")
rows = []
for k in ("qwen_zero", "qwen_fs_comp", "qwen_fs_full"):
    p, d = preds(FILES[k])
    inv = ~np.isin(p, C); ok = p == yte
    raw = d["raw_output"].fillna("").astype(str)[inv]
    ntok = raw.apply(lambda t: len(tk.encode(t, add_special_tokens=False).ids))
    rows.append(dict(config=k, n_unparseable=int(inv.sum()),
                     begin_with_based_on=int(raw.str.startswith("Based on").sum()),
                     name_a_class=int(raw.str.upper().apply(lambda t: bool(NAMES.search(t))).sum()),
                     at_least_19_tokens=int((ntok >= 19).sum()), tokens_min=int(ntok.min()), tokens_max=int(ntok.max()),
                     accuracy=round(float(ok.mean()), 6),
                     accuracy_on_parsed_outputs=round(float(ok[~inv].mean()), 6),
                     accuracy_if_all_unparseable_correct=round(float((ok.sum() + inv.sum()) / len(yte)), 6)))
T25 = pd.DataFrame(rows)
T25.to_csv(os.path.join(DST, "T25_unparseable_outputs.csv"), index=False)
print(T24.to_string(index=False)); print(T25.to_string(index=False))
