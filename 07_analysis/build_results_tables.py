# -*- coding: utf-8 -*-
"""Builds every results table of the paper from per-record predictions and run logs.

Inputs : workdir/output/  (split JSON files, per-record predictions, training_curve.json),
         workdir/memory_table.csv and memory_components.csv (06_memory/make_memory_table.py),
         workdir/results_master.csv (07_analysis/results_master.py, consistency check only)
Output : workdir/output/results_tables/T01..T14  (aggregate tables only: no note text, no
         record IDs); the released copies are in results/
Run    : from the workspace root, python /path/to/repo/07_analysis/build_results_tables.py
"""
import json, os, platform, re
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import binomtest, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.svm import LinearSVC

BASE = "./workdir"  # workspace/workdir: run from the workspace root (see README)
OUT = os.path.join(BASE, "output")
DST = os.path.join(OUT, "results_tables")
os.makedirs(DST, exist_ok=True)
C = ["SEPSIS", "STROKE", "ATRIAL_FIBRILLATION", "AKI", "PNEUMONIA", "UTI", "HEART_FAILURE",
     "GI_BLEEDING", "DIABETES_COMPLICATION", "COPD_EXACERBATION", "OTHER"]
SEED, B = 0, 2000
J = lambda n: json.load(open(os.path.join(OUT, "qlora_json", n), encoding="utf-8"))
test, val = J("qlora_v4_test.json"), J("qlora_v4_val.json")
ids = {"test": [str(r["note_id"]) for r in test], "val": [str(r["note_id"]) for r in val]}
gold = {"test": np.array([r["output"] for r in test]), "val": np.array([r["output"] for r in val])}

def load(path, split, by_order=False):
    d = pd.read_csv(os.path.join(OUT, path))
    g = next(c for c in ("gold", "label", "output", "gercek") if c in d)
    p = next(c for c in ("pred", "parsed_pred", "tahmin") if c in d)
    if "note_id" in d and not by_order:
        d["note_id"] = d["note_id"].astype(str)
        d = d.set_index("note_id").loc[ids[split]]
    assert (d[g].values == gold[split]).all(), f"label order mismatch: {path}"
    return d[p].fillna("INVALID").astype(str).values

def wilson(k, n, z=1.959964):
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h

def mf1(y, p): return f1_score(y, p, labels=C, average="macro", zero_division=0)

rng = np.random.default_rng(SEED)
BOOT = {s: rng.integers(0, len(gold[s]), (B, len(gold[s]))) for s in gold}

def metrics(p, split):
    y = gold[split]; k = int((p == y).sum()); n = len(y)
    lo, hi = wilson(k, n)
    bs = [mf1(y[i], p[i]) for i in BOOT[split]]
    return dict(n=n, n_invalid=int((~np.isin(p, C)).sum()), accuracy=round(k / n, 4),
                acc_ci_low=round(lo, 4), acc_ci_high=round(hi, 4), macro_f1=round(mf1(y, p), 4),
                mf1_ci_low=round(np.percentile(bs, 2.5), 4), mf1_ci_high=round(np.percentile(bs, 97.5), 4))

# ---- configurations ---------------------------------------------------------------
TEST = [  # key, model, method, precision, hardware, fits_8gb, file
 ("qwen_zero", "Qwen3.5-4B", "zero-shot", "4-bit NF4", "RTX 4060 8 GB", "yes", "zeroshot/zeroshot_v4_test.csv"),
 ("qwen_fs_comp", "Qwen3.5-4B", "few-shot, compressed (33 ex.)", "4-bit NF4", "A100 80 GB", "yes (2,024-token prompt)", "b1_compressed_a100/b1_compressed_test.csv"),
 ("qwen_fs_full", "Qwen3.5-4B", "few-shot, full (33 ex.)", "4-bit NF4", "A100 80 GB", "no (~23.8k-token prompt)", "b2_b5_verify/b2_full3shot_test.csv"),
 ("qwen_qlora", "Qwen3.5-4B", "QLoRA fair (r=8, q/v, 2 ep.)", "4-bit NF4", "RTX 4060 8 GB", "yes", "qlora_eval_fullctx/test_predictions.csv"),
 ("gemma_zero_bf16", "Gemma-4-E4B", "zero-shot", "bf16", "A100 80 GB", "no", "colab_downloads/gemma_bf16_zeroshot_test.csv"),
 ("gemma_zero_4bit", "Gemma-4-E4B", "zero-shot", "4-bit NF4", "A100 80 GB", "no", "colab_downloads/gemma_4bit_zeroshot_test.csv"),
 ("gemma_fs_full", "Gemma-4-E4B", "few-shot, full (33 ex.)", "bf16", "A100 80 GB", "no", "gemma_full3shot_test.csv"),
 ("gemma_qlora_fair", "Gemma-4-E4B", "QLoRA fair (r=8, q/v, 2 ep.)", "4-bit NF4", "A100 80 GB", "no", "b2_b5_verify/b5_gemma_qlora_test.csv"),
 ("gemma_qlora_opt", "Gemma-4-E4B", "QLoRA optimized (r=16, all linear, 3 ep.)", "4-bit NF4", "A100 80 GB", "no", None),
 ("svm", "TF-IDF + linear SVM", "supervised (classical)", "-", "CPU", "yes (no GPU)", "classical_ml_v4/classical_ml_v4_predictions.csv"),
]
P = {}
for k, *_, f in TEST:
    if k == "gemma_qlora_opt":
        q = pd.read_csv(os.path.join(OUT, "qlora_eval_fullctx/test_predictions.csv"))
        go = pd.read_csv(os.path.join(OUT, "gemma_qlora_test_preds.csv"))
        assert (q["gold"].values == go["output"].values).all()
        m = dict(zip(q["note_id"].astype(str), go["pred"].astype(str)))
        P[k] = np.array([m[i] for i in ids["test"]])
    else:
        P[k] = load(f, "test", by_order=(k == "svm"))
P["qwen_qlora_trunc"] = load("qlora_eval/v4_s3_full_2epoch_lr1e4/test_predictions.csv", "test")

rows = []
for k, model, method, prec, hw, fits, _ in TEST:
    rows.append(dict(config=k, model=model, method=method, precision=prec, hardware=hw, fits_8gb=fits, **metrics(P[k], "test")))
cl = pd.read_csv(os.path.join(OUT, "classical_ml_v4/classical_ml_v4_results.csv"))
for name, key, cfg in [("TF-IDF + logistic regression", "TF-IDF + LR", "lr"), ("TF-IDF + random forest", "TF-IDF + RF", "rf")]:
    r = cl[cl["Yontem"] == key].iloc[0]; k = int(round(r["Test_Acc"] * 1650)); lo, hi = wilson(k, 1650)
    rows.append(dict(config=cfg, model=name, method="supervised (classical)",
                     precision="-", hardware="CPU", fits_8gb="yes (no GPU)", n=1650, n_invalid=0,
                     accuracy=round(r["Test_Acc"], 4), acc_ci_low=round(lo, 4), acc_ci_high=round(hi, 4),
                     macro_f1=round(r["Test_MacroF1"], 4), mf1_ci_low=np.nan, mf1_ci_high=np.nan))
T01 = pd.DataFrame(rows)
T01.to_csv(os.path.join(DST, "T01_main_results_test.csv"), index=False)

# ---- validation --------------------------------------------------------------------
VAL = [  # key, model, method, precision, hardware, file
 ("qwen_zero_val", "Qwen3.5-4B", "zero-shot", "4-bit NF4", "RTX 4060 8 GB", "zeroshot/zeroshot_v4_val.csv"),
 ("qwen_fs_comp_val_4060", "Qwen3.5-4B", "few-shot, compressed (33 ex.)", "4-bit NF4", "RTX 4060 8 GB", "fewshot/fewshot_v4_3shot_ultracompact_val.csv"),
 ("qwen_fs_comp_val_a100", "Qwen3.5-4B", "few-shot, compressed (33 ex.)", "4-bit NF4", "A100 80 GB", "b1_compressed_a100/b1_compressed_val.csv"),
 ("qwen_fs_full_val", "Qwen3.5-4B", "few-shot, full (33 ex.)", "4-bit NF4", "A100 80 GB", "colab_downloads/full_3shot_val.csv"),
 ("gemma_zero_val_bf16", "Gemma-4-E4B", "zero-shot", "bf16", "A100 80 GB", "colab_downloads/gemma_zeroshot_val.csv"),
 ("gemma_zero_val_4bit", "Gemma-4-E4B", "zero-shot", "4-bit NF4", "A100 80 GB", "colab_downloads/gemma_4bit_zeroshot_val.csv")]
vr = []
for k, model, method, prec, hw, f in VAL:
    P[k] = load(f, "val"); vr.append(dict(config=k, model=model, method=method, precision=prec, hardware=hw, **metrics(P[k], "val")))
pd.DataFrame(vr).to_csv(os.path.join(DST, "T02_validation_results.csv"), index=False)

# ---- paired tests ------------------------------------------------------------------
def mcn(a, b, split="test"):
    y = gold[split]; ca, cb = P[a] == y, P[b] == y
    n10, n01 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    return dict(acc_a=round(ca.mean(), 4), acc_b=round(cb.mean(), 4), diff_points=round(100 * (ca.mean() - cb.mean()), 2),
                a_right_b_wrong=n10, a_wrong_b_right=n01, p_exact=binomtest(n10, n10 + n01, 0.5).pvalue)
PAIRS = [("QLoRA gain, Qwen", "qwen_qlora", "qwen_zero", "test"),
         ("QLoRA gain, Gemma (4-bit to 4-bit)", "gemma_qlora_fair", "gemma_zero_4bit", "test"),
         ("Few-shot benefit, Qwen (full)", "qwen_fs_full", "qwen_zero", "test"),
         ("Compression cost, Qwen", "qwen_fs_comp", "qwen_fs_full", "test"),
         ("Compressed few-shot vs zero-shot, Qwen", "qwen_fs_comp", "qwen_zero", "test"),
         ("Few-shot benefit, Gemma (bf16)", "gemma_fs_full", "gemma_zero_bf16", "test"),
         ("Quantization cost, Gemma", "gemma_zero_4bit", "gemma_zero_bf16", "test"),
         ("Model gap, fair QLoRA", "qwen_qlora", "gemma_qlora_fair", "test"),
         ("Gemma optimized vs Qwen QLoRA", "gemma_qlora_opt", "qwen_qlora", "test"),
         ("Truncated evaluation input, Qwen QLoRA", "qwen_qlora_trunc", "qwen_qlora", "test"),
         ("Device control (compressed few-shot, val)", "qwen_fs_comp_val_4060", "qwen_fs_comp_val_a100", "val")]
PAIRS += [(f"{k} vs TF-IDF SVM", k, "svm", "test") for k, *_ in TEST if k != "svm"]
T03 = pd.DataFrame([dict(comparison=n, a=a, b=b, split=s, **mcn(a, b, s)) for n, a, b, s in PAIRS])
order = T03["p_exact"].sort_values().index; m = len(T03); prev = 0; holm = {}
for i, ix in enumerate(order):
    prev = max(prev, min(1.0, (m - i) * T03.at[ix, "p_exact"])); holm[ix] = prev
T03["p_holm"] = pd.Series(holm); T03["significant_holm_0.05"] = T03["p_holm"] < 0.05
for c in ("p_exact", "p_holm"): T03[c] = T03[c].map(lambda v: float(f"{v:.4g}"))
T03.to_csv(os.path.join(DST, "T03_paired_tests.csv"), index=False)

# ---- per-class F1, confusion, overlap ----------------------------------------------
keys = [k for k, *_ in TEST]
T04 = pd.DataFrame({k: f1_score(gold["test"], P[k], labels=C, average=None, zero_division=0).round(3) for k in keys}, index=C)
T04.index.name = "class"; T04.to_csv(os.path.join(DST, "T04_per_class_f1_test.csv"))
pc = pd.read_csv(os.path.join(OUT, "classical_ml_v4/classical_ml_v4_per_class_f1.csv"))
T05 = pc.rename(columns={"Sinif": "class"})
assert np.allclose(T05.set_index("class").loc[C, "TF-IDF + SVM"].values, T04.loc[C, "svm"].values, atol=1e-3), \
    "SVM per-class F1: stored aggregate differs from the per-record recomputation"
T05.to_csv(os.path.join(DST, "T05_classical_per_class_f1.csv"), index=False)
long = []
for k in keys:  # every configuration with per-record predictions
    cm = confusion_matrix(gold["test"], P[k], labels=C)
    for i, a in enumerate(C):
        for j, b in enumerate(C): long.append(dict(config=k, true_class=a, predicted_class=b, count=int(cm[i, j])))
T06 = pd.DataFrame(long); T06.to_csv(os.path.join(DST, "T06_confusion_matrices_long.csv"), index=False)
KEY4 = ["qwen_zero", "qwen_qlora", "gemma_qlora_opt", "svm"]  # configurations discussed in the error analysis
T07 = (T06[(T06.true_class != T06.predicted_class) & T06.config.isin(KEY4)].sort_values(["config", "count"], ascending=[True, False])
       .groupby("config").head(8))
T07.to_csv(os.path.join(DST, "T07_top_confusions.csv"), index=False)
W = np.c_[P["svm"] != gold["test"], P["qwen_qlora"] != gold["test"], P["gemma_qlora_opt"] != gold["test"]]
rho, prho = spearmanr(T04["svm"], T04["qwen_qlora"])
pd.DataFrame([dict(metric="all three wrong (SVM, Qwen QLoRA, Gemma QLoRA optimized), %", value=round(100 * W.all(1).mean(), 1)),
              dict(metric="at least one correct (oracle upper bound), %", value=round(100 * (~W).any(1).mean(), 1)),
              dict(metric="Spearman rho, per-class F1 of SVM vs Qwen QLoRA", value=round(rho, 3)),
              dict(metric="Spearman p", value=float(f"{prho:.2g}"))]).to_csv(os.path.join(DST, "T09_error_overlap.csv"), index=False)

# ---- [DIAG_MASKED] control (same environment for both runs) ------------------------
tr = J("qlora_v4_train_S3.json"); Xtr, ytr = [r["input"] for r in tr], [r["output"] for r in tr]; Xte = [r["input"] for r in test]
def svm(Xa, Xb):
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True, lowercase=True)
    return LinearSVC(C=1.0, class_weight="balanced", random_state=42).fit(v.fit_transform(Xa), ytr).predict(v.transform(Xb))
tok = "[DIAG_MASKED]"; ct, ce = np.array([t.count(tok) for t in Xtr]), np.array([t.count(tok) for t in Xte])
oh = lambda c: (np.minimum(c, 10)[:, None] == np.arange(11)[None, :]).astype(float)
only = LogisticRegression(max_iter=1000).fit(oh(ct), ytr).predict(oh(ce))
P["svm_env_with_mask"], P["svm_env_without_mask"] = svm(Xtr, Xte), svm([t.replace(tok, " ") for t in Xtr], [t.replace(tok, " ") for t in Xte])
t = mcn("svm_env_without_mask", "svm_env_with_mask")
pd.DataFrame([dict(metric="test notes containing the placeholder, %", value=round(100 * (ce > 0).mean(), 1)),
              dict(metric="placeholders per test note, mean", value=round(ce.mean(), 2)),
              dict(metric="accuracy from placeholder count alone", value=round((only == gold["test"]).mean(), 4)),
              dict(metric="chance accuracy (11 balanced classes)", value=round(1 / 11, 4)),
              dict(metric="SVM accuracy with placeholder (this environment)", value=t["acc_b"]),
              dict(metric="SVM accuracy without placeholder (this environment)", value=t["acc_a"]),
              dict(metric="difference, points", value=t["diff_points"]),
              dict(metric="McNemar exact p", value=float(f"{t['p_exact']:.3g}")),
              dict(metric="scikit-learn version of this control", value=sklearn.__version__)]
             ).to_csv(os.path.join(DST, "T08_mask_control.csv"), index=False)

# ---- leakage audit and dataset ----------------------------------------------------
z = pd.read_csv(os.path.join(OUT, "zeroshot/zeroshot_v4_test.csv"))
z["leakage_severity_final"].value_counts().rename_axis("leakage_category").reset_index(name="test_records") \
    .to_csv(os.path.join(DST, "T10_leakage_audit_test.csv"), index=False)
ds = []
for name, f in [("train S1 (nested)", "qlora_v4_train_S1.json"), ("train S2 (nested)", "qlora_v4_train_S2.json"),
                ("train S3 (used)", "qlora_v4_train_S3.json"), ("validation", "qlora_v4_val.json"), ("test", "qlora_v4_test.json")]:
    d = pd.Series([r["output"] for r in J(f)]).value_counts()
    ds.append(dict(split=name, n=int(d.sum()), per_class_min=int(d.min()), per_class_max=int(d.max()), classes=len(d)))
pd.DataFrame(ds).to_csv(os.path.join(DST, "T11_dataset_splits.csv"), index=False)

# ---- memory (measured by the memory scripts) ---------------------------------------
curve = json.load(open(os.path.join(OUT, "qlora_adapters/v4_s3_full_2epoch_lr1e4/training_curve.json"), encoding="utf-8"))
FITS = {"EVET": "yes", "HAYIR": "no", "HAYIR (temiz OOM)": "no (clean CUDA OOM)",
        "EVET (%55)": "yes (peak at 55% of 8188 MiB)", "HAYIR (4-bit'te bile)": "no (not even at 4-bit)"}
mt = pd.read_csv(os.path.join(BASE, "memory_table.csv"), encoding="utf-8-sig").drop(columns=["kaynak"])
mt.columns = ["model", "precision", "hardware", "weights_MiB", "non_quantized_MiB", "peak_inference_MiB", "activations_MiB",
              "fits_8gb", "quantization_saving_pct", "non_quantized_share_pct"]
assert mt["fits_8gb"].isin(list(FITS)).all(), set(mt["fits_8gb"]) - set(FITS)
mt["fits_8gb"] = mt["fits_8gb"].map(FITS)
mt["hardware"] = mt["hardware"].str.replace("80GB", "80 GB")
mt["peak_training_MiB"] = np.where((mt["model"] == "Qwen3.5-4B") & (mt["precision"] == "4-bit NF4"),
                                   round(curve["peak_gb"] * 1024, 1), np.nan)
mt["measurement"] = np.where(mt["model"].str.startswith("Qwen"), "06_memory/profile_memory.py (RTX 4060)",
                             "06_memory/colab_B3_gemma_memory.py (A100)")
mt.to_csv(os.path.join(DST, "T12_memory.csv"), index=False)
mc = pd.read_csv(os.path.join(BASE, "memory_components.csv"), encoding="utf-8-sig")
mc.columns = ["model", "precision", "component", "parameters", "MiB"]
mc["precision"] = mc["precision"].replace({"4bit": "4-bit NF4"})
mc["component"] = mc["component"].str.replace("linear/diger", "linear/other").str.replace("standart", "standard")
mc.to_csv(os.path.join(DST, "T13_memory_components_gemma.csv"), index=False)

# ---- prompt length and run time, from per-record logs ------------------------------
META = {k: (model, method, prec, hw) for k, model, method, prec, hw, *_ in TEST}
META.update({k: (model, method, prec, hw) for k, model, method, prec, hw, _ in VAL})
META["qwen_qlora_trunc"] = ("Qwen3.5-4B", "QLoRA fair, truncated evaluation input (ablation)", "4-bit NF4", "RTX 4060 8 GB")
RUNS = [("qwen_zero", "test", "zeroshot/zeroshot_v4_test.csv"), ("qwen_zero_val", "val", "zeroshot/zeroshot_v4_val.csv"),
        ("qwen_fs_comp_val_4060", "val", "fewshot/fewshot_v4_3shot_ultracompact_val.csv"),
        ("qwen_fs_comp_val_a100", "val", "b1_compressed_a100/b1_compressed_val.csv"),
        ("qwen_fs_comp", "test", "b1_compressed_a100/b1_compressed_test.csv"),
        ("qwen_fs_full_val", "val", "colab_downloads/full_3shot_val.csv"),
        ("qwen_fs_full", "test", "b2_b5_verify/b2_full3shot_test.csv"),
        ("qwen_qlora", "test", "qlora_eval_fullctx/test_predictions.csv"),
        ("qwen_qlora_trunc", "test", "qlora_eval/v4_s3_full_2epoch_lr1e4/test_predictions.csv"),
        ("gemma_zero_val_bf16", "val", "colab_downloads/gemma_zeroshot_val.csv"),
        ("gemma_zero_val_4bit", "val", "colab_downloads/gemma_4bit_zeroshot_val.csv"),
        ("gemma_zero_bf16", "test", "colab_downloads/gemma_bf16_zeroshot_test.csv"),
        ("gemma_zero_4bit", "test", "colab_downloads/gemma_4bit_zeroshot_test.csv"),
        ("gemma_fs_full", "test", "gemma_full3shot_test.csv"),
        ("gemma_qlora_fair", "test", "b2_b5_verify/b5_gemma_qlora_test.csv"),
        ("gemma_qlora_opt", "test", "gemma_qlora_test_preds.csv")]
rr = []
for k, split, f in RUNS:
    d = pd.read_csv(os.path.join(OUT, f)); tc = next((c for c in ("in_tokens", "input_tokens") if c in d), None)
    model, method, prec, hw = META[k]
    row = dict(config=k, model=model, method=method, precision=prec, hardware=hw, split=split, n=len(d),
               prompt_tokens_mean=round(d[tc].mean(), 1) if tc else np.nan,
               latency_s_per_record=np.nan, inference_min=np.nan, prompt_tokens_per_s=np.nan)
    if "sure_s" in d:
        row.update(latency_s_per_record=round(d["sure_s"].mean(), 3), inference_min=round(d["sure_s"].sum() / 60, 1),
                   prompt_tokens_per_s=round(d[tc].mean() / d["sure_s"].mean()))
    row.update(training_h=round(curve["train_sec"] / 3600, 2) if k == "qwen_qlora" else np.nan,
               timing_source="per-record timing" if "sure_s" in d else "not recorded")
    rr.append(row)
pd.DataFrame(rr).to_csv(os.path.join(DST, "T14_resources_latency.csv"), index=False)

# ---- self-check: aggregates only, English labels -----------------------------------
BAD_COLS = {"note_id", "subject_id", "hadm_id", "raw_output", "raw", "input", "text"}
TRW = ("EVET", "HAYIR", "olcum", "sikistir", "adil", "gomme", "yerelde", "sigmadi", "duyarlik", "Sinif", "Yontem", "diger", "standart")
for f in sorted(x for x in os.listdir(DST) if x.endswith(".csv")):
    d = pd.read_csv(os.path.join(DST, f), dtype=str).fillna("")
    blob = " ".join(list(d.columns) + d.values.ravel().tolist())
    assert not BAD_COLS & set(d.columns), f"record-level column in {f}"
    assert not re.search(r"(?<![\d.])[12]\d{7}(?![\d.])|-DS-\d", blob), f"ID-like value in {f}"
    assert not [w for w in TRW if w in blob], f"untranslated label in {f}"
    print(f"  {f:36s} {d.shape[0]:4d} rows x {d.shape[1]:2d} cols")

# ---- consistency with results_master.csv -------------------------------------------
rm = pd.read_csv(os.path.join(BASE, "results_master.csv")).set_index("yapilandirma")
ref = {"qwen_zero": "Qwen zero-shot", "qwen_qlora": "Qwen QLoRA adil", "gemma_zero_bf16": "Gemma zero-shot bf16",
       "gemma_zero_4bit": "Gemma zero-shot 4bit", "gemma_fs_full": "Gemma tam 3-shot", "gemma_qlora_opt": "Gemma QLoRA optimize",
       "qwen_fs_comp": "Qwen 3-shot sikistirilmis (A100)", "qwen_fs_full": "Qwen tam 3-shot (yeniden)", "gemma_qlora_fair": "Gemma QLoRA adil (yeniden)"}
bad = [(k, a) for k, a in ref.items()
       if abs(T01.set_index("config").at[k, "accuracy"] - rm.at[a, "accuracy"]) > 5e-4
       or abs(T01.set_index("config").at[k, "macro_f1"] - rm.at[a, "macro_f1_DUZELTILMIS"]) > 5e-4]
import scipy
print("rows inconsistent with results_master.csv:", bad or "none")
print("environment: Python", platform.python_version(), "| pandas", pd.__version__, "| numpy", np.__version__,
      "| scipy", scipy.__version__, "| scikit-learn", sklearn.__version__)
