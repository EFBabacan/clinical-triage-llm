"""Final state of the leakage screen per split: counts only, no note text is read.

Input : output/discharge_split.parquet (columns split, strict_current_leakage_severity_final and
        manual_review_decision_13f only)
Output: output/results_tables/T23_leakage_screen.csv

flagged_final: records that still hold a sentence in one of the three non-historical tiers
(CURRENT_HIGH, CURRENT_MEDIUM, REVIEW_UNCERTAIN) after the last screen. rule_review_*: the
decisions of the sentence-level rules of BLOK 13F (13f_manual_review_cleanup.py), a step that
was applied to the test split only.
Run   : from the workspace root, python /path/to/repo/07_analysis/build_leak_screen_table.py
        (paths above are relative to workdir/)
"""
import os
import pandas as pd

BASE = "./workdir"  # workspace/workdir: run from the workspace root (see README)
OUT = os.path.join(BASE, "output")
DST = os.path.join(OUT, "results_tables")
d = pd.read_parquet(os.path.join(OUT, "discharge_split.parquet"),
                    columns=["split", "strict_current_leakage_severity_final", "manual_review_decision_13f"])
FLAG = ["CURRENT_HIGH", "CURRENT_MEDIUM", "REVIEW_UNCERTAIN"]
rows = []
for sp, g in d.groupby("split"):
    sev = g["strict_current_leakage_severity_final"].value_counts()
    dec = g["manual_review_decision_13f"].value_counts()
    rows.append(dict(split=sp, n=len(g),
                     current_high=int(sev.get("CURRENT_HIGH", 0)),
                     current_medium=int(sev.get("CURRENT_MEDIUM", 0)),
                     review_uncertain=int(sev.get("REVIEW_UNCERTAIN", 0)),
                     flagged_final=int(sum(sev.get(k, 0) for k in FLAG)),
                     rule_review_kept_historical=int(dec.get("HISTORICAL_ALLOWED", 0)),
                     rule_review_records_cleaned=int(dec.get("REMOVE_CURRENT_LEAKAGE", 0)),
                     rule_review_needs_human=int(dec.get("NEEDS_HUMAN", 0))))
T = pd.DataFrame(rows)
T.to_csv(os.path.join(DST, "T23_leakage_screen.csv"), index=False)
print(T.to_string(index=False))
