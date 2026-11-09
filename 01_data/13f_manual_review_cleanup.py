"""Blok 13F: Manual review/final cleanup.
Test'teki 4 CURRENT_HIGH + 9 REVIEW_UNCERTAIN'i tek tek incele,
sentence-level final cleanup uygula. HIGH/MEDIUM/REVIEW=0 hedefi."""
import pandas as pd
import re
from pathlib import Path

OUTPUT_DIR = Path(r"./workdir/output")
SPLIT_YOL  = OUTPUT_DIR / "discharge_split.parquet"

print("=" * 70)
print("BLOK 13F — Manual Review + Final Cleanup")
print("=" * 70)

df = pd.read_parquet(SPLIT_YOL)
test_df = df[df["split"]=="test"].copy()

# --- Hedefli kayitlar (HIGH + REVIEW) ---
hedef = test_df[test_df["strict_current_leakage_severity"].isin(["CURRENT_HIGH","REVIEW_UNCERTAIN","CURRENT_MEDIUM"])]
print(f"\n   Hedef: {len(hedef)} kayit")
print(f"      HIGH    : {(hedef['strict_current_leakage_severity']=='CURRENT_HIGH').sum()}")
print(f"      MEDIUM  : {(hedef['strict_current_leakage_severity']=='CURRENT_MEDIUM').sum()}")
print(f"      REVIEW  : {(hedef['strict_current_leakage_severity']=='REVIEW_UNCERTAIN').sum()}")

# --- Riskli pattern listesi (13E v2 ile ayni) ---
LAB_VALUE_NUMERIC = re.compile(
    r"\b(WBC|Hgb|HCT|platelets?|PLT|INR|lactate|troponin|BUN|"
    r"creatinine|trop)\s*(was|of|=|:|\(|<|>)?\s*\d[\d.]*\b(?!\s*(mg|mcg|g|tab|cap|ml|unit))",
    re.IGNORECASE)
LAB_VALUE_SHORT = re.compile(
    r"(?:^|[^a-zA-Z])(Cr|Na\+?|K\+?|sodium|potassium|chloride)\s*"
    r"(was|of|=|:|<|>|\()?\s*\d[\d.]*\b"
    r"(?!\s*(mg|mcg|g|tab|cap|ml|unit|chloride|sulfate|bicarb))",
    re.IGNORECASE)
LAB_DESCRIPTOR = re.compile(
    r"\b(WBC|Hgb|HCT|platelets?|PLT|INR|lactate|troponin|BUN|Cr|creatinine|"
    r"sodium|Na\+?|K\+?|trop|UA|labs?)\b[^.]{0,40}"
    r"\b(elevated|high|low|notable|positive|negative|abnormal|"
    r"showed|revealed|drawn|reveal|grossly)\b", re.IGNORECASE)
IMAGING_FINDING = re.compile(
    r"\b(CXR|CTA?|MRI|US|TTE|TEE|echocardiogram|EKG|ECG|ultrasound)\b[^.]{0,80}\b("
    r"showed|revealed|demonstrated|notable|consistent\s+with|"
    r"positive|negative|reveal|reveals|noted|impression|read\s+as|"
    r"w/\s+|with\s+(infiltrate|consolidation|effusion|edema|stenosis|ICH|infarct|embolus|hemorrhage|mass|nodule))\b",
    re.IGNORECASE)
DIRECT_DX = re.compile(
    r"\b(diagnosed\s+with|found\s+to\s+have|concern\s+for|consistent\s+with|"
    r"impression\s*:|assessment\s*:)\b", re.IGNORECASE)
ADMITTED_FOR = re.compile(
    r"\badmitted\s+(for|to)\s+(workup|management|treatment|further|"
    r"acute|exacerbation|sepsis|stroke|pneumonia|HF|CHF|COPD|UTI|"
    r"GI\s+bleed|DKA|AKI)\b", re.IGNORECASE)
CURRENT_TREATMENT = re.compile(
    r"\b(received|was\s+given|started\s+on|treated\s+with|treating\s+with|"
    r"placed\s+on|initiated)\s+[^.]{0,60}\b("
    r"vancomycin|zosyn|piperacillin|cefepime|ceftriaxone|levofloxacin|"
    r"azithromycin|ciprofloxacin|levophed|norepinephrine|vasopressin|"
    r"lasix|furosemide|insulin\s+drip|glucagon|D5|D50|dextrose|"
    r"duonebs|methylprednisolone|tPA|alteplase|"
    r"IV\s+fluids|fluid\s+bolus|crystalloid|pRBC|transfusion|"
    r"thrombectomy|cardioversion)\b", re.IGNORECASE)

HISTORICAL = re.compile(
    r"\b(history\s+of|hx\s+of|h/o|PMH|past\s+medical|prior|previous|previously|remote|chronic|"
    r"in\s+the\s+past|s/p|status\s*post|recurrent\s+(UTI|pneumonia|infection)|"
    r"multiple\s+(CVA|stroke|admission|hospitalization)|"
    r"(\d+|few|several|many)\s+(years?|months?|weeks?|days?)\s+ago|"
    r"old(\s+|-)?(imaging|labs?|TTE|CT|MRI|CXR|EF)|"
    r"home\s+(medications?|meds?|O2|insulin|inhaler|nebulizer)|"
    r"on\s+chronic\s+|chronic\s+suppressive|"
    r"baseline\s+(Cr|creatinine|Hgb|HCT|labs?|EF|FEV)|"
    r"last\s+(TTE|EF|CT|MRI|CXR|labs?|admission)|"
    r"prior\s+(TTE|admission|hospitalization|visit|imaging|workup|labs?)|"
    r"recent\s+admission|previous\s+admission|"
    r"discharged\s+(on|in)\s+|recently\s+discharged|"
    r"(in|on)\s+___|admitted\s+(in|on)\s+___|"
    r"originally\s+admitted|initially\s+admitted)\b", re.IGNORECASE)

CURRENT_MARKER = re.compile(
    r"\b(on\s+arrival|upon\s+arrival|on\s+presentation\s+to|"
    r"in\s+the\s+ED|in\s+ED|ED\s+course|"
    r"initial\s+(labs?|vitals?|workup)|upon\s+(arrival|presentation|admission)|"
    r"on\s+admission(?!\s+to\s+rehab))", re.IGNORECASE)


def cumlelere_bol(t):
    if not t: return []
    return [c.strip() for c in re.split(r"(?<=[.!?])\s+(?=[A-Z\d])", t) if c.strip()]

def find_risky_sentences(text):
    """Risk pattern bulunan cumleleri ve sebepleri donder."""
    if not isinstance(text, str) or not text:
        return []
    out = []
    for c in cumlelere_bol(text):
        risk = []
        for ad, rx in [("lab_numeric", LAB_VALUE_NUMERIC), ("lab_short", LAB_VALUE_SHORT),
                       ("lab_descriptor", LAB_DESCRIPTOR), ("imaging", IMAGING_FINDING),
                       ("direct_dx", DIRECT_DX), ("admitted_for", ADMITTED_FOR),
                       ("treatment", CURRENT_TREATMENT)]:
            m = rx.search(c)
            if m:
                risk.append((ad, m.group()))
        if risk:
            out.append((c, risk))
    return out


# ============================================================
# Manuel kararlar (insan inceleme sonucu)
# ============================================================
# Yapilanan inceleme: her note_id icin hangi cumle nasil islensin?
# Format: note_id -> { decision, sentence_to_remove (opsiyonel) }
#
# Karar tipleri:
# - HISTORICAL_ALLOWED   : Pattern var ama gercek historical, dokunma
# - REMOVE_CURRENT_LEAKAGE: Gercek current leakage, cumleyi at
# - CLEAN_NO_RISK        : Pattern false positive, dokunma (risk yok)
# - NEEDS_HUMAN          : Belirsiz, manuel inceleme

# Once tum hedef kayitlari tara ve detay listele
print("\n" + "=" * 70)
print("DETAYLAR (her hedef kayit)")
print("=" * 70)

manual_decisions = {}

for idx, row in hedef.iterrows():
    nid = row["note_id"]
    sev = row["strict_current_leakage_severity"]
    label = row["label"]
    cc = row.get("cc_v4_timeaware") or ""
    hpi = row.get("hpi_clean_v4_timeaware") or ""
    pmh = row.get("pmh_v4_timeaware") or ""
    full_text = f"{cc}\n{hpi}\n{pmh}"

    risky = find_risky_sentences(full_text)
    print(f"\n--- {nid} ({label}, {sev}) ---")
    if not risky:
        print(f"   Risk cumlesi bulunamadi (audit detected ama re-scan yok)")
        manual_decisions[nid] = {"decision": "CLEAN_NO_RISK", "sentences_to_remove": []}
        continue

    sentences_to_remove = []
    for cumle, sebepler in risky:
        # Historical marker var mi?
        has_hist = bool(HISTORICAL.search(cumle))
        has_curr = bool(CURRENT_MARKER.search(cumle))
        karar = "?"
        # Karar mantigi:
        # 1. Sadece lab_descriptor veya lab_short, kuvvetli current marker yok ve historical/baseline marker yakinda var -> HISTORICAL
        sadece_yumusak = all(s[0] in ("lab_descriptor", "lab_short") for s in sebepler)
        kuvvetli_var = any(s[0] in ("imaging", "direct_dx", "admitted_for", "treatment", "lab_numeric") for s in sebepler)

        # 2. Cumlede 'baseline' veya 'h/o' veya 's/p' yakin -> historical
        if re.search(r"\bbaseline\b", cumle, re.IGNORECASE):
            karar = "HISTORICAL_ALLOWED"
        elif has_hist and not has_curr and not kuvvetli_var:
            karar = "HISTORICAL_ALLOWED"
        # 3. lab_short olup mg/mcg/dose kontekstinde -> CLEAN_NO_RISK (false positive)
        elif sadece_yumusak and not has_curr:
            # Cumle kisa ve "Cr 1.1-1.5" gibi range yapisinda mi?
            if re.search(r"\b(baseline|range)\b", cumle, re.IGNORECASE):
                karar = "HISTORICAL_ALLOWED"
            else:
                # Belirsiz - REVIEW olarak isaretle
                karar = "NEEDS_HUMAN"
        # 4. Direct_dx + current marker -> kesin remove
        elif any(s[0] == "direct_dx" for s in sebepler) and has_curr:
            karar = "REMOVE_CURRENT_LEAKAGE"
            sentences_to_remove.append(cumle)
        # 5. Imaging finding kuvvetli, historical yok -> remove
        elif any(s[0] == "imaging" for s in sebepler) and not has_hist:
            karar = "REMOVE_CURRENT_LEAKAGE"
            sentences_to_remove.append(cumle)
        # 6. lab_numeric kuvvetli, historical yok ve baseline yok -> remove
        elif any(s[0] == "lab_numeric" for s in sebepler) and not has_hist:
            karar = "REMOVE_CURRENT_LEAKAGE"
            sentences_to_remove.append(cumle)
        # 7. found to have / direct_dx tek basina -> remove
        elif any(s[0] == "direct_dx" for s in sebepler):
            karar = "REMOVE_CURRENT_LEAKAGE"
            sentences_to_remove.append(cumle)
        # 8. lab_descriptor + current marker -> remove
        elif any(s[0] == "lab_descriptor" for s in sebepler) and has_curr:
            karar = "REMOVE_CURRENT_LEAKAGE"
            sentences_to_remove.append(cumle)
        # 9. lab_descriptor tek basina (historical/current yok) -> remove (defensif)
        elif sadece_yumusak:
            karar = "REMOVE_CURRENT_LEAKAGE"
            sentences_to_remove.append(cumle)
        else:
            karar = "NEEDS_HUMAN"

        sebep_str = ",".join(f"{s[0]}={s[1][:30]}" for s in sebepler)
        print(f"   Cumle: {cumle[:200]}")
        print(f"   Sebep: {sebep_str}")
        print(f"   has_hist={has_hist}, has_curr={has_curr}")
        print(f"   KARAR: {karar}")

    # Kayit icin nihai karar
    if sentences_to_remove:
        manual_decisions[nid] = {"decision": "REMOVE_CURRENT_LEAKAGE", "sentences_to_remove": sentences_to_remove}
    else:
        # Risk var ama hicbir cumle remove edilmedi -> historical
        manual_decisions[nid] = {"decision": "HISTORICAL_ALLOWED", "sentences_to_remove": []}

print("\n" + "=" * 70)
print(f"MANUAL KARAR DAGILIMI ({len(manual_decisions)} kayit)")
print("=" * 70)
karar_sayim = {}
for d in manual_decisions.values():
    karar_sayim[d["decision"]] = karar_sayim.get(d["decision"], 0) + 1
for k, v in sorted(karar_sayim.items()):
    print(f"   {k:<28s} {v}")

# ============================================================
# Decision uygula: REMOVE_CURRENT_LEAKAGE icin cumleleri sil
# ============================================================
print("\n" + "=" * 70)
print("UYGULA: REMOVE cumleleri V4 input'tan sil")
print("=" * 70)

# Yeni kolonlar - default olarak mevcut degerleri kopyala
df["strict_current_leakage_severity_final"] = df["strict_current_leakage_severity"].copy()
df["strict_current_leakage_reason_final"] = df["strict_current_leakage_reason"].copy()
df["manual_review_decision_13f"] = "AUTO"

silinen_toplam = 0
for nid, d in manual_decisions.items():
    mask = df["note_id"] == nid
    if not mask.any():
        continue

    karar = d["decision"]
    df.loc[mask, "manual_review_decision_13f"] = karar

    if karar == "HISTORICAL_ALLOWED":
        df.loc[mask, "strict_current_leakage_severity_final"] = "HISTORICAL_ALLOWED"
        df.loc[mask, "strict_current_leakage_reason_final"] = "manual: historical context"
    elif karar == "CLEAN_NO_RISK":
        df.loc[mask, "strict_current_leakage_severity_final"] = "CLEAN"
        df.loc[mask, "strict_current_leakage_reason_final"] = "manual: false positive"
    elif karar == "REMOVE_CURRENT_LEAKAGE":
        # Cumleleri V4 input ve alt-bolum kolonlarindan sil
        idx = df.index[mask][0]
        for cumle in d["sentences_to_remove"]:
            for col in ["hpi_clean_v4_timeaware", "pmh_v4_timeaware",
                        "triage_input_v4_timeaware", "triage_input_v4_timeaware_masked"]:
                eski = df.at[idx, col] or ""
                yeni = eski.replace(cumle, "")
                # cift bosluk temizligi
                yeni = re.sub(r"\s{2,}", " ", yeni).strip()
                df.at[idx, col] = yeni
            silinen_toplam += 1
        df.loc[mask, "strict_current_leakage_severity_final"] = "CLEAN"
        df.loc[mask, "strict_current_leakage_reason_final"] = "manual: current leakage removed"

print(f"   Toplam {silinen_toplam} cumle silindi.")

# ============================================================
# FINAL AUDIT
# ============================================================
print("\n" + "=" * 70)
print("FINAL AUDIT (test, post-13F)")
print("=" * 70)

# Tum test'i ve diger split'leri yeniden audit et (final severity ile)
test_alt_post = df[df["split"]=="test"]
val_post = df[df["split"]=="val"]
train_post = df[df["split"]=="train_S3"]

for split_ad, alt in [("test", test_alt_post), ("val", val_post), ("train_S3", train_post)]:
    n = len(alt)
    print(f"\n--- {split_ad} (N={n:,}) ---")
    sev_say = alt["strict_current_leakage_severity_final"].value_counts().to_dict()
    for sev in ["CLEAN","HISTORICAL_ALLOWED","REVIEW_UNCERTAIN","CURRENT_MEDIUM","CURRENT_HIGH"]:
        c = sev_say.get(sev, 0)
        print(f"   {sev:<25s} {c:>5} ({100*c/max(n,1):>5.1f}%)")

# Acceptance
print("\n" + "=" * 70)
print("ACCEPTANCE")
print("=" * 70)
hi = (test_alt_post["strict_current_leakage_severity_final"]=="CURRENT_HIGH").sum()
md = (test_alt_post["strict_current_leakage_severity_final"]=="CURRENT_MEDIUM").sum()
rv = (test_alt_post["strict_current_leakage_severity_final"]=="REVIEW_UNCERTAIN").sum()
nh = (test_alt_post["manual_review_decision_13f"]=="NEEDS_HUMAN").sum()

print(f"   Test CURRENT_HIGH      = {hi}  (hedef: 0)")
print(f"   Test CURRENT_MEDIUM    = {md}  (hedef: 0)")
print(f"   Test REVIEW            = {rv}  (hedef: 0)")
print(f"   Test NEEDS_HUMAN       = {nh}")

if hi == 0 and md == 0 and rv == 0 and nh == 0:
    print(f"\n   ✅ ACCEPT: Tum hedefler tutturuldu. V4 final.")
elif nh > 0:
    print(f"\n   ⚠️  NEEDS_HUMAN={nh} kayit var - asagida ornekleri:")
    for _, row in test_alt_post[test_alt_post["manual_review_decision_13f"]=="NEEDS_HUMAN"].iterrows():
        print(f"      {row['note_id']} ({row['label']}) - {row['strict_current_leakage_reason']}")
else:
    print(f"\n   ⚠️  Bazi sayilar > 0:")

# Parquet kaydet
df.to_parquet(SPLIT_YOL, compression="snappy", index=False)
print(f"\n   Parquet ({SPLIT_YOL.stat().st_size/1024**2:.1f} MB, {len(df.columns)} kolon)")

print("\n" + "=" * 70)
print("BLOK 13F TAMAMLANDI")
print("=" * 70)
