"""Zero-shot final parser (8B'de sabitlendi)."""
ALLOWED_CLASSES = ['SEPSIS', 'STROKE', 'ATRIAL_FIBRILLATION', 'AKI', 'PNEUMONIA', 'UTI', 'HEART_FAILURE', 'GI_BLEEDING', 'DIABETES_COMPLICATION', 'COPD_EXACERBATION', 'OTHER']
FUZZY_MAP = {'SEPS': 'SEPSIS', 'SEPT': 'SEPSIS', 'PNEUM': 'PNEUMONIA', 'STROK': 'STROKE', 'CVA': 'STROKE', 'ATRIAL': 'ATRIAL_FIBRILLATION', 'AFIB': 'ATRIAL_FIBRILLATION', 'FIBRIL': 'ATRIAL_FIBRILLATION', 'AKI': 'AKI', 'KIDNEY': 'AKI', 'RENAL': 'AKI', 'UTI': 'UTI', 'URIN': 'UTI', 'HEART': 'HEART_FAILURE', 'CHF': 'HEART_FAILURE', 'GI': 'GI_BLEEDING', 'BLEED': 'GI_BLEEDING', 'DIAB': 'DIABETES_COMPLICATION', 'DKA': 'DIABETES_COMPLICATION', 'COPD': 'COPD_EXACERBATION', 'OTHER': 'OTHER', 'NONE': 'OTHER'}

def parse_class(raw):
    if not raw:
        return "INVALID", "empty"
    text = raw.strip().upper()
    siralama = sorted(ALLOWED_CLASSES, key=len, reverse=True)
    for cls in siralama:
        if cls in text[:80]:
            return cls, "direct"
    for k, v in FUZZY_MAP.items():
        if k in text[:80]:
            return v, "fuzzy"
    return "INVALID", "invalid"
