# predictor_xgb_rf.py

import numpy as np
import torch  # only if you still want future CNN; otherwise can be removed
import joblib

from data_preprocess_final import SCALER_PATH

# ====== Paths ======
RF_MODEL_PATH  = "best_rf.pkl"
XGB_MODEL_PATH = "best_xgb.pkl"

# These reflectivity ranges are from your synthetic physics
REFL_MIN = 0.35
REFL_MAX = 1.05

# HLK raw amplitude range (device-specific)
HLK_MIN  = 4000.0
HLK_MAX  = 26000.0

# ============================================================
# 1. Load models + scaler
# ============================================================

rf = joblib.load(RF_MODEL_PATH)
print("✓ RF model loaded:", RF_MODEL_PATH)

xgb = joblib.load(XGB_MODEL_PATH)
print("✓ XGBoost model loaded:", XGB_MODEL_PATH)

scaler = joblib.load(SCALER_PATH)
print("✓ Scaler loaded:", SCALER_PATH)


# ============================================================
# 2. Map HLK reflectivity → synthetic reflectivity
# ============================================================

def hlk_to_dataset_refl(raw_hlk):
    """
    Convert HLK raw readings (e.g., 4000–26000) into the synthetic
    reflectivity range [0.35, 1.05] used in training.
    """
    r = np.array(raw_hlk, dtype=float)
    mapped = REFL_MIN + (r - HLK_MIN) / (HLK_MAX - HLK_MIN) * (REFL_MAX - REFL_MIN)
    return np.clip(mapped, REFL_MIN, REFL_MAX)


# ============================================================
# 3. Build dataset-style array for a live bag
# ============================================================

def build_dataset_array(live):
    """
    Convert live_data dict into (4,10,2) array:

    Order: front, back, left, right

    feature 0 = ultrasonic (raw cm)
    feature 1 = reflectivity (synthetic domain)
    """
    order = ["front", "back", "left", "right"]
    out = []

    for side in order:
        ultra = np.array(live[side]["ultra"], dtype=float)
        refl  = hlk_to_dataset_refl(live[side]["refl"])
        out.append(np.column_stack([ultra, refl]))

    return np.stack(out, axis=0)   # (4,10,2)


# ============================================================
# 4. Build RF/XGB features with SAME scaler as training
# ============================================================

def build_ml_features(live):
    """
    Build standardized 80-D feature vector for RF and XGB:
    (4,10,2) → (40,2) → scaler.transform → (80,)
    """
    arr = build_dataset_array(live)
    flat = arr.reshape(-1, 2)              # (40,2)
    scaled = scaler.transform(flat)        # standardized
    return scaled.flatten()                # (80,)


# ============================================================
# 5. Predict with RF + XGBoost ensemble
# ============================================================

def predict_bag(live_data):
    feat = build_ml_features(live_data)

    # RF prediction
    prob_rf = float(rf.predict_proba([feat])[0, 1])

    # XGB prediction
    prob_xgb = float(xgb.predict_proba([feat])[0, 1])

    # Simple 50/50 ensemble (you can tune weights)
    prob_ens = 0.5 * prob_rf + 0.5 * prob_xgb

    label = "SUSPICIOUS / HIDDEN COMPARTMENT" if prob_ens >= 0.5 else "NORMAL"

    print("=====================================================")
    print(" LIVE LUGGAGE SCAN PREDICTION (RF + XGBoost) ")
    print("=====================================================")
    print(f"RF Probability:        {prob_rf:.4f}")
    print(f"XGBoost Probability:   {prob_xgb:.4f}")
    print(f"Ensemble (50/50):      {prob_ens:.4f} → {label}")
    print("=====================================================")
    print(" FINAL RESULT:", label)
    print("=====================================================")

    return prob_ens, label


# ============================================================
# 6. Example usage
# ============================================================


# Put your latest readings here:
live_data = {
"front": {
    "ultra": [13.42, 14.23, 13.07, 14.18, 14.73, 14.04, 14.59, 13.99, 13.75, 13.93],
    "refl":  [25632, 25631, 25600, 25600, 25600, 25632, 25600, 25600, 25600, 25600]
},
"back": {
    "ultra": [21.4, 16.3, 15.93, 15.59, 15.37, 15.68, 15.28, 15.62, 16.22, 15.23],
    "refl":  [25600, 25600, 25600, 25600, 25600, 25600, 25600, 25600, 25600, 25600]
},
"left": {
    "ultra": [20.44, 19.78, 20.48, 20.77, 21.16, 20.54, 20.3, 21.53, 21.28, 21.34],
    "refl":  [25600, 25600, 25600, 25600, 25600, 25600, 25630, 25600, 25600, 25600]
},
"right": {
    "ultra": [17.79, 11.56, 16.85, 16.64, 17.36, 17.54, 16.86, 16.99, 16.85, 17.31],
    "refl":  [25600, 25600, 25600, 25600, 25600, 25600, 25632, 25632, 25600, 25634]
}
}
predict_bag(live_data)
