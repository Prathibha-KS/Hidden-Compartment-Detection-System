import numpy as np

# ------------------------------------------------------------
# GLOBAL NORMALIZATION ACROSS ALL 4 DIRECTIONS (CORRECT)
# ------------------------------------------------------------
def normalize_live_data(live_data):
    """
    live_data format:
      {
        "front": {"ultra": [...10], "refl": [...10]},
        "back":  {"ultra": [...10], "refl": [...10]},
        "left":  {"ultra": [...10], "refl": [...10]},
        "right": {"ultra": [...10], "refl": [...10]}
      }
    """

    # Collect all ultra + reflectivity values
    all_ultra = []
    all_refl = []

    for side in live_data.values():
        all_ultra.extend(side["ultra"])
        all_refl.extend(side["refl"])

    all_ultra = np.array(all_ultra, dtype=float)
    all_refl = np.array(all_refl, dtype=float)

    # Global min/max for the bag
    ultra_min, ultra_max = all_ultra.min(), all_ultra.max()
    refl_min,  refl_max  = all_refl.min(),  all_refl.max()

    # Avoid division by zero
    if ultra_min == ultra_max: ultra_max += 1
    if refl_min == refl_max: refl_max += 1

    normalized = {}

    for key, side in live_data.items():
        ultra_norm = (np.array(side["ultra"]) - ultra_min) / (ultra_max - ultra_min)
        refl_norm  = (np.array(side["refl" ]) - refl_min ) / (refl_max  - refl_min )
        normalized[key] = {
            "ultra": ultra_norm,
            "refl":  refl_norm
        }

    return normalized


# ------------------------------------------------------------
# BUILD TENSOR FOR CNN/RF
# ------------------------------------------------------------
def build_tensor(normalized_data):
    order = ["front", "back", "left", "right"]
    output = []

    for side in order:
        u = normalized_data[side]["ultra"]
        r = normalized_data[side]["refl"]
        output.append(np.column_stack([u, r]))  # (10,2)

    return np.array(output)  # (4,10,2)
live_data = {
    "front": {   # Direction 1
        "ultra": [13.42, 14.23, 13.07, 14.18, 14.73, 14.04, 14.59, 13.99, 13.75, 13.93],
        "refl":  [25632, 25631, 25600, 25600, 25600, 25632, 25600, 25600, 25600, 25600]
    },
    "back": {    # Direction 2
        "ultra": [21.4, 16.3, 15.93, 15.59, 15.37, 15.68, 15.28, 15.62, 16.22, 15.23],
        "refl":  [25600, 25600, 25600, 25600, 25600, 25600, 25600, 25600, 25600, 25600]
    },
    "left": {    # Direction 3
        "ultra": [20.44, 19.78, 20.48, 20.77, 21.16, 20.54, 20.3, 21.53, 21.28, 21.34],
        "refl":  [25600, 25600, 25600, 25600, 25600, 25600, 25630, 25600, 25600, 25600]
    },
    "right": {   # Direction 4
        "ultra": [17.79, 11.56, 16.85, 16.64, 17.36, 17.54, 16.86, 16.99, 16.85, 17.31],
        "refl":  [25600, 25600, 25600, 25600, 25600, 25600, 25632, 25632, 25600, 25634]
    }
}

normalized_data = normalize_live_data(live_data)
final_tensor = build_tensor(normalized_data)  # (4,10,2)