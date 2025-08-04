"""
radii_bootstrap.py

This script analyzes Days on Market (DOM) for homes in Buncombe County, comparing properties inside a 
specified geographic circle to those outside. It tests whether the mean DOM (or DOM level) inside the region 
differs significantly from the rest of the sample using bootstrap resampling.

Usage:
- Set the input CSV file (either "1m_plus.csv" or "500k_to_1m.csv").
- The script automatically selects the appropriate DOM binning and region points for each price range.
- Results include mean difference, bootstrap p-value, and DOM level distribution inside the circle.

Author: Andrew Smith
Date: July 2025
"""

import pandas as pd
import numpy as np

# --------------------------
# 0. Load Data
# --------------------------
file = "1m_plus.csv" 
df = pd.read_csv(file)

# --------------------------
# 1. Create Categorical DOM Levels
# --------------------------
def dom_to_level_lower(dom):
    """
    Categorize DOM for $500k–$1M price range.
    Returns:
        0: 1st quartile (<8 days)
        1: 2nd quartile (8–30 days)
        2: 3rd quartile (31–90 days)
        3: 4th quartile (91+ days)
    """
    if dom < 8:
        return 0
    elif dom < 31:
        return 1
    elif dom < 91:
        return 2
    else:
        return 3

def dom_to_level_upper(dom):
    """
    Categorize DOM for $1M+ price range.
    Returns:
        0: 1st quartile (<15 days)
        1: 2nd quartile (15–90 days)
        2: 3rd quartile (91–150 days)
        3: 4th quartile (151+ days)
    """
    if dom < 15:
        return 0
    elif dom < 91:
        return 1
    elif dom < 151:
        return 2
    else:
        return 3

# Apply appropriate DOM binning based on file
if file == "500k_to_1m.csv":
    df["dom level"] = df["days on market"].apply(dom_to_level_lower)
else:
    df["dom level"] = df["days on market"].apply(dom_to_level_upper)

# --------------------------
# 2. Build the inside‑circle mask
# --------------------------
EARTH_MI = 3958.8

def haversine_miles(lat1, lon1, lat2, lon2):
    """
    Calculate great-circle distance (in miles) between two lat/lon points using the haversine formula.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * EARTH_MI * np.arcsin(np.sqrt(a))

# Define region centers and radii for each price range
points_1m_plus = [
    (35.458777,-82.606037, 1.9),    # SW AVL, p = 0.0044
    (35.627376, -82.341875, 1.05),  # Black Mtn, p = 0.0486
    (35.48822, -82.552515, 0.6),    # Biltmore Park, p = 0.0076
    (35.500295460168374, -82.54420712288918, 0.6), # Biltmore Park Northern NBHD, p = 0.0527
    (35.620721970896, -82.554717705398, 0.8),      # Kimberly Ave, p = 0.0540
    (35.644537, -82.533116, 1.15),  # North AVL, p = 0.1479
    (35.477892,-82.474507, 1.25),   # Concord Mountain/Fletcher, p = 0.1089
]

points_500k_to_1m = [
    (35.576573,-82.598217, 1.35),   # West AVL, p = 0.0042
    (35.466258,-82.511709, 1.15),   # Sweeten creek/hendersonville road, p = 0.0135
    (35.43492,-82.581852, 3),       # Brevard Road, p = 0.0265
    (35.59132926644027, -82.51655371719325, 0.8), # Tunnel Road off I-240, p = 0.0068
    (35.591478,-82.496632, 0.45),   # Tunnel Road, p = 0.0505
    (35.626173,-82.554985, 0.5),    # North Merrimon, p = 0.0924
    (35.60404139247464, -82.55854002710929, 0.4), # South Broadway, p = 0.0880
    (35.608052,-82.571744, 0.4),    # Montford Hills (next to UNCA), p = 0.0402
    (35.673091,-82.587463, 0.55),   # Weaverville I-26 ramp, p = 0.0936
    (35.528896,-82.417601, 1.2),    # Charlotte HWY/SE AVL, p = 0.1000
    (35.546736,-82.624108, 0.3),    # West Oakview road (West AVL), p = 0.0451
]

# Select region points based on file
if file == "1m_plus.csv":
    points = points_1m_plus
else:
    points = points_500k_to_1m

# Select a point to use as the region center
center_lat, center_lon = points[1][:2]
radius_mi              = points[1][2]

# Calculate distance from center for each property
df = df.copy()
df["dist"] = haversine_miles(df["lat"], df["lon"], center_lat, center_lon)
mask = df["dist"] <= radius_mi           # True = inside, False = outside

# --------------------------
# 3. Bootstrap mean test
# --------------------------
def bootstrap_mean_test(series, mask, B=10_000, seed=42):
    """
    Perform a bootstrap test for the difference in means between inside and outside groups.

    Args:
        series: pd.Series of values to test (e.g., DOM level)
        mask: Boolean mask for "inside" group
        B: Number of bootstrap samples
        seed: Random seed

    Returns:
        delta_obs: Observed mean difference (inside - outside)
        p_val: Bootstrap p-value for the observed difference
    """
    clean = series.dropna()
    m = mask.loc[clean.index]

    x = clean[m].to_numpy()      # inside group
    y = clean[~m].to_numpy()     # outside group

    if len(x) < 8:
        raise ValueError("Need ≥8 values in inside group")

    delta_obs = np.mean(x) - np.mean(y)

    rng = np.random.default_rng(seed)
    pooled = np.concatenate([x, y])
    n_x, n_y = len(x), len(y)

    deltas = np.empty(B)
    for b in range(B):
        # Resample from pooled data to simulate null hypothesis
        samp = rng.choice(pooled, size=n_x + n_y, replace=True)
        deltas[b] = (np.mean(samp[:n_x]) -
                     np.mean(samp[n_x:]))

    # p-value: fraction of bootstrap samples with difference >= observed
    p_val = np.mean(np.abs(deltas) >= np.abs(delta_obs))
    return delta_obs, p_val

# --------------------------
# 4. Run Test and Report
# --------------------------
dom_series = df["dom level"]
delta_mean, p_mean = bootstrap_mean_test(dom_series, mask)

mean_inside = dom_series[mask].mean()
mean_outside = dom_series[~mask].mean()
pct_change = (mean_inside - mean_outside) / mean_outside

print(f"\nΔ mean days on market level (inside − outside): {delta_mean:.2f}")
print(f"Bootstrap p‑value                               : {p_mean:.4f}")
print(f"Mean inside circle                              : {mean_inside:.1f}")
print(f"Mean outside circle                             : {mean_outside:.1f}")
print(f"Mean DOM % change                               : {pct_change:.2%}")

n_inside = mask.sum()
n_outside = len(mask) - n_inside
print(f"\nListings inside circle                    : {n_inside}")
print(f"Listings outside circle                   : {n_outside}")

# --------------------------
# 5. DOM Level Distribution 
# --------------------------
print("\nDOM level distribution inside circle:")
level_counts = df.loc[mask, "dom level"].value_counts(normalize=True).sort_index()
for level, pct in level_counts.items():
    label = {
        0: "0 (1st quartile)",
        1: "1 (2nd quartile)",
        2: "2 (3rd quartile)",
        3: "3 (4th quartile)"
    }.get(level, str(level))
    print(f"{label:<20}: {pct:.1%}")
