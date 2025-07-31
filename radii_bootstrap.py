import pandas as pd
import numpy as np

# --------------------------
# 0. Load Data
# --------------------------
df = pd.read_csv("500k_to_1m.csv")
# df = df[df["address"] != "353 Dingle Creek Ln, Asheville, NC 28803"]

# --------------------------
# 1. Create Categorical DOM Levels
# --------------------------
def dom_to_level(dom):
    # 500k
    if dom < 8:
        return 0
    elif dom < 31:
        return 1
    elif dom < 91:
        return 2
    else:
        return 3

    # 1m
    # if dom < 15:
    #     return 0
    # elif dom < 91:
    #     return 1
    # elif dom < 151:
    #     return 2
    # else:
    #     return 3

df["dom_level"] = df["days on market"].apply(dom_to_level)

# --------------------------
# 2. Build the inside‑circle mask
# --------------------------
EARTH_MI = 3958.8
def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * EARTH_MI * np.arcsin(np.sqrt(a))


# points = [
#     (35.458777,-82.606037, 1.9), # SW AVL, 11
#     (35.627376, -82.341875, 0.2), # Black Mtn, 11
#     (35.625971,-82.339054, 0.9), # Black Mtn, 00
#     (35.48822, -82.552515, 0.55), # Biltmore Park, 11
#     (35.484176438776274, -82.55149026049794, 0.82), # Biltmore Park, 01
#     (35.644537, -82.533116, 1.3), # North AVL, 00
#     (35.641046, -82.530512, 0.8), # North AVL (small sample with unsolds constituting large portion), 10
#     (35.622805217005855, -82.55688511721898, 0.93), # Kimberly Ave, 01
#     (35.49043,-82.409025, 1.5), # SE AVL, 00
#     (35.538216537934034, -82.55150461942004, 0.9), # Biltmore Forest, 01
#     (35.540133,-82.536944, 1.05), # Biltmore Forest, 00
#     (35.546226,-82.460486, 1.25), # East 00
#     (35.722444,-82.550109, 3.1), # Weaverville 11
# ]

# points = [
#     (35.458777,-82.606037, 1.9), # SW AVL, 11
#     (35.627376, -82.341875, 0.2), # Black Mtn, 11
#     (35.48822, -82.552515, 0.55), # Biltmore Park, 01
#     (35.484176438776274, -82.55149026049794, 0.82), # Biltmore Park, 01
#     (35.622805217005855, -82.55688511721898, 0.93), # Kimberly Ave, 01
#     (35.538216537934034, -82.55150461942004, 1.06), # Biltmore Forest, 01
#     (35.722444,-82.550109, 3.1), # Weaverville 10
#     (35.500295460168374, -82.54420712288918, 0.6), # Biltmore Park Northern NBHD, 11
# ]

points = [
    (35.576573,-82.598217, 1.35), # West AVL, 11
    (35.483876,-82.54999, 1.375), # Biltmore Park, 00
    (35.55384,-82.503086, 1.3), # Sweeten Creek/BRP, 00
    (35.546736,-82.624108, 0.3), # West AVL neighborhood, 11
    (35.673091,-82.587463, 0.55), # Under Weaverville, 00 close
    (35.608052,-82.571744, 0.4), # Montford Hills (next to UNCA), 01
    (35.614111,-82.557102, 0.45), # UP/Merrimon, 10
    (35.6127734974028, -82.50413559279191, 0.41), # Rolling Green (next to BRP, outskirts of Asheville), 00
    (35.591478,-82.496632, 0.45), # East AVL (Bull MTN), 01
    (35.466258,-82.511709, 1.15), # Sweeten creek hendersonville road, 11
    (35.43492,-82.581852, 3), # West of Biltmore Park, 01
    (35.45613,-82.537469, 3.2), # Biltmore Park/sweeten creek, hendersonville road, 01
    (35.626173,-82.554985, 0.5), # North Merrimon, 00 close
    (35.60404139247464, -82.55854002710929, 0.4), # South Broadway, 00 close
    (35.591478,-82.496632, 0.45), # Tunnel Road, 01
    (35.59132926644027, -82.51655371719325, 0.8), # Tunnel Road west, 01
    (35.528896,-82.417601, 1.2), # SE AVL/Charlotte HWY, 00 close
    (35.621048,-82.334177, 0.17),

]

center_lat, center_lon = points[0][:2]
radius_mi              = points[0][2]

df = df.copy()
df["dist"] = haversine_miles(df["lat"], df["lon"], center_lat, center_lon)
mask = df["dist"] <= radius_mi           # True = inside, False = outside

# --------------------------
# 3. Bootstrap variance test
# --------------------------
def bootstrap_std_test(series, mask, B=10_000, seed=42):
    clean = series.dropna()
    m = mask.loc[clean.index]

    x = clean[m].to_numpy()      # inside
    y = clean[~m].to_numpy()     # outside

    if len(x) < 2 or len(y) < 2:
        raise ValueError("Need ≥2 values in both groups")

    delta_obs = np.std(x, ddof=1) - np.std(y, ddof=1)

    rng = np.random.default_rng(seed)
    pooled = np.concatenate([x, y])
    n_x, n_y = len(x), len(y)

    deltas = np.empty(B)
    for b in range(B):
        samp = rng.choice(pooled, size=n_x + n_y, replace=True)
        deltas[b] = (np.std(samp[:n_x], ddof=1) -
                     np.std(samp[n_x:],  ddof=1))

    p_val = np.mean(np.abs(deltas) >= np.abs(delta_obs))
    return delta_obs, p_val

# --------------------------
# 4. Bootstrap mean test
# --------------------------
def bootstrap_mean_test(series, mask, B=10_000, seed=42):
    clean = series.dropna()
    m = mask.loc[clean.index]

    x = clean[m].to_numpy()      # inside
    y = clean[~m].to_numpy()     # outside

    if len(x) < 2 or len(y) < 2:
        raise ValueError("Need ≥2 values in both groups")

    delta_obs = np.mean(x) - np.mean(y)

    rng = np.random.default_rng(seed)
    pooled = np.concatenate([x, y])
    n_x, n_y = len(x), len(y)

    deltas = np.empty(B)
    for b in range(B):
        samp = rng.choice(pooled, size=n_x + n_y, replace=True)
        deltas[b] = (np.mean(samp[:n_x]) -
                     np.mean(samp[n_x:]))

    p_val = np.mean(np.abs(deltas) >= np.abs(delta_obs))
    return delta_obs, p_val


# --------------------------
# 5. Run Test and Report
# --------------------------
dom_series = df["dom_level"]
delta_std, p = bootstrap_std_test(dom_series, mask)

# Compute raw variances for context
std_inside = dom_series[mask].std(ddof=1)
std_outside = dom_series[~mask].std(ddof=1)
pct_decrease = (std_outside - std_inside) / std_outside

print(f"\nΔ std of dom_level (inside − outside): {delta_std:,.4f}")
print(f"Bootstrap p‑value                         : {p:.4f}")
print(f"Std inside circle                    : {std_inside:.4f}")
print(f"Std outside circle                   : {std_outside:.4f}")
print(f"Std decrease percentage              : {pct_decrease:.2%}")


delta_mean, p_mean = bootstrap_mean_test(dom_series, mask)

mean_inside = dom_series[mask].mean()
mean_outside = dom_series[~mask].mean()
pct_change = (mean_inside - mean_outside) / mean_outside

print(f"\nΔ mean days on market (inside − outside): {delta_mean:.2f}")
print(f"Bootstrap p‑value                         : {p_mean:.4f}")
print(f"Mean inside circle                        : {mean_inside:.1f}")
print(f"Mean outside circle                       : {mean_outside:.1f}")
print(f"Mean DOM % change                         : {pct_change:.2%}")


n_inside = mask.sum()
n_outside = len(mask) - n_inside
print(f"\nListings inside circle                    : {n_inside}")
print(f"Listings outside circle                   : {n_outside}")

# --------------------------
# 5. DOM Level Distribution 
# --------------------------
print("\nDOM level distribution inside circle:")
level_counts = df.loc[mask, "dom_level"].value_counts(normalize=True).sort_index()
for level, pct in level_counts.items():
    label = {
        0: "0 (Fast)",
        1: "1 (Moderate)",
        2: "2 (Slow)",
        3: "3 (Very slow)"
    }.get(level, str(level))
    print(f"{label:<20}: {pct:.1%}")