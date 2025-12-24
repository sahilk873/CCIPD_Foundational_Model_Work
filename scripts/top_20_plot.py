#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# -------------------------
# CONFIG — update these
# -------------------------
SHAP_CACHE_DIR = "shap_cache"
OUT_FIG = "top20_feature_importance_remade.png"

# These must match your training setup
dim_musk  = 1024
dim_hopt  = 1536
dim_conch = 512

topk = 20

# -------------------------
# Load SHAP values
# -------------------------
shap_vals = np.load(os.path.join(SHAP_CACHE_DIR, "shap_values.npy"))

# If interaction-shaped, collapse to main effects
if shap_vals.ndim == 3:
    print("[INFO] Collapsing interaction SHAP values")
    shap_vals = shap_vals.mean(axis=2)

abs_shap = np.abs(shap_vals)

# -------------------------
# Compute feature importance
# -------------------------
feat_imp = abs_shap.mean(axis=0)
D = feat_imp.shape[0]
topk_eff = min(topk, D)

idx = np.argsort(feat_imp)[-topk_eff:][::-1]
idx = [int(i) for i in idx]
vals = feat_imp[idx]

# -------------------------
# Determine FM source
# -------------------------
b0 = dim_musk
b1 = dim_musk + dim_hopt

sources = []
labels = []

for i in idx:
    if i < b0:
        sources.append("MUSK")
        labels.append(f"MUSK[{i}]")
    elif i < b1:
        sources.append("H-Hoptimus-1")
        labels.append(f"HOPT[{i - b0}]")
    else:
        sources.append("CONCHv1_5")
        labels.append(f"CONCH[{i - b1}]")

colors = {
    "MUSK": "#1f77b4",
    "H-Hoptimus-1": "#ff7f0e",
    "CONCHv1_5": "#2ca02c",
}

# -------------------------
# Plot
# -------------------------
plt.figure(figsize=(12, 7))
y = np.arange(topk_eff)

plt.barh(y, vals, color=[colors[s] for s in sources])

# Bold y-axis tick labels
plt.yticks(y, labels, fontweight="bold")

# Bold x-axis label
plt.xlabel(
    "Mean SHAP Value Importance",
    fontweight="bold"
)

# Bold title
plt.title(
    f"Top {topk_eff} Individual Feature Importances",
    fontweight="bold"
)

plt.gca().invert_yaxis()

# Bold x-axis tick labels (THIS is what was missing)
for tick in plt.gca().get_xticklabels():
    tick.set_fontweight("bold")

# Bold legend text
plt.legend(
    handles=[Patch(color=v, label=k) for k, v in colors.items()],
    loc="lower right",
    prop={"weight": "bold"}
)

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
plt.close()

print(f"[DONE] Saved {OUT_FIG}")
