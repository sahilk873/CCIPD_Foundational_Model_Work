#!/usr/bin/env bash
# Example: SHAP vs dimension stability for the fused CONCH (3 resolutions) model.
# Uses fused_conch_all.pth trained on 5x, 10x, 20x CONCH features.
# Cache is written to a dedicated dir so this run regenerates SHAP from scratch.

set -euo pipefail

WORKSPACE="/scratch/pioneer/users/sxk2517"
cd "$WORKSPACE"

# Fusion model checkpoint (train with fusion_model.py --dirs 5x 10x 20x conch --save_path fused_conch_all.pth)
MODEL="${WORKSPACE}/fused_conch_all.pth"

# Three CONCH resolution H5 dirs (same order as training)
DIR_5X="${WORKSPACE}/trident_processed/5x_512px_0px_overlap/features_conch_v15"
DIR_10X="${WORKSPACE}/trident_processed/10x_1024px_0px_overlap/features_conch_v15"
DIR_20X="${WORKSPACE}/trident_processed/20x_2048px_0px_overlap/features_conch_v15"

# Stability CSV from resolution_similarity_embeddings.py
STABILITY_CSV="${WORKSPACE}/resolution_similarity_out/per_dimension_stability.csv"

# Fresh cache for this job (no prior SHAP cache; will be created)
CACHE_DIR="${WORKSPACE}/shap_stability_cache_conch"
OUT_DIR="${WORKSPACE}/shap_stability_out_conch"

# Model names in same order as --dirs (all three blocks are conch_v15)
MODEL_NAMES="conch_v15 conch_v15 conch_v15"

if [[ ! -f "$MODEL" ]]; then
  echo "ERROR: Model not found: $MODEL"
  echo "Train first with:"
  echo "  python scripts/fusion_model.py --dirs $DIR_5X $DIR_10X $DIR_20X --epochs 30 --batch_size 256 --lr 1e-4 --save_path $MODEL"
  exit 1
fi

if [[ ! -d "$DIR_5X" || ! -d "$DIR_10X" || ! -d "$DIR_20X" ]]; then
  echo "ERROR: One or more H5 dirs missing. Check: $DIR_5X, $DIR_10X, $DIR_20X"
  exit 1
fi

if [[ ! -f "$STABILITY_CSV" ]]; then
  echo "ERROR: Stability CSV not found: $STABILITY_CSV"
  echo "Run resolution_similarity_embeddings.py first to generate per_dimension_stability.csv"
  exit 1
fi

echo "=== SHAP vs stability (fused CONCH) ==="
echo "Model:       $MODEL"
echo "Dirs:        $DIR_5X | $DIR_10X | $DIR_20X"
echo "Cache (new): $CACHE_DIR"
echo "Out:         $OUT_DIR"
echo "================================"

# Regenerate cache: use dedicated cache dir (empty or fresh)
# Optional: remove old cache to force full recompute
# rm -rf "$CACHE_DIR"
mkdir -p "$CACHE_DIR"

python scripts/shap_stability_correlation.py \
  --model "$MODEL" \
  --dirs "$DIR_5X" "$DIR_10X" "$DIR_20X" \
  --model_names $MODEL_NAMES \
  --stability_csv "$STABILITY_CSV" \
  --out_dir "$OUT_DIR" \
  --cache_dir "$CACHE_DIR" \
  --instability_col mean_mad \
  --background_size 50 \
  --to_explain_size 200 \
  --nsamples 500 \
  --save_csv

echo "Done. Plots and CSV in $OUT_DIR; SHAP cache in $CACHE_DIR"
