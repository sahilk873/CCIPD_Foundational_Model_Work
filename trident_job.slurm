#!/bin/bash
#SBATCH --job-name=trident_gpu
#SBATCH --output=test%j.out
#SBATCH --error=test%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# If your site supports requeueing, this helps you recover from bad nodes
#SBATCH --requeue

set -euo pipefail

# ---- Environment ----
source /scratch/pioneer/users/sxk2517/trident_env/bin/activate
export HUGGINGFACE_TOKEN=""

# Make CUDA errors show at the correct line (helpful for diagnosing)
export CUDA_LAUNCH_BLOCKING=1

# Optional: reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ---- Logging (super useful when ECC happens) ----
echo "=== SLURM JOB INFO ==="
echo "JobID: ${SLURM_JOB_ID}"
echo "Node:  ${SLURM_NODELIST}"
echo "GPUs:  ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Date:  $(date)"
echo "======================"

echo "=== NVIDIA-SMI (pre) ==="
nvidia-smi || true
echo "========================"

# ---- Run ----
cd /scratch/pioneer/users/sxk2517/scripts

python attention_from_h5.py \
  --h5 /scratch/pioneer/users/sxk2517/images/TCGA-KL-8323-01Z-00-DX1.01d72f6a-cc87-4082-8af4-738250ec4d9c_patches_images.h5 \
  --out-dir /scratch/pioneer/users/sxk2517/hoptimus_attn_maps \
  --model hoptimus \
  --max-tiles 100 \
  --layer-idx -1

echo "=== NVIDIA-SMI (post) ==="
nvidia-smi || true
echo "========================="

#python total_overlay.py --hopt_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_hoptimus1 --musk_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_musk --conch_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_conch_v15 --slides_dir /scratch/pioneer/users/sxk2517/trident_processed/thumbnails --hopt_model /scratch/pioneer/users/sxk2517/model_weights/hoptimus_base.pth --musk_model /scratch/pioneer/users/sxk2517/model_weights/musk_base.pth --conch_model /scratch/pioneer/users/sxk2517/model_weights/conch_base.pth --ensemble_model  /scratch/pioneer/users/sxk2517/model_weights/fused_all.pth --save_dir /scratch/pioneer/users/sxk2517/overlays_total --max_overlays 150
#python fusion_model.py --dirs /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_conch_v15 /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_musk /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_hoptimus1 --epochs 30 --batch_size 256 --lr 1e-4 --save_path fused_all.pth

#python fusion_model.py --dirs /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_conch_v15 /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_musk /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_hoptimus1 --epochs 30 --batch_size 256 --lr 1e-4 --save_path fused_all.pth


#python append_labels_to_h5_batch.py --h5_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_conch_v15 --xml_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology-Segmentations --patch_size 512
#python append_labels_to_h5_batch.py --h5_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_musk --xml_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology-Segmentations --patch_size 512
#python append_labels_to_h5_batch.py --h5_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_hoptimus1 --xml_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology-Segmentations --patch_size 512
#python mlp_classifier_from_h5.py --h5 /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_conch_v15 --epochs 40 --batch_size 256 --lr 0.002 --save_path /scratch/pioneer/users/sxk2517/model_weights/conch_base.pth
#python mlp_classifier_from_h5.py --h5 /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_musk --epochs 40 --batch_size 256 --lr 0.002 --save_path /scratch/pioneer/users/sxk2517/model_weights/musk_base.pth
#python mlp_classifier_from_h5.py --h5 /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_hoptimus1 --epochs 40 --batch_size 256 --lr 0.002 --save_path /scratch/pioneer/users/sxk2517/model_weights/hoptimus_base.pth

#python majority_vote_classifier.py --dirs /scratch/pioneer/users/sxk2517/trident_processed/5x_512px_0px_overlap/features_conch_v15 /scratch/pioneer/users/sxk2517/trident_processed/10x_1024px_0px_overlap/features_conch_v15 /scratch/pioneer/users/sxk2517/trident_processed/20x_2048px_0px_overlap/features_conch_v15
#python majority_vote_classifier.py --dirs /scratch/pioneer/users/sxk2517/trident_processed/5x_512px_0px_overlap/features_musk /scratch/pioneer/users/sxk2517/trident_processed/10x_1024px_0px_overlap/features_musk /scratch/pioneer/users/sxk2517/trident_processed/20x_2048px_0px_overlap/features_musk
#python majority_vote_classifier.py --dirs /scratch/pioneer/users/sxk2517/trident_processed/5x_512px_0px_overlap/features_hoptimus1 /scratch/pioneer/users/sxk2517/trident_processed/10x_1024px_0px_overlap/features_hoptimus1 /scratch/pioneer/users/sxk2517/trident_processed/20x_2048px_0px_overlap/features_hoptimus1



#python total_overlay.py --musk_dir /scratch/pioneer/users/sxk2517/trident_processed/z20x_512px_0px_overlap_old/features_musk_kich --hopt_dir /scratch/pioneer/users/sxk2517/trident_processed/z20x_512px_0px_overlap_old/features_hoptimus1_kich --conch_dir /scratch/pioneer/users/sxk2517/trident_processed/z20x_512px_0px_overlap_old/features_conch_v15_kich --slides_dir /scratch/pioneer/users/sxk2517/trident_processed/thumbnails --musk_model /scratch/pioneer/users/sxk2517/model_weights/musk_kich_mlp.pth --ensemble_model /scratch/pioneer/users/sxk2517/model_weights/ensemble_kich_mlp.pth --tile_size 512 --save_dir /scratch/pioneer/users/sxk2517/overlays_top_musk --rank_by tp_delta --top_k 10 --device cuda && python overlay_top.py --hopt_dir /scratch/pioneer/users/sxk2517/trident_processed/z20x_512px_0px_overlap_old/features_hoptimus1_kich --musk_dir /scratch/pioneer/users/sxk2517/trident_processed/z20x_512px_0px_overlap_old/features_musk_kich --conch_dir /scratch/pioneer/users/sxk2517/trident_processed/z20x_512px_0px_overlap_old/features_conch_v15_kich --slides_dir /scratch/pioneer/users/sxk2517/trident_processed/thumbnails --hopt_model /scratch/pioneer/users/sxk2517/model_weights/hoptimus_kich_mlp.pth --ensemble_model /scratch/pioneer/users/sxk2517/model_weights/ensemble_kich_mlp.pth --tile_size 512 --save_dir /scratch/pioneer/users/sxk2517/overlays_top_hoptimus1 --rank_by tp_delta --top_k 3 --device cuda --debug


#python super_majority_vote.py --dirs /scratch/pioneer/users/sxk2517/trident_processed/5x_512px_0px_overlap/features_conch_v15 /scratch/pioneer/users/sxk2517/trident_processed/5x_512px_0px_overlap/features_musk /scratch/pioneer/users/sxk2517/trident_processed/5x_512px_0px_overlap/features_hoptimus1 /scratch/pioneer/users/sxk2517/trident_processed/10x_1024px_0px_overlap/features_conch_v15 /scratch/pioneer/users/sxk2517/trident_processed/10x_1024px_0px_overlap/features_musk /scratch/pioneer/users/sxk2517/trident_processed/10x_1024px_0px_overlap/features_hoptimus1 /scratch/pioneer/users/sxk2517/trident_processed/20x_2048px_0px_overlap/features_conch_v15 /scratch/pioneer/users/sxk2517/trident_processed/20x_2048px_0px_overlap/features_musk /scratch/pioneer/users/sxk2517/trident_processed/20x_2048px_0px_overlap/features_hoptimus1



#python majority_vote_classifier.py --dirs /scratch/pioneer/users/sxk2517/trident_processed/5x_512px_0px_overlap/features_conch_v15 /scratch/pioneer/users/sxk2517/trident_processed/5x_512px_0px_overlap/features_musk /scratch/pioneer/users/sxk2517/trident_processed/5x_512px_0px_overlap/features_hoptimus1
#python majority_vote_classifier.py --dirs /scratch/pioneer/users/sxk2517/trident_processed/10x_1024px_0px_overlap/features_conch_v15 /scratch/pioneer/users/sxk2517/trident_processed/10x_1024px_0px_overlap/features_musk /scratch/pioneer/users/sxk2517/trident_processed/10x_1024px_0px_overlap/features_hoptimus1
#python majority_vote_classifier.py --dirs /scratch/pioneer/users/sxk2517/trident_processed/20x_2048px_0px_overlap/features_conch_v15 /scratch/pioneer/users/sxk2517/trident_processed/20x_2048px_0px_overlap/features_musk /scratch/pioneer/users/sxk2517/trident_processed/20x_2048px_0px_overlap/features_hoptimus1

# Conch
#python fusion_model.py --dirs /scratch/pioneer/users/sxk2517/trident_processed/5x_512px_0px_overlap/features_conch_v15 /scratch/pioneer/users/sxk2517/trident_processed/10x_1024px_0px_overlap/features_conch_v15 /scratch/pioneer/users/sxk2517/trident_processed/20x_2048px_0px_overlap/features_conch_v15 --epochs 30 --batch_size 256 --lr 1e-4 --save_path fused_conch_all.pth

# MUSK
#python fusion_model.py --dirs /scratch/pioneer/users/sxk2517/trident_processed/5x_512px_0px_overlap/features_musk /scratch/pioneer/users/sxk2517/trident_processed/10x_1024px_0px_overlap/features_musk /scratch/pioneer/users/sxk2517/trident_processed/20x_2048px_0px_overlap/features_musk --epochs 30 --batch_size 256 --lr 1e-4 --save_path fused_musk_all.pth

# HOPTIMUS
#python fusion_model.py --dirs /scratch/pioneer/users/sxk2517/trident_processed/5x_512px_0px_overlap/features_hoptimus1 /scratch/pioneer/users/sxk2517/trident_processed/10x_1024px_0px_overlap/features_hoptimus1 /scratch/pioneer/users/sxk2517/trident_processed/20x_2048px_0px_overlap/features_hoptimus1 --epochs 30 --batch_size 256 --lr 1e-4 --save_path fused_hoptimus_all.pth


#python fusion_model.py --dirs /scratch/pioneer/users/sxk2517/trident_processed/5x_512px_0px_overlap/features_conch_v15 /scratch/pioneer/users/sxk2517/trident_processed/5x_512px_0px_overlap/features_musk /scratch/pioneer/users/sxk2517/trident_processed/5x_512px_0px_overlap/features_hoptimus1 --epochs 30 --batch_size 256 --lr 1e-4 --save_path fused_model_5x.pth
#python fusion_model.py --dirs /scratch/pioneer/users/sxk2517/trident_processed/10x_1024px_0px_overlap/features_conch_v15 /scratch/pioneer/users/sxk2517/trident_processed/10x_1024px_0px_overlap/features_musk /scratch/pioneer/users/sxk2517/trident_processed/10x_1024px_0px_overlap/features_hoptimus1 --epochs 30 --batch_size 256 --lr 1e-4 --save_path fused_model_10x.pth
#python fusion_model.py --dirs /scratch/pioneer/users/sxk2517/trident_processed/20x_2048px_0px_overlap/features_conch_v15 /scratch/pioneer/users/sxk2517/trident_processed/20x_2048px_0px_overlap/features_musk /scratch/pioneer/users/sxk2517/trident_processed/20x_2048px_0px_overlap/features_hoptimus1 --epochs 30 --batch_size 256 --lr 1e-4 --save_path fused_model_20x.pth

#python overlay_top.py --musk_dir /scratch/pioneer/users/sxk2517/trident_processed/z20x_512px_0px_overlap_old/features_musk_kich --hopt_dir /scratch/pioneer/users/sxk2517/trident_processed/z20x_512px_0px_overlap_old/features_hoptimus1_kich --conch_dir /scratch/pioneer/users/sxk2517/trident_processed/z20x_512px_0px_overlap_old/features_conch_v15_kich --slides_dir /scratch/pioneer/users/sxk2517/trident_processed/thumbnails --musk_model /scratch/pioneer/users/sxk2517/model_weights/musk_kich_mlp.pth --ensemble_model /scratch/pioneer/users/sxk2517/model_weights/ensemble_kich_mlp.pth --tile_size 512 --save_dir /scratch/pioneer/users/sxk2517/overlays_top_musk --rank_by tp_delta --top_k 10 --device cuda
#python overlay_top.py --hopt_dir /scratch/pioneer/users/sxk2517/trident_processed/z20x_512px_0px_overlap_old/features_hoptimus1_kich --musk_dir /scratch/pioneer/users/sxk2517/trident_processed/z20x_512px_0px_overlap_old/features_musk_kich --conch_dir /scratch/pioneer/users/sxk2517/trident_processed/z20x_512px_0px_overlap_old/features_conch_v15_kich --slides_dir /scratch/pioneer/users/sxk2517/trident_processed/thumbnails --hopt_model /scratch/pioneer/users/sxk2517/model_weights/hoptimus_kich_mlp.pth --ensemble_model /scratch/pioneer/users/sxk2517/model_weights/ensemble_kich_mlp.pth --tile_size 512 --save_dir /scratch/pioneer/users/sxk2517/overlays_top_hoptimus1 --rank_by tp_delta --top_k 3 --device cuda --debug

#python mlp_classifier_from_h5.py --h5 /scratch/pioneer/users/sxk2517/trident_processed/5x_512px_0px_overlap/features_conch_v15 --epochs 40 --batch_size 256 --lr 0.002 --save_path /scratch/pioneer/users/sxk2517/model_weights/conch_5x.pth
#python mlp_classifier_from_h5.py --h5 /scratch/pioneer/users/sxk2517/trident_processed/10x_1024px_0px_overlap/features_conch_v15 --epochs 40 --batch_size 256 --lr 0.002 --save_path /scratch/pioneer/users/sxk2517/model_weights/conch_10x.pth
#python mlp_classifier_from_h5.py --h5 /scratch/pioneer/users/sxk2517/trident_processed/20x_2048px_0px_overlap/features_conch_v15 --epochs 40 --batch_size 256 --lr 0.002 --save_path /scratch/pioneer/users/sxk2517/model_weights/conch_20x.pth

#python mlp_classifier_from_h5.py --h5 /scratch/pioneer/users/sxk2517/trident_processed/5x_512px_0px_overlap/features_musk --epochs 40 --batch_size 256 --lr 0.002 --save_path /scratch/pioneer/users/sxk2517/model_weights/musk_5x.pth
#python mlp_classifier_from_h5.py --h5 /scratch/pioneer/users/sxk2517/trident_processed/10x_1024px_0px_overlap/features_musk --epochs 40 --batch_size 256 --lr 0.002 --save_path /scratch/pioneer/users/sxk2517/model_weights/musk_10x.pth
#python mlp_classifier_from_h5.py --h5 /scratch/pioneer/users/sxk2517/trident_processed/20x_2048px_0px_overlap/features_musk --epochs 40 --batch_size 256 --lr 0.002 --save_path /scratch/pioneer/users/sxk2517/model_weights/musk_20x.pth

#python mlp_classifier_from_h5.py --h5 /scratch/pioneer/users/sxk2517/trident_processed/5x_512px_0px_overlap/features_hoptimus1 --epochs 40 --batch_size 256 --lr 0.002 --save_path /scratch/pioneer/users/sxk2517/model_weights/hoptimus_5x.pth
#python mlp_classifier_from_h5.py --h5 /scratch/pioneer/users/sxk2517/trident_processed/10x_1024px_0px_overlap/features_hoptimus1 --epochs 40 --batch_size 256 --lr 0.002 --save_path /scratch/pioneer/users/sxk2517/model_weights/hoptimus_10x.pth
#python mlp_classifier_from_h5.py --h5 /scratch/pioneer/users/sxk2517/trident_processed/20x_2048px_0px_overlap/features_hoptimus1 --epochs 40 --batch_size 256 --lr 0.002 --save_path /scratch/pioneer/users/sxk2517/model_weights/hoptimus_20x.pth


#python append_labels_to_h5_batch.py --h5_dir /scratch/pioneer/users/sxk2517/trident_processed/5x_512px_0px_overlap/features_conch_v15 --xml_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology-Segmentations --patch_size 512
#python append_labels_to_h5_batch.py --h5_dir /scratch/pioneer/users/sxk2517/trident_processed/5x_512px_0px_overlap/features_hoptimus1 --xml_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology-Segmentations --patch_size 512
#python append_labels_to_h5_batch.py --h5_dir /scratch/pioneer/users/sxk2517/trident_processed/5x_512px_0px_overlap/features_musk --xml_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology-Segmentations --patch_size 512


#python append_labels_to_h5_batch.py --h5_dir /scratch/pioneer/users/sxk2517/trident_processed/10x_1024px_0px_overlap/features_hoptimus1 --xml_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology-Segmentations --patch_size 512
#python append_labels_to_h5_batch.py --h5_dir /scratch/pioneer/users/sxk2517/trident_processed/10x_1024px_0px_overlap/features_musk --xml_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology-Segmentations --patch_size 512

#python append_labels_to_h5_batch.py --h5_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_2048px_0px_overlap/features_conch_v15 --xml_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology-Segmentations --patch_size 512
#python append_labels_to_h5_batch.py --h5_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_2048px_0px_overlap/features_hoptimus1 --xml_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology-Segmentations --patch_size 512
#python append_labels_to_h5_batch.py --h5_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_2048px_0px_overlap/features_musk --xml_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology-Segmentations --patch_size 512

#cd ..
#cd trident
#python run_batch_of_slides.py --task all --wsi_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology --job_dir /scratch/pioneer/users/sxk2517/trident_processed --patch_encoder conch_v15 --mag 10 --patch_size 1024 --gpu 0 --max_workers 16 --feat_batch_size 64

#cd ..
#cd scripts

#python append_labels_to_h5_batch.py --h5_dir /scratch/pioneer/users/sxk2517/trident_processed/10x_1024px_0px_overlap/features_conch_v15 --xml_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology-Segmentations --patch_size 512





#python append_labels_to_h5_batch.py --h5_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_224px_0px_overlap/features_hoptimus1 --xml_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology-Segmentations --patch_size 224
#python append_labels_to_h5_batch.py --h5_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_280px_0px_overlap/features_hoptimus1 --xml_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology-Segmentations --patch_size 280
#python append_labels_to_h5_batch.py --h5_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_336px_0px_overlap/features_hoptimus1 --xml_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology-Segmentations --patch_size 336
#python append_labels_to_h5_batch.py --h5_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_392px_0px_overlap/features_hoptimus1 --xml_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology-Segmentations --patch_size 392
#python append_labels_to_h5_batch.py --h5_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_448px_0px_overlap/features_hoptimus1 --xml_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology-Segmentations --patch_size 448

#python append_labels_to_h5_batch.py --h5_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_384px_0px_overlap/features_musk --xml_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology-Segmentations --patch_size 384
#python append_labels_to_h5_batch.py --h5_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_480px_0px_overlap/features_musk --xml_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology-Segmentations --patch_size 480
#python append_labels_to_h5_batch.py --h5_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_576px_0px_overlap/features_musk --xml_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology-Segmentations --patch_size 576
#python append_labels_to_h5_batch.py --h5_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_672px_0px_overlap/features_musk --xml_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology-Segmentations --patch_size 672
#python append_labels_to_h5_batch.py --h5_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_768px_0px_overlap/features_musk --xml_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology-Segmentations --patch_size 768

#python append_labels_to_h5_batch.py --h5_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_conch_v15 --xml_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology-Segmentations --patch_size 512
#python append_labels_to_h5_batch.py --h5_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_640px_0px_overlap/features_conch_v15 --xml_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology-Segmentations --patch_size 640
#python append_labels_to_h5_batch.py --h5_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_768px_0px_overlap/features_conch_v15 --xml_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology-Segmentations --patch_size 768
#python append_labels_to_h5_batch.py --h5_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_896px_0px_overlap/features_conch_v15 --xml_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology-Segmentations --patch_size 896
#python append_labels_to_h5_batch.py --h5_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_1024px_0px_overlap/features_conch_v15 --xml_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology-Segmentations --patch_size 1024




#python run_batch_of_slides.py --task all --wsi_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology --job_dir /scratch/pioneer/users/sxk2517/trident_processed --patch_encoder musk --mag 5 --patch_size 512 --gpu 0 --max_workers 16 --feat_batch_size 64
#python run_batch_of_slides.py --task all --wsi_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology --job_dir /scratch/pioneer/users/sxk2517/trident_processed --patch_encoder musk --mag 10 --patch_size 1024 --gpu 0 --max_workers 16 --feat_batch_size 64
#python run_batch_of_slides.py --task all --wsi_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology --job_dir /scratch/pioneer/users/sxk2517/trident_processed --patch_encoder musk --mag 20 --patch_size 2048 --gpu 0 --max_workers 16 --feat_batch_size 64
#python run_batch_of_slides.py --task all --wsi_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology --job_dir /scratch/pioneer/users/sxk2517/trident_processed --patch_encoder musk --mag 20 --patch_size 672 --gpu 0 --max_workers 16 --feat_batch_size 64
#python run_batch_of_slides.py --task all --wsi_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology --job_dir /scratch/pioneer/users/sxk2517/trident_processed --patch_encoder musk --mag 20 --patch_size 768 --gpu 0 --max_workers 16 --feat_batch_size 64

#python run_batch_of_slides.py --task all --wsi_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology --job_dir /scratch/pioneer/users/sxk2517/trident_processed --patch_encoder hoptimus1 --mag 5 --patch_size 512 --gpu 0 --max_workers 16 --feat_batch_size 64
#python run_batch_of_slides.py --task all --wsi_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology --job_dir /scratch/pioneer/users/sxk2517/trident_processed --patch_encoder hoptimus1 --mag 10 --patch_size 1024 --gpu 0 --max_workers 16 --feat_batch_size 64
#python run_batch_of_slides.py --task all --wsi_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology --job_dir /scratch/pioneer/users/sxk2517/trident_processed --patch_encoder hoptimus1 --mag 20 --patch_size 2048 --gpu 0 --max_workers 16 --feat_batch_size 64
#python run_batch_of_slides.py --task all --wsi_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology --job_dir /scratch/pioneer/users/sxk2517/trident_processed --patch_encoder hoptimus1 --mag 20 --patch_size 392 --gpu 0 --max_workers 16 --feat_batch_size 64
#python run_batch_of_slides.py --task all --wsi_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology --job_dir /scratch/pioneer/users/sxk2517/trident_processed --patch_encoder hoptimus1 --mag 20 --patch_size 448 --gpu 0 --max_workers 16 --feat_batch_size 64

#python run_batch_of_slides.py --task all --wsi_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology --job_dir /scratch/pioneer/users/sxk2517/trident_processed --patch_encoder conch_v15 --mag 5 --patch_size 512 --gpu 0 --max_workers 16 --feat_batch_size 64
#python run_batch_of_slides.py --task all --wsi_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology --job_dir /scratch/pioneer/users/sxk2517/trident_processed --patch_encoder conch_v15 --mag 10 --patch_size 1048 --gpu 0 --max_workers 16 --feat_batch_size 64
#python run_batch_of_slides.py --task all --wsi_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology --job_dir /scratch/pioneer/users/sxk2517/trident_processed --patch_encoder conch_v15 --mag 20 --patch_size 2048 --gpu 0 --max_workers 16 --feat_batch_size 64
#python run_batch_of_slides.py --task all --wsi_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology --job_dir /scratch/pioneer/users/sxk2517/trident_processed --patch_encoder conch_v15 --mag 20 --patch_size 896 --gpu 0 --max_workers 16 --feat_batch_size 64
#python run_batch_of_slides.py --task all --wsi_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology --job_dir /scratch/pioneer/users/sxk2517/trident_processed --patch_encoder conch_v15 --mag 20 --patch_size 1024 --gpu 0 --max_workers 16 --feat_batch_size 64


#python fairness_concatenated_test.py --h5_dir1 /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/merged_conch --h5_dir2 /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/merged_musk --h5_dir3 /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/merged_hoptimus1 --model_path /scratch/pioneer/users/sxk2517/model_weights/concatenated_all.pth --out_csv /scratch/pioneer/users/sxk2517/fairness_evals/concatenated_all.csv

#python attention_from_h5.py

#python train_lora_from_lmdb.py --lmdb_path /scratch/pioneer/users/sxk2517/kich_patches.lmdb --out_dir /scratch/pioneer/users/sxk2517/lora_output

#python train_lora_from_lmdb.py --lmdb_path /scratch/pioneer/users/sxk2517/kich_patches.lmdb --out_dir /scratch/pioneer/users/sxk2517/trial_lora --batch_size 8 


#python convert_h5_to_ldmb.py --h5_dir /scratch/pioneer/users/sxk2517/kich_patch_images --lmdb_out /scratch/pioneer/users/sxk2517/kich_patches.lmdb --map_size_gb 500

#python make_patches_from_coords.py --coords_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/patches --svs_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology --xml_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology-Segmentations --out_dir /scratch/pioneer/users/sxk2517/kich_patch_images --patch_size 512

#python fairness_voting.py --conch_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/merged_conch --musk_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/merged_musk --hoptimus_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/merged_hoptimus1 --conch_model /scratch/pioneer/users/sxk2517/model_weights/conch_all.pth --musk_model /scratch/pioneer/users/sxk2517/model_weights/musk_all.pth --hoptimus_model /scratch/pioneer/users/sxk2517/model_weights/hoptimus_all.pth --out_csv /scratch/pioneer/users/sxk2517/fairness_evals/voting_all.csv

#python fairness_concatenated.py --dir1 /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/merged_conch --dir2 /scratch/pioneer/users/sxk2517/tridssssent_processed/20x_512px_0px_overlap/merged_musk --dir3 /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/merged_hoptimus1 --epochs 30 --save_path /scratch/pioneer/users/sxk2517/model_weights/concatenated_all.pth

#python evaluate.py --h5_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/trial --model_path /scratch/pioneer/users/sxk2517/model_weights/musk_kich_mlp.pth --batch_size 128

#python eval_demographics.py --h5_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_conch_v15 --model_path /scratch/pioneer/users/sxk2517/model_weights/conch_kich_mlp.pth --out_csv /scratch/pioneer/users/sxk2517/fairness_evals/conch_eval_single.csv

#python eval_demographics.py --h5_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_hoptimus1 --model_path /scratch/pioneer/users/sxk2517/model_weights/hoptimus_kich_mlp.pth --out_csv /scratch/pioneer/users/sxk2517/fairness_evals/hoptimus_eval_single.csv

#python ensemble_mlp_classifier.py --musk /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_musk_kich --hopt /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_hoptimus1_kich --conch /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_conch_v15_kich --epochs 30 --batch_size 128 --lr 0.002 --save_path /scratch/pioneer/users/sxk2517/model_weights/ensemble_kich_mlp.pth

#python overlay.py --conch_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_conch_v15_kich --musk_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_musk_kich --hopt_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_hoptimus1_kich --slides_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/visualization --conch_model /scratch/pioneer/users/sxk2517/model_weights/conch_kich_mlp.pth --ensemble_model /scratch/pioneer/users/sxk2517/model_weights/ensemble_kich_mlp.pth --tile_size 512 --save_dir /scratch/pioneer/users/sxk2517/overlays --max_overlays 5 --device cuda --debug

#python overlay_top.py --conch_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_conch_v15_kich --musk_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_musk_kich --hopt_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_hoptimus1_kich --slides_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/visualization --conch_model /scratch/pioneer/users/sxk2517/model_weights/conch_kich_mlp.pth --ensemble_model /scratch/pioneer/users/sxk2517/model_weights/ensemble_kich_mlp.pth --tile_size 512 --save_dir /scratch/pioneer/users/sxk2517/overlays_top --top_k 3 --device cuda

#python overlay.py --conch_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_conch_v15_kich --musk_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_musk_kich --hopt_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_hoptimus1_kich --slides_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/visualization --conch_model /scratch/pioneer/users/sxk2517/model_weights/conch_kich_mlp.pth --ensemble_model /scratch/pioneer/users/sxk2517/model_weights/ensemble_kich_mlp.pth --tile_size 512 --save_dir /scratch/pioneer/users/sxk2517/overlays --rank_by tp_ensemble --top_k 3 --device cuda

#python overlay_top.py --conch_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_conch_v15_kich --musk_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_musk_kich --hopt_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_hoptimus1_kich --slides_dir /scratch/pioneer/users/sxk2517/trident_processed/thumbnails --conch_model /scratch/pioneer/users/sxk2517/model_weights/conch_kich_mlp.pth --ensemble_model /scratch/pioneer/users/sxk2517/model_weights/ensemble_kich_mlp.pth --tile_size 512 --coords_mode center --alpha 0.5 --save_dir /scratch/pioneer/users/sxk2517/overlays_TPR --rank_by tp_delta --top_k 3 --device cuda --debug

#python fairness_ensemble.py --dir1 /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_hoptimus1 --dir2 /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_hoptimus1_kich --epochs 30 --batch_size 64 --lr 0.001 --save_path /scratch/pioneer/users/sxk2517/model_weights/hoptimus_all.pth

#python fairness_ensemble_test.py --h5_dir1 /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_hoptimus1 --h5_dir2 /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_hoptimus1_kich --model_path /scratch/pioneer/users/sxk2517/model_weights/hoptimus_all.pth --out_csv /scratch/pioneer/users/sxk2517/fairness_evals/hoptimus_all.csv
#python fairness_ensemble_test.py --h5_dir1 /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_conch_v15 --h5_dir2 /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_conch_v15_kich --model_path /scratch/pioneer/users/sxk2517/model_weights/conch_all.pth --out_csv /scratch/pioneer/users/sxk2517/fairness_evals/conch_all.csv
#python fairness_ensemble_test.py --h5_dir1 /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_musk --h5_dir2 /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_musk_kich --model_path /scratch/pioneer/users/sxk2517/model_weights/musk_all.pth --out_csv /scratch/pioneer/users/sxk2517/fairness_evals/musk_all.csv

#python fairness_ensemble_train.py --groupA /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_conch_v15 /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_hoptimus1 /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_musk --groupB /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_conch_v15_kich /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_hoptimus1_kich /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_musk_kich --epochs 30 --batch_size 64 --lr 0.001 --save_path /scratch/pioneer/users/sxk2517/model_weights/ensemble_all.pth


#python append_labels_to_h5_batch.py --h5_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_conch_v15_kich --xml_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology-Segmentations --patch_size 512

#python append_labels_to_h5_batch.py --h5_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_hoptimus1_kich --xml_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology-Segmentations --patch_size 512

#python append_labels_to_h5_batch.py --h5_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_musk_kich --xml_dir /scratch/pioneer/users/sxk2517/TCGA-KICH-Pathology-Segmentations --patch_size 512

#python run_batch_of_slides.py --task feat --wsi_dir /scratch/users/sxk2517/WM4237 --job_dir /scratch/users/sxk2517/trident_processed --patch_encoder hoptimus1 --mag 20 --patch_size 256 --gpu 0 --max_workers 2 --feat_batch_size 64

#python run_batch_of_slides.py --task all --wsi_dir /scratch/users/sxk2517/WM4237 --job_dir /scratch/users/sxk2517/trident_processed --segmenter hest --mag 20 --patch_size 512 --gpu 0 --max_workers 12

#python run_batch_of_slides.py --task all --wsi_dir /scratch/users/sxk2517/WM4237 --job_dir /scratch/users/sxk2517/trident_processed --segmenter hest --mag 20 --patch_size 256 --gpu 0 --max_workers 12

#cd /scratch/pioneer/users/sxk2517/scripts

#python majority_vote_mlp.py

#python wm_ensemble.py

#python wm_ensemble_slide.py

#python mlp_wm_classifier_slide.py

#python mlp_classifier_from_h5.py --h5 /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_conch_v15_kich --epochs 40 --batch_size 256 --lr 0.002 --save_path /scratch/pioneer/users/sxk2517/model_weights/conch_kich_mlp.pth

#python mlp_classifier_from_h5.py --h5 /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_hoptimus1_kich --epochs 40 --batch_size 256 --lr 0.002 --save_path /scratch/pioneer/users/sxk2517/model_weights/hoptimus_kich_mlp.pth

#python mlp_classifier_from_h5.py --h5 /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_musk_kich --epochs 40 --batch_size 256 --lr 0.002 --save_path /scratch/pioneer/users/sxk2517/model_weights/musk_kich_mlp.pth

#python mlp_classifier_from_h5.py --h5 "/scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_hoptimus1" --epochs 40 --batch_size 256 --lr 0.002

#python mlp_classifier_from_h5.py --h5 /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_musk --epochs 40 --batch_size 256 --lr 0.002

#python ensemble_mlp_per_slide.py --musk /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_musk --hopt /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_hoptimus1 --conch /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_conch_v15 --holdout_json /scratch/pioneer/users/sxk2517/holdout_slides.json --model_path "/scratch/pioneer/users/sxk2517/trained_model.pth" --epochs 50 --batch_size 256 --lr 0.001

#python shap_quick.py --cached_data /scratch/pioneer/users/sxk2517/cached_dataset.pt --model_path /scratch/pioneer/users/sxk2517/best_model.pt --load_model --shap_only

#python shap_quick.py --musk /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_musk --hopt /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_hoptimus1 --conch /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_conch_v15 --cached_data /scratch/pioneer/users/sxk2517/cached_dataset.pt --model_path /scratch/pioneer/users/sxk2517/best_model.pt --cache_dir /scratch/pioneer/users/sxk2517/shap_cache

#python per_slide_eval.py --h5_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_conch_v15 --model_path /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_conch_v15/trained_model.pth --output_csv per_slide_metrics_conch.csv

#python per_slide_eval.py --h5_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_hoptimus1 --model_path /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_hoptimus1/trained_model.pth --output_csv per_slide_metrics_hoptimus.csv

#python per_slide_eval.py --h5_dir /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_musk --model_path /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_musk/trained_model.pth --output_csv /scratch/pioneer/users/sxk2517/per_slide_metrics_musk.csv --holdout_json /scratch/pioneer/users/sxk2517/holdout_slides.json

#python ensemble_per_slide_eval.py --musk /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_musk --hopt /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_hoptimus1 --conch /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_conch_v15 --holdout_json /scratch/pioneer/users/sxk2517/holdout_slides.json --model_path /scratch/pioneer/users/sxk2517/fusion_trained_model.pth --output_csv /scratch/pioneer/users/sxk2517/ensemble_per_slide_metrics.csv

#python majority_vote_per_slide.py --musk /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_musk --hopt /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_hoptimus1 --conch /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_conch_v15 --holdout_json /scratch/pioneer/users/sxk2517/holdout_slides.json --models /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_conch_v15/trained_model.pth /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_hoptimus1/trained_model.pth /scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_hoptimus1/trained_model.pth --output_csv /scratch/pioneer/users/sxk2517/scripts/majority_vote_results.csv
