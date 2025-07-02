#!/bin/bash
#SBATCH --job-name=trident_gpu
#SBATCH --output=trident_%j.out
#SBATCH --error=trident_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# Load Python module and activate virtual environment
module load Python/3.11.5-GCCcore-13.2.0
source /scratch/users/sxk2517/trident_env/bin/activate
#python zenodo_uploader.py

#cd /scratch/users/sxk2517/TRIDENT

export HUGGINGFACE_TOKEN="insert token here"

#python run_batch_of_slides.py --task feat --wsi_dir /scratch/users/sxk2517/WM4237 --job_dir /scratch/users/sxk2517/trident_processed --patch_encoder musk --mag 20 --patch_size 256 --gpu 0 --max_workers 2 --feat_batch_size 64

#python run_batch_of_slides.py --task feat --wsi_dir /scratch/users/sxk2517/WM4237 --job_dir /scratch/users/sxk2517/trident_processed --patch_encoder hoptimus1 --mag 20 --patch_size 256 --gpu 0 --max_workers 2 --feat_batch_size 64

#python run_batch_of_slides.py --task all --wsi_dir /scratch/users/sxk2517/WM4237 --job_dir /scratch/users/sxk2517/trident_processed --segmenter hest --mag 20 --patch_size 512 --gpu 0 --max_workers 12

#python run_batch_of_slides.py --task all --wsi_dir /scratch/users/sxk2517/WM4237 --job_dir /scratch/users/sxk2517/trident_processed --segmenter hest --mag 20 --patch_size 256 --gpu 0 --max_workers 12

cd /scratch/users/sxk2517/scripts

#python mlp_wm_classifier.py

#python ensemble_mlp_classifier.py --musk /scratch/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_musk --hopt /scratch/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_hoptimus1 --conch /scratch/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_conch_v15 --epochs 30 --batch_size 256 --lr 0.001

python shap_quick.py --cached_data /scratch/users/sxk2517/cached_dataset.pt --model_path /scratch/users/sxk2517/best_model.pt --load_model --shap_only

