import h5py

path = "/scratch/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_musk/TCGA-KL-8326-01Z-00-DX1.0fe86143-728f-4447-98a5-4a6d87ef398b.h5"
with h5py.File(path, 'r') as f:
    print(list(f.keys()))
