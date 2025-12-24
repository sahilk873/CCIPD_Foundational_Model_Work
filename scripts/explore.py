# explore.py
import h5py

path = "/scratch/pioneer/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_conch_v15/TCGA-KL-8323-01Z-00-DX1.01d72f6a-cc87-4082-8af4-738250ec4d9c.h5"
with h5py.File(path, "r") as f:
    print("Datasets:", list(f.keys()))

    # Count number of patches based on coords or labels
    num_patches = None
    if "coords" in f:
        coords = f["coords"]
        print("Shape of coords:", coords.shape)
        print("First few coords:")
        print(coords[:5])
        num_patches = coords.shape[0]

    if "labels" in f:
        labels = f["labels"]
        print("First few labels:")
        print(labels[:5])
        # Safety check — sometimes labels may differ in size
        if num_patches is None:
            num_patches = labels.shape[0]
        elif num_patches != labels.shape[0]:
            print(f"⚠️ Warning: coords and labels differ in count ({num_patches} vs {labels.shape[0]})")

    if num_patches is not None:
        print(f"\n✅ Total number of patches in file: {num_patches}")
    else:
        print("\n⚠️ Could not determine patch count — no coords or labels dataset found.")
