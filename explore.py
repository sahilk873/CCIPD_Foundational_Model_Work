import h5py

# Replace with your actual path to a feature file
filepath = "/scratch/users/sxk2517/trident_processed/20x_512px_0px_overlap/features_conch_v15/TCGA-KL-8323-01Z-00-DX1.01d72f6a-cc87-4082-8af4-738250ec4d9c.h5"

with h5py.File(filepath, "r") as f:
    print("Top-level keys:", list(f.keys()))

    if "coords" in f:
        coords = f["coords"]
        print("\n[coords]")
        print("  Shape:", coords.shape)
        print("  Sample:", coords[0])

    if "features" in f:
        features = f["features"]
        print("\n[features]")
        print("  Shape:", features.shape)
        print("  Dtype:", features.dtype)
        print("  Sample vector:", features[0])

    if "labels" in f:
        labels = f["labels"]
        print("\n[labels]")
        print("  Shape:", labels.shape)
        print("  Values (first 10):", labels[:10])
    else:
        print("\n[labels] not found in the file.")
