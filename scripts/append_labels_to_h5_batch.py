import os
import h5py
import numpy as np
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon
from tqdm import tqdm
import argparse


def extract_tumor_polygons(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    polygons = []
    for region in root.findall('.//Region'):
        vertices = [(float(v.attrib['X']), float(v.attrib['Y'])) for v in region.findall('.//Vertex')]
        if len(vertices) >= 3:
            poly = Polygon(vertices)
            if poly.is_valid:
                polygons.append(poly)
    return polygons


def append_labels(h5_path, tumor_polygons, patch_size=512, label_dataset_name="labels"):
    with h5py.File(h5_path, "r+") as h5_file:
        coords = h5_file["coords"][:]
        labels = []
        for x, y in coords:
            patch_poly = Polygon([
                (x, y),
                (x + patch_size, y),
                (x + patch_size, y + patch_size),
                (x, y + patch_size)
            ])
            is_tumor = any(poly.intersects(patch_poly) for poly in tumor_polygons)
            labels.append(1 if is_tumor else 0)
        labels = np.array(labels, dtype=np.uint8)

        if label_dataset_name in h5_file:
            del h5_file[label_dataset_name]
        h5_file.create_dataset(label_dataset_name, data=labels)


def normalize_slide_id(path):
    name = os.path.splitext(os.path.basename(path))[0]
    name = name.replace("_patches_images", "")
    name = name.replace("_patches", "")
    name = name.replace("_coords", "")
    name = name.replace("_features", "")
    name = name.replace("_images", "")
    name = name.replace(".tumor", "")
    return name.lower()


def match_slide_ids(h5_files, xml_files):
    matched = []
    xml_map = {normalize_slide_id(xml_path): xml_path for xml_path in xml_files}

    for h5_path in h5_files:
        h5_id = normalize_slide_id(h5_path)
        xml_path = xml_map.get(h5_id)
        if xml_path:
            matched.append((h5_path, xml_path))
            continue

        # fallback: substring match in case of extra characters
        for xml_path in xml_files:
            if h5_id in normalize_slide_id(xml_path):
                matched.append((h5_path, xml_path))
                break
    return matched


def main(h5_dir, xml_dir, patch_size):
    h5_files = [os.path.join(h5_dir, f) for f in os.listdir(h5_dir) if f.endswith(".h5")]
    xml_files = [os.path.join(xml_dir, f) for f in os.listdir(xml_dir) if f.endswith(".tumor.xml")]

    matched_pairs = match_slide_ids(h5_files, xml_files)

    print(f"[INFO] Found {len(matched_pairs)} matched .h5 and .tumor.xml pairs.")

    for h5_path, xml_path in tqdm(matched_pairs, desc="Processing slides"):
        try:
            polygons = extract_tumor_polygons(xml_path)
            append_labels(h5_path, polygons, patch_size=patch_size)
            print(xml_path)
        except Exception as e:
            print(f"[ERROR] Failed for {h5_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch-append tumor labels to H5 files using corresponding XML annotation directories.")
    parser.add_argument("--h5_dir", required=True, help="Directory containing .h5 feature files.")
    parser.add_argument("--xml_dir", required=True, help="Directory containing .tumor.xml annotation files.")
    parser.add_argument("--patch_size", type=int, default=512, help="Patch size used during feature extraction (default: 512).")
    args = parser.parse_args()

    main(args.h5_dir, args.xml_dir, args.patch_size)
