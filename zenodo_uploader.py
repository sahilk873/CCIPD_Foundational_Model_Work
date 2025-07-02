import os
import requests
import hashlib
from tqdm import tqdm

# List of files to download: (filename, md5 checksum)
FILES = [
    ("WM4237_T4_S1_ST.ome.tiff", "1edff718bce0905ec7a3646e20991fd8"),
    ("WM4237_T4_S2_ST.ome.tiff", "45ef3438750f02f0742e2cf7f7c7a030"),
    ("WM4237_TC_S1_ST.ome.tiff", "624311cc19da71b097425939de2bcbd1"),
    ("WM4237_TC_S2_ST.ome.tiff", "5aa9ba3d5ce34600e6494e4776b66ead"),
]

ZENODO_BASE = "https://zenodo.org/records/15263298/files/"

def md5sum(filename, chunk_size=8192):
    h = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def download_with_resume(url: str, output_path: str):
    # determine existing file size (for resume)
    temp_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
    headers = {"Range": f"bytes={temp_size}-"} if temp_size > 0 else {}

    with requests.get(url, stream=True, headers=headers) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0)) + temp_size
        mode = "ab" if temp_size > 0 else "wb"
        with open(output_path, mode) as f, tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            initial=temp_size,
            desc=os.path.basename(output_path)
        ) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

def main():
    for fname, expected_md5 in FILES:
        url = f"{ZENODO_BASE}{fname}?download=1"
        print(f"\nStarting download: {fname}")
        try:
            download_with_resume(url, fname)
        except Exception as e:
            print(f"❌ Error downloading {fname}: {e}")
            continue

        print("Verifying checksum...", end=" ")
        actual_md5 = md5sum(fname)
        if actual_md5 == expected_md5:
            print("✅ OK")
        else:
            print(f"❌ MD5 mismatch!\n  expected: {expected_md5}\n  actual:   {actual_md5}")

if __name__ == "__main__":
    main()
