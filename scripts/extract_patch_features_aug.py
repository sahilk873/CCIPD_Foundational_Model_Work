#!/usr/bin/env python3
"""
extract_patch_features_aug_batch.py

Batch TRIDENT patch feature extraction with deterministic augmentations applied
BEFORE the encoder's eval-time preprocessing transform.

Coords auto-matching (your convention):
  slide:  <ID>.<UUID>.svs
  coords: <ID>.<UUID>_patches.h5   (anywhere under coords_dir)

Output structure:
  out_dir/
    none/
    affine_L0/ affine_L1/ affine_L2/
    hsv_L0/    hsv_L1/    hsv_L2/
    noise_L0/  noise_L1/  noise_L2/          (Gaussian pixel noise)
    gaussian_L0/ gaussian_L1/ gaussian_L2/    (Gaussian blur)
    elastic_L0/  elastic_L1/  elastic_L2/     (Elastic deformation)
    he_L0/       he_L1/       he_L2/          (H&E stain jitter)
(each folder contains one .h5 per slide)

Fix included:
- Works for MUSK and CONCH (and other encoders) by choosing the correct callable model:
  Prefer using the encoder wrapper (base) so custom forward() logic is preserved (MUSK needs this).
"""

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter
import torch
import torch.nn.functional as F

import trident
from trident.patch_encoder_models import encoder_factory, CustomInferenceEncoder


# -----------------------------
# Fixed augmentations (deterministic)
# -----------------------------
class FixedHSVShift:
    def __init__(self, hue_delta: float = 0.0, sat_mult: float = 1.0, val_mult: float = 1.0):
        self.hue_delta = float(hue_delta)
        self.sat_mult = float(sat_mult)
        self.val_mult = float(val_mult)

    def __call__(self, img: Image.Image) -> Image.Image:
        hsv = img.convert("HSV")
        arr = np.array(hsv).astype(np.float32)
        arr[..., 0] = (arr[..., 0] + self.hue_delta * 255.0) % 255.0
        arr[..., 1] = np.clip(arr[..., 1] * self.sat_mult, 0, 255)
        arr[..., 2] = np.clip(arr[..., 2] * self.val_mult, 0, 255)
        return Image.fromarray(arr.astype(np.uint8), mode="HSV").convert("RGB")


class FixedGaussianNoise:
    def __init__(self, sigma: float = 0.0, seed: int = 0):
        self.sigma = float(sigma)
        self.seed = int(seed)

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.sigma <= 0:
            return img
        arr = np.array(img).astype(np.float32)
        rng = np.random.default_rng(self.seed)
        noise = rng.normal(0.0, self.sigma, size=arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")


class FixedGaussianBlur:
    """Gaussian blur using PIL (deterministic)."""
    def __init__(self, radius: float = 0.0):
        self.radius = float(radius)

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.radius <= 0:
            return img
        return img.filter(ImageFilter.GaussianBlur(radius=self.radius))


class FixedAffine:
    def __init__(self, angle_deg: float = 0.0, translate_px: Tuple[int, int] = (0, 0), scale: float = 1.0):
        self.angle_deg = float(angle_deg)
        self.translate_px = (int(translate_px[0]), int(translate_px[1]))
        self.scale = float(scale)

    def __call__(self, img: Image.Image) -> Image.Image:
        out = img.rotate(self.angle_deg, resample=Image.BILINEAR, expand=False)
        if self.translate_px != (0, 0):
            canvas = Image.new("RGB", out.size)
            canvas.paste(out, self.translate_px)
            out = canvas
        if self.scale != 1.0:
            w, h = out.size
            nw, nh = max(1, int(w * self.scale)), max(1, int(h * self.scale))
            out = out.resize((nw, nh), resample=Image.BILINEAR).resize((w, h), resample=Image.BILINEAR)
        return out


def _pil_to_torch(img: Image.Image) -> torch.Tensor:
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0  # C,H,W in [0,1]
    return t


def _torch_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.clamp(0, 1)
    arr = (t * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


def _gaussian_kernel_1d(sigma: float, device: torch.device) -> torch.Tensor:
    sigma = float(sigma)
    if sigma <= 0:
        return torch.tensor([1.0], device=device)

    radius = max(1, int(math.ceil(3.0 * sigma)))
    xs = torch.arange(-radius, radius + 1, device=device).float()
    kernel = torch.exp(-(xs ** 2) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel


def _separable_gaussian_blur_2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    x: (1, C, H, W)
    returns same shape
    """
    device = x.device
    k1 = _gaussian_kernel_1d(sigma, device=device)  # (K,)
    kx = k1.view(1, 1, 1, -1)  # (1,1,1,K)
    ky = k1.view(1, 1, -1, 1)  # (1,1,K,1)

    c = x.shape[1]
    kx = kx.repeat(c, 1, 1, 1)  # (C,1,1,K)
    ky = ky.repeat(c, 1, 1, 1)  # (C,1,K,1)

    pad_x = (kx.shape[-1] // 2, kx.shape[-1] // 2, 0, 0)  # left,right,top,bottom
    pad_y = (0, 0, ky.shape[-2] // 2, ky.shape[-2] // 2)

    x = F.pad(x, pad_x, mode="reflect")
    x = F.conv2d(x, kx, groups=c)
    x = F.pad(x, pad_y, mode="reflect")
    x = F.conv2d(x, ky, groups=c)
    return x


class FixedElasticDeform:
    """
    Elastic deformation using torch grid_sample.
    Deterministic via fixed seed. Same warp per image size.
    """
    def __init__(self, alpha_px: float, sigma_px: float, seed: int = 0):
        self.alpha_px = float(alpha_px)
        self.sigma_px = float(sigma_px)
        self.seed = int(seed)

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.alpha_px <= 0:
            return img

        t = _pil_to_torch(img)  # C,H,W
        c, h, w = t.shape
        device = torch.device("cpu")  # keep on CPU for DataLoader workers
        t = t.to(device).unsqueeze(0)  # 1,C,H,W

        gen = torch.Generator(device=device)
        gen.manual_seed(self.seed + h * 1000 + w)  # deterministic per size

        # Random displacement fields in pixels
        dx = (torch.rand((1, 1, h, w), generator=gen, device=device) * 2.0 - 1.0) * self.alpha_px
        dy = (torch.rand((1, 1, h, w), generator=gen, device=device) * 2.0 - 1.0) * self.alpha_px

        if self.sigma_px > 0:
            dx = _separable_gaussian_blur_2d(dx, self.sigma_px)
            dy = _separable_gaussian_blur_2d(dy, self.sigma_px)

        # Base grid in [-1,1]
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=device),
            torch.linspace(-1.0, 1.0, w, device=device),
            indexing="ij",
        )
        grid = torch.stack([xx, yy], dim=-1).unsqueeze(0)  # 1,H,W,2

        # Convert pixel displacement to normalized coordinates
        dx_norm = dx.squeeze(1) / max(1.0, (w - 1) / 2.0)
        dy_norm = dy.squeeze(1) / max(1.0, (h - 1) / 2.0)
        grid[..., 0] = grid[..., 0] + dx_norm
        grid[..., 1] = grid[..., 1] + dy_norm

        warped = F.grid_sample(
            t,
            grid.clamp(-1, 1),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(0)  # C,H,W

        return _torch_to_pil(warped)


class FixedHEJitter:
    """
    H&E jitter via a simple optical-density (OD) model with a fixed stain matrix.
    This is a practical deterministic approximation (no Macenko fitting per-slide).

    It works best on H&E slides. On non-H&E stains, colors may shift oddly.
    """
    # Commonly used approximate stain vectors (rows are H, E), in RGB OD space
    # These are normalized internally.
    DEFAULT_STAIN_MATRIX = np.array(
        [
            [0.650, 0.704, 0.286],  # H
            [0.072, 0.990, 0.105],  # E
        ],
        dtype=np.float32,
    )

    def __init__(self, h_mult: float = 1.0, e_mult: float = 1.0, seed: int = 0):
        self.h_mult = float(h_mult)
        self.e_mult = float(e_mult)
        self.seed = int(seed)

        m = FixedHEJitter.DEFAULT_STAIN_MATRIX.copy()
        m = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-8)
        self.M = m  # (2,3)

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img.convert("RGB"), dtype=np.float32)  # H,W,3 in [0,255]
        # OD transform
        I = np.clip(arr, 1.0, 255.0)
        OD = -np.log(I / 255.0)  # H,W,3

        h, w, _ = OD.shape
        od2 = OD.reshape(-1, 3).T  # 3,N

        # Solve for concentrations C in least squares: OD ~= M^T * C, where M is (2,3)
        # Rearranged: (M.T) is (3,2), so C is (2,N)
        MT = self.M.T  # 3,2
        C, _, _, _ = np.linalg.lstsq(MT, od2, rcond=None)  # 2,N

        # Deterministic scaling of stain concentrations
        C2 = C.copy()
        C2[0, :] *= self.h_mult
        C2[1, :] *= self.e_mult

        # Reconstruct OD and invert
        od_recon = (MT @ C2).T.reshape(h, w, 3)  # H,W,3
        I_recon = 255.0 * np.exp(-od_recon)
        I_recon = np.clip(I_recon, 0.0, 255.0).astype(np.uint8)

        return Image.fromarray(I_recon, mode="RGB")


def build_aug(name: str, level: int) -> Optional[object]:
    if name == "none":
        return None
    if level not in (0, 1, 2):
        raise ValueError("--level must be 0, 1, or 2 for these policies.")

    if name == "affine":
        angles = [1.0, 2.0, 4.0]
        shifts = [(1, 1), (2, 2), (4, 4)]
        return FixedAffine(angle_deg=angles[level], translate_px=shifts[level], scale=1.0)

    if name == "hsv":
        hue = [0.01, 0.02, 0.04]
        sat = [1.05, 1.10, 1.20]
        val = [1.03, 1.06, 1.12]
        return FixedHSVShift(hue_delta=hue[level], sat_mult=sat[level], val_mult=val[level])

    if name == "noise":
        sigmas = [2.0, 5.0, 10.0]  # pixel-space noise (stddev)
        return FixedGaussianNoise(sigma=sigmas[level], seed=123)

    if name == "gaussian":
        # Gaussian blur radius in pixels (PIL)
        radii = [0.5, 1.0, 1.5]
        return FixedGaussianBlur(radius=radii[level])

    if name == "elastic":
        # Small elastic warp: alpha controls displacement magnitude (pixels),
        # sigma controls smoothness of the displacement field (pixels).
        alphas = [2.0, 4.0, 6.0]
        sigmas = [18.0, 14.0, 10.0]
        return FixedElasticDeform(alpha_px=alphas[level], sigma_px=sigmas[level], seed=1337)

    if name == "he":
        # H&E concentration scaling. Conservative by default.
        h_mults = [1.05, 1.10, 1.20]
        e_mults = [1.05, 1.10, 1.20]
        return FixedHEJitter(h_mult=h_mults[level], e_mult=e_mults[level], seed=7)

    raise ValueError(f"Unknown aug name: {name}")


def compose_transforms(base_eval_transform, aug):
    if aug is None:
        return base_eval_transform

    def _t(img):
        return base_eval_transform(aug(img))

    return _t


# -----------------------------
# Batch helpers
# -----------------------------
WSI_EXTS = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn", ".svslide", ".vms", ".vmu"}


def list_slides(slides_dir: Path, recursive: bool) -> List[Path]:
    it = slides_dir.rglob("*") if recursive else slides_dir.iterdir()
    slides: List[Path] = []
    for p in it:
        if p.is_file() and p.suffix.lower() in WSI_EXTS:
            slides.append(p)
    slides.sort()
    return slides


def index_coords_files(coords_dir: Path, recursive: bool, coords_ext: str) -> Dict[str, Path]:
    coords_ext = coords_ext.lower()
    it = coords_dir.rglob(f"*{coords_ext}") if recursive else coords_dir.glob(f"*{coords_ext}")
    idx: Dict[str, Path] = {}
    for p in it:
        if not p.is_file():
            continue
        stem = p.stem
        if stem not in idx:
            idx[stem] = p
        else:
            if len(str(p)) < len(str(idx[stem])):
                idx[stem] = p
    return idx


def resolve_coords_for_slide(slide_path: Path, coords_idx: Dict[str, Path], coords_ext: str) -> Optional[Path]:
    stem = slide_path.stem
    prefix = slide_path.name.split(".", 1)[0]

    candidates = [
        f"{stem}_patches",
        stem,
        f"{prefix}_patches",
        prefix,
        f"{stem}_coords",
        f"{prefix}_coords",
    ]

    for key in candidates:
        p = coords_idx.get(key)
        if p is not None and p.is_file():
            return p

    for key in candidates:
        candidate = slide_path.parent / f"{key}{coords_ext}"
        if candidate.is_file():
            return candidate

    return None


def augmentation_plan(include_none: bool, types: List[str], levels: List[int]) -> List[Tuple[str, Optional[int], str]]:
    plan: List[Tuple[str, Optional[int], str]] = []
    if include_none:
        plan.append(("none", None, "none"))
    for t in types:
        for lvl in levels:
            plan.append((t, lvl, f"{t}_L{lvl}"))
    return plan


def pick_inference_model(base_encoder: torch.nn.Module) -> torch.nn.Module:
    """
    MUSK requires the encoder wrapper (base_encoder) because it defines a custom forward()
    that returns a single tensor. If we use base_encoder.model, it may return a tuple which
    breaks TRIDENT downstream (.cpu()).
    """
    if isinstance(base_encoder, torch.nn.Module) and callable(getattr(base_encoder, "forward", None)):
        return base_encoder

    underlying = getattr(base_encoder, "model", None)
    if isinstance(underlying, torch.nn.Module):
        return underlying

    raise RuntimeError("Could not determine a valid inference model to pass into CustomInferenceEncoder.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides_dir", required=True, help="Directory containing WSI files")
    ap.add_argument("--coords_dir", required=True, help="Directory containing coords .h5 files (expects dataset 'coords')")
    ap.add_argument("--out_dir", required=True, help="Root output directory for feature subfolders")
    ap.add_argument("--patch_encoder", required=True, help="Encoder name, e.g. conch_v15, musk, hoptimus0, uni_v1, etc.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num_workers", type=int, default=16, help="Dataloader workers (lower if you see slowdowns/freezes)")
    ap.add_argument("--recursive", action="store_true", help="Recursively search slides_dir and coords_dir")
    ap.add_argument("--coords_ext", default=".h5", help="Coords file extension (default: .h5)")
    ap.add_argument("--skip_existing", action="store_true", help="Skip (slide, aug) if output feature file already exists")
    ap.add_argument(
        "--aug_types",
        nargs="+",
        default=["affine", "hsv", "noise", "gaussian", "elastic", "he"],
        choices=["affine", "hsv", "noise", "gaussian", "elastic", "he"],
        help="Augmentation types to run"
    )
    ap.add_argument("--levels", nargs="+", type=int, default=[0, 1, 2], help="Intensity levels to run (default: 0 1 2)")
    ap.add_argument("--no_baseline", action="store_true", help="Do not run the baseline (none) extraction")
    args = ap.parse_args()

    slides_dir = Path(args.slides_dir)
    coords_dir = Path(args.coords_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if not slides_dir.is_dir():
        raise SystemExit(f"--slides_dir not found or not a directory: {slides_dir}")
    if not coords_dir.is_dir():
        raise SystemExit(f"--coords_dir not found or not a directory: {coords_dir}")

    slides = list_slides(slides_dir, recursive=args.recursive)
    if not slides:
        raise SystemExit(f"No slide files found in {slides_dir} (recursive={args.recursive})")

    coords_idx = index_coords_files(coords_dir, recursive=args.recursive, coords_ext=args.coords_ext)

    include_none = not args.no_baseline
    plan = augmentation_plan(include_none=include_none, types=args.aug_types, levels=args.levels)

    # Build base encoder wrapper once
    base = encoder_factory(args.patch_encoder)
    base_tf = getattr(base, "eval_transforms", None)
    if base_tf is None or not callable(base_tf):
        raise RuntimeError(
            f"Encoder {args.patch_encoder} did not expose callable eval_transforms. "
            f"Got eval_transforms={base_tf}"
        )

    infer_model = pick_inference_model(base)

    print(f"[INFO] Found {len(slides)} slides in {slides_dir} (recursive={args.recursive})")
    print(f"[INFO] Indexed {len(coords_idx)} coords files under {coords_dir} (recursive={args.recursive})")
    print(f"[INFO] Encoder={args.patch_encoder} | device={args.device} | workers={args.num_workers}")
    print(f"[INFO] Runs per slide: {len(plan)}  ->  {[p[2] for p in plan]}")

    missing_coords = 0
    processed = 0
    skipped = 0
    failed = 0

    for slide_path in slides:
        coords_path = resolve_coords_for_slide(slide_path, coords_idx, coords_ext=args.coords_ext)
        if coords_path is None:
            missing_coords += 1
            print(f"[WARN] No coords found for slide: {slide_path.name}")
            continue

        try:
            wsi = trident.load_wsi(str(slide_path), lazy_init=False)
        except Exception as e:
            failed += 1
            print(f"[ERROR] Failed to load slide {slide_path}: {e}")
            continue

        for aug_type, level, subdir in plan:
            out_dir = out_root / subdir
            out_dir.mkdir(parents=True, exist_ok=True)

            if args.skip_existing:
                existing = list(out_dir.glob(f"{slide_path.stem}*.h5"))
                if existing:
                    skipped += 1
                    continue

            if aug_type == "none":
                aug = None
                enc_name = f"{args.patch_encoder}_none"
            else:
                assert level is not None
                aug = build_aug(aug_type, level)
                enc_name = f"{args.patch_encoder}_{aug_type}_L{level}"

            composed = compose_transforms(base_tf, aug)
            wrapped = CustomInferenceEncoder(
                enc_name=enc_name,
                model=infer_model,
                transforms=composed,
                precision=base.precision,
            )

            extract_kwargs = dict(
                patch_encoder=wrapped,
                coords_path=str(coords_path),
                save_features=str(out_dir),
                device=args.device,
                saveas="h5",
                verbose=True,
            )

            try:
                try:
                    out_path = wsi.extract_patch_features(**extract_kwargs, num_workers=args.num_workers)
                except TypeError:
                    out_path = wsi.extract_patch_features(**extract_kwargs)

                processed += 1
                print(f"[OK] {slide_path.name} -> {subdir} | {out_path}")
            except Exception as e:
                failed += 1
                print(f"[ERROR] Feature extraction failed for {slide_path.name} ({subdir}): {e}")

    print("\n========== SUMMARY ==========")
    print(f"Slides found:          {len(slides)}")
    print(f"Slides missing coords: {missing_coords}")
    print(f"Runs processed:        {processed}")
    print(f"Runs skipped:          {skipped}")
    print(f"Runs failed:           {failed}")
    print("=============================\n")


if __name__ == "__main__":
    main()
