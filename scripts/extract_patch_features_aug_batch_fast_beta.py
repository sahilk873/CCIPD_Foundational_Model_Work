#!/usr/bin/env python3
"""
extract_patch_features_aug_batch_fast.py

Batch TRIDENT patch feature extraction with deterministic augmentations applied
BEFORE encoder eval-time preprocessing.

Supports ONLY levels: L0, L1, L2

Speedups:
1) Pre-build encoders once per augmentation policy and reuse across slides.
2) Pre-create output subdirs once; skip-existing checks expected output.
3) Fast inference settings (inference_mode, TF32, cudnn.benchmark).
4) Cheaper augmentations: cached elastic fields per (H,W); cached RNG; HE pinv precomputed.

Adds:
- New augmentation families (1–4):
  (1) cutout
  (2) jpeg
  (3) downsample
  (4) lowpass/highpass
- Logging to console + optional --log_file for debugging.
"""

import argparse
import io
import logging
import math
import os
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

import torch
import torch.nn.functional as F

import trident
from trident.patch_encoder_models import encoder_factory, CustomInferenceEncoder


# ============================================================
# Logging
# ============================================================
def setup_logger(log_file: Optional[str], debug_console: bool) -> logging.Logger:
    logger = logging.getLogger("extract_aug")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if debug_console else logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file, mode="w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"))
        logger.addHandler(fh)

    return logger


# ============================================================
# Deterministic augmentations (fast)
# ============================================================
class FixedHSVShift:
    def __init__(self, hue_delta: float = 0.0, sat_mult: float = 1.0, val_mult: float = 1.0):
        self.hue_delta = float(hue_delta)
        self.sat_mult = float(sat_mult)
        self.val_mult = float(val_mult)

    def __call__(self, img: Image.Image) -> Image.Image:
        hsv = img.convert("HSV")
        arr = np.asarray(hsv, dtype=np.uint8)
        f = arr.astype(np.float32)
        f[..., 0] = (f[..., 0] + self.hue_delta * 255.0) % 255.0
        f[..., 1] = np.clip(f[..., 1] * self.sat_mult, 0, 255)
        f[..., 2] = np.clip(f[..., 2] * self.val_mult, 0, 255)
        return Image.fromarray(f.astype(np.uint8), mode="HSV").convert("RGB")


class FixedGaussianNoise:
    def __init__(self, sigma: float = 0.0, seed: int = 0):
        self.sigma = float(sigma)
        self._rng = np.random.default_rng(int(seed))

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.sigma <= 0:
            return img
        arr = np.asarray(img.convert("RGB"), dtype=np.uint8).astype(np.float32)
        noise = self._rng.normal(0.0, self.sigma, size=arr.shape).astype(np.float32)
        out = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(out, mode="RGB")


class FixedGaussianBlur:
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
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)


def _torch_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.clamp(0, 1)
    arr = (t * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


def _gaussian_kernel_1d(sigma: float, device: torch.device) -> torch.Tensor:
    if sigma <= 0:
        return torch.tensor([1.0], device=device)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    xs = torch.arange(-radius, radius + 1, device=device).float()
    kernel = torch.exp(-(xs ** 2) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def _separable_gaussian_blur_2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    device = x.device
    k1 = _gaussian_kernel_1d(float(sigma), device=device)
    kx = k1.view(1, 1, 1, -1)
    ky = k1.view(1, 1, -1, 1)
    c = x.shape[1]
    kx = kx.repeat(c, 1, 1, 1)
    ky = ky.repeat(c, 1, 1, 1)

    pad_x = (kx.shape[-1] // 2, kx.shape[-1] // 2, 0, 0)
    pad_y = (0, 0, ky.shape[-2] // 2, ky.shape[-2] // 2)

    x = F.pad(x, pad_x, mode="reflect")
    x = F.conv2d(x, kx, groups=c)
    x = F.pad(x, pad_y, mode="reflect")
    x = F.conv2d(x, ky, groups=c)
    return x


class FixedElasticDeform:
    def __init__(self, alpha_px: float, sigma_px: float, seed: int = 0):
        self.alpha_px = float(alpha_px)
        self.sigma_px = float(sigma_px)
        self.seed = int(seed)
        self._cache: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def _get_cached(self, h: int, w: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (h, w)
        if key in self._cache:
            return self._cache[key]

        device = torch.device("cpu")
        gen = torch.Generator(device=device)
        gen.manual_seed(self.seed + h * 1000 + w)

        dx = (torch.rand((1, 1, h, w), generator=gen, device=device) * 2.0 - 1.0) * self.alpha_px
        dy = (torch.rand((1, 1, h, w), generator=gen, device=device) * 2.0 - 1.0) * self.alpha_px

        if self.sigma_px > 0:
            dx = _separable_gaussian_blur_2d(dx, self.sigma_px)
            dy = _separable_gaussian_blur_2d(dy, self.sigma_px)

        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=device),
            torch.linspace(-1.0, 1.0, w, device=device),
            indexing="ij",
        )
        grid = torch.stack([xx, yy], dim=-1).unsqueeze(0)
        self._cache[key] = (grid, dx, dy)
        return self._cache[key]

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.alpha_px <= 0:
            return img

        t = _pil_to_torch(img)
        _, h, w = t.shape
        device = torch.device("cpu")

        t = t.to(device).unsqueeze(0)
        grid, dx, dy = self._get_cached(h, w)

        dx_norm = dx.squeeze(1) / max(1.0, (w - 1) / 2.0)
        dy_norm = dy.squeeze(1) / max(1.0, (h - 1) / 2.0)

        warped_grid = grid.clone()
        warped_grid[..., 0] = warped_grid[..., 0] + dx_norm
        warped_grid[..., 1] = warped_grid[..., 1] + dy_norm

        warped = F.grid_sample(
            t,
            warped_grid.clamp(-1, 1),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(0)
        return _torch_to_pil(warped)


class FixedHEJitter:
    DEFAULT_STAIN_MATRIX = np.array(
        [
            [0.650, 0.704, 0.286],
            [0.072, 0.990, 0.105],
        ],
        dtype=np.float32,
    )

    def __init__(self, h_mult: float = 1.0, e_mult: float = 1.0):
        self.h_mult = float(h_mult)
        self.e_mult = float(e_mult)

        m = FixedHEJitter.DEFAULT_STAIN_MATRIX.copy()
        m = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-8)
        self.M = m
        self.MT = self.M.T
        self.MT_pinv = np.linalg.pinv(self.MT).astype(np.float32)

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.asarray(img.convert("RGB"), dtype=np.uint8).astype(np.float32)
        I = np.clip(arr, 1.0, 255.0)
        OD = -np.log(I / 255.0)

        h, w, _ = OD.shape
        od2 = OD.reshape(-1, 3).T
        C = (self.MT_pinv @ od2).astype(np.float32)

        C[0, :] *= self.h_mult
        C[1, :] *= self.e_mult

        od_recon = (self.MT @ C).T.reshape(h, w, 3)
        I_recon = 255.0 * np.exp(-od_recon)
        I_recon = np.clip(I_recon, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(I_recon, mode="RGB")


# ============================================================
# New augmentation families (1–4)
# ============================================================
class FixedCutout:
    def __init__(self, frac: float, seed: int = 0):
        self.frac = float(frac)
        self.seed = int(seed)

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.frac <= 0:
            return img
        w, h = img.size
        side = max(1, int(min(w, h) * self.frac))
        rng = np.random.default_rng(self.seed + w * 31 + h * 97)
        x0 = int(rng.integers(0, max(1, w - side + 1)))
        y0 = int(rng.integers(0, max(1, h - side + 1)))
        arr = np.asarray(img.convert("RGB"), dtype=np.uint8).copy()
        arr[y0:y0 + side, x0:x0 + side, :] = 0
        return Image.fromarray(arr, mode="RGB")


class FixedJPEG:
    def __init__(self, quality: int):
        self.quality = int(quality)

    def __call__(self, img: Image.Image) -> Image.Image:
        q = int(np.clip(self.quality, 5, 100))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=q, optimize=False)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class FixedDownsampleUpsample:
    def __init__(self, scale: float):
        self.scale = float(scale)

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.scale >= 1.0:
            return img
        w, h = img.size
        nw = max(1, int(w * self.scale))
        nh = max(1, int(h * self.scale))
        small = img.resize((nw, nh), resample=Image.BILINEAR)
        return small.resize((w, h), resample=Image.BILINEAR)


class FixedFreqFilter:
    def __init__(self, mode: str, cutoff_frac: float):
        assert mode in ("lowpass", "highpass")
        self.mode = mode
        self.cutoff_frac = float(cutoff_frac)

    def __call__(self, img: Image.Image) -> Image.Image:
        cf = float(np.clip(self.cutoff_frac, 0.001, 0.5))
        arr = np.asarray(img.convert("RGB"), dtype=np.float32)
        h, w, c = arr.shape

        cy, cx = h // 2, w // 2
        rad = int(cf * min(h, w))

        yy, xx = np.ogrid[:h, :w]
        mask = ((yy - cy) ** 2 + (xx - cx) ** 2 <= rad * rad).astype(np.float32)

        out = np.zeros_like(arr)
        for ch in range(c):
            Fch = np.fft.fft2(arr[..., ch])
            Fsh = np.fft.fftshift(Fch)
            if self.mode == "lowpass":
                Fsh_f = Fsh * mask
            else:
                Fsh_f = Fsh * (1.0 - mask)
            inv = np.fft.ifft2(np.fft.ifftshift(Fsh_f))
            out[..., ch] = np.real(inv)

        out = np.clip(out, 0, 255).astype(np.uint8)
        return Image.fromarray(out, mode="RGB")


# ============================================================
# Augmentation builder (ONLY L0/L1/L2)
# ============================================================
def build_aug(name: str, level: int) -> Optional[object]:
    if name == "none":
        return None
    if level not in (0, 1, 2):
        raise ValueError("--levels must be 0, 1, or 2")

    # Original families
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
        sigmas = [2.0, 5.0, 10.0]
        return FixedGaussianNoise(sigma=sigmas[level], seed=123 + level)

    if name == "gaussian":
        radii = [0.5, 1.0, 1.5]
        return FixedGaussianBlur(radius=radii[level])

    if name == "elastic":
        alphas = [2.0, 4.0, 6.0]
        sigmas = [18.0, 14.0, 10.0]
        return FixedElasticDeform(alpha_px=alphas[level], sigma_px=sigmas[level], seed=1337 + level)

    if name == "he":
        h_mults = [1.05, 1.10, 1.20]
        e_mults = [1.05, 1.10, 1.20]
        return FixedHEJitter(h_mult=h_mults[level], e_mult=e_mults[level])

    # New families (1–4), 3 intensities
    if name == "cutout":
        fracs = [0.07, 0.12, 0.18]
        return FixedCutout(frac=fracs[level], seed=2027 + level)

    if name == "jpeg":
        qualities = [80, 55, 30]
        return FixedJPEG(quality=qualities[level])

    if name == "downsample":
        scales = [0.85, 0.65, 0.45]
        return FixedDownsampleUpsample(scale=scales[level])

    if name == "lowpass":
        cutoffs = [0.28, 0.18, 0.12]
        return FixedFreqFilter(mode="lowpass", cutoff_frac=cutoffs[level])

    if name == "highpass":
        cutoffs = [0.28, 0.18, 0.12]
        return FixedFreqFilter(mode="highpass", cutoff_frac=cutoffs[level])

    if name == "he_h_suppress":
        h = [0.7, 0.6, 0.5][level]
        return FixedHEJitter(h_mult=h, e_mult=1.0)

    if name == "he_e_suppress":
        e = [0.7, 0.6, 0.5][level]
        return FixedHEJitter(h_mult=1.0, e_mult=e)


    raise ValueError(f"Unknown aug name: {name}")


def compose_transforms(base_eval_transform, aug):
    if aug is None:
        return base_eval_transform

    def _t(img):
        return base_eval_transform(aug(img))

    return _t


# ============================================================
# Batch helpers
# ============================================================
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
        if stem not in idx or len(str(p)) < len(str(idx[stem])):
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
    if isinstance(base_encoder, torch.nn.Module) and callable(getattr(base_encoder, "forward", None)):
        return base_encoder
    underlying = getattr(base_encoder, "model", None)
    if isinstance(underlying, torch.nn.Module):
        return underlying
    raise RuntimeError("Could not determine a valid inference model to pass into CustomInferenceEncoder.")


def trident_expected_h5(out_dir: Path, slide_path: Path) -> Optional[Path]:
    matches = list(out_dir.glob(f"{slide_path.stem}*.h5"))
    return matches[0] if matches else None


def enable_fast_inference(device: str, logger: logging.Logger):
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        logger.debug("torch.set_float32_matmul_precision not available")

    if device.startswith("cuda"):
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            logger.debug("TF32 toggles not available")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides_dir", required=True, help="Directory containing WSI files")
    ap.add_argument("--coords_dir", required=True, help="Directory containing coords .h5 files (expects dataset 'coords')")
    ap.add_argument("--out_dir", required=True, help="Root output directory for feature subfolders")
    ap.add_argument("--patch_encoder", required=True, help="Encoder name, e.g. conch_v15, musk, hoptimus0, etc.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--recursive", action="store_true", help="Recursively search slides_dir and coords_dir")
    ap.add_argument("--coords_ext", default=".h5", help="Coords file extension (default: .h5)")
    ap.add_argument("--skip_existing", action="store_true", help="Skip (slide, aug) if output feature file already exists")
    ap.add_argument("--batch_limit", type=int, default=512, help="TRIDENT batch_limit")
    ap.add_argument("--verbose_trident", action="store_true", help="Pass verbose=True to TRIDENT extractor")

    ap.add_argument("--log_file", default=None, help="Write DEBUG logs to this file")
    ap.add_argument("--debug", action="store_true", help="More console logging")

    ap.add_argument(
        "--aug_types",
        nargs="+",
        default=["affine", "hsv", "noise", "gaussian", "elastic", "he"],
        choices=[
            "affine", "hsv", "noise", "gaussian", "elastic", "he",
            "cutout", "jpeg", "downsample", "lowpass", "highpass", "he_h_suppress", "he_e_suppress"
        ],
        help="Augmentation types to run",
    )
    ap.add_argument(
        "--levels",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        choices=[0, 1, 2],
        help="Intensity levels to run (ONLY: 0 1 2)",
    )
    ap.add_argument("--no_baseline", action="store_true", help="Do not run baseline (none)")
    args = ap.parse_args()

    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = "8"
    if "MKL_NUM_THREADS" not in os.environ:
        os.environ["MKL_NUM_THREADS"] = "8"

    logger = setup_logger(args.log_file, debug_console=args.debug)

    slides_dir = Path(args.slides_dir)
    coords_dir = Path(args.coords_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    logger.info(f"slides_dir={slides_dir}")
    logger.info(f"coords_dir={coords_dir}")
    logger.info(f"out_dir={out_root}")
    logger.info(f"patch_encoder={args.patch_encoder} device={args.device} batch_limit={args.batch_limit}")
    logger.info(f"aug_types={args.aug_types}")
    logger.info(f"levels={args.levels}")
    logger.info(f"recursive={args.recursive} skip_existing={args.skip_existing} no_baseline={args.no_baseline}")

    if not slides_dir.is_dir():
        raise SystemExit(f"--slides_dir not found or not a directory: {slides_dir}")
    if not coords_dir.is_dir():
        raise SystemExit(f"--coords_dir not found or not a directory: {coords_dir}")

    enable_fast_inference(args.device, logger)

    slides = list_slides(slides_dir, recursive=args.recursive)
    logger.info(f"Discovered {len(slides)} slide files (extensions={sorted(WSI_EXTS)})")
    if not slides:
        raise SystemExit(
            f"No slide files found in {slides_dir} with extensions {sorted(WSI_EXTS)}. "
            f"If nested, add --recursive."
        )

    coords_idx = index_coords_files(coords_dir, recursive=args.recursive, coords_ext=args.coords_ext)
    logger.info(f"Indexed {len(coords_idx)} coords files under {coords_dir} (coords_ext={args.coords_ext})")

    include_none = not args.no_baseline
    plan = augmentation_plan(include_none=include_none, types=args.aug_types, levels=args.levels)
    subdirs = [p[2] for p in plan]
    logger.info(f"Runs per slide: {len(plan)}")
    logger.info(f"Output subdirs: {subdirs}")

    logger.info("Loading encoder...")
    base = encoder_factory(args.patch_encoder)
    base_tf = getattr(base, "eval_transforms", None)
    if base_tf is None or not callable(base_tf):
        raise RuntimeError(f"Encoder {args.patch_encoder} did not expose callable eval_transforms.")
    infer_model = pick_inference_model(base)
    logger.info("Encoder loaded.")

    out_dirs: Dict[str, Path] = {}
    for subdir in subdirs:
        d = out_root / subdir
        d.mkdir(parents=True, exist_ok=True)
        out_dirs[subdir] = d

    logger.info("Pre-building CustomInferenceEncoder objects...")
    encoders: Dict[str, CustomInferenceEncoder] = {}
    for aug_type, level, subdir in plan:
        try:
            if aug_type == "none":
                aug = None
                enc_name = f"{args.patch_encoder}_none"
            else:
                assert level is not None
                aug = build_aug(aug_type, level)
                enc_name = f"{args.patch_encoder}_{aug_type}_L{level}"

            composed = compose_transforms(base_tf, aug)
            encoders[subdir] = CustomInferenceEncoder(
                enc_name=enc_name,
                model=infer_model,
                transforms=composed,
                precision=base.precision,
            )
            logger.debug(f"Built encoder: {subdir} ({enc_name})")
        except Exception as e:
            logger.error(f"Failed to build encoder for {subdir}: {e}")
            logger.debug(traceback.format_exc())
            raise

    missing_coords = 0
    processed = 0
    skipped = 0
    failed = 0
    slides_with_coords = 0

    for si, slide_path in enumerate(slides):
        logger.info(f"[{si+1}/{len(slides)}] Slide: {slide_path.name}")

        coords_path = resolve_coords_for_slide(slide_path, coords_idx, coords_ext=args.coords_ext)
        if coords_path is None:
            missing_coords += 1
            logger.warning(f"No coords found for slide: {slide_path.name} (stem={slide_path.stem})")
            continue

        slides_with_coords += 1
        logger.info(f"Coords matched: {coords_path.name}")

        try:
            wsi = trident.load_wsi(str(slide_path), lazy_init=False)
            logger.debug("Loaded WSI successfully.")
        except Exception as e:
            failed += 1
            logger.error(f"Failed to load slide {slide_path}: {e}")
            logger.debug(traceback.format_exc())
            continue

        for _aug_type, _level, subdir in plan:
            out_dir = out_dirs[subdir]

            if args.skip_existing:
                existing = trident_expected_h5(out_dir, slide_path)
                if existing is not None and existing.is_file():
                    skipped += 1
                    logger.info(f"[SKIP] exists: {slide_path.stem} -> {subdir} ({existing.name})")
                    continue

            wrapped = encoders[subdir]
            logger.info(f"[RUN] {slide_path.stem} -> {subdir}")

            try:
                with torch.inference_mode():
                    out_path = wsi.extract_patch_features(
                        patch_encoder=wrapped,
                        coords_path=str(coords_path),
                        save_features=str(out_dir),
                        device=args.device,
                        saveas="h5",
                        batch_limit=int(args.batch_limit),
                        verbose=bool(args.verbose_trident),
                    )
                processed += 1
                logger.info(f"[OK] {slide_path.name} -> {subdir} | out={out_path}")
            except Exception as e:
                failed += 1
                logger.error(f"[ERROR] Extraction failed for {slide_path.name} ({subdir}): {e}")
                logger.debug(traceback.format_exc())

        try:
            wsi.release()
            logger.debug("Released WSI handle.")
        except Exception:
            logger.debug("WSI release() not available/failed (ignored).")

    logger.info("========== SUMMARY ==========")
    logger.info(f"Slides discovered:       {len(slides)}")
    logger.info(f"Slides w/ coords:        {slides_with_coords}")
    logger.info(f"Slides missing coords:   {missing_coords}")
    logger.info(f"Runs processed:          {processed}")
    logger.info(f"Runs skipped (existing): {skipped}")
    logger.info(f"Runs failed:             {failed}")
    logger.info("=============================")


if __name__ == "__main__":
    main()
