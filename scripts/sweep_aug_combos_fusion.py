#!/usr/bin/env python3
"""
sweep_aug_combos_fusion_single.py

Sweeps over augmentation *families* (each family includes L0/L1/L2 together),
builds stability .h5 outputs ONCE per combo, then trains ONE fusion mode
(specified by --fusion) on that same stability dir, writing per-combo metrics JSON,
and finally deletes the intermediate stability .h5 files (so the sweep doesn't blow up disk).

✅ Option A implemented: auto-resume / skip completed combos
- If a per-combo metrics JSON already exists and is parseable, the combo is skipped.
- If sweep_results.json already exists, we load it and keep appending to it.
- Ranked-by-AUROC JSON is regenerated after each successful combo (and at the end).

Assumes:
- build_stability_embeddings.py is on PATH / in cwd
- fusion_stability_classifier.py is available (supports --fusion and --out_json)
"""

import argparse
import itertools
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional


def L012(base: str):
    return [f"{base}_L0", f"{base}_L1", f"{base}_L2"]


AUG_GROUPS: Dict[str, List[str]] = {
    "gaussian": L012("gaussian"),
    # "affine": L012("affine"),
    # "cutout": L012("cutout"),
    "elastic": L012("elastic"),
    "hsv": L012("hsv"),
    "he_e_suppress": L012("he_e_suppress"),
    "he_h_suppress": L012("he_h_suppress"),
    "he": L012("he"),

    # New clinically relevant degradations
    "downsample": L012("downsample"),
    "jpeg": L012("jpeg"),
    # "highpass": L012("highpass"),
}


def powerset_groups(keys):
    keys = list(keys)
    for r in range(1, len(keys) + 1):
        for comb in itertools.combinations(keys, r):
            yield list(comb)


def run_and_log(cmd: List[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        f.write("CMD: " + " ".join(cmd) + "\n\n")
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    return int(p.returncode)


def _safe_rmtree(path: Path) -> None:
    if path.exists() and path.is_dir():
        shutil.rmtree(path, ignore_errors=True)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def _load_existing_results(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _metrics_ok(metrics_path: Path) -> Optional[Dict[str, Any]]:
    """
    Returns payload if metrics JSON exists + parseable + looks valid, else None.
    """
    if not metrics_path.exists():
        return None
    try:
        payload = _read_json(metrics_path)
        if not isinstance(payload, dict):
            return None
        m = payload.get("best_val_metrics", {})
        if not isinstance(m, dict):
            return None
        # At least one key metric should exist; AUROC commonly present.
        if ("auroc" in m) or ("f1" in m) or ("acc" in m):
            return payload
        return None
    except Exception:
        return None


def _rank_successes(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = [
        r for r in results
        if r.get("ok") and isinstance(r.get("best_val_metrics", {}).get("auroc", None), (int, float))
    ]
    ranked.sort(key=lambda r: float(r["best_val_metrics"]["auroc"]), reverse=True)
    return ranked


def main() -> None:
    ap = argparse.ArgumentParser()

    # ---- build inputs ----
    ap.add_argument("--baseline_dir", required=True, help="Directory of baseline (normal) .h5 feature files")
    ap.add_argument("--aug_root", required=True, help="Root directory with augmentation subfolders")

    # ---- outputs ----
    ap.add_argument("--out_root", required=True, help="Root folder to write per-combo logs + metrics (and temp h5 dirs)")

    # ---- classifier ----
    ap.add_argument(
        "--classifier_script",
        default="fusion_stability_classifier.py",
        help="Path to fusion classifier script (default: fusion_stability_classifier.py)",
    )
    ap.add_argument(
        "--fusion",
        type=str,
        required=True,
        choices=["concat", "multihead", "gated"],
        help="Fusion mode to run per combo (single).",
    )
    ap.add_argument(
        "--use",
        nargs="+",
        default=["features", "cv", "iqr", "mean", "std"],
        help="Blocks to use. Example: --use features cv iqr mean std",
    )

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    # Classifier-supported hyperparams
    ap.add_argument("--z_dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.25)

    # Gated temperature scheduling passthrough
    ap.add_argument("--temp_schedule", default="linear", choices=["linear", "cosine", "exp"])
    ap.add_argument("--t_start", type=float, default=2.0)
    ap.add_argument("--t_end", type=float, default=1.0)

    # ---- build_stability_embeddings.py options ----
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--max_combos", type=int, default=0, help="0 means run all combos")
    ap.add_argument("--min_k", type=int, default=2, help="Pass-through to build_stability_embeddings.py")

    # ---- cleanup behavior ----
    ap.add_argument(
        "--keep_stability_h5",
        action="store_true",
        help="If set, do NOT delete intermediate stability dirs after training.",
    )
    ap.add_argument(
        "--delete_on_failure",
        action="store_true",
        help="If set, delete stability dir even if classify fails. Default keeps the dir on failure for inspection.",
    )

    args = ap.parse_args()

    out_root = Path(args.out_root)
    logs_dir = out_root / "_logs"
    metrics_dir = out_root / "_metrics"
    tmp_stability_root = out_root / "_tmp_stability_h5"

    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tmp_stability_root.mkdir(parents=True, exist_ok=True)

    results_path = out_root / "sweep_results.json"
    ranked_path = out_root / "sweep_ranked_by_auroc.json"

    results: List[Dict[str, Any]] = _load_existing_results(results_path)

    combos = list(powerset_groups(AUG_GROUPS.keys()))
    if args.max_combos and args.max_combos > 0:
        combos = combos[: args.max_combos]

    fusion = str(args.fusion)

    for groups in combos:
        combo_name = "+".join(groups)
        combo_out_dir = tmp_stability_root / combo_name
        build_log = logs_dir / f"{combo_name}.build.log"
        cls_log = logs_dir / f"{combo_name}.classify.{fusion}.log"
        metrics_path = metrics_dir / f"{combo_name}.best_val.{fusion}.json"

        # -----------------------
        # Option A: auto-resume
        # -----------------------
        payload_existing = _metrics_ok(metrics_path)
        if payload_existing is not None:
            already = any(
                r.get("combo") == combo_name and r.get("fusion") == fusion and r.get("ok") is True
                for r in results
            )
            if not already:
                best_metrics = payload_existing.get("best_val_metrics", {})
                results.append({
                    "combo": combo_name,
                    "ok": True,
                    "stage": "done",
                    "fusion": fusion,
                    "use_blocks": list(payload_existing.get("use_blocks", args.use)),
                    "groups": groups,
                    "aug_folders": None,
                    "stability_dir": None,
                    "build_log": str(build_log),
                    "classify_log": str(cls_log),
                    "metrics_json": str(metrics_path),
                    "best_val_metrics": best_metrics,
                    "best_epoch": payload_existing.get("best_epoch", None),
                    "n_train": payload_existing.get("n_train", None),
                    "n_val": payload_existing.get("n_val", None),
                    "n_total": payload_existing.get("n_total", None),
                    "resumed_from_existing_metrics": True,
                })
                _write_json(results_path, results)
                _write_json(ranked_path, _rank_successes(results))
            print(f"[SKIP] {combo_name} ({fusion}) already has metrics: {metrics_path}")
            continue

        # -----------------------
        # 0) Decide aug folders
        # -----------------------
        aug_include: List[str] = []
        for g in groups:
            aug_include.extend(AUG_GROUPS[g])

        # -----------------------
        # 1) Build stability h5
        # -----------------------
        combo_out_dir.mkdir(parents=True, exist_ok=True)

        build_cmd = [
            "python", "build_stability_embeddings.py",
            "--baseline_dir", args.baseline_dir,
            "--aug_root", args.aug_root,
            "--out_dir", str(combo_out_dir),
            "--aug_include", ",".join(aug_include),
            "--require_all_augs",
            "--min_k", str(args.min_k),
        ]
        if args.recursive:
            build_cmd.append("--recursive")

        rc = run_and_log(build_cmd, build_log)
        if rc != 0:
            _safe_rmtree(combo_out_dir)
            results.append({
                "combo": combo_name,
                "ok": False,
                "stage": "build",
                "fusion": fusion,
                "use_blocks": list(args.use),
                "groups": groups,
                "aug_folders": aug_include,
                "build_log": str(build_log),
            })
            _write_json(results_path, results)
            _write_json(ranked_path, _rank_successes(results))
            continue

        # -----------------------
        # 2) Classify (single fusion)
        # -----------------------
        cls_cmd = [
            "python", args.classifier_script,
            "--h5_dir", str(combo_out_dir),
            "--use", *args.use,
            "--fusion", fusion,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--seed", str(args.seed),
            "--val_frac", str(args.val_frac),
            "--lr", str(args.lr),
            "--weight_decay", str(args.weight_decay),
            "--z_dim", str(args.z_dim),
            "--dropout", str(args.dropout),
            "--out_json", str(metrics_path),
        ]

        # Only pass gated scheduling args if gated (safe either way, but keep logs clean)
        if fusion == "gated":
            cls_cmd.extend([
                "--temp_schedule", str(args.temp_schedule),
                "--t_start", str(args.t_start),
                "--t_end", str(args.t_end),
            ])

        rc2 = run_and_log(cls_cmd, cls_log)
        ok2 = (rc2 == 0) and metrics_path.exists()

        if not ok2:
            results.append({
                "combo": combo_name,
                "ok": False,
                "stage": "classify",
                "fusion": fusion,
                "use_blocks": list(args.use),
                "groups": groups,
                "aug_folders": aug_include,
                "stability_dir": str(combo_out_dir),
                "build_log": str(build_log),
                "classify_log": str(cls_log),
                "metrics_json": str(metrics_path),
            })
            _write_json(results_path, results)
            _write_json(ranked_path, _rank_successes(results))

            if (not args.keep_stability_h5) and args.delete_on_failure:
                _safe_rmtree(combo_out_dir)
            continue

        payload = _read_json(metrics_path)
        best_metrics = payload.get("best_val_metrics", {})

        results.append({
            "combo": combo_name,
            "ok": True,
            "stage": "done",
            "fusion": fusion,
            "use_blocks": list(payload.get("use_blocks", args.use)),
            "groups": groups,
            "aug_folders": aug_include,
            "stability_dir": str(combo_out_dir),
            "build_log": str(build_log),
            "classify_log": str(cls_log),
            "metrics_json": str(metrics_path),
            "best_val_metrics": best_metrics,
            "best_epoch": payload.get("best_epoch", None),
            "n_train": payload.get("n_train", None),
            "n_val": payload.get("n_val", None),
            "n_total": payload.get("n_total", None),
        })

        _write_json(results_path, results)
        _write_json(ranked_path, _rank_successes(results))

        # -----------------------
        # 3) Cleanup intermediate stability h5
        # -----------------------
        if not args.keep_stability_h5:
            _safe_rmtree(combo_out_dir)

    # final regen
    _write_json(results_path, results)
    _write_json(ranked_path, _rank_successes(results))

    print(f"[DONE] Wrote:\n  - {results_path}\n  - {ranked_path}")
    print(f"[INFO] Temp stability root (may be empty if cleanup ran): {tmp_stability_root}")


if __name__ == "__main__":
    main()
