#!/usr/bin/env python3
"""
sweep_aug_combos.py  (RESUME-SAFE + TEMP H5)

Sweeps over augmentation *families* (each family includes L0/L1/L2 together),
builds stability .h5 outputs for each combo (ONLY in a temporary directory),
trains the stability classifier (if missing),
and writes per-combo BEST-checkpoint metrics JSON.

Behavior:
- If metrics JSON exists -> EVERYTHING for that combo is skipped (no temp h5 created).
- If metrics JSON missing -> build stability h5 into a per-combo temp dir, train classifier,
  then DELETE the temp h5 dir after metrics are successfully written.

Failure behavior:
- If build/classifier fails -> temp dir is PRESERVED (moved into out_root/stability_h5_failed/<combo>)
  so you can debug.
"""

import argparse
import itertools
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any

# -------------------------
# Augmentation families
# -------------------------
def L012(base: str):
    return [f"{base}_L0", f"{base}_L1", f"{base}_L2"]


AUG_GROUPS: Dict[str, List[str]] = {
    "gaussian": L012("gaussian"),
    #"affine": L012("affine"),
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
    return p.returncode


def load_json_safe(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def short_metrics_str(best_val_metrics: Dict[str, Any]) -> str:
    if not isinstance(best_val_metrics, dict):
        return ""
    keys = ["auroc", "f1", "accuracy", "sens", "spec", "precision", "recall"]
    parts = []
    for k in keys:
        v = best_val_metrics.get(k, None)
        if isinstance(v, (int, float)):
            parts.append(f"{k}={v:.4f}")
    return ", ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_dir", required=True)
    ap.add_argument("--aug_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--use", nargs="+", default=["features", "iqr", "mean", "std"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--max_combos", type=int, default=0)
    ap.add_argument("--min_k", type=int, default=2)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    logs_dir = out_root / "_logs"
    metrics_dir = out_root / "_metrics"

    # temp h5 parent (kept tidy; combo-specific dirs are auto-deleted on success)
    tmp_root = out_root / "_tmp_stability_h5"
    failed_root = out_root / "stability_h5_failed"

    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)
    failed_root.mkdir(parents=True, exist_ok=True)

    combos = list(powerset_groups(AUG_GROUPS.keys()))
    if args.max_combos and args.max_combos > 0:
        combos = combos[: args.max_combos]

    results: List[Dict[str, Any]] = []

    for groups in combos:
        combo_name = "+".join(groups)

        aug_include: List[str] = []
        for g in groups:
            aug_include.extend(AUG_GROUPS[g])

        build_log = logs_dir / f"{combo_name}.build.log"
        cls_log = logs_dir / f"{combo_name}.classify.log"
        metrics_path = metrics_dir / f"{combo_name}.best_val.json"

        # -----------------------
        # 0) If metrics exist, skip everything for this combo
        # -----------------------
        if metrics_path.exists():
            payload = load_json_safe(metrics_path)
            results.append({
                "combo": combo_name,
                "ok": True,
                "groups": groups,
                "aug_folders": aug_include,
                "metrics_json": str(metrics_path),
                "best_val_metrics": payload.get("best_val_metrics", {}),
                "best_epoch": payload.get("best_epoch", None),
                "n_train": payload.get("n_train", None),
                "n_val": payload.get("n_val", None),
                "n_total": payload.get("n_total", None),
                "skipped": True,
            })
            print(f"[SKIP] {combo_name} -> metrics already exist: {metrics_path.name}")
            (out_root / "sweep_results.json").write_text(json.dumps(results, indent=2))
            continue

        # -----------------------
        # 1) Create per-combo TEMP dir for stability h5
        # -----------------------
        with tempfile.TemporaryDirectory(prefix=f"{combo_name}__", dir=str(tmp_root)) as tmp_dir_str:
            combo_tmp_dir = Path(tmp_dir_str)

            # -----------------------
            # 2) Build stability h5 into TEMP dir
            # -----------------------
            build_cmd = [
                "python", "build_stability_embeddings.py",
                "--baseline_dir", args.baseline_dir,
                "--aug_root", args.aug_root,
                "--out_dir", str(combo_tmp_dir),
                "--aug_include", ",".join(aug_include),
                "--require_all_augs",
                "--min_k", str(args.min_k),
            ]
            if args.recursive:
                build_cmd.append("--recursive")

            rc = run_and_log(build_cmd, build_log)
            if rc != 0:
                # preserve temp dir for debugging
                keep_dir = failed_root / combo_name
                if keep_dir.exists():
                    shutil.rmtree(keep_dir)
                shutil.copytree(combo_tmp_dir, keep_dir)

                results.append({
                    "combo": combo_name,
                    "ok": False,
                    "stage": "build",
                    "groups": groups,
                    "aug_folders": aug_include,
                    "temp_stability_dir_preserved": str(keep_dir),
                    "build_log": str(build_log),
                })
                (out_root / "sweep_results.json").write_text(json.dumps(results, indent=2))
                print(f"[FAIL] Build failed for {combo_name} (temp preserved at {keep_dir})")
                continue

            # -----------------------
            # 3) Train classifier using TEMP dir -> writes metrics JSON (persistent)
            # -----------------------
            cls_cmd = [
                "python", "stability_classifier.py",
                "--h5_dir", str(combo_tmp_dir),
                "--use", *args.use,
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--seed", str(args.seed),
                "--val_frac", str(args.val_frac),
                "--lr", str(args.lr),
                "--weight_decay", str(args.weight_decay),
                "--out_json", str(metrics_path),
            ]

            rc = run_and_log(cls_cmd, cls_log)
            if rc != 0 or not metrics_path.exists():
                # preserve temp dir for debugging
                keep_dir = failed_root / combo_name
                if keep_dir.exists():
                    shutil.rmtree(keep_dir)
                shutil.copytree(combo_tmp_dir, keep_dir)

                results.append({
                    "combo": combo_name,
                    "ok": False,
                    "stage": "classify",
                    "groups": groups,
                    "aug_folders": aug_include,
                    "temp_stability_dir_preserved": str(keep_dir),
                    "classify_log": str(cls_log),
                })
                (out_root / "sweep_results.json").write_text(json.dumps(results, indent=2))
                print(f"[FAIL] Classifier failed for {combo_name} (temp preserved at {keep_dir})")
                continue

            payload = load_json_safe(metrics_path)
            best_val_metrics = payload.get("best_val_metrics", {}) if isinstance(payload, dict) else {}

            results.append({
                "combo": combo_name,
                "ok": True,
                "groups": groups,
                "aug_folders": aug_include,
                "metrics_json": str(metrics_path),
                "best_val_metrics": best_val_metrics,
                "best_epoch": payload.get("best_epoch", None),
                "n_train": payload.get("n_train", None),
                "n_val": payload.get("n_val", None),
                "n_total": payload.get("n_total", None),
            })

            # "final metrics printed" (per combo)
            msg = short_metrics_str(best_val_metrics)
            if msg:
                print(f"[OK] {combo_name} -> {msg}")
            else:
                print(f"[OK] {combo_name} -> metrics written: {metrics_path.name}")

            # TEMP DIR IS AUTO-DELETED HERE by TemporaryDirectory()

        # -----------------------
        # Rolling writes (persistent)
        # -----------------------
        (out_root / "sweep_results.json").write_text(json.dumps(results, indent=2))

        ranked = [
            r for r in results
            if r.get("ok") and isinstance(r.get("best_val_metrics", {}).get("auroc", None), (int, float))
        ]
        ranked.sort(key=lambda r: float(r["best_val_metrics"]["auroc"]), reverse=True)
        (out_root / "sweep_ranked_by_auroc.json").write_text(json.dumps(ranked, indent=2))

    print("[DONE]")
    print(f"  - {out_root / 'sweep_results.json'}")
    print(f"  - {out_root / 'sweep_ranked_by_auroc.json'}")
    print(f"  - temp h5 root (should be empty unless runs are active): {tmp_root}")
    print(f"  - failed temp h5 preserved here (only on failure): {failed_root}")


if __name__ == "__main__":
    main()
