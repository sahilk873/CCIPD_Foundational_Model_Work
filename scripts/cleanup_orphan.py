import json
from pathlib import Path
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", required=True, help="Path to _logs directory")
    ap.add_argument("--metrics_dir", required=True, help="Path to _metrics directory")
    ap.add_argument("--results_json", required=True, help="Path to sweep_results.json")
    ap.add_argument("--dry_run", action="store_true", help="Only print what would be deleted")
    args = ap.parse_args()

    logs_dir = Path(args.logs_dir)
    metrics_dir = Path(args.metrics_dir)
    results_path = Path(args.results_json)

    # Load valid combos
    data = json.loads(results_path.read_text())
    valid_combos = set(item["combo"] for item in data if "combo" in item)

    removed_logs = 0
    kept_logs = 0
    removed_metrics = 0
    kept_metrics = 0

    # --- Clean logs ---
    for log_file in logs_dir.glob("*.log"):
        combo = log_file.name.replace(".build.log", "").replace(".classify.log", "")
        if combo not in valid_combos:
            if args.dry_run:
                print(f"[DRY] delete log {log_file}")
            else:
                log_file.unlink()
                print(f"[DEL] log {log_file}")
            removed_logs += 1
        else:
            kept_logs += 1

    # --- Clean metrics ---
    for json_file in metrics_dir.glob("*.json"):
        combo = json_file.name.replace(".best_val.json", "")
        if combo not in valid_combos:
            if args.dry_run:
                print(f"[DRY] delete metrics {json_file}")
            else:
                json_file.unlink()
                print(f"[DEL] metrics {json_file}")
            removed_metrics += 1
        else:
            kept_metrics += 1

    print("\n===== CLEANUP SUMMARY =====")
    print(f"Kept logs:     {kept_logs}")
    print(f"Removed logs:  {removed_logs}")
    print(f"Kept metrics:  {kept_metrics}")
    print(f"Removed metrics: {removed_metrics}")

if __name__ == "__main__":
    main()






