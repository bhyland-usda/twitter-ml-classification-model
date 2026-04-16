#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/bryan/Documents/rust/classifier"
INPUT_CSV="${1:-$PROJECT_ROOT/ml_artifacts/raw_comments_export_predictions_response_guardrailed.csv}"
OUT_DIR="${2:-$PROJECT_ROOT/ml_artifacts/queues}"

if ! command -v uv >/dev/null 2>&1; then
  echo "[ERROR] uv is required but was not found in PATH."
  exit 1
fi

mkdir -p "$OUT_DIR"

uv run python - "$INPUT_CSV" "$OUT_DIR" <<'PY'
import csv
import sys
from collections import Counter
from datetime import datetime, UTC
from pathlib import Path

project_root = Path("/home/bryan/Documents/rust/classifier")
input_csv = Path(sys.argv[1]) if len(sys.argv) > 1 else project_root / "ml_artifacts" / "raw_comments_export_predictions_response_guardrailed.csv"
out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else project_root / "ml_artifacts" / "queues"
out_dir.mkdir(parents=True, exist_ok=True)

if not input_csv.exists():
    raise SystemExit(f"[ERROR] missing input CSV: {input_csv}")

timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
actionable_out = out_dir / f"actionable_queue_{timestamp}.csv"
non_actionable_out = out_dir / f"non_actionable_log_{timestamp}.csv"
summary_out = out_dir / f"queue_summary_{timestamp}.txt"

with input_csv.open("r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

if not rows:
    raise SystemExit("[ERROR] input has zero rows")

required = {
    "comment_id",
    "text",
    "predicted_label",
    "confidence",
    "guardrail_version",
    "guardrail_triggered",
    "guardrail_rule_ids",
    "guardrail_categories",
    "final_label",
    "final_confidence",
}
missing = required - set(rows[0].keys())
if missing:
    raise SystemExit(f"[ERROR] input missing columns: {sorted(missing)}")

actionable_rows = [r for r in rows if (r.get("final_label") or "").strip() == "actionable"]
non_actionable_rows = [r for r in rows if (r.get("final_label") or "").strip() != "actionable"]

def write(path: Path, data):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(data)

write(actionable_out, actionable_rows)
write(non_actionable_out, non_actionable_rows)

guardrail_triggered = sum(
    1 for r in rows if (r.get("guardrail_triggered") or "").strip().lower() == "true"
)
category_counts = Counter()
for r in rows:
    cats = [c for c in (r.get("guardrail_categories") or "").split("|") if c]
    for c in cats:
        category_counts[c] += 1

with summary_out.open("w", encoding="utf-8") as f:
    f.write(f"total_rows={len(rows)}\n")
    f.write(f"actionable_rows={len(actionable_rows)}\n")
    f.write(f"non_actionable_rows={len(non_actionable_rows)}\n")
    f.write(f"guardrail_triggered_rows={guardrail_triggered}\n")
    f.write("guardrail_category_counts:\n")
    for cat, count in sorted(category_counts.items(), key=lambda x: (-x[1], x[0])):
        f.write(f"  {cat}={count}\n")
    f.write(f"actionable_queue_file={actionable_out}\n")
    f.write(f"non_actionable_log_file={non_actionable_out}\n")

print(f"[INFO] actionable_queue={actionable_out}")
print(f"[INFO] non_actionable_log={non_actionable_out}")
print(f"[INFO] summary={summary_out}")
PY
