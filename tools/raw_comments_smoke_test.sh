#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/bryan/Documents/rust/classifier"
RAW_INPUT="/home/bryan/rapid-response/cio-rapid-response/cmd/bot-scraper/raw_comments_export.csv"

MODEL_PATH="$PROJECT_ROOT/model_binary_nb.json"
TRAIN_CSV="$PROJECT_ROOT/train_binary.csv"
VAL_CSV="$PROJECT_ROOT/val_binary.csv"

OUTPUT_DIR="/home/bryan/rapid-response/cio-rapid-response/cmd/bot-scraper/ml_artifacts"
PREDICTIONS_CSV="$OUTPUT_DIR/raw_comments_export_predictions.csv"

mkdir -p "$OUTPUT_DIR"

if [[ ! -f "$RAW_INPUT" ]]; then
  echo "[ERROR] raw input CSV not found: $RAW_INPUT" >&2
  exit 2
fi

if [[ ! -f "$TRAIN_CSV" || ! -f "$VAL_CSV" ]]; then
  echo "[ERROR] missing training split files in $PROJECT_ROOT" >&2
  echo "Expected: train_binary.csv and val_binary.csv" >&2
  exit 2
fi

cd "$PROJECT_ROOT"

# Build once for faster repeated runs.
echo "[STEP] cargo build --release"
cargo build --release

# Train only if model is missing.
if [[ ! -f "$MODEL_PATH" ]]; then
  echo "[STEP] model not found; training baseline model"
  cargo run --release -- train \
    --train-csv "$TRAIN_CSV" \
    --val-csv "$VAL_CSV" \
    --model-out "$MODEL_PATH" \
    --min-df 2 \
    --max-features 20000 \
    --alpha 1.0
else
  echo "[INFO] using existing model: $MODEL_PATH"
fi

echo "[STEP] running inference on raw comments export"
cargo run --release -- predict-csv \
  --model-path "$MODEL_PATH" \
  --input-csv "$RAW_INPUT" \
  --output-csv "$PREDICTIONS_CSV"

echo "[STEP] summarizing predictions"
python3 - <<'PY'
import csv
from collections import Counter
from pathlib import Path

pred_path = Path("/home/bryan/rapid-response/cio-rapid-response/cmd/bot-scraper/ml_artifacts/raw_comments_export_predictions.csv")
if not pred_path.exists():
    raise SystemExit(f"[ERROR] missing predictions file: {pred_path}")

with pred_path.open("r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

if not rows:
    raise SystemExit("[ERROR] predictions CSV has zero rows")

required = {"comment_id", "text", "predicted_label", "confidence"}
missing = required - set(rows[0].keys())
if missing:
    raise SystemExit(f"[ERROR] predictions CSV missing columns: {sorted(missing)}")

counts = Counter(r["predicted_label"].strip() for r in rows)
total = len(rows)
actionable = counts.get("actionable", 0)
non_actionable = counts.get("non_actionable", 0)

print(f"[INFO] total_rows={total}")
print(f"[INFO] actionable={actionable} ({(actionable/total*100):.2f}%)")
print(f"[INFO] non_actionable={non_actionable} ({(non_actionable/total*100):.2f}%)")

# Top actionable by confidence
actionable_rows = []
for r in rows:
    if r["predicted_label"].strip() != "actionable":
        continue
    try:
        confidence = float(r["confidence"])
    except ValueError:
        confidence = 0.0
    actionable_rows.append((confidence, r))

actionable_rows.sort(key=lambda x: x[0], reverse=True)
top_n = actionable_rows[:20]

print("\n[INFO] top_actionable_by_confidence (up to 20):")
for i, (conf, r) in enumerate(top_n, start=1):
    snippet = (r["text"] or "").replace("\n", " ").strip()
    snippet = snippet[:180]
    print(f"{i:02d}. conf={conf:.6f} id={r['comment_id']} text={snippet}")
PY

echo "[DONE] predictions written to: $PREDICTIONS_CSV"
