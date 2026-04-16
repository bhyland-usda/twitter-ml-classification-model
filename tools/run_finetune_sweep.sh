#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./run_finetune_sweep.sh \
    --train-csv /path/train_binary.csv \
    --val-csv /path/val_binary.csv \
    --outdir /path/finetune_runs \
    [--test-csv /path/test_binary.csv] \
    [--alphas "0.25,0.5,1.0,2.0"] \
    [--min-dfs "1,2,3"] \
    [--max-features "5000,10000,20000"] \
    [--cargo-bin "cargo run --release --"]

Requires classifier CLI subcommands:
  train      --train-csv --val-csv --model-out --min-df --max-features --alpha
  predict-csv --model-path --input-csv --output-csv

Outputs:
  <outdir>/leaderboard.csv
  <outdir>/leaderboard.sorted.csv
  <outdir>/best_config.env
  <outdir>/best_val_metrics.json
  <outdir>/best_test_metrics.json (if --test-csv provided)
USAGE
}

TRAIN_CSV=""
VAL_CSV=""
TEST_CSV=""
OUTDIR=""
ALPHAS="0.25,0.5,1.0,2.0"
MIN_DFS="1,2,3"
MAX_FEATURES="5000,10000,20000"
CARGO_BIN="cargo run --release --"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-csv) TRAIN_CSV="${2:-}"; shift 2 ;;
    --val-csv) VAL_CSV="${2:-}"; shift 2 ;;
    --test-csv) TEST_CSV="${2:-}"; shift 2 ;;
    --outdir) OUTDIR="${2:-}"; shift 2 ;;
    --alphas) ALPHAS="${2:-}"; shift 2 ;;
    --min-dfs) MIN_DFS="${2:-}"; shift 2 ;;
    --max-features) MAX_FEATURES="${2:-}"; shift 2 ;;
    --cargo-bin) CARGO_BIN="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$TRAIN_CSV" || -z "$VAL_CSV" || -z "$OUTDIR" ]]; then
  echo "[ERROR] --train-csv, --val-csv, and --outdir are required" >&2
  usage
  exit 2
fi

if [[ ! -f "$TRAIN_CSV" ]]; then
  echo "[ERROR] train CSV not found: $TRAIN_CSV" >&2
  exit 2
fi
if [[ ! -f "$VAL_CSV" ]]; then
  echo "[ERROR] val CSV not found: $VAL_CSV" >&2
  exit 2
fi
if [[ -n "$TEST_CSV" && ! -f "$TEST_CSV" ]]; then
  echo "[ERROR] test CSV not found: $TEST_CSV" >&2
  exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[ERROR] uv is required but was not found in PATH." >&2
  exit 2
fi

mkdir -p "$OUTDIR"

LEADERBOARD="$OUTDIR/leaderboard.csv"
SORTED_LEADERBOARD="$OUTDIR/leaderboard.sorted.csv"
BEST_ENV="$OUTDIR/best_config.env"
BEST_VAL_METRICS="$OUTDIR/best_val_metrics.json"
BEST_TEST_METRICS="$OUTDIR/best_test_metrics.json"

echo "run_id,alpha,min_df,max_features,macro_f1,actionable_f1,accuracy,status,model_path,val_pred_path,train_log,predict_log" > "$LEADERBOARD"

# Split comma-separated strings into arrays
IFS=',' read -r -a ALPHA_ARR <<< "$ALPHAS"
IFS=',' read -r -a MIN_DF_ARR <<< "$MIN_DFS"
IFS=',' read -r -a MAX_FEAT_ARR <<< "$MAX_FEATURES"

run_counter=0

calc_metrics_py() {
  local truth_csv="$1"
  local pred_csv="$2"
  local out_json="$3"

  uv run python - "$truth_csv" "$pred_csv" "$out_json" <<'PY'
import csv
import json
import math
import sys
from collections import defaultdict

truth_csv, pred_csv, out_json = sys.argv[1], sys.argv[2], sys.argv[3]

with open(truth_csv, "r", encoding="utf-8", newline="") as f:
    truth_rows = list(csv.DictReader(f))
with open(pred_csv, "r", encoding="utf-8", newline="") as f:
    pred_rows = list(csv.DictReader(f))

if not truth_rows:
    raise SystemExit("truth CSV has zero rows")
if not pred_rows:
    raise SystemExit("prediction CSV has zero rows")

# Validate columns
truth_cols = set(truth_rows[0].keys())
pred_cols = set(pred_rows[0].keys())
required_truth = {"comment_id", "text", "label"}
required_pred = {"comment_id", "text", "predicted_label", "confidence"}

missing_truth = required_truth - truth_cols
missing_pred = required_pred - pred_cols
if missing_truth:
    raise SystemExit(f"truth CSV missing columns: {sorted(missing_truth)}")
if missing_pred:
    raise SystemExit(f"pred CSV missing columns: {sorted(missing_pred)}")

# Prefer comment_id join if unique and complete; otherwise row-order join
truth_ids = [r["comment_id"].strip() for r in truth_rows]
pred_ids = [r["comment_id"].strip() for r in pred_rows]

use_id_join = True
if len(set(truth_ids)) != len(truth_ids):
    use_id_join = False
if len(set(pred_ids)) != len(pred_ids):
    use_id_join = False
if set(truth_ids) != set(pred_ids):
    use_id_join = False

pairs = []
if use_id_join:
    pred_map = {r["comment_id"].strip(): r["predicted_label"].strip() for r in pred_rows}
    for r in truth_rows:
        cid = r["comment_id"].strip()
        y_true = r["label"].strip()
        y_pred = pred_map[cid]
        pairs.append((y_true, y_pred))
else:
    if len(truth_rows) != len(pred_rows):
        raise SystemExit("row-order fallback failed: truth/pred row counts differ")
    for t, p in zip(truth_rows, pred_rows):
        pairs.append((t["label"].strip(), p["predicted_label"].strip()))

labels = sorted(set(y for y, _ in pairs) | set(y for _, y in pairs))
label_to_idx = {l: i for i, l in enumerate(labels)}
n = len(labels)
cm = [[0 for _ in range(n)] for _ in range(n)]  # [true][pred]

for y_true, y_pred in pairs:
    cm[label_to_idx[y_true]][label_to_idx[y_pred]] += 1

total = len(pairs)
correct = sum(cm[i][i] for i in range(n))
accuracy = correct / total if total else 0.0

per_class = []
f1s = []
for i, lbl in enumerate(labels):
    tp = cm[i][i]
    fp = sum(cm[r][i] for r in range(n) if r != i)
    fn = sum(cm[i][c] for c in range(n) if c != i)
    support = sum(cm[i])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    per_class.append({
        "label": lbl,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })
    f1s.append(f1)

macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
actionable_f1 = 0.0
for m in per_class:
    if m["label"] == "actionable":
        actionable_f1 = m["f1"]
        break

payload = {
    "total": total,
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "actionable_f1": actionable_f1,
    "labels": labels,
    "confusion_matrix": cm,
    "per_class": per_class,
}

with open(out_json, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)

print(f"{macro_f1}\t{actionable_f1}\t{accuracy}")
PY
}

echo "[INFO] Starting sweep..."
for alpha in "${ALPHA_ARR[@]}"; do
  for min_df in "${MIN_DF_ARR[@]}"; do
    for max_features in "${MAX_FEAT_ARR[@]}"; do
      run_counter=$((run_counter + 1))
      run_id=$(printf "run_%03d" "$run_counter")
      run_dir="$OUTDIR/$run_id"
      mkdir -p "$run_dir"

      model_path="$run_dir/model.json"
      val_pred="$run_dir/val_predictions.csv"
      train_log="$run_dir/train.log"
      predict_log="$run_dir/predict.log"
      val_metrics_json="$run_dir/val_metrics.json"

      echo "[INFO] $run_id alpha=$alpha min_df=$min_df max_features=$max_features"

      set +e
      eval "$CARGO_BIN train --train-csv \"$TRAIN_CSV\" --val-csv \"$VAL_CSV\" --model-out \"$model_path\" --min-df \"$min_df\" --max-features \"$max_features\" --alpha \"$alpha\"" >"$train_log" 2>&1
      train_rc=$?
      set -e

      if [[ $train_rc -ne 0 ]]; then
        echo "[WARN] $run_id train failed (rc=$train_rc)"
        echo "$run_id,$alpha,$min_df,$max_features,0,0,0,train_failed,$model_path,$val_pred,$train_log,$predict_log" >> "$LEADERBOARD"
        continue
      fi

      set +e
      eval "$CARGO_BIN predict-csv --model-path \"$model_path\" --input-csv \"$VAL_CSV\" --output-csv \"$val_pred\"" >"$predict_log" 2>&1
      pred_rc=$?
      set -e

      if [[ $pred_rc -ne 0 ]]; then
        echo "[WARN] $run_id predict failed (rc=$pred_rc)"
        echo "$run_id,$alpha,$min_df,$max_features,0,0,0,predict_failed,$model_path,$val_pred,$train_log,$predict_log" >> "$LEADERBOARD"
        continue
      fi

      metrics_line=$(calc_metrics_py "$VAL_CSV" "$val_pred" "$val_metrics_json")
      macro_f1=$(echo "$metrics_line" | awk -F $'\t' '{print $1}')
      actionable_f1=$(echo "$metrics_line" | awk -F $'\t' '{print $2}')
      accuracy=$(echo "$metrics_line" | awk -F $'\t' '{print $3}')

      echo "$run_id,$alpha,$min_df,$max_features,$macro_f1,$actionable_f1,$accuracy,ok,$model_path,$val_pred,$train_log,$predict_log" >> "$LEADERBOARD"
    done
  done
done

uv run python - "$LEADERBOARD" "$SORTED_LEADERBOARD" <<'PY'
import csv
import sys

in_csv, out_csv = sys.argv[1], sys.argv[2]

with open(in_csv, "r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

ok_rows = [r for r in rows if r["status"] == "ok"]
bad_rows = [r for r in rows if r["status"] != "ok"]

def f(x): return float(x)

ok_rows.sort(
    key=lambda r: (
        f(r["macro_f1"]),
        f(r["actionable_f1"]),
        f(r["accuracy"]),
    ),
    reverse=True,
)

with open(out_csv, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    for r in ok_rows + bad_rows:
        w.writerow(r)

if not ok_rows:
    print("[ERROR] no successful runs")
    sys.exit(1)

best = ok_rows[0]
print(best["run_id"])
print(best["alpha"])
print(best["min_df"])
print(best["max_features"])
print(best["model_path"])
PY

best_meta=$(uv run python - "$SORTED_LEADERBOARD" <<'PY'
import csv, sys
p = sys.argv[1]
with open(p, "r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
best = next((r for r in rows if r["status"] == "ok"), None)
if best is None:
    raise SystemExit(1)
print(f'{best["run_id"]}\t{best["alpha"]}\t{best["min_df"]}\t{best["max_features"]}\t{best["model_path"]}')
PY
)

BEST_RUN_ID=$(echo "$best_meta" | awk -F $'\t' '{print $1}')
BEST_ALPHA=$(echo "$best_meta" | awk -F $'\t' '{print $2}')
BEST_MIN_DF=$(echo "$best_meta" | awk -F $'\t' '{print $3}')
BEST_MAX_FEATURES=$(echo "$best_meta" | awk -F $'\t' '{print $4}')
BEST_MODEL_PATH=$(echo "$best_meta" | awk -F $'\t' '{print $5}')
BEST_RUN_DIR="$OUTDIR/$BEST_RUN_ID"

cp "$BEST_RUN_DIR/val_metrics.json" "$BEST_VAL_METRICS"

cat > "$BEST_ENV" <<EOF
BEST_RUN_ID=$BEST_RUN_ID
BEST_ALPHA=$BEST_ALPHA
BEST_MIN_DF=$BEST_MIN_DF
BEST_MAX_FEATURES=$BEST_MAX_FEATURES
BEST_MODEL_PATH=$BEST_MODEL_PATH
BEST_VAL_METRICS=$BEST_VAL_METRICS
EOF

if [[ -n "$TEST_CSV" ]]; then
  BEST_TEST_PRED="$OUTDIR/best_test_predictions.csv"
  BEST_TEST_LOG="$OUTDIR/best_test_predict.log"

  eval "$CARGO_BIN predict-csv --model-path \"$BEST_MODEL_PATH\" --input-csv \"$TEST_CSV\" --output-csv \"$BEST_TEST_PRED\"" >"$BEST_TEST_LOG" 2>&1
  calc_metrics_py "$TEST_CSV" "$BEST_TEST_PRED" "$BEST_TEST_METRICS" >/dev/null
fi

echo "[DONE] Finetune sweep complete"
echo "[INFO] leaderboard: $LEADERBOARD"
echo "[INFO] sorted leaderboard: $SORTED_LEADERBOARD"
echo "[INFO] best config: $BEST_ENV"
echo "[INFO] best val metrics: $BEST_VAL_METRICS"
if [[ -n "$TEST_CSV" ]]; then
  echo "[INFO] best test metrics: $BEST_TEST_METRICS"
fi
