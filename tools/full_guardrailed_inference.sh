#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/bryan/Documents/rust/classifier"
RAW_INPUT="/home/bryan/rapid-response/cio-rapid-response/cmd/bot-scraper/raw_comments_export.csv"

MODEL_PATH="$PROJECT_ROOT/model_binary_response_nb.json"
RULES_PATH="$PROJECT_ROOT/guardrail_rules.yaml"

OUTPUT_DIR="$PROJECT_ROOT/ml_artifacts"
BASE_PRED="$OUTPUT_DIR/raw_comments_export_predictions_response.csv"
GUARDRAILED_STRICT_PRED="$OUTPUT_DIR/raw_comments_export_predictions_response_guardrailed_strict.csv"
GUARDRAILED_COMPAT_PRED="$OUTPUT_DIR/raw_comments_export_predictions_response_guardrailed.csv"
CHANGES_CSV="$OUTPUT_DIR/raw_comments_export_predictions_changes_strict.csv"

WINNOTIFY_DIR_DEFAULT="/home/bryan/Documents/go/winnotify"
WINNOTIFY_DIR="${WINNOTIFY_DIR:-$WINNOTIFY_DIR_DEFAULT}"

if [ -d "$WINNOTIFY_DIR" ]; then
  export PATH="$WINNOTIFY_DIR:$PATH"
fi

WINNOTIFY_BIN="${WINNOTIFY_BIN:-}"
if [ -z "$WINNOTIFY_BIN" ] && command -v winnotify >/dev/null 2>&1; then
  WINNOTIFY_BIN="$(command -v winnotify)"
fi

notify() {
  if [ -n "${WINNOTIFY_BIN:-}" ]; then
    "$WINNOTIFY_BIN" -title "Classifier Pipeline" "$1" || true
  fi
}

on_error() {
  notify "Full strict guardrailed pipeline failed"
}
trap on_error ERR

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "[ERROR] uv is required but not found in PATH."
  exit 1
fi

if [ ! -f "$RAW_INPUT" ]; then
  echo "[ERROR] Raw input CSV not found: $RAW_INPUT"
  exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
  echo "[ERROR] Model artifact not found: $MODEL_PATH"
  exit 1
fi

if [ ! -f "$RULES_PATH" ]; then
  echo "[ERROR] Guardrail rules not found: $RULES_PATH"
  exit 1
fi

notify "Full strict guardrailed pipeline started"
echo "[STEP] Running model inference on raw input..."
cargo run --release -- predict-csv \
  --model-path "$MODEL_PATH" \
  --input-csv "$RAW_INPUT" \
  --output-csv "$BASE_PRED"

echo "[STEP] Applying strict guardrails..."
uv run python "$PROJECT_ROOT/policy_guardrail_with_rules.py" \
  --input "$BASE_PRED" \
  --output "$GUARDRAILED_STRICT_PRED" \
  --rules "$RULES_PATH"

# Keep compatibility with scripts expecting the non-strict filename.
cp "$GUARDRAILED_STRICT_PRED" "$GUARDRAILED_COMPAT_PRED"

echo "[STEP] Building changes report..."
uv run python "$PROJECT_ROOT/diff_report.py" \
  --before "$BASE_PRED" \
  --after "$GUARDRAILED_STRICT_PRED" \
  --output "$CHANGES_CSV" \
  --max-print 40

echo "[STEP] Exporting actionable/non-actionable queues..."
bash "$PROJECT_ROOT/export_final_queues.sh"

notify "Full strict guardrailed pipeline complete"
echo "[DONE] Full strict guardrailed pipeline complete."
echo "[DONE] Base predictions:      $BASE_PRED"
echo "[DONE] Strict guardrailed:    $GUARDRAILED_STRICT_PRED"
echo "[DONE] Compatibility output:  $GUARDRAILED_COMPAT_PRED"
echo "[DONE] Changes report:        $CHANGES_CSV"
