#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./run_finetune_track.sh --track <name> [--outdir <dir>] [--cargo-bin "<cmd>"] [--dry-run]

Tracks:
  baseline
    General response model sweep on train_binary_response.csv / val_binary_response.csv / test_binary_response.csv

  strictness
    High-regularization sweep intended to reduce raw actionable volume

  guardrail_hn
    Sweep on hard-negative-augmented training set (broad structural+policy downgrade reasons)

  guardrail_safety
    Sweep on hard-negative-augmented training set (safety/scope-focused reasons)

Optional overrides:
  --train-csv <path>
  --val-csv <path>
  --test-csv <path>
  --alphas "<csv>"
  --min-dfs "<csv>"
  --max-features "<csv>"

Examples:
  ./run_finetune_track.sh --track baseline
  ./run_finetune_track.sh --track strictness --outdir ml_artifacts/finetune_strictness_v2
  ./run_finetune_track.sh --track guardrail_safety --dry-run
USAGE
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRACK=""
OUTDIR=""
CARGO_BIN="cargo run --release --"
DRY_RUN="0"

# Optional user overrides
TRAIN_CSV_OVERRIDE=""
VAL_CSV_OVERRIDE=""
TEST_CSV_OVERRIDE=""
ALPHAS_OVERRIDE=""
MIN_DFS_OVERRIDE=""
MAX_FEATURES_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --track) TRACK="${2:-}"; shift 2 ;;
    --outdir) OUTDIR="${2:-}"; shift 2 ;;
    --cargo-bin) CARGO_BIN="${2:-}"; shift 2 ;;
    --train-csv) TRAIN_CSV_OVERRIDE="${2:-}"; shift 2 ;;
    --val-csv) VAL_CSV_OVERRIDE="${2:-}"; shift 2 ;;
    --test-csv) TEST_CSV_OVERRIDE="${2:-}"; shift 2 ;;
    --alphas) ALPHAS_OVERRIDE="${2:-}"; shift 2 ;;
    --min-dfs) MIN_DFS_OVERRIDE="${2:-}"; shift 2 ;;
    --max-features) MAX_FEATURES_OVERRIDE="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN="1"; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$TRACK" ]]; then
  echo "[ERROR] --track is required" >&2
  usage
  exit 2
fi

if [[ ! -f "$PROJECT_ROOT/run_finetune_sweep.sh" ]]; then
  echo "[ERROR] Missing required script: $PROJECT_ROOT/run_finetune_sweep.sh" >&2
  exit 2
fi

# Track defaults
TRAIN_CSV=""
VAL_CSV=""
TEST_CSV=""
ALPHAS=""
MIN_DFS=""
MAX_FEATURES=""

case "$TRACK" in
  baseline)
    TRAIN_CSV="$PROJECT_ROOT/train_binary_response.csv"
    VAL_CSV="$PROJECT_ROOT/val_binary_response.csv"
    TEST_CSV="$PROJECT_ROOT/test_binary_response.csv"
    ALPHAS="0.1,0.25,0.5,1.0,2.0,4.0"
    MIN_DFS="1,2,3,5"
    MAX_FEATURES="5000,10000,20000"
    [[ -z "$OUTDIR" ]] && OUTDIR="$PROJECT_ROOT/ml_artifacts/finetune_response_track"
    ;;
  strictness)
    TRAIN_CSV="$PROJECT_ROOT/train_binary_response.csv"
    VAL_CSV="$PROJECT_ROOT/val_binary_response.csv"
    TEST_CSV="$PROJECT_ROOT/test_binary_response.csv"
    ALPHAS="2.0,4.0,8.0,16.0,32.0"
    MIN_DFS="5,8,12"
    MAX_FEATURES="2000,5000"
    [[ -z "$OUTDIR" ]] && OUTDIR="$PROJECT_ROOT/ml_artifacts/finetune_strictness_track"
    ;;
  guardrail_hn)
    TRAIN_CSV="$PROJECT_ROOT/ml_artifacts/guardrail_finetune/train_binary_response_guardrail_hn.csv"
    VAL_CSV="$PROJECT_ROOT/val_binary_response.csv"
    TEST_CSV="$PROJECT_ROOT/test_binary_response.csv"
    ALPHAS="0.25,0.5,1.0,2.0,4.0"
    MIN_DFS="2,3,5"
    MAX_FEATURES="5000,10000,20000"
    [[ -z "$OUTDIR" ]] && OUTDIR="$PROJECT_ROOT/ml_artifacts/finetune_guardrail_hn_track"
    ;;
  guardrail_safety)
    TRAIN_CSV="$PROJECT_ROOT/ml_artifacts/guardrail_finetune/train_binary_response_guardrail_safety.csv"
    VAL_CSV="$PROJECT_ROOT/val_binary_response.csv"
    TEST_CSV="$PROJECT_ROOT/test_binary_response.csv"
    ALPHAS="0.25,0.5,1.0,2.0"
    MIN_DFS="2,3,5"
    MAX_FEATURES="5000,10000,20000"
    [[ -z "$OUTDIR" ]] && OUTDIR="$PROJECT_ROOT/ml_artifacts/finetune_guardrail_safety_track"
    ;;
  *)
    echo "[ERROR] Unknown track: $TRACK" >&2
    usage
    exit 2
    ;;
esac

# Apply user overrides
[[ -n "$TRAIN_CSV_OVERRIDE" ]] && TRAIN_CSV="$TRAIN_CSV_OVERRIDE"
[[ -n "$VAL_CSV_OVERRIDE" ]] && VAL_CSV="$VAL_CSV_OVERRIDE"
[[ -n "$TEST_CSV_OVERRIDE" ]] && TEST_CSV="$TEST_CSV_OVERRIDE"
[[ -n "$ALPHAS_OVERRIDE" ]] && ALPHAS="$ALPHAS_OVERRIDE"
[[ -n "$MIN_DFS_OVERRIDE" ]] && MIN_DFS="$MIN_DFS_OVERRIDE"
[[ -n "$MAX_FEATURES_OVERRIDE" ]] && MAX_FEATURES="$MAX_FEATURES_OVERRIDE"

# Validate key inputs
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

echo "[INFO] track=$TRACK"
echo "[INFO] outdir=$OUTDIR"
echo "[INFO] train_csv=$TRAIN_CSV"
echo "[INFO] val_csv=$VAL_CSV"
echo "[INFO] test_csv=$TEST_CSV"
echo "[INFO] alphas=$ALPHAS"
echo "[INFO] min_dfs=$MIN_DFS"
echo "[INFO] max_features=$MAX_FEATURES"
echo "[INFO] cargo_bin=$CARGO_BIN"

cmd=(
  bash "$PROJECT_ROOT/run_finetune_sweep.sh"
  --train-csv "$TRAIN_CSV"
  --val-csv "$VAL_CSV"
  --outdir "$OUTDIR"
  --alphas "$ALPHAS"
  --min-dfs "$MIN_DFS"
  --max-features "$MAX_FEATURES"
  --cargo-bin "$CARGO_BIN"
)

if [[ -n "$TEST_CSV" ]]; then
  cmd+=(--test-csv "$TEST_CSV")
fi

if [[ "$DRY_RUN" == "1" ]]; then
  printf '[DRY-RUN] %q ' "${cmd[@]}"
  echo
  exit 0
fi

"${cmd[@]}"

echo "[DONE] Track sweep complete: $TRACK"
echo "[INFO] results_dir=$OUTDIR"
