#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./run_binary_pipeline.sh \
    --raw-input /path/to/raw_comments_export.csv \
    --workdir /path/to/output_dir \
    --seed 42 \
    --train-frac 0.70 \
    --val-frac 0.15 \
    --test-frac 0.15 \
    --min-class-count 60 \
    --max-majority-ratio 0.70

Optional flags:
  --skip-annotate         Skip auto-annotation step (requires raw_comments_labeled.csv to exist in workdir)

Expected scripts in current directory:
  - annotate_from_rubric.py
  - qa_validator.py
  - make_binary_and_split.py
USAGE
}

RAW_INPUT=""
WORKDIR=""
SEED="42"
TRAIN_FRAC="0.70"
VAL_FRAC="0.15"
TEST_FRAC="0.15"
MIN_CLASS_COUNT="60"
MAX_MAJORITY_RATIO="0.70"
SKIP_ANNOTATE="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --raw-input)
      RAW_INPUT="${2:-}"; shift 2 ;;
    --workdir)
      WORKDIR="${2:-}"; shift 2 ;;
    --seed)
      SEED="${2:-}"; shift 2 ;;
    --train-frac)
      TRAIN_FRAC="${2:-}"; shift 2 ;;
    --val-frac)
      VAL_FRAC="${2:-}"; shift 2 ;;
    --test-frac)
      TEST_FRAC="${2:-}"; shift 2 ;;
    --min-class-count)
      MIN_CLASS_COUNT="${2:-}"; shift 2 ;;
    --max-majority-ratio)
      MAX_MAJORITY_RATIO="${2:-}"; shift 2 ;;
    --skip-annotate)
      SKIP_ANNOTATE="1"; shift 1 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "[ERROR] unknown argument: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ -z "$RAW_INPUT" || -z "$WORKDIR" ]]; then
  echo "[ERROR] --raw-input and --workdir are required" >&2
  usage
  exit 2
fi

if [[ ! -f "$RAW_INPUT" ]]; then
  echo "[ERROR] raw input not found: $RAW_INPUT" >&2
  exit 2
fi

for script in annotate_from_rubric.py qa_validator.py make_binary_and_split.py; do
  if [[ ! -f "$script" ]]; then
    echo "[ERROR] required script not found in current dir: $script" >&2
    exit 2
  fi
done

mkdir -p "$WORKDIR"

LABELED_CSV="$WORKDIR/raw_comments_labeled.csv"
CLEAN_CSV="$WORKDIR/raw_comments_labeled.clean.csv"
BINARY_CSV="$WORKDIR/raw_comments_binary.csv"
TRAIN_CSV="$WORKDIR/train_binary.csv"
VAL_CSV="$WORKDIR/val_binary.csv"
TEST_CSV="$WORKDIR/test_binary.csv"

run_py() {
  if command -v uv >/dev/null 2>&1; then
    uv run python "$@"
  else
    python3 "$@"
  fi
}

echo "[INFO] pipeline starting"
echo "[INFO] raw_input=$RAW_INPUT"
echo "[INFO] workdir=$WORKDIR"
echo "[INFO] seed=$SEED"
echo "[INFO] split=train:$TRAIN_FRAC val:$VAL_FRAC test:$TEST_FRAC"

if [[ "$SKIP_ANNOTATE" == "1" ]]; then
  if [[ ! -f "$LABELED_CSV" ]]; then
    echo "[ERROR] --skip-annotate set, but labeled file missing: $LABELED_CSV" >&2
    exit 2
  fi
  echo "[INFO] skipping annotation (using existing $LABELED_CSV)"
else
  echo "[STEP] annotation -> $LABELED_CSV"
  run_py annotate_from_rubric.py \
    --input "$RAW_INPUT" \
    --output "$LABELED_CSV"
fi

echo "[STEP] QA + clean -> $CLEAN_CSV"
run_py qa_validator.py \
  --input "$LABELED_CSV" \
  --write-clean "$CLEAN_CSV" \
  --min-class-count "$MIN_CLASS_COUNT" \
  --max-majority-ratio "$MAX_MAJORITY_RATIO" \
  --train-frac "$TRAIN_FRAC" \
  --val-frac "$VAL_FRAC" \
  --test-frac "$TEST_FRAC"

echo "[STEP] binary mapping + stratified split"
run_py make_binary_and_split.py \
  --input "$CLEAN_CSV" \
  --out-binary "$BINARY_CSV" \
  --train-out "$TRAIN_CSV" \
  --val-out "$VAL_CSV" \
  --test-out "$TEST_CSV" \
  --train-frac "$TRAIN_FRAC" \
  --val-frac "$VAL_FRAC" \
  --test-frac "$TEST_FRAC" \
  --seed "$SEED"

echo "[STEP] binary integrity check"
run_py - <<PY
import csv
from pathlib import Path

paths = {
    "binary": Path(r"$BINARY_CSV"),
    "train": Path(r"$TRAIN_CSV"),
    "val": Path(r"$VAL_CSV"),
    "test": Path(r"$TEST_CSV"),
}
allowed = {"actionable", "non_actionable"}

for name, p in paths.items():
    if not p.exists():
        raise SystemExit(f"[ERROR] missing file: {p}")
    with p.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"[ERROR] {name} split has zero rows: {p}")
    required = {"comment_id", "text", "parent_text", "label"}
    if set(rows[0].keys()) != required:
        raise SystemExit(f"[ERROR] {name} columns mismatch: {rows[0].keys()} != {required}")
    bad = [r["label"] for r in rows if r["label"] not in allowed]
    if bad:
        raise SystemExit(f"[ERROR] {name} has invalid labels, sample={bad[:5]}")
    a = sum(1 for r in rows if r["label"] == "actionable")
    n = sum(1 for r in rows if r["label"] == "non_actionable")
    print(f"[INFO] {name}: rows={len(rows)} actionable={a} non_actionable={n}")

print("[PASS] binary outputs validated")
PY

cat <<EOF
[DONE] Binary training files are ready:

  $BINARY_CSV
  $TRAIN_CSV
  $VAL_CSV
  $TEST_CSV

Next (Rust training):
  cargo run --release -- train \\
    --train-csv "$TRAIN_CSV" \\
    --val-csv "$VAL_CSV" \\
    --model-out "$WORKDIR/model_binary_nb.json" \\
    --min-df 2 --max-features 20000 --alpha 1.0
EOF
