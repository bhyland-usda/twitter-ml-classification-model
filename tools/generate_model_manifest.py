#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reproducible model artifact manifest metadata."
    )
    parser.add_argument(
        "--model-path",
        default="ml_artifacts/guardrail_distill/sweep_fullfit/run_010/model.json",
        help="Path to model artifact JSON.",
    )
    parser.add_argument(
        "--train-csv",
        default="ml_artifacts/guardrail_distill/guardrail_teacher_labels_full.csv",
        help="Training CSV used for this model (comment_id,text,parent_text,label).",
    )
    parser.add_argument(
        "--known-actionable-csv",
        default="",
        help=(
            "Known-good actionable CSV path. If omitted, latest "
            "ml_artifacts/queues/actionable_queue_*.csv is used if present."
        ),
    )
    parser.add_argument(
        "--guardrail-rules",
        default="guardrail_rules.yaml",
        help="Guardrail rules YAML path.",
    )
    parser.add_argument(
        "--teacher-output-csv",
        default="ml_artifacts/raw_comments_export_predictions_response_guardrailed_strict.csv",
        help=(
            "Optional teacher output CSV used for alignment checks. "
            "Expected columns include comment_id and final_label."
        ),
    )
    parser.add_argument(
        "--output",
        default="ml_artifacts/model_manifest.current.json",
        help="Output manifest JSON path.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_guardrail_version(path: Path) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r'^\s*version:\s*["\']?([^"\']+)["\']?\s*$', text, flags=re.MULTILINE)
    return m.group(1).strip() if m else ""


def latest_known_actionable_csv(explicit: str) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None

    queues = Path("ml_artifacts/queues")
    if not queues.exists():
        return None
    candidates = sorted(queues.glob("actionable_queue_*.csv"))
    return candidates[-1] if candidates else None


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def label_distribution(
    rows: list[dict[str, str]], label_col: str = "label"
) -> dict[str, int]:
    counts = Counter((r.get(label_col) or "").strip() for r in rows)
    return {k: int(v) for k, v in sorted(counts.items()) if k}


def known_actionable_ids(path: Optional[Path]) -> set[str]:
    if not path or not path.exists():
        return set()
    rows = csv_rows(path)
    ids = set()
    for r in rows:
        cid = (r.get("comment_id") or "").strip()
        if cid:
            ids.add(cid)
    return ids


def teacher_alignment(
    teacher_csv: Path,
    known_ids: set[str],
) -> dict[str, Any]:
    if not teacher_csv.exists():
        return {
            "available": False,
            "reason": f"missing_file:{teacher_csv}",
        }

    rows = csv_rows(teacher_csv)
    if not rows:
        return {
            "available": False,
            "reason": f"empty_file:{teacher_csv}",
        }

    cols = set(rows[0].keys())
    if "comment_id" not in cols:
        return {
            "available": False,
            "reason": "missing_column:comment_id",
        }

    label_col = (
        "final_label"
        if "final_label" in cols
        else ("predicted_label" if "predicted_label" in cols else "")
    )
    if not label_col:
        return {
            "available": False,
            "reason": "missing_label_column(final_label|predicted_label)",
        }

    pred_actionable = {
        (r.get("comment_id") or "").strip()
        for r in rows
        if (r.get(label_col) or "").strip() == "actionable"
        and (r.get("comment_id") or "").strip()
    }

    summary: dict[str, Any] = {
        "available": True,
        "teacher_csv": str(teacher_csv),
        "teacher_label_column": label_col,
        "teacher_rows": len(rows),
        "teacher_actionable_count": len(pred_actionable),
    }

    if known_ids:
        inter = pred_actionable & known_ids
        only_teacher = pred_actionable - known_ids
        only_known = known_ids - pred_actionable
        summary.update(
            {
                "known_good_count": len(known_ids),
                "known_good_intersection_count": len(inter),
                "known_good_only_teacher_count": len(only_teacher),
                "known_good_only_reference_count": len(only_known),
                "known_good_exact_set_match": pred_actionable == known_ids,
                "known_good_intersection_sample": sorted(inter)[:25],
                "known_good_only_teacher_sample": sorted(only_teacher)[:25],
                "known_good_only_reference_sample": sorted(only_known)[:25],
            }
        )

    return summary


def infer_hyperparams_from_leaderboard(model_path: Path) -> dict[str, Any]:
    """
    Attempts to infer run metadata from nearby leaderboard files.
    Works for paths like: .../run_010/model.json
    """
    out: dict[str, Any] = {"available": False}
    run_dir = model_path.parent
    run_id = run_dir.name
    parent = run_dir.parent

    candidates = [
        parent / "leaderboard_fullfit.csv",
        parent / "leaderboard.sorted.csv",
        parent / "leaderboard.csv",
    ]
    board_path = next((p for p in candidates if p.exists()), None)
    if board_path is None:
        return out

    try:
        rows = csv_rows(board_path)
    except Exception:
        return out

    row = None
    for r in rows:
        if (r.get("run_id") or "").strip() == run_id:
            row = r
            break

    if row is None:
        return out

    out = {
        "available": True,
        "leaderboard_path": str(board_path),
        "run_id": run_id,
        "alpha": row.get("alpha", ""),
        "min_df": row.get("min_df", ""),
        "max_features": row.get("max_features", ""),
        "status": row.get("status", ""),
        "validation_macro_f1": row.get("macro_f1", ""),
        "validation_actionable_f1": row.get("actionable_f1", ""),
        "validation_accuracy": row.get("accuracy", ""),
    }
    return out


def main() -> int:
    args = parse_args()

    model_path = Path(args.model_path)
    train_csv = Path(args.train_csv)
    rules_path = Path(args.guardrail_rules)
    teacher_csv = Path(args.teacher_output_csv)
    output_path = Path(args.output)
    known_csv = latest_known_actionable_csv(args.known_actionable_csv)

    if not model_path.exists():
        raise SystemExit(f"[ERROR] missing model artifact: {model_path}")
    if not train_csv.exists():
        raise SystemExit(f"[ERROR] missing train CSV: {train_csv}")

    train_rows = csv_rows(train_csv)
    if not train_rows:
        raise SystemExit(f"[ERROR] train CSV has zero rows: {train_csv}")

    train_cols = list(train_rows[0].keys())
    if "label" not in train_cols:
        raise SystemExit(f"[ERROR] train CSV missing 'label' column: {train_csv}")

    known_ids = known_actionable_ids(known_csv)
    alignment = teacher_alignment(teacher_csv, known_ids)

    manifest: dict[str, Any] = {
        "manifest_version": "1.0.0",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "generator": {
            "script": "generate_model_manifest.py",
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "model": {
            "artifact_path": str(model_path),
            "artifact_sha256": file_sha256(model_path),
            "artifact_size_bytes": model_path.stat().st_size,
            "type": "naive_bayes_text_classifier",
            "hyperparameters": infer_hyperparams_from_leaderboard(model_path),
        },
        "training_data": {
            "train_csv_path": str(train_csv),
            "train_csv_sha256": file_sha256(train_csv),
            "row_count": len(train_rows),
            "columns": train_cols,
            "label_distribution": label_distribution(train_rows, "label"),
        },
        "guardrails": {
            "rules_path": str(rules_path),
            "rules_exists": rules_path.exists(),
            "rules_sha256": file_sha256(rules_path) if rules_path.exists() else "",
            "rules_version": parse_guardrail_version(rules_path),
            "known_good_actionable_csv": str(known_csv) if known_csv else "",
            "known_good_actionable_count": len(known_ids),
            "teacher_alignment": alignment,
        },
        "artifacts": {
            "output_manifest_path": str(output_path),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[INFO] wrote manifest: {output_path}")
    print(f"[INFO] train_rows={manifest['training_data']['row_count']}")
    print(f"[INFO] rules_version={manifest['guardrails']['rules_version']}")
    print(
        "[INFO] teacher_alignment_available="
        f"{manifest['guardrails']['teacher_alignment'].get('available', False)}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
