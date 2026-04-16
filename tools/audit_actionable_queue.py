#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ACTION_CUE_PATTERN = re.compile(
    r"\?|"
    r"\b("
    r"please|can we|could we|will you|would you|"
    r"we need|need to|should|must|"
    r"how about|what about|when are you|why are you|"
    r"support|help|audit|investigate|fix|stop|ban|reform|"
    r"eliminate|address|improve|expand|enforce|review|"
    r"contact|come to|reply|respond|explain"
    r")\b",
    re.IGNORECASE,
)

CONCRETE_ACTION_PATTERN = re.compile(
    r"\b("
    r"please|we need|need to|stop|ban|fix|investigate|audit|"
    r"reform|eliminate|expand|enforce|review|contact|come to|"
    r"reply|respond|explain|make (?:it|this)|pass (?:a|the) law|"
    r"change (?:the|this)|update (?:the|this)"
    r")\b",
    re.IGNORECASE,
)

RISKY_CONTENT_PATTERN = re.compile(
    r"\b("
    r"epstein|pedo|pedophile|rapist|traitor|treason|"
    r"all islamic countries|mosques?|islamic|islamists?|"
    r"foreigners?|illegals?|taxloot|ngojobs|fraud|corrupt|"
    r"shut the fuck up|fuck|idiot|moron|scum|liar"
    r")\b",
    re.IGNORECASE,
)

GENERIC_OR_CHATTER_PATTERN = re.compile(
    r"\b("
    r"great job|fantastic|amazing|warm and fuzzy|america first|"
    r"lol|lmao|hahaha|thanks for sharing|good point"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AuditDecision:
    likely_false_positive: bool
    reasons: list[str]
    score: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit actionable queue and flag likely false positives."
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Path to actionable queue CSV.",
    )
    parser.add_argument(
        "--max-print",
        type=int,
        default=25,
        help="Max flagged examples to print (default: 25).",
    )
    parser.add_argument(
        "--output-csv",
        default="",
        help="Optional path to write flagged candidate rows with reasons.",
    )
    return parser.parse_args()


def text_has_action_cue(text: str) -> bool:
    return bool(ACTION_CUE_PATTERN.search(text))


def text_has_concrete_action(text: str) -> bool:
    return bool(CONCRETE_ACTION_PATTERN.search(text))


def text_has_risky_content(text: str) -> bool:
    return bool(RISKY_CONTENT_PATTERN.search(text))


def text_is_generic_chatter(text: str) -> bool:
    return bool(GENERIC_OR_CHATTER_PATTERN.search(text))


def assess_row(text: str) -> AuditDecision:
    normalized = (text or "").strip()
    reasons: list[str] = []
    score = 0

    has_action_cue = text_has_action_cue(normalized)
    has_concrete_action = text_has_concrete_action(normalized)
    has_risky_content = text_has_risky_content(normalized)
    is_generic_chatter = text_is_generic_chatter(normalized)
    has_question_mark = "?" in normalized

    if not has_action_cue:
        reasons.append("no_action_cue")
        score += 2

    if has_question_mark and not has_concrete_action:
        reasons.append("question_without_concrete_request")
        score += 2

    if has_risky_content and not has_concrete_action:
        reasons.append("risky_or_rant_content_without_request")
        score += 2

    if is_generic_chatter and not has_concrete_action:
        reasons.append("generic_chatter_without_request")
        score += 1

    if len(normalized) < 24 and not has_concrete_action:
        reasons.append("very_short_without_request")
        score += 1

    return AuditDecision(
        likely_false_positive=score >= 2,
        reasons=reasons,
        score=score,
    )


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"[ERROR] input has no header row: {path}")
        rows = list(reader)
    if not rows:
        raise SystemExit(f"[ERROR] input has zero rows: {path}")
    return rows


def write_flagged_csv(path: Path, rows: Iterable[dict[str, str]]) -> None:
    row_list = list(rows)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "comment_id",
                "score",
                "reasons",
                "confidence",
                "text",
            ],
        )
        writer.writeheader()
        writer.writerows(row_list)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise SystemExit(f"[ERROR] missing input CSV: {input_path}")

    rows = load_rows(input_path)

    flagged: list[dict[str, str]] = []
    likely_actionable_count = 0

    for row in rows:
        text = (row.get("text") or "").strip()
        decision = assess_row(text)
        if decision.likely_false_positive:
            flagged.append(
                {
                    "comment_id": (row.get("comment_id") or "").strip(),
                    "score": str(decision.score),
                    "reasons": "|".join(decision.reasons),
                    "confidence": (
                        row.get("final_confidence")
                        or row.get("confidence")
                        or ""
                    ).strip(),
                    "text": text,
                }
            )
        else:
            likely_actionable_count += 1

    flagged.sort(key=lambda r: int(r["score"]), reverse=True)

    total = len(rows)
    flagged_count = len(flagged)

    print(f"[INFO] input_file={input_path}")
    print(f"[INFO] total_actionable_queue={total}")
    print(f"[INFO] likely_actionable={likely_actionable_count}")
    print(f"[INFO] likely_non_actionable_candidates={flagged_count}")

    max_print = max(0, int(args.max_print))
    if flagged and max_print > 0:
        print(f"[INFO] sample_candidates_up_to={max_print}")
        for i, row in enumerate(flagged[:max_print], start=1):
            snippet = row["text"].replace("\n", " ")[:190]
            print(
                f"{i:02d}. id={row['comment_id']} score={row['score']} "
                f"reasons={row['reasons']} text={snippet}"
            )

    output_csv = (args.output_csv or "").strip()
    if output_csv:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_flagged_csv(output_path, flagged)
        print(f"[INFO] wrote_candidates_csv={output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
