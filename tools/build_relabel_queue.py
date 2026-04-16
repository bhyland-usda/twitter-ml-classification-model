#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

RISK_RE = re.compile(
    r"\b(rape supporter|rapist|pedo|pedophile|traitor|scum|vermin|subhuman|kill|hang|lynch|eradicate|all islamic countries)\b",
    re.IGNORECASE,
)
REQUEST_RE = re.compile(
    r"\?|^\s*(why|what|when|where|who|how|does|do|did|is|are|can|could|should|would|will)\b|\b(please|you should|we need to|can you|will you|source|evidence)\b",
    re.IGNORECASE,
)
FEEDBACK_RE = re.compile(
    r"\b(should|must|need to|ban|stop|eliminate|federal law|congress|policy|program|subsid|fraud|waste|abuse|private property|ownership|taxpayers|foreign investors)\b",
    re.IGNORECASE,
)

def score_text(text: str) -> tuple[int, str]:
    t = (text or "").strip()
    score = 0
    reasons = []
    if len(t) >= 80:
        score += 2
        reasons.append("len>=80")
    elif len(t) >= 40:
        score += 1
        reasons.append("len>=40")

    if RISK_RE.search(t):
        score += 5
        reasons.append("risk_cue")
    if REQUEST_RE.search(t):
        score += 3
        reasons.append("question_or_request_cue")
    if FEEDBACK_RE.search(t):
        score += 3
        reasons.append("feedback_cue")

    # penalize near-emoji/noise
    alnum = sum(ch.isalnum() for ch in t)
    if alnum <= 3:
        score -= 4
        reasons.append("very_low_alnum")

    return score, ",".join(reasons)

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="raw_comments_labeled.clean.csv")
    p.add_argument("--output", required=True, help="rows for manual relabel")
    p.add_argument("--top-k", type=int, default=200)
    args = p.parse_args()

    inp = Path(args.input)
    out = Path(args.output)

    with inp.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    candidates = []
    for r in rows:
        if r.get("label") != "non_actionable_noise":
            continue
        text = r.get("text", "")
        score, reason = score_text(text)
        if score >= 3:
            candidates.append({
                "comment_id": r.get("comment_id", ""),
                "text": text,
                "current_label": r.get("label", ""),
                "suggested_review_priority_score": str(score),
                "reason_flags": reason,
                "new_label": "",
            })

    candidates.sort(key=lambda x: int(x["suggested_review_priority_score"]), reverse=True)
    top = candidates[: args.top_k]

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "comment_id",
                "text",
                "current_label",
                "suggested_review_priority_score",
                "reason_flags",
                "new_label",
            ],
        )
        writer.writeheader()
        writer.writerows(top)

    print(f"[INFO] candidates_total={len(candidates)}")
    print(f"[INFO] wrote_top_k={len(top)} to {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
