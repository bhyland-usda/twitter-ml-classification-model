#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

LABEL_MOD = "moderation_risk"
LABEL_QR = "question_or_request"
LABEL_AF = "actionable_feedback"
LABEL_NA = "non_actionable_noise"

ALLOWED = {LABEL_MOD, LABEL_QR, LABEL_AF, LABEL_NA}

# --- Moderation-risk cues (highest precedence) ---
MOD_PATTERNS = [
    r"\b(rape supporter|rapist|pedo|pedophile|scum|traitor|vermin|subhuman)\b",
    r"\b(kill|hang|lynch|exterminate|eradicate)\b",
    r"\b(hate\s+(them|those people|all\s+\w+))\b",
    r"\b(go back to your country|all islamic countries)\b",
    r"\b(filthy|disgusting)\b",
]
MOD_RE = [re.compile(p, re.IGNORECASE) for p in MOD_PATTERNS]

# --- Question/request cues (2nd precedence) ---
QUESTION_WORD_RE = re.compile(
    r"^\s*(why|what|when|where|who|whom|whose|which|how|does|do|did|is|are|can|could|should|would|will)\b",
    re.IGNORECASE,
)
REQUEST_PATTERNS = [
    r"\bplease\b",
    r"\byou should\b",
    r"\bwe need (you )?to\b",
    r"\bpost (a|the)?\s*list\b",
    r"\bsource\??$",
    r"\bevidence\??$",
    r"\bcan you\b",
    r"\bwill you\b",
]
REQUEST_RE = [re.compile(p, re.IGNORECASE) for p in REQUEST_PATTERNS]

# --- Actionable feedback cues (3rd precedence) ---
AF_PATTERNS = [
    r"\bshould\b",
    r"\bneed to\b",
    r"\bmust\b",
    r"\bban\b",
    r"\bstop\b",
    r"\beliminate\b",
    r"\bpass (a|the)?\s*law\b",
    r"\bfederal law\b",
    r"\bcongress\b",
    r"\bprogram\b",
    r"\bpolicy\b",
    r"\bsubsid(y|ies)\b",
    r"\bforeign(ers| investor| ownership)\b",
    r"\bamerican (land|soil|taxpayers)\b",
    r"\bpoison(ing)? (our )?crops\b",
    r"\bprivate property rights\b",
    r"\bfraud\b|\bwaste\b|\babuse\b",
]
AF_RE = [re.compile(p, re.IGNORECASE) for p in AF_PATTERNS]

# Very short/generic cheer/slogan reactions likely non-actionable
NOISE_PATTERNS = [
    r"^\s*[\W_]+$",
    r"^\s*(great|good|nice|awesome|wow|yes|no|america first)\s*[!.\s🇺🇸👏👍🤢🙌💯]*$",
]
NOISE_RE = [re.compile(p, re.IGNORECASE) for p in NOISE_PATTERNS]


def clean_text(s: str) -> str:
    return (s or "").replace("\ufeff", "").replace("\x00", "").strip()


def is_emoji_or_symbol_heavy(text: str) -> bool:
    if not text:
        return True
    alnum = sum(ch.isalnum() for ch in text)
    return alnum <= max(2, int(len(text) * 0.15))


def has_moderation_risk(text: str) -> bool:
    t = text.lower()
    for rx in MOD_RE:
        if rx.search(t):
            return True
    # abusive second-person insults
    if re.search(r"\byou( are|'re)\s+(an?\s+)?(idiot|moron|liar|scum|trash)\b", t):
        return True
    return False


def has_question_or_request(text: str) -> bool:
    t = text.strip()
    if "?" in t:
        return True
    if QUESTION_WORD_RE.search(t):
        return True
    for rx in REQUEST_RE:
        if rx.search(t):
            return True
    return False


def has_actionable_feedback(text: str) -> bool:
    t = text.lower()
    hits = 0
    for rx in AF_RE:
        if rx.search(t):
            hits += 1
    # Require some substance
    token_count = len(re.findall(r"[A-Za-z0-9']+", t))
    if hits >= 1 and token_count >= 6:
        return True
    return False


def is_non_actionable_noise(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    for rx in NOISE_RE:
        if rx.search(t):
            return True
    if is_emoji_or_symbol_heavy(t):
        return True
    # link-only / near link-only
    if re.fullmatch(r"(https?://\S+\s*){1,3}", t, flags=re.IGNORECASE):
        return True
    return False


def assign_label(text: str) -> str:
    # Strict precedence from rubric:
    # 1) moderation_risk
    # 2) question_or_request
    # 3) actionable_feedback
    # 4) non_actionable_noise
    if has_moderation_risk(text):
        return LABEL_MOD
    if has_question_or_request(text):
        return LABEL_QR
    if has_actionable_feedback(text):
        return LABEL_AF
    return LABEL_NA


def annotate(input_csv: Path, output_csv: Path) -> None:
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Input CSV is missing a header")
        cols = [c.strip() for c in reader.fieldnames]

        required = {"comment_id", "text", "parent_text"}
        missing = sorted(required - set(cols))
        if missing:
            raise ValueError(
                f"Input CSV missing required columns: {missing}. Found: {cols}"
            )

        rows = list(reader)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["comment_id", "text", "parent_text", "label"]
        )
        writer.writeheader()

        counts = {k: 0 for k in ALLOWED}
        for row in rows:
            comment_id = clean_text(row.get("comment_id", ""))
            text = clean_text(row.get("text", ""))
            parent_text = clean_text(row.get("parent_text", ""))
            label = assign_label(text)
            counts[label] += 1
            writer.writerow(
                {
                    "comment_id": comment_id,
                    "text": text,
                    "parent_text": parent_text,
                    "label": label,
                }
            )

    total = sum(counts.values())
    print(f"[INFO] wrote: {output_csv}")
    print(f"[INFO] rows: {total}")
    for k in [LABEL_MOD, LABEL_QR, LABEL_AF, LABEL_NA]:
        c = counts[k]
        pct = (100.0 * c / total) if total else 0.0
        print(f"[INFO] {k}: {c} ({pct:.2f}%)")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Annotate comments using strict rubric precedence."
    )
    p.add_argument(
        "--input",
        required=True,
        help="Path to raw comments CSV (must include comment_id,text,parent_text)",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Path to labeled output CSV (comment_id,text,parent_text,label)",
    )
    args = p.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    annotate(in_path, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
