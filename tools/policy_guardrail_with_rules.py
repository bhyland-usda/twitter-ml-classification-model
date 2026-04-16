#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml

EXPECTED_INPUT_COLUMNS = {"comment_id", "text", "predicted_label", "confidence"}
ALLOWED_FLAG_NAMES = {"IGNORECASE", "MULTILINE", "DOTALL", "ASCII", "VERBOSE"}

THRESHOLD_REASON_ID = "system_low_actionable_confidence"
THRESHOLD_CATEGORY = "system_threshold"

MISSING_ACTION_CUE_REASON_ID = "system_missing_action_cue"
MISSING_ACTION_CUE_CATEGORY = "system_action_cue"

MISSING_CONCRETE_ACTION_REASON_ID = "system_missing_concrete_action"
MISSING_CONCRETE_ACTION_CATEGORY = "system_actionability"

CIVILITY_GATE_REASON_ID = "system_civility_gate"
CIVILITY_GATE_CATEGORY = "system_civility"

QUESTION_ONLY_REASON_ID = "system_question_without_concrete_action"
QUESTION_ONLY_CATEGORY = "system_question_gate"

SAFETY_VETO_REASON_ID = "system_safety_veto"
SAFETY_VETO_CATEGORY = "system_safety"

SCOPE_VETO_REASON_ID = "system_scope_veto"
SCOPE_VETO_CATEGORY = "system_scope"

MISSING_SCOPE_SIGNAL_REASON_ID = "system_missing_scope_signal"
MISSING_SCOPE_SIGNAL_CATEGORY = "system_scope"

ACTION_CUE_PATTERN = re.compile(
    r"\?|"
    r"\b(please|can we|could we|will you|would you|"
    r"we need|need to|should|must|"
    r"how about|what about|when are you|why are you|"
    r"support|help|audit|investigate|fix|stop|ban|reform)\b",
    re.IGNORECASE,
)

REQUEST_IMPERATIVE_PATTERN = re.compile(
    r"^\s*(?:@\w+\s+)*(?:please\s+|kindly\s+)?"
    r"(?:stop|ban|fix|investigate|audit|reform|eliminate|expand|enforce|"
    r"review|contact|reply|respond|explain|change|update|reduce|increase|"
    r"fund|allow|prohibit|divest|deregulate|suspend|remove|open|boost|"
    r"protect|support|halt)\b",
    re.IGNORECASE,
)

REQUEST_MODAL_PATTERN = re.compile(
    r"\b(?:please|can you|could you|will you|would you|we need(?:\s+you)?\s+to|"
    r"you need to|you must|you should|must|should|let's|how about you|"
    r"i urge you to|do something to)\s+"
    r"(?:stop|ban|fix|investigate|audit|reform|eliminate|expand|enforce|"
    r"review|contact|reply|respond|explain|change|update|reduce|increase|"
    r"fund|allow|prohibit|divest|deregulate|suspend|remove|open|boost|"
    r"protect|support|halt)\b",
    re.IGNORECASE,
)

NO_MORE_REQUEST_PATTERN = re.compile(
    r"^\s*(?:@\w+\s+)*(?:please\s+)?no more\b",
    re.IGNORECASE,
)

NEED_TO_ACTION_PATTERN = re.compile(
    r"\b(?:we|you|usda|congress|government|states?)\s+need\s+to\s+"
    r"(?:stop|ban|fix|investigate|audit|reform|eliminate|expand|"
    r"enforce|review|respond|contact|help|reduce|increase|fund|"
    r"allow|prohibit|change|update|pass|remove|halt)\b",
    re.IGNORECASE,
)

OPERATIONAL_OBJECT_PATTERN = re.compile(
    r"\b("
    r"law|policy|program|regulation|regulations|mandate|mandates|"
    r"fraud|prices?|soil|crop|crops|food|farmers?|farmland|land|water|"
    r"grazing|monopoly|monopolies|cartel|subsid(?:y|ies)|snap|ebt|"
    r"vaccine|vaccines|outbreak|disease|poison|poisons|glyphosate|"
    r"chemtrails?|gmo|hormones?|beef|cattle|ranch(?:ing|ers?)|horses?|"
    r"solar|data centers?|ownership|label(?:ing)?|imports?|china|origin"
    r")\b",
    re.IGNORECASE,
)

META_DISCOURSE_PATTERN = re.compile(
    r"\b(gaslight(?:ing)?|pr\s+lies?|just stop|stop with this)\b",
    re.IGNORECASE,
)

USDA_SCOPE_PATTERN = re.compile(
    r"\b("
    r"usda|agri(?:culture|cultural)?|farm(?:er|ers|ing|land)?|ranch(?:er|ers|ing)?|"
    r"livestock|cattle|beef|poultry|dairy|crop|crops|soil|water|irrigation|"
    r"snap|ebt|food(?:\s+supply)?|label(?:ing)?|meat|monopoly|cartel|subsid(?:y|ies)"
    r")\b",
    re.IGNORECASE,
)

CIVILITY_BLOCK_PATTERN = re.compile(
    r"\b("
    r"fuck|f\*+ck|shut the fuck up|asses out|"
    r"idiot|moron|scum|disgrace|pathetic|liar|"
    r"epstein|pedo|pedophile|rapist|sexual predator|"
    r"treason|taxloot|ngojobs|"
    r"all islamic countries|no illegals?|no foreigners?|"
    r"mosques?|islamists?|foreigners? owning land|foreign ownership"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CompiledRule:
    rule_id: str
    category: str
    regex: re.Pattern[str]


@dataclass(frozen=True)
class GuardrailConfig:
    version: str
    apply_only_when_predicted_label_in: List[str]
    final_label: str
    final_confidence: float
    min_actionable_confidence_for_passthrough: float
    require_action_cue_for_actionable_passthrough: bool
    require_concrete_action_for_actionable_passthrough: bool
    require_concrete_action_for_question_passthrough: bool
    require_scope_signal_for_actionable_passthrough: bool
    strict_civility_gate_enabled: bool
    safety_veto_categories: List[str]
    scope_veto_categories: List[str]
    compiled_rules: List[CompiledRule]


def _to_regex_flags(flag_names: List[str]) -> int:
    flags_value = 0
    for flag_name in flag_names:
        normalized = str(flag_name).strip().upper()
        if normalized not in ALLOWED_FLAG_NAMES:
            raise ValueError(
                f"Unsupported regex flag '{flag_name}'. Allowed: {sorted(ALLOWED_FLAG_NAMES)}"
            )
        flags_value |= getattr(re, normalized)
    return flags_value


def _parse_probability(value: str) -> float:
    try:
        parsed = float(value)
    except Exception:
        return 0.0
    if not math.isfinite(parsed):
        return 0.0
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed


def _parse_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _parse_string_list(value, default: List[str]) -> List[str]:
    if value is None:
        return list(default)
    if not isinstance(value, list):
        return list(default)
    parsed = [str(item).strip() for item in value if str(item).strip()]
    if not parsed:
        return list(default)
    return parsed


def _has_action_request_cue(text: str) -> bool:
    return bool(ACTION_CUE_PATTERN.search((text or "").strip()))


def _has_concrete_action_request(text: str) -> bool:
    normalized = (text or "").strip()
    lowered = normalized.lower()

    if not normalized:
        return False

    # Exclude conditional/rhetorical "need to" framing.
    if re.search(r"\bif\s+(?:i|we|you|they)\s+need\s+to\b", lowered):
        return False
    if re.search(r"\bshould\s+never\s+need\s+to\b", lowered):
        return False

    has_request_form = (
        bool(REQUEST_IMPERATIVE_PATTERN.search(normalized))
        or bool(REQUEST_MODAL_PATTERN.search(normalized))
        or bool(NEED_TO_ACTION_PATTERN.search(normalized))
        or bool(NO_MORE_REQUEST_PATTERN.search(normalized))
    )
    if not has_request_form:
        return False

    # Filter meta-discourse asks unless an operational object is also present.
    if META_DISCOURSE_PATTERN.search(
        normalized
    ) and not OPERATIONAL_OBJECT_PATTERN.search(normalized):
        return False

    # Require a concrete operational target for most requests.
    if OPERATIONAL_OBJECT_PATTERN.search(normalized):
        return True

    # Allow direct engagement asks even without policy nouns.
    if re.search(r"\b(?:come to|contact|reply|respond)\b", lowered):
        return True

    return False


def _is_question_without_concrete_action(text: str) -> bool:
    normalized = (text or "").strip()
    return "?" in normalized and not _has_concrete_action_request(normalized)


def _violates_civility_gate(text: str) -> bool:
    return bool(CIVILITY_BLOCK_PATTERN.search((text or "").strip()))


def _has_usda_scope_signal(text: str) -> bool:
    return bool(USDA_SCOPE_PATTERN.search((text or "").strip()))


def load_guardrail_config(yaml_path: Path) -> GuardrailConfig:
    if not yaml_path.exists():
        raise FileNotFoundError(f"rules file not found: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as file_handle:
        raw = yaml.safe_load(file_handle)

    if not isinstance(raw, dict):
        raise ValueError("rules YAML must be a mapping at top-level")

    version = str(raw.get("version", "")).strip()
    if not version:
        raise ValueError("rules YAML missing non-empty 'version'")

    override = raw.get("override")
    if not isinstance(override, dict):
        raise ValueError("rules YAML missing mapping 'override'")

    apply_only_when_predicted_label_in = override.get(
        "apply_only_when_predicted_label_in"
    )
    if (
        not isinstance(apply_only_when_predicted_label_in, list)
        or not apply_only_when_predicted_label_in
    ):
        raise ValueError(
            "override.apply_only_when_predicted_label_in must be a non-empty list"
        )
    apply_only_when_predicted_label_in = [
        str(label_name).strip()
        for label_name in apply_only_when_predicted_label_in
        if str(label_name).strip()
    ]
    if not apply_only_when_predicted_label_in:
        raise ValueError(
            "override.apply_only_when_predicted_label_in contains no valid labels"
        )

    final_label = str(override.get("final_label", "")).strip()
    if not final_label:
        raise ValueError("override.final_label must be non-empty")

    final_confidence_raw = override.get("final_confidence", "")
    try:
        final_confidence = float(str(final_confidence_raw).strip())
    except Exception as exc:
        raise ValueError("override.final_confidence must be numeric") from exc
    if final_confidence < 0.0 or final_confidence > 1.0:
        raise ValueError("override.final_confidence must be in [0.0, 1.0]")

    min_actionable_confidence_for_passthrough_raw = override.get(
        "min_actionable_confidence_for_passthrough", 0.0
    )
    try:
        min_actionable_confidence_for_passthrough = float(
            str(min_actionable_confidence_for_passthrough_raw).strip()
        )
    except Exception as exc:
        raise ValueError(
            "override.min_actionable_confidence_for_passthrough must be numeric"
        ) from exc
    if (
        min_actionable_confidence_for_passthrough < 0.0
        or min_actionable_confidence_for_passthrough > 1.0
    ):
        raise ValueError(
            "override.min_actionable_confidence_for_passthrough must be in [0.0, 1.0]"
        )

    require_action_cue_for_actionable_passthrough = _parse_bool(
        override.get("require_action_cue_for_actionable_passthrough"), True
    )
    require_concrete_action_for_actionable_passthrough = _parse_bool(
        override.get("require_concrete_action_for_actionable_passthrough"), True
    )
    require_concrete_action_for_question_passthrough = _parse_bool(
        override.get("require_concrete_action_for_question_passthrough"), True
    )
    require_scope_signal_for_actionable_passthrough = _parse_bool(
        override.get("require_scope_signal_for_actionable_passthrough"), False
    )
    strict_civility_gate_enabled = _parse_bool(
        override.get("strict_civility_gate_enabled"), True
    )

    safety_veto_categories = _parse_string_list(
        override.get("safety_veto_categories"),
        default=[
            "xenophobia",
            "abuse",
            "violence",
            "conspiracy",
            "scam_spam",
            "health_misinformation",
        ],
    )
    scope_veto_categories = _parse_string_list(
        override.get("scope_veto_categories"),
        default=["off_topic_geopolitics", "out_of_scope"],
    )

    rules = raw.get("rules")
    if not isinstance(rules, list) or not rules:
        raise ValueError("rules YAML must contain non-empty list 'rules'")

    compiled_rules: List[CompiledRule] = []
    seen_rule_ids = set()

    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise ValueError(f"rules[{index}] must be a mapping")

        enabled = bool(rule.get("enabled", True))
        if not enabled:
            continue

        rule_id = str(rule.get("id", "")).strip()
        category = str(rule.get("category", "")).strip()
        pattern = str(rule.get("pattern", "")).strip()
        flags_list = rule.get("flags", [])

        if not rule_id:
            raise ValueError(f"rules[{index}].id must be non-empty")
        if rule_id in seen_rule_ids:
            raise ValueError(f"duplicate rule id: {rule_id}")
        seen_rule_ids.add(rule_id)

        if not category:
            raise ValueError(f"rules[{index}].category must be non-empty")
        if not pattern:
            raise ValueError(f"rules[{index}].pattern must be non-empty")
        if flags_list is None:
            flags_list = []
        if not isinstance(flags_list, list):
            raise ValueError(f"rules[{index}].flags must be a list")

        regex_flags = _to_regex_flags(flags_list)
        try:
            compiled_pattern = re.compile(pattern, regex_flags)
        except re.error as exc:
            raise ValueError(f"invalid regex for rule '{rule_id}': {exc}") from exc

        compiled_rules.append(
            CompiledRule(rule_id=rule_id, category=category, regex=compiled_pattern)
        )

    if not compiled_rules:
        raise ValueError("no enabled rules found in YAML")

    return GuardrailConfig(
        version=version,
        apply_only_when_predicted_label_in=apply_only_when_predicted_label_in,
        final_label=final_label,
        final_confidence=final_confidence,
        min_actionable_confidence_for_passthrough=min_actionable_confidence_for_passthrough,
        require_action_cue_for_actionable_passthrough=require_action_cue_for_actionable_passthrough,
        require_concrete_action_for_actionable_passthrough=require_concrete_action_for_actionable_passthrough,
        require_concrete_action_for_question_passthrough=require_concrete_action_for_question_passthrough,
        require_scope_signal_for_actionable_passthrough=require_scope_signal_for_actionable_passthrough,
        strict_civility_gate_enabled=strict_civility_gate_enabled,
        safety_veto_categories=safety_veto_categories,
        scope_veto_categories=scope_veto_categories,
        compiled_rules=compiled_rules,
    )


def detect_rule_matches(
    text: str, compiled_rules: List[CompiledRule]
) -> List[CompiledRule]:
    content = (text or "").strip()
    matched_rules: List[CompiledRule] = []
    for compiled_rule in compiled_rules:
        if compiled_rule.regex.search(content):
            matched_rules.append(compiled_rule)
    return matched_rules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply strict guardrail overrides using versioned YAML rule config."
    )
    parser.add_argument(
        "--input", required=True, help="Predictions CSV from classifier"
    )
    parser.add_argument(
        "--output", required=True, help="Output CSV with final_label applied"
    )
    parser.add_argument("--rules", required=True, help="Path to guardrail_rules.yaml")
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()

    input_path = Path(arguments.input)
    output_path = Path(arguments.output)
    rules_path = Path(arguments.rules)

    guardrail_config = load_guardrail_config(rules_path)

    if not input_path.exists():
        raise SystemExit(f"input CSV not found: {input_path}")

    with input_path.open("r", encoding="utf-8", newline="") as input_file:
        csv_reader = csv.DictReader(input_file)
        if csv_reader.fieldnames is None:
            raise SystemExit("input CSV has no header row")

        input_columns = {column_name.strip() for column_name in csv_reader.fieldnames}
        missing_columns = EXPECTED_INPUT_COLUMNS - input_columns
        if missing_columns:
            raise SystemExit(
                f"input CSV missing required columns: {sorted(missing_columns)}"
            )

        input_rows = list(csv_reader)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_columns = [
        "comment_id",
        "text",
        "parent_text",
        "predicted_label",
        "confidence",
        "guardrail_version",
        "guardrail_triggered",
        "guardrail_decision_source",
        "guardrail_rule_ids",
        "guardrail_categories",
        "final_label",
        "final_confidence",
    ]

    total_rows = 0
    rows_overridden = 0
    actionable_before = 0
    actionable_after = 0
    overrides_by_rule_match = 0
    overrides_by_conf_threshold = 0
    overrides_by_missing_action_cue = 0
    overrides_by_missing_concrete_action = 0
    overrides_by_question_gate = 0
    overrides_by_civility_gate = 0
    overrides_by_safety_veto = 0
    overrides_by_scope_veto = 0
    overrides_by_missing_scope_signal = 0

    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        csv_writer = csv.DictWriter(output_file, fieldnames=output_columns)
        csv_writer.writeheader()

        for row in input_rows:
            total_rows += 1

            comment_id = (row.get("comment_id") or "").strip()
            text = (row.get("text") or "").strip()
            parent_text = (row.get("parent_text") or "").strip()
            predicted_label = (row.get("predicted_label") or "").strip()
            confidence_text = (row.get("confidence") or "").strip() or "0.000000"
            confidence_value = _parse_probability(confidence_text)

            if predicted_label == "actionable":
                actionable_before += 1

            matched_rules = detect_rule_matches(text, guardrail_config.compiled_rules)
            matched_rule_categories = {
                matched_rule.category for matched_rule in matched_rules
            }
            rule_ids = [matched_rule.rule_id for matched_rule in matched_rules]
            categories = sorted(matched_rule_categories)

            rule_based_override = (
                predicted_label in guardrail_config.apply_only_when_predicted_label_in
                and len(matched_rules) > 0
            )

            confidence_threshold_override = (
                predicted_label == "actionable"
                and confidence_value
                < guardrail_config.min_actionable_confidence_for_passthrough
            )

            missing_action_cue_override = (
                predicted_label == "actionable"
                and guardrail_config.require_action_cue_for_actionable_passthrough
                and not _has_action_request_cue(text)
            )

            missing_concrete_action_override = (
                predicted_label == "actionable"
                and guardrail_config.require_concrete_action_for_actionable_passthrough
                and not _has_concrete_action_request(text)
            )

            question_only_override = (
                predicted_label == "actionable"
                and guardrail_config.require_concrete_action_for_question_passthrough
                and _is_question_without_concrete_action(text)
            )

            civility_gate_override = (
                predicted_label == "actionable"
                and guardrail_config.strict_civility_gate_enabled
                and _violates_civility_gate(text)
            )

            safety_veto_override = predicted_label == "actionable" and any(
                category_name in guardrail_config.safety_veto_categories
                for category_name in matched_rule_categories
            )

            scope_category_override = predicted_label == "actionable" and any(
                category_name in guardrail_config.scope_veto_categories
                for category_name in matched_rule_categories
            )
            missing_scope_signal_override = (
                predicted_label == "actionable"
                and guardrail_config.require_scope_signal_for_actionable_passthrough
                and _has_action_request_cue(text)
                and not _has_usda_scope_signal(text)
            )
            scope_veto_override = (
                scope_category_override or missing_scope_signal_override
            )

            should_apply_override = (
                rule_based_override
                or confidence_threshold_override
                or missing_action_cue_override
                or missing_concrete_action_override
                or question_only_override
                or civility_gate_override
                or safety_veto_override
                or scope_veto_override
            )

            decision_sources: List[str] = []
            if rule_based_override:
                decision_sources.append("rule_match")
            if confidence_threshold_override:
                decision_sources.append("confidence_threshold")
                if THRESHOLD_REASON_ID not in rule_ids:
                    rule_ids.append(THRESHOLD_REASON_ID)
                if THRESHOLD_CATEGORY not in categories:
                    categories.append(THRESHOLD_CATEGORY)

            if missing_action_cue_override:
                decision_sources.append("missing_action_cue")
                if MISSING_ACTION_CUE_REASON_ID not in rule_ids:
                    rule_ids.append(MISSING_ACTION_CUE_REASON_ID)
                if MISSING_ACTION_CUE_CATEGORY not in categories:
                    categories.append(MISSING_ACTION_CUE_CATEGORY)

            if missing_concrete_action_override:
                decision_sources.append("missing_concrete_action")
                if MISSING_CONCRETE_ACTION_REASON_ID not in rule_ids:
                    rule_ids.append(MISSING_CONCRETE_ACTION_REASON_ID)
                if MISSING_CONCRETE_ACTION_CATEGORY not in categories:
                    categories.append(MISSING_CONCRETE_ACTION_CATEGORY)

            if question_only_override:
                decision_sources.append("question_without_concrete_action")
                if QUESTION_ONLY_REASON_ID not in rule_ids:
                    rule_ids.append(QUESTION_ONLY_REASON_ID)
                if QUESTION_ONLY_CATEGORY not in categories:
                    categories.append(QUESTION_ONLY_CATEGORY)

            if civility_gate_override:
                decision_sources.append("civility_gate")
                if CIVILITY_GATE_REASON_ID not in rule_ids:
                    rule_ids.append(CIVILITY_GATE_REASON_ID)
                if CIVILITY_GATE_CATEGORY not in categories:
                    categories.append(CIVILITY_GATE_CATEGORY)

            if safety_veto_override:
                decision_sources.append("safety_veto")
                if SAFETY_VETO_REASON_ID not in rule_ids:
                    rule_ids.append(SAFETY_VETO_REASON_ID)
                if SAFETY_VETO_CATEGORY not in categories:
                    categories.append(SAFETY_VETO_CATEGORY)

            if scope_veto_override:
                decision_sources.append("scope_veto")
                if SCOPE_VETO_REASON_ID not in rule_ids:
                    rule_ids.append(SCOPE_VETO_REASON_ID)
                if SCOPE_VETO_CATEGORY not in categories:
                    categories.append(SCOPE_VETO_CATEGORY)
                if missing_scope_signal_override:
                    if MISSING_SCOPE_SIGNAL_REASON_ID not in rule_ids:
                        rule_ids.append(MISSING_SCOPE_SIGNAL_REASON_ID)
                    if MISSING_SCOPE_SIGNAL_CATEGORY not in categories:
                        categories.append(MISSING_SCOPE_SIGNAL_CATEGORY)

            categories.sort()

            if should_apply_override:
                final_label = guardrail_config.final_label
                final_confidence = f"{guardrail_config.final_confidence:.6f}"
                rows_overridden += 1
                if rule_based_override:
                    overrides_by_rule_match += 1
                if confidence_threshold_override:
                    overrides_by_conf_threshold += 1
                if missing_action_cue_override:
                    overrides_by_missing_action_cue += 1
                if missing_concrete_action_override:
                    overrides_by_missing_concrete_action += 1
                if question_only_override:
                    overrides_by_question_gate += 1
                if civility_gate_override:
                    overrides_by_civility_gate += 1
                if safety_veto_override:
                    overrides_by_safety_veto += 1
                if scope_veto_override:
                    overrides_by_scope_veto += 1
                if missing_scope_signal_override:
                    overrides_by_missing_scope_signal += 1
            else:
                final_label = predicted_label
                final_confidence = f"{confidence_value:.6f}"

            if final_label == "actionable":
                actionable_after += 1

            csv_writer.writerow(
                {
                    "comment_id": comment_id,
                    "text": text,
                    "parent_text": parent_text,
                    "predicted_label": predicted_label,
                    "confidence": f"{confidence_value:.6f}",
                    "guardrail_version": guardrail_config.version,
                    "guardrail_triggered": "true" if should_apply_override else "false",
                    "guardrail_decision_source": "|".join(decision_sources),
                    "guardrail_rule_ids": "|".join(rule_ids),
                    "guardrail_categories": "|".join(categories),
                    "final_label": final_label,
                    "final_confidence": final_confidence,
                }
            )

    print(f"[INFO] rules_version={guardrail_config.version}")
    print(f"[INFO] rules_file={rules_path}")
    print(f"[INFO] output_file={output_path}")
    print(f"[INFO] total_rows={total_rows}")
    print(f"[INFO] actionable_before={actionable_before}")
    print(f"[INFO] actionable_after={actionable_after}")
    print(f"[INFO] overridden_rows={rows_overridden}")
    print(f"[INFO] overrides_rule_match={overrides_by_rule_match}")
    print(f"[INFO] overrides_confidence_threshold={overrides_by_conf_threshold}")
    print(f"[INFO] overrides_missing_action_cue={overrides_by_missing_action_cue}")
    print(
        f"[INFO] overrides_missing_concrete_action={overrides_by_missing_concrete_action}"
    )
    print(
        f"[INFO] overrides_question_without_concrete_action={overrides_by_question_gate}"
    )
    print(f"[INFO] overrides_civility_gate={overrides_by_civility_gate}")
    print(f"[INFO] overrides_safety_veto={overrides_by_safety_veto}")
    print(f"[INFO] overrides_scope_veto={overrides_by_scope_veto}")
    print(f"[INFO] overrides_missing_scope_signal={overrides_by_missing_scope_signal}")
    print(
        "[INFO] min_actionable_confidence_for_passthrough="
        f"{guardrail_config.min_actionable_confidence_for_passthrough:.6f}"
    )
    print(
        "[INFO] require_action_cue_for_actionable_passthrough="
        f"{guardrail_config.require_action_cue_for_actionable_passthrough}"
    )
    print(
        "[INFO] require_concrete_action_for_actionable_passthrough="
        f"{guardrail_config.require_concrete_action_for_actionable_passthrough}"
    )
    print(
        "[INFO] require_concrete_action_for_question_passthrough="
        f"{guardrail_config.require_concrete_action_for_question_passthrough}"
    )
    print(
        "[INFO] strict_civility_gate_enabled="
        f"{guardrail_config.strict_civility_gate_enabled}"
    )
    print(
        "[INFO] require_scope_signal_for_actionable_passthrough="
        f"{guardrail_config.require_scope_signal_for_actionable_passthrough}"
    )
    print(
        "[INFO] safety_veto_categories="
        f"{'|'.join(guardrail_config.safety_veto_categories)}"
    )
    print(
        "[INFO] scope_veto_categories="
        f"{'|'.join(guardrail_config.scope_veto_categories)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
