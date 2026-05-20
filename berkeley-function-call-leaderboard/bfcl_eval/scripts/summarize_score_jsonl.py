import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected a JSON object on line {line_no}, got {type(parsed)}")
            parsed["_line_no"] = line_no
            records.append(parsed)
    return records


def find_first_difference(
    differences: Any,
    path_prefix: str = "",
) -> tuple[str, Any, Any] | None:
    if isinstance(differences, dict):
        if "model" in differences and "ground_truth" in differences:
            return path_prefix or "root", differences.get("model"), differences.get("ground_truth")

        for key, value in differences.items():
            child_path = f"{path_prefix}.{key}" if path_prefix else str(key)
            found = find_first_difference(value, child_path)
            if found is not None:
                return found

    return None


def compact_text(value: Any, max_len: int) -> str:
    text = str(value).replace("\n", "\\n")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def find_first_leaf_mismatch(
    model_value: Any,
    gt_value: Any,
    path_prefix: str,
) -> tuple[str, Any, Any] | None:
    if isinstance(model_value, dict) and isinstance(gt_value, dict):
        keys = sorted(set(model_value.keys()) | set(gt_value.keys()), key=str)
        for key in keys:
            child_path = f"{path_prefix}.{key}" if path_prefix else str(key)
            in_model = key in model_value
            in_gt = key in gt_value
            if in_model and not in_gt:
                return child_path, model_value[key], "<missing>"
            if in_gt and not in_model:
                return child_path, "<missing>", gt_value[key]
            found = find_first_leaf_mismatch(model_value[key], gt_value[key], child_path)
            if found is not None:
                return found
        return None

    if isinstance(model_value, list) and isinstance(gt_value, list):
        common_len = min(len(model_value), len(gt_value))
        for i in range(common_len):
            child_path = f"{path_prefix}[{i}]"
            found = find_first_leaf_mismatch(model_value[i], gt_value[i], child_path)
            if found is not None:
                return found
        if len(model_value) != len(gt_value):
            return path_prefix, model_value, gt_value
        return None

    if model_value != gt_value:
        return path_prefix or "root", model_value, gt_value
    return None


def explain_string_mismatch(model_text: str, gt_text: str) -> str:
    if model_text == "" and gt_text != "":
        return "empty output/content where non-empty ground truth is expected"
    if model_text != "" and gt_text == "":
        return "non-empty output/content where empty ground truth is expected"
    if normalize_whitespace(model_text) == normalize_whitespace(gt_text):
        return "formatting mismatch (whitespace/newline differences)"
    if sorted(model_text.split()) == sorted(gt_text.split()):
        return "token/phrase order mismatch"
    return "text content mismatch"


def explain_mismatch(diff_path: str, model_value: Any, gt_value: Any) -> str:
    leaf_mismatch = find_first_leaf_mismatch(model_value, gt_value, diff_path)
    if leaf_mismatch is not None:
        leaf_path, leaf_model, leaf_gt = leaf_mismatch
    else:
        leaf_path, leaf_model, leaf_gt = diff_path, model_value, gt_value

    if isinstance(leaf_model, str) and isinstance(leaf_gt, str):
        reason = explain_string_mismatch(leaf_model, leaf_gt)
        return (
            f"{reason} at {leaf_path} "
            f"({compact_text(leaf_model, 90)} vs expected {compact_text(leaf_gt, 90)})."
        )

    if type(leaf_model) != type(leaf_gt):
        return (
            f"type mismatch at {leaf_path} "
            f"({type(leaf_model).__name__} vs expected {type(leaf_gt).__name__})."
        )

    return (
        f"value mismatch at {leaf_path} "
        f"({compact_text(leaf_model, 90)} vs expected {compact_text(leaf_gt, 90)})."
    )


def summarize(records: list[dict[str, Any]], max_diff_len: int, score_file_name: str) -> str:
    lines: list[str] = []

    summary = records[0] if records else {}
    has_summary = all(k in summary for k in ("accuracy", "correct_count", "total_count"))

    if has_summary:
        total = int(summary["total_count"])
        correct = int(summary["correct_count"])
        accuracy = float(summary["accuracy"])
        lines.append("=== Overall ===")
        lines.append(f"accuracy: {accuracy:.6f} ({correct}/{total})")
    else:
        lines.append("=== Overall ===")
        lines.append("No aggregate summary row found.")

    detail_records = records[1:] if has_summary else records
    failures = [r for r in detail_records if r.get("valid") is False]
    successes = [r for r in detail_records if r.get("valid") is True]

    lines.append("")
    lines.append("=== Record Counts ===")
    lines.append(f"detail rows: {len(detail_records)}")
    lines.append(f"passed rows: {len(successes)}")
    lines.append(f"failed rows: {len(failures)}")

    error_counter = Counter(
        (r.get("error", {}) or {}).get("error_type", "unknown") for r in failures
    )

    lines.append("")
    lines.append("=== Failure Types ===")
    if not error_counter:
        lines.append("none")
    else:
        for error_type, count in error_counter.most_common():
            lines.append(f"{error_type}: {count}")

    lines.append("")
    lines.append("=== Failed Cases ===")
    if not failures:
        lines.append("none")
        return "\n".join(lines)

    natural_language_lines: list[str] = []

    for idx, item in enumerate(failures, start=1):
        case_id = item.get("id", "unknown_id")
        error = item.get("error", {}) or {}
        error_type = error.get("error_type", "unknown")
        error_message = error.get("error_message", "")

        lines.append(f"{idx}. id={case_id}")
        lines.append(f"   type={error_type}")
        if error_message:
            lines.append(f"   message={compact_text(error_message, max_diff_len)}")

        details = error.get("details", {}) if isinstance(error.get("details"), dict) else {}
        differences = details.get("differences")
        first_diff = find_first_difference(differences)
        if first_diff is not None:
            diff_path, model_value, gt_value = first_diff
            lines.append(f"   first_diff_path={diff_path}")
            lines.append(f"   model={compact_text(model_value, max_diff_len)}")
            lines.append(f"   ground_truth={compact_text(gt_value, max_diff_len)}")

            line_no = item.get("_line_no", "?")
            mismatch_explanation = explain_mismatch(diff_path, model_value, gt_value)
            natural_language_lines.append(
                f"{case_id} ({score_file_name}:{line_no}): {mismatch_explanation}"
            )

    lines.append("")
    lines.append("=== Natural-language mismatch summary ===")
    for line in natural_language_lines:
        lines.append(line)

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize BFCL score JSONL output with failure breakdown.",
    )
    parser.add_argument(
        "score_jsonl",
        type=Path,
        help="Path to score JSONL file (e.g. BFCL_v4_multi_turn_base_score.json)",
    )
    parser.add_argument(
        "--max-diff-len",
        type=int,
        default=180,
        help="Maximum printed length for model/ground-truth difference previews.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_diff_len < 20:
        raise ValueError("--max-diff-len must be at least 20")

    records = load_jsonl(args.score_jsonl)
    if not records:
        raise ValueError(f"No records found in {args.score_jsonl}")

    print(summarize(records, args.max_diff_len, args.score_jsonl.name))


if __name__ == "__main__":
    main()
