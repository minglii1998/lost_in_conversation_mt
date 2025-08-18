import argparse
import json
import os
from typing import Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate accuracies from JSONL logs per task/setting and overall."
    )
    parser.add_argument(
        "--logs",
        required=True,
        help="Absolute path to logs root directory. Structure: <logs>/<task>/<setting>/*.jsonl",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name used in filenames, e.g., Qwen_Qwen3-1.7B. We match files ending with _{model}.jsonl",
    )
    return parser.parse_args()


def coerce_is_correct(value) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "correct"}:
            return True
        if normalized in {"false", "0", "no", "n", "incorrect"}:
            return False
    return None


def read_jsonl_counts(file_path: str) -> Tuple[int, int]:
    total = 0
    correct = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "is_correct" not in record:
                continue
            coerced = coerce_is_correct(record["is_correct"])
            if coerced is None:
                continue
            total += 1
            if coerced:
                correct += 1
    return correct, total


def read_jsonl_score(file_path: str) -> Tuple[float, int]:
    score_sum: float = 0.0
    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Prefer explicit score if present
            if "score" in record:
                score_value = record["score"]
                try:
                    score_float = float(score_value)
                except (TypeError, ValueError):
                    continue
                score_sum += score_float
                count += 1
                continue
            # Fallback: derive score from is_correct if present
            if "is_correct" in record:
                coerced = coerce_is_correct(record["is_correct"])
                if coerced is None:
                    continue
                score_sum += 1.0 if coerced else 0.0
                count += 1
    return score_sum, count


def find_model_jsonl(setting_dir: str, model_name: str) -> Optional[str]:
    target_suffix = f"_{model_name}.jsonl"
    candidates: List[str] = []
    for entry in os.scandir(setting_dir):
        if not entry.is_file():
            continue
        if not entry.name.endswith(".jsonl"):
            continue
        name = entry.name
        # Prefer exact suffix match if available
        if name.endswith(target_suffix):
            candidates.append(entry.path)
            continue
        # Fallback: accept files that contain _{model_name} anywhere before .jsonl
        needle = f"_{model_name}"
        if needle in name:
            candidates.append(entry.path)
    if not candidates:
        return None
    # If multiple candidates, pick the lexicographically last (often the most specific), though
    # in well-formed logs there should be exactly one.
    candidates.sort()
    return candidates[-1]


def calculate_accuracies(logs_root: str, model_name: str) -> Tuple[Dict, Dict, Dict[str, Dict], Dict[str, Dict[str, Dict]]]:
    # Unified scoring for all tasks using mean of `score`
    overall_acc = {"correct": 0, "total": 0}  # kept for compatibility
    overall_score = {"score_sum": 0.0, "count": 0}
    per_task: Dict[str, Dict[str, float]] = {}
    per_setting: Dict[str, Dict[str, Dict[str, float]]] = {}

    if not os.path.isdir(logs_root):
        raise FileNotFoundError(f"Logs directory not found: {logs_root}")

    # tasks are subdirectories under logs_root
    for task_entry in sorted(os.scandir(logs_root), key=lambda e: e.name):
        if not task_entry.is_dir():
            continue
        task_name = task_entry.name
        task_score_sum = 0.0
        task_count = 0
        per_setting.setdefault(task_name, {})

        # settings are subdirectories under each task
        for setting_entry in sorted(os.scandir(task_entry.path), key=lambda e: e.name):
            if not setting_entry.is_dir():
                continue
            setting_name = setting_entry.name
            jsonl_path = find_model_jsonl(setting_entry.path, model_name)
            if jsonl_path is None:
                continue

            score_sum, count = read_jsonl_score(jsonl_path)
            per_setting[task_name][setting_name] = {"score_sum": score_sum, "count": count}
            task_score_sum += score_sum
            task_count += count
            overall_score["score_sum"] += score_sum
            overall_score["count"] += count

        per_task[task_name] = {"score_sum": task_score_sum, "count": task_count}

    return overall_acc, overall_score, per_task, per_setting


def safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def format_percentage(value: float) -> str:
    return f"{value * 100:.2f}%"


def print_report(overall_acc: Dict, overall_score: Dict, per_task: Dict[str, Dict], per_setting: Dict[str, Dict[str, Dict]]):

    mean_score_overall = safe_divide(float(overall_score["score_sum"]), int(overall_score["count"])) if overall_score["count"] > 0 else 0.0
    print("Overall (score):")
    print(
        f"  count={overall_score['count']} mean_score={mean_score_overall:.4f}"
    )

    # Structured 3x6 table (rows: full, concat, sharded; cols: code, database, actions, math, data2text, summary)
    # print("\nMatrix (rows: full, concat, sharded | cols: code, database, actions, math, data2text, summary):")
    tasks_order = ["code", "database", "actions", "math", "data2text", "summary"]
    settings_order = ["full", "concat", "sharded"]
    score_tasks = {"data2text", "summary"}

    # Header (Markdown table)
    header_cells = ["setting"] + tasks_order
    print("| " + " | ".join(header_cells) + " |")
    print("| " + " | ".join(["---"] * len(header_cells)) + " |")

    for setting_name in settings_order:
        row_cells = [setting_name]
        for task_name in tasks_order:
            stats = per_setting.get(task_name, {}).get(setting_name)
            cell = "-"
            if stats:
                count = int(stats.get("count", 0))
                if count > 0:
                    mean_score = safe_divide(float(stats.get("score_sum", 0.0)), count)
                    cell = f"{mean_score:.4f}"
            row_cells.append(cell)
        print("| " + " | ".join(row_cells) + " |")

    # Ratio row: sharded / concat
    ratio_cells = ["ratio"]
    for task_name in tasks_order:
        stats_concat = per_setting.get(task_name, {}).get("concat")
        stats_sharded = per_setting.get(task_name, {}).get("sharded")
        ratio_cell = "-"
        if stats_concat and stats_sharded:
            cnt_c = int(stats_concat.get("count", 0))
            cnt_s = int(stats_sharded.get("count", 0))
            if cnt_c > 0 and cnt_s >= 0:
                mean_c = safe_divide(float(stats_concat.get("score_sum", 0.0)), cnt_c)
                mean_s = safe_divide(float(stats_sharded.get("score_sum", 0.0)), cnt_s) if cnt_s > 0 else 0.0
                if mean_c > 0:
                    ratio_cell = f"{(mean_s / mean_c):.2f}"
        ratio_cells.append(ratio_cell)
    print("| " + " | ".join(ratio_cells) + " |")


def main():
    args = parse_args()
    logs_root = args.logs
    model_name = args.model

    overall_acc, overall_score, per_task, per_setting = calculate_accuracies(logs_root, model_name)
    print(f"Model: {model_name}")
    print_report(overall_acc, overall_score, per_task, per_setting)


if __name__ == "__main__":
    main()


