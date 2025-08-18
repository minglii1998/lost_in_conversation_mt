import argparse
import os
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete JSONL log files for a specific model across all 6 tasks and 3 settings."
    )
    parser.add_argument(
        "--logs",
        required=True,
        help="Absolute path to logs root directory (same structure used by cal_acc.py).",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name to match in filenames, e.g., Qwen_Qwen3-1.7B (matches *_<model>.jsonl).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, only print the files that would be deleted without actually deleting them.",
    )
    return parser.parse_args()


def list_candidate_files(setting_dir: str, model_name: str) -> List[str]:
    """Return a list of JSONL files in *setting_dir* that belong to *model_name*."""
    target_suffix = f"_{model_name}.jsonl"
    candidates: List[str] = []
    for entry in os.scandir(setting_dir):
        if not entry.is_file():
            continue
        if not entry.name.endswith(".jsonl"):
            continue
        name = entry.name
        # Accept exact suffix match first
        if name.endswith(target_suffix):
            candidates.append(entry.path)
            continue
        # Fallback: any occurrence of _<model_name> before extension
        if f"_{model_name}" in name:
            candidates.append(entry.path)
    return candidates


def delete_files(files: List[str], dry_run: bool = False):
    for fp in files:
        if dry_run:
            print(f"[DRY-RUN] Would delete: {fp}")
        else:
            try:
                os.remove(fp)
                print(f"Deleted: {fp}")
            except OSError as e:
                print(f"Failed to delete {fp}: {e}")


def remove_empty_dirs(root_dir: str):
    """Recursively remove empty directories under *root_dir*."""
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        # Skip the root itself; we don't want to delete logs_root accidentally
        if dirpath == root_dir:
            continue
        # If directory is now empty (no subdirs, no files)
        if not dirnames and not filenames:
            try:
                os.rmdir(dirpath)
                print(f"Removed empty directory: {dirpath}")
            except OSError:
                pass


def main():
    args = parse_args()
    logs_root = args.logs
    model_name = args.model

    tasks_order = [
        "code",
        "database",
        "actions",
        "math",
        "data2text",
        "summary",
    ]
    settings_order = ["full", "concat", "sharded"]

    if not os.path.isdir(logs_root):
        raise FileNotFoundError(f"Logs directory not found: {logs_root}")

    total_deleted = 0
    for task in tasks_order:
        task_dir = os.path.join(logs_root, task)
        if not os.path.isdir(task_dir):
            continue
        for setting in settings_order:
            setting_dir = os.path.join(task_dir, setting)
            if not os.path.isdir(setting_dir):
                continue
            files_to_delete = list_candidate_files(setting_dir, model_name)
            total_deleted += len(files_to_delete)
            delete_files(files_to_delete, dry_run=args.dry_run)

    if total_deleted == 0:
        print("No matching files found.")
    else:
        print(f"Total {'would be ' if args.dry_run else ''}deleted: {total_deleted} files")

    if not args.dry_run:
        # Clean up empty dirs created by deletion
        remove_empty_dirs(logs_root)


if __name__ == "__main__":
    main()
