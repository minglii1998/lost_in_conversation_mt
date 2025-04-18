import tqdm, argparse, random, multiprocessing, os, json
from simulator_lazy import ConversationSimulatorLazy
from simulator_full import ConversationSimulatorFull
from concurrent.futures import ThreadPoolExecutor
from utils_log import get_run_counts
from collections import Counter

def single_run(todo):
    dataset_fn = todo["dataset_fn"]
    try:
        assistant_temp = todo.get("assistant_temperature", 1.0)
        user_temp = todo.get("user_temperature", 1.0)
        if todo["conv_type"].startswith("full"):
            conversation_simulator = ConversationSimulatorFull(todo["sample"], assistant_model=todo["assistant_model"], system_model=todo["system_model"], temperature=assistant_temp, dataset_fn=dataset_fn, log_folder=args.log_folder)
        elif todo["conv_type"].startswith("concat"):
            conversation_simulator = ConversationSimulatorFull(todo["sample"], assistant_model=todo["assistant_model"], system_model=todo["system_model"], run_concat=True, temperature=assistant_temp, dataset_fn=dataset_fn, log_folder=args.log_folder)
        elif todo["conv_type"].startswith("lazy"):
            conversation_simulator = ConversationSimulatorLazy(todo["sample"], assistant_model=todo["assistant_model"], system_model=todo["system_model"], user_model=todo["user_model"], assistant_temperature=assistant_temp, user_temperature=user_temp, dataset_fn=dataset_fn, log_folder=args.log_folder)
        conversation_simulator.run(verbose=args.verbose)

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        tqdm.tqdm.write(f"\033[91m [Error on {todo['sample']['task_id']}; {todo['assistant_model']}; {todo['conv_type']}]:\n{error_msg}\033[0m")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_file", type=str, default=None, help="Dataset file to use")

    parser.add_argument("--N_full_runs", type=int, default=10, help="Number of full runs per model")
    parser.add_argument("--N_concat_runs", type=int, default=10, help="Number of concat runs per model")
    parser.add_argument("--N_lazy_runs", type=int, default=10, help="Number of lazy runs per model")
    parser.add_argument("--models", nargs="+", default=["gpt-4o-mini", "gpt-4o", "gemini-1.5-flash", "claude3-haiku", "claude3.5-sonnet", "gemini-1.5-pro"],
                        help="List of models to run experiments with")
    parser.add_argument("--system_model", type=str, default="gpt-4o-mini", help="System model to use")
    parser.add_argument("--user_model", type=str, default="gpt-4o-mini", help="User model to use")
    parser.add_argument("--N_workers", type=int, default=7, help="Number of workers to run experiments with")
    parser.add_argument("--log_folder", type=str, default="logs", help="Log folder to use")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")

    parser.add_argument("--assistant_temperature", type=float, default=1.0, help="Temperature to use for assistant models")
    parser.add_argument("--user_temperature", type=float, default=1.0, help="Temperature to use for user models")

    args = parser.parse_args()

    # windows fix dataset_file to be unix format
    dataset_fn = args.dataset_file
    if args.dataset_file.startswith(".\\"):
        dataset_fn = args.dataset_file[2:]
    dataset_fn = dataset_fn.replace("\\", "/")

    if os.environ.get("USE_TRAPI", "0") == "1":
        if args.system_model.startswith("gpt"):
            args.system_model = "t-"+args.system_model
            print("Switched system model to TRAPI backend")
        if args.user_model.startswith("gpt"):
            args.user_model = "t-"+args.user_model
            print("Switched user model to TRAPI backend")

    with open(dataset_fn, "r") as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} samples")
    random.shuffle(samples)
    all_todos = []

    sharded_extra = f"-at{args.assistant_temperature}-ut{args.user_temperature}" if args.assistant_temperature != 1.0 or args.user_temperature != 1.0 else ""
    st_extra = f"-t{args.assistant_temperature}" if args.assistant_temperature != 1.0 else ""
    sharded_ct, full_ct, concat_ct = f"lazy{sharded_extra}", f"full{st_extra}", f"concat{st_extra}"

    all_tasks = list(set([sample["task"] for sample in samples]))
    for assistant_model in args.models:
        lazy_run_counts, full_run_counts, concat_run_counts = Counter(), Counter(), Counter()
        for task in all_tasks:
            lazy_run_counts.update(get_run_counts(sharded_ct, task, assistant_model, dataset_fn, log_folder=args.log_folder))
            full_run_counts.update(get_run_counts(full_ct, task, assistant_model, dataset_fn, log_folder=args.log_folder))
            concat_run_counts.update(get_run_counts(concat_ct, task, assistant_model, dataset_fn, log_folder=args.log_folder))
        print(f"Lazy run counts: {lazy_run_counts}")
        print(f"Full run counts: {full_run_counts}")
        print(f"Concat run counts: {concat_run_counts}")

        for sample in samples:
            all_todos += [{"sample": sample, "assistant_model": assistant_model, "conv_type": full_ct, "system_model": args.system_model, "dataset_fn": dataset_fn}] * (args.N_full_runs - full_run_counts[sample["task_id"]])
            all_todos += [{"sample": sample, "assistant_model": assistant_model, "conv_type": concat_ct, "system_model": args.system_model, "dataset_fn": dataset_fn}] * (args.N_concat_runs - concat_run_counts[sample["task_id"]])
            all_todos += [{"sample": sample, "assistant_model": assistant_model, "conv_type": sharded_ct, "system_model": args.system_model, "user_model": args.user_model, "dataset_fn": dataset_fn}] * (args.N_lazy_runs - lazy_run_counts[sample["task_id"]])

    if args.assistant_temperature != 1.0 or args.user_temperature != 1.0:
        # update todos with temperature
        for todo in all_todos:
            todo["assistant_temperature"] = args.assistant_temperature
            todo["user_temperature"] = args.user_temperature

    random.shuffle(all_todos)

    print(f"Running {len(all_todos)} conversations")
    print(Counter([todo["assistant_model"] for todo in all_todos]))
    print(Counter([todo["conv_type"] for todo in all_todos]))

    with ThreadPoolExecutor(max_workers=args.N_workers) as executor:
        list(tqdm.tqdm(executor.map(single_run, all_todos), total=len(all_todos)))
