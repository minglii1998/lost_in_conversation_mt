from utils_log import load_results_from
from tasks import get_task
import os, numpy as np

def extract_success_or_score(d):
    if "is_correct" in d and type(d["is_correct"]) == bool:
        return d["is_correct"]
    elif "is_correct" in d and type(d["is_correct"]) == int:
        return d["is_correct"] == 1
    elif "score" in d and type(d["score"]) == float:
        return d["score"]
    raise ValueError(f"Unknown format for success or score: {d}")

def get_convs(model_data, task_id, conv_type, model):
    return [d for d in model_data[conv_type].get(model, []) if d["task_id"] == task_id]

def compute_results(task_name):
    task = get_task(task_name) # version is by default now

    dataset_fn = task.get_dataset_file()

    conv_types = [ct for ct in os.listdir(f"logs/{task_name}") if ct not in ["lazy-tree", "shuffle-concat"] and not ("-ut" in ct and "-at" in ct)]
    model_data = {conv_type: load_results_from(f"logs/{task_name}/{conv_type}/", dataset_fn) for conv_type in conv_types}
    conv_types = sorted([conv_type for conv_type in conv_types if len(model_data[conv_type]) > 0])

    samples = task.get_samples()
    print(f"Number of samples: {len(samples)}")

    task_ids = set([d["task_id"] for d in samples])
    models = list(model_data["full"].keys())

    main_results, per_sample_results, N_samples,  = [], [], []
    all_sample_convs, all_sample_results, per_sample_avg_result = {}, {}, {}
    for task_id in task_ids:
        per_sample_results.append({"task_id": task_id})
        N_samples.append({"task_id": task_id})
        for model in models:
            for conv_type in conv_types:
                convs = get_convs(model_data, task_id, conv_type, model)
                success_or_scores = [extract_success_or_score(d) for d in convs]
                all_sample_results[(task_id, conv_type, model)] = success_or_scores
                all_sample_convs[(task_id, conv_type, model)] = convs
                per_sample_avg_result[(task_id, conv_type, model)] = np.nan if len(convs) == 0 else 100.0*np.mean(success_or_scores)
                per_sample_results[-1][f"{conv_type.capitalize()}_Acc_{model}"] = per_sample_avg_result[(task_id, conv_type, model)]
                N_samples[-1][f"N_{conv_type}_{model}"] = len(convs)

    for model in models:
        main_results.append({"model": model})
        for conv_type in conv_types:
            main_results[-1][conv_type] = np.mean([per_sample_avg_result[(task_id, conv_type, model)] for task_id in task_ids if not np.isnan(per_sample_avg_result[(task_id, conv_type, model)])])

    condensed_counts_table = []
    for model in models:
        row = {"model": model}
        for conv_type in conv_types:
            row[conv_type] = sum([len(get_convs(model_data, task_id, conv_type, model)) for task_id in task_ids])
        condensed_counts_table.append(row)
    return main_results, condensed_counts_table, per_sample_results, N_samples, all_sample_results, all_sample_convs