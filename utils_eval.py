import ast, tokenize, re, pandas as pd, numpy as np, re, os
from IPython.display import display
from io import StringIO
from utils_results import compute_results


def interpolate_color(val, min_val=-40.0, mid_val=0.0, max_val=40.0, min_color=(0.90, 0.52, 0.52), mid_color=(1.0, 1.0, 1.0), max_color=(0.5, 0.65, 0.9)):
    # Clamp the value to the min-max range
    val = max(min_val, min(max_val, val))
    
    if val <= mid_val:
        # Interpolate between min and mid color
        t = (val - min_val) / (mid_val - min_val) if (mid_val - min_val) != 0 else 0
        r = min_color[0] + t * (mid_color[0] - min_color[0])
        g = min_color[1] + t * (mid_color[1] - min_color[1])
        b = min_color[2] + t * (mid_color[2] - min_color[2])
    else:
        # Interpolate between mid and max color
        t = (val - mid_val) / (max_val - mid_val) if (max_val - mid_val) != 0 else 0
        r = mid_color[0] + t * (max_color[0] - mid_color[0])
        g = mid_color[1] + t * (max_color[1] - mid_color[1])
        b = mid_color[2] + t * (max_color[2] - mid_color[2])
    
    return r, g, b

def create_gradient_color(val, min_val=-40.0, mid_val=0.0, max_val=40.0):
    r, g, b = interpolate_color(val, min_val, mid_val, max_val)
    backcolor = '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))
    return f"background-color: {backcolor}; color: black;"


def get_task_distribution_stats(result_map, task):
    values = np.array([result_map[task][model][conv_type] for model in result_map[task] for conv_type in result_map[task][model]])
    return np.percentile(values, 0), np.percentile(values, 40), np.percentile(values, 100)

def plot_2d_table(result_map, plot_title="", model_order=None, add_fraction_columns=True):
    tasks = sorted(result_map.keys())
    models = set(model for task in tasks for model in result_map[task])    
    if model_order is not None:
        models = sorted(models, key=lambda m: model_order[m], reverse=False)
    
    conv_types = sorted(set(conv_type for task in tasks for model in models for conv_type in result_map[task].get(model, {})))
    columns = [("", "Model Name")] + [(task, conv_type) for task in tasks for conv_type in conv_types]
    if add_fraction_columns:
        columns += [("", "concat / full"), ("", "shard / full")]

    rows = []

    concat_over_full, sharded_over_full = {model: [] for model in models}, {model: [] for model in models}
    for model in models:
        row = [model]
        for task in tasks:
            # if both values are not nan, then add the ratio to the list
            model_vals = result_map[task].get(model, {})
            if "full" in model_vals and "concat" in model_vals:
                concat_over_full[model].append((model_vals["concat"] / model_vals["full"]))
            if "full" in model_vals and "shard" in model_vals:
                sharded_over_full[model].append((model_vals["shard"] / model_vals["full"]))

            for conv_type in conv_types:
                row.append(model_vals.get(conv_type, np.nan))
        rows.append(row)

    concat_over_full_avg = {model: 100.0 * np.mean(concat_over_full[model]).item() for model in concat_over_full}
    sharded_over_full_avg = {model: 100.0 * np.mean(sharded_over_full[model]).item() for model in sharded_over_full}

    if add_fraction_columns:
        for model, row in zip(models, rows):
            row += [concat_over_full_avg[model], sharded_over_full_avg[model]]

    df = pd.DataFrame(rows, columns=pd.MultiIndex.from_tuples(columns))

    # set the index to model name
    df = df.set_index(("", "Model Name"))

    styled_df = df.style.set_table_styles(
        [
            {'selector': 'th', 'props': [('text-align', 'center')]},
            {'selector': ", ".join([f'td:nth-child({len(conv_types)*i+1})' for i in range(len(tasks))]), 'props': [('border-right', '5px solid black')]},
        ],
        overwrite=False
    )
    
    for task in tasks: # one separate gradient per task
        task_cols = [(task, conv_type) for conv_type in conv_types]
        min_val, mid_val, max_val = get_task_distribution_stats(result_map, task)
        def make_style_task_values(task_min, task_mid, task_max):
            def style_task_values(val):
                if pd.isna(val):
                    return ''
                return create_gradient_color(val, task_min, task_mid, task_max)
            return style_task_values

        style_task_values = make_style_task_values(min_val, mid_val, max_val)
        styled_df = styled_df.applymap(style_task_values, subset=task_cols)

    styled_df = styled_df.set_properties(**{'text-align': 'center'}).format(lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else x).set_caption(plot_title)

    display(styled_df)
    return styled_df


def plot_2d_table_flip(result_map, plot_title="", model_order=None, add_fraction_columns=True):
    task_order = ["python", "database", "apis", "data2text", "math", "summary"]
    tasks = sorted(result_map.keys(), key=lambda x: task_order.index(x))
    models = set(model for task in tasks for model in result_map[task])    
    if model_order is not None:
        models = sorted(models, key=lambda m: model_order[m], reverse=False)
    
    # conv_types = sorted(set(conv_type for task in tasks for model in models for conv_type in result_map[task].get(model, {})))
    conv_types = ["full", "concat", "shard"]

    task2short_task = {"summary": "Sum", "python": "Pyth", "data2text": "D2T", "math": "Math", "database": "DB", "apis": "API"}

    columns = [("", "Model Name")] + [(conv_type, task2short_task[task]) for conv_type in conv_types for task in tasks]
    if add_fraction_columns:
        columns += [("", "concat / full"), ("", "shard / full")]

    rows = []

    task_model_degradation = {task: {model: {} for model in models} for task in tasks}
    task_model_degradation_color = {task: {model: {} for model in models} for task in tasks}

    for task in tasks:
        for model in models:
            if model not in result_map[task]:
                continue
            model_perf = result_map[task].get(model, {})
            full_perf = model_perf.get("full", np.nan)
            if np.isnan(full_perf):
                continue
            for conv_type in conv_types:
                conv_perf = model_perf.get(conv_type, np.nan)
                if np.isnan(conv_perf):
                    continue
                task_model_degradation[task][model][conv_type] = (100.0 * (conv_perf - full_perf) / full_perf).item()
                task_model_degradation_color[task][model][conv_type] = create_gradient_color(task_model_degradation[task][model][conv_type])

    concat_over_full, sharded_over_full = {model: [] for model in models}, {model: [] for model in models}
    for model in models:
        row = [model]
        for task in tasks:
            # if both values are not nan, then add the ratio to the list
            model_vals = result_map[task].get(model, {})
            if "full" in model_vals and "concat" in model_vals:
                concat_over_full[model].append((model_vals["concat"] / model_vals["full"]))
            if "full" in model_vals and "shard" in model_vals:
                sharded_over_full[model].append((model_vals["shard"] / model_vals["full"]))

        for conv_type in conv_types:
            for task in tasks:
                row.append(result_map[task].get(model, {}).get(conv_type, np.nan))
        rows.append(row)

    if add_fraction_columns:
        concat_over_full_avg = {model: 100.0 * np.mean(concat_over_full[model]).item() for model in concat_over_full}
        sharded_over_full_avg = {model: 100.0 * np.mean(sharded_over_full[model]).item() for model in sharded_over_full}
        for model, row in zip(models, rows):
            row += [concat_over_full_avg[model], sharded_over_full_avg[model]]

    df = pd.DataFrame(rows, columns=pd.MultiIndex.from_tuples(columns))

    # set the index to model name
    df = df.set_index(("", "Model Name"))

    styled_df = df.style.set_table_styles(
        [
            {'selector': 'th', 'props': [('text-align', 'center')]},
            {'selector': ", ".join([f'td:nth-child({len(tasks)*i+1})' for i in range(len(tasks))]), 'props': [('border-right', '5px solid black')]},
        ],
        overwrite=False
    )

    # Apply the styling using apply
    def style_cells(x):
        # Create a DataFrame of the same shape as x
        df = pd.DataFrame('', index=x.index, columns=x.columns)
        for task in tasks:
            for conv_type in conv_types:
                for model in models:
                    col = (conv_type, task2short_task[task])
                    if model in task_model_degradation_color[task] and conv_type in task_model_degradation_color[task][model]:
                        color = task_model_degradation_color[task][model][conv_type]
                        df.loc[model, col] = color
        return df

    styled_df = styled_df.apply(style_cells, axis=None)

    styled_df = styled_df.set_properties(**{'text-align': 'center'}).format(lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else x).set_caption(plot_title)

    display(styled_df)
    return styled_df

def strip_comments_and_docstrings(code_string):
    try:
        # First pass: remove comments using tokenize
        tokens = tokenize.generate_tokens(StringIO(code_string).readline)
        tokens = [t for t in tokens if t.type not in (tokenize.COMMENT, tokenize.NL)]
        code_without_comments = tokenize.untokenize(tokens)
        
        # Second pass: remove docstrings using ast
        tree = ast.parse(code_without_comments)
        
        class DocstringRemover(ast.NodeTransformer):
            def visit_Module(self, node):
                # Remove module docstring if it exists
                if (node.body and isinstance(node.body[0], ast.Expr) and 
                    isinstance(node.body[0].value, ast.Str)):
                    node.body.pop(0)
                return self.generic_visit(node)
                
            def visit_FunctionDef(self, node):
                # Remove function docstring if it exists
                if (node.body and isinstance(node.body[0], ast.Expr) and 
                    isinstance(node.body[0].value, ast.Str)):
                    node.body.pop(0)
                return self.generic_visit(node)
                
            def visit_ClassDef(self, node):
                # Remove class docstring if it exists
                if (node.body and isinstance(node.body[0], ast.Expr) and 
                    isinstance(node.body[0].value, ast.Str)):
                    node.body.pop(0)
                return self.generic_visit(node)
        
        # Apply the transformation
        tree = DocstringRemover().visit(tree)
        return ast.unparse(tree)
    except:
        # If we can't parse the code, return it as-is
        # This handles cases with invalid syntax like unparenthesized generator expressions
        return code_string.strip()

def clean_humaneval_answer(answer):
    return strip_comments_and_docstrings(answer)

def clean_spider_answer(answer):
    # replace any multi-space / newlines with a single space
    answer = re.sub(r'\s+', ' ', answer)
    return answer.strip()

def clean_answer(answer, task):
    if task == "humaneval":
        return clean_humaneval_answer(answer)
    elif task == "spider":
        return clean_spider_answer(answer)
    return answer



if __name__ == "__main__":
    import json, os, numpy as np, pandas as pd
    from IPython.display import display

    def print_results(task_name, only_main=False):
        main_results, condensed_counts_table, per_sample_results, N_samples, all_sample_results, all_sample_convs = compute_results(task_name)
        return main_results, condensed_counts_table, per_sample_results, N_samples, all_sample_results, all_sample_convs

    tasks = ["math", "data2text", "apis", "database", "python", "summary"] # , "data2text", "apis", "database"

    all_datasets_avg_results = {}
    all_datasets_all_results = {}

    all_datasets_all_convs = {}
    consolidated_results = {}
    for task in tasks:
        main_results, condensed_counts_table, per_sample_results, N_samples, all_sample_results, all_sample_convs = print_results(task, only_main=True) # version=version
        all_datasets_all_results[task] = all_sample_results
        all_datasets_all_convs[task] = all_sample_convs
        all_datasets_avg_results[task] = per_sample_results
        consolidated_results[task] = main_results

    tasks = sorted(consolidated_results.keys())

    models = sorted(set([row["model"] for task in tasks for row in consolidated_results[task]]))
    avg_full_perf = {model: np.mean([row["full"] for task in tasks for row in consolidated_results[task] if row["model"] == model]).item() for model in models}
    model_order = {model: avg_full_perf[model] for model in models}

    conv_types = ["concat", "full", "lazy"]
    result_map = {}
    for task in tasks:
        result_map[task] = {}
        for row in consolidated_results[task]:
            model = row["model"]
            if model not in result_map[task]:
                result_map[task][model] = {}
            for conv_type in conv_types:
                clean_conv_type = conv_type if conv_type != "lazy" else "shard"
                result_map[task][model][clean_conv_type] = row[conv_type]

    # sort the models by avg_full_perf
    latex = plot_2d_table(result_map, plot_title=f"Average Performance of Models on {len(tasks)} Tasks in Concat, Full and Sharded Conversation Modes", model_order=model_order)
    # latex = latex.replace("\\multicolumn{3}{r}{", "\\multicolumn{3}{c}{")
    # latex = latex.replace("concat", "\\concat").replace("full", "\\full").replace("shard", "\\sharded")

    # print(latex)
