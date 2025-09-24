import json
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from rliable import library as rly
from rliable import metrics
from rliable import plot_utils


def prepare_runs(env_id, project_id, context_id=None, method_filter=None, task_filter=None, max_steps=100000, custom_method_fun=None):
    runs = []
    metric_dirs = glob.glob(f"../../contexts/metrics/{project_id}/*{env_id}*")
    for metric_dir in metric_dirs:
        metrics = []
        filename = f"{metric_dir}/metrics.json"
        if not Path(filename).is_file():
            continue
        with open(filename, 'rb') as f:
            for lines in f:
                metrics.append(json.loads(lines))

        filename = f"{metric_dir}/metadata.json"
        with open(filename, 'rb') as f:
            metadata = json.load(f)

        for task_id, task_name in zip(
            ["rl_returns_eval/eval_train_return", "rl_returns_eval/eval_in_return", "rl_returns_eval/eval_out_return"],
            ["Training", "Eval-in", "Eval-out"],
        ):
            if context_id is not None and metadata["context_id"] != context_id:
                continue

            xs = [m["step"] for m in metrics]
            ys = [m[task_id] for m in metrics]
            if xs[-1] != max_steps:
                continue

            if custom_method_fun is not None:
                method = custom_method_fun(metadata)
            else:
                method = metadata["method"]
            if method_filter is not None and method not in method_filter:
                continue
            if task_filter is not None and task_name not in task_filter:
                continue

            run_dict = {
                "method": method,
                "seed": metadata["seed"],
                "task": task_name,
                "ys": max(ys[-2:])
            }
            runs.append(run_dict)
    return runs


def compute_iqm_scores(project_id, task_filter, method_filter, env_normalizing_factors, max_steps, custom_method_fun=None):
    aggregate_scores_dict = {method: [] for method in method_filter}
    aggregate_score_cis_dict = {method: [] for method in method_filter}
    for task in task_filter:
        score_dict = {method: [] for method in method_filter}
        for env_id, normalizing_factor in env_normalizing_factors.items():
            runs = prepare_runs(env_id, project_id, task_filter=[task], method_filter=method_filter, max_steps=max_steps[env_id], custom_method_fun=custom_method_fun)
            df = pd.DataFrame(runs)
            df = df[df["task"] == task]
            for method in score_dict.keys():
                returns = df[df["method"] == method].sort_values(by="seed").drop_duplicates(inplace=False)["ys"].values.reshape(-1, 1)
                if returns.shape[0] == 0:  # empty, e.g. if env not available
                    continue
                assert returns.shape[0] == 10, f"{returns.shape[0]}, {task}, {env_id}, {method}"
                if normalizing_factor < 0:
                    normalized_returns = 1 - returns / normalizing_factor
                    normalized_returns = (returns - normalizing_factor) / - normalizing_factor
                else:
                    normalized_returns = returns / normalizing_factor
                score_dict[method].append(normalized_returns)
        for method in score_dict.keys():
            score_dict[method] = np.concatenate(score_dict[method], axis=1)

        def aggregate_func(x):
            return np.array([metrics.aggregate_iqm(x)])
        aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
            score_dict, aggregate_func, reps=5000
        )
        for method in method_filter:
            aggregate_scores_dict[method].append(aggregate_scores[method])
            aggregate_score_cis_dict[method].append(aggregate_score_cis[method])

    for method in method_filter:
        aggregate_scores_dict[method] = np.concatenate(aggregate_scores_dict[method], axis=0)
        aggregate_score_cis_dict[method] = np.concatenate(aggregate_score_cis_dict[method], axis=1)
    return aggregate_scores_dict, aggregate_score_cis_dict


def compute_poi_scores(project_id, task_filter, method_filter, env_normalizing_factors, compare_dict, max_steps, custom_method_fun=None):

    score_dict = {method: [] for method in method_filter}
    for task in task_filter:
        for env_id, normalizing_factor in env_normalizing_factors.items():
            runs = prepare_runs(env_id, project_id, task_filter=[task], method_filter=method_filter, max_steps=max_steps[env_id], custom_method_fun=custom_method_fun)
            df = pd.DataFrame(runs)
            df = df[df["task"] == task]
            for method in score_dict.keys():
                returns = df[df["method"] == method].sort_values(by="seed").drop_duplicates(inplace=False)["ys"].values.reshape(-1, 1)
                if returns.shape[0] == 0:  # empty, e.g. if env not available
                    continue
                assert returns.shape[0] == 10, f"{returns.shape[0]}, {task}, {env_id}, {method}"
                if normalizing_factor < 0:
                    normalized_returns = 1 - returns / normalizing_factor
                    normalized_returns = (returns - normalizing_factor) / - normalizing_factor
                else:
                    normalized_returns = returns / normalizing_factor
                score_dict[method].append(normalized_returns)

    for method in score_dict.keys():
        score_dict[method] = np.concatenate(score_dict[method], axis=1)

    return score_dict


def compute_aer_scores(project_id, task_filter, method_filter, env_normalizing_factors, max_steps, normalize=False, custom_method_fun=None):
    task_score_dict = {method: [] for method in method_filter}
    task_score_std_dict = {method: [] for method in method_filter}
    for task in task_filter:
        score_dict = {method: [] for method in method_filter}
        score_std_dict = {method: [] for method in method_filter}
        for env_id, normalizing_factor in env_normalizing_factors.items():
            runs = prepare_runs(env_id, project_id, task_filter=[task], method_filter=method_filter, max_steps=max_steps[env_id], custom_method_fun=custom_method_fun)
            df = pd.DataFrame(runs)
            df = df[df["task"] == task]
            for method in score_dict.keys():
                returns = df[df["method"] == method].sort_values(by="seed").drop_duplicates(inplace=False)["ys"].values.reshape(-1, 1)
                if returns.shape[0] == 0:  # empty, e.g. if env not available
                    continue
                assert returns.shape[0] == 10, f"{returns.shape[0]}, {task}, {env_id}, {method}"
                if normalize:
                    if normalizing_factor < 0:
                        returns = (returns - normalizing_factor) / - normalizing_factor
                    else:
                        returns = returns / normalizing_factor
                score_dict[method].append(returns.mean())
                score_std_dict[method].append(returns.std())

        for method in method_filter:
            task_score_dict[method].append(score_dict[method])
            task_score_std_dict[method].append(score_std_dict[method])
    for method in method_filter:
        # task_score_dict[method] = np.array(task_score_dict[method]).mean(axis=0)
        task_score_dict[method] = np.array(task_score_dict[method])
        task_score_std_dict[method] = np.array(task_score_std_dict[method])
    return task_score_dict, task_score_std_dict


def run_comparisons_Main():
    project_id = "DMASH-Main-v0"
    task_filter = ["Training", "Eval-in", "Eval-out"]
    method_filter = [
        "DMA*-SH",
        "DMA*",
        "DMA-Pearl",
        "DMA",
        # "Amago",
        "DR",
        "DA",
        "Concat",
    ]

    env_normalizing_factors = {
        "DI-sparse-v0": 100,
        "DI-friction-sparse-v0": 100,
        "ODE-v0": 200,
        "cartpole-balance-v0": 1000,
        "ball_in_cup-catch-v0": 1000,
        "walker-walk-v0": 1000,
    }
    env_max_steps = {
        "DI-sparse-v0": 100000,
        "DI-friction-sparse-v0": 100000,
        "ODE-v0": 100000,
        "cartpole-balance-v0": 100000,
        "ball_in_cup-catch-v0": 200000,
        "walker-walk-v0": 200000,
    }

    def custom_method_fun(metadata):
        if metadata["method"] == "unaware_dr":
            method = "DR"
        elif metadata["method"] == "aware_concat":
            method = "Concat"
        elif metadata["method"] == "aware_hypernet":
            method = "DA"
        elif metadata["method"] == "inferred_plain_concat":
            method = "DMA"
        elif metadata["method"] == "inferred_plain_pearl":
            method = "DMA-Pearl"
        elif metadata["method"] == "inferred_concat":
            method = "DMA*"
        elif metadata["method"] == "inferred_hypernet_shared":
            method = "DMA*-SH"
        elif metadata["method"] == "Amago":
            method = "Amago"
        return method

    aggregate_scores_dict, aggregate_score_cis_dict = compute_iqm_scores(
        project_id, task_filter, method_filter, env_normalizing_factors, max_steps=env_max_steps, custom_method_fun=custom_method_fun
    )
    print(aggregate_scores_dict)

    fig, axes = plot_utils.plot_interval_estimates(
        aggregate_scores_dict, aggregate_score_cis_dict,
        metric_names=task_filter,
        algorithms=list(aggregate_scores_dict.keys()),
        xlabel='IQM normalized scores',
        xlabel_y_coordinate=-0.1,
    )
    fig.savefig(f"iqm_{project_id}.pdf", bbox_inches="tight")

    # get aer table
    scores_dict_all, scores_std_dict_all = compute_aer_scores(
        project_id, task_filter, method_filter, env_normalizing_factors, max_steps=env_max_steps, custom_method_fun=custom_method_fun
    )
    task_filter.append("all")
    for t, task in enumerate(task_filter):
        if task == "all":
            scores_dict = {ks: scores_dict_all[ks].mean(0) for ks in method_filter}
            scores_std_dict = {ks: scores_std_dict_all[ks].mean(0) for ks in method_filter}
        else:
            scores_dict = {ks: scores_dict_all[ks][t] for ks in method_filter}
            scores_std_dict = {ks: scores_std_dict_all[ks][t] for ks in method_filter}
        print()
        print(task)
        print(list(reversed(method_filter)))
        print()
        for i in range(6):
            # ks = ["unaware_dr", "amago", "aware_concat", "aware_hypernet", "inferred_plain_concat", "inferred_plain_pearl", "inferred_concat", "inferred_hypernet_shared"]
            ks = list(reversed(method_filter))
            li = []
            for k in ks:
                s = scores_dict[k][i]
                s_std = scores_std_dict[k][i]
                all_s = [scores_dict[ink][i] for ink in ks]
                print_score = int(s.round(0))
                print_score_std = int(s_std.round(0))
                if s >= 0.99 * max(all_s):
                    li.append(r"& \textbf{" + f"{print_score}" + r"$\pm$" + f"{print_score_std}" + "}")
                elif s >= 1.01 * max(all_s):
                    li.append(r"& \textbf{" + f"{print_score}" + r"$\pm$" + f"{print_score_std}" + "}")
                else:
                    li.append(f"& {print_score}" + r"$\pm$" + f"{print_score_std}")
            li.append(r"\\")
            print(" ".join(li))

        ks = list(reversed(method_filter))
        li = []
        for k in ks:
            s = (scores_dict[k] / np.array([100, 100, 200, 1000, 1000, 1000])).mean()
            s_std = (scores_dict[k] / np.array([100, 100, 200, 1000, 1000, 1000])).std()
            all_s = [(scores_dict[ink] / np.array([100, 100, 200, 1000, 1000, 1000])).mean() for ink in ks]
            print_score = float(s.round(2))
            print_score_std = float(s_std.round(2))
            if s >= 0.99 * max(all_s):
                # li.append(r"& \textbf{" + f"{print_score}" + r"$\pm$" + f"{print_score_std}" + "}")
                li.append(r"& \textbf{" + f"{print_score}" + "}")
            elif s >= 1.01 * max(all_s):
                # li.append(r"& \textbf{" + f"{print_score}" + r"$\pm$" + f"{print_score_std}" + "}")
                li.append(r"& \textbf{" + f"{print_score}" + "}")
            else:
                # li.append(f"& {print_score}" + r"$\pm$" + f"{print_score_std}")
                li.append(f"& {print_score}")
        li.append(r"\\")
        print(" ".join(li))


if __name__ == "__main__":
    run_comparisons_Main()

