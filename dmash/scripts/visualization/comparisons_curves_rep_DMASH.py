import json
import glob
from pathlib import Path

import numpy as np

from rliable import metrics

from dmash.common import plotting


def prepare_runs(env_id, project_id, context_id=None, method_filter=None, task_filter=None, max_steps=100000, custom_method_fun=None, task_ids=[]):
    runs = []
    metric_dirs = glob.glob(f"../../contexts/metrics/{project_id}/*{env_id}*")
    for metric_dir in metric_dirs:
        filename = f"{metric_dir}/metrics.json"
        if not Path(filename).is_file():
            continue
        with open(filename, 'rb') as f:
            for lines in f:
                metrics.append(json.loads(lines))

        filename = f"{metric_dir}/metadata.json"
        with open(filename, 'rb') as f:
            metadata = json.load(f)

        assert len(task_ids) == 3

        for task_id, task_name in zip(
            task_ids,
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
                "xs": xs,
                "ys": ys
            }
            runs.append(run_dict)
    return runs


def save_returns(
        env_ids, project_id, context_id="all", method_filter=None, task_filter=None, figsize=(2.4, 2.5), suffix="returns", env_xlim=None, env_ylim=None, titles=None, env_max_steps=None, env_name_map=None, custom_method_fun=None
):
    modified_colors = {"DR": "k", "Amago": "k", "DMA": "k", "DMA-Pearl": "k", "DMA*": "k", "DMA*-SH": "r"}
    modified_linestyles = {"DMA*": "--", "DMA": ":", "DMA*-SH": "-"}
    for e, env_id in enumerate(env_ids):
        runs = prepare_runs(
            env_id, project_id, context_id, method_filter, task_filter, max_steps=env_max_steps[env_id], custom_method_fun=custom_method_fun, task_ids=["rl_returns_eval/eval_train_return", "rl_returns_eval/eval_in_return", "rl_returns_eval/eval_out_return"],
        )
        xlim = (0, max(runs[0]["xs"])) if env_xlim is None else env_xlim[env_id]
        bins = np.linspace(*xlim, 10 + 1, endpoint=True)
        tasks = task_filter

        tensor, tasks, methods, seeds = plotting.tensor(runs, bins, tasks=tasks, methods=method_filter)

        fig, axes = plotting.plots(len(tasks), cols=len(tasks), size=figsize)

        if titles is not None:
            assert len(titles) == len(tasks)

        for i, task in enumerate(tasks if titles is None else titles):
            ax = axes[i]
            ax.set_xlim(*xlim)
            ax.xaxis.set_major_formatter(plotting.smart_format)
            if env_ylim is not None:
                ax.set_ylim(*env_ylim[env_id])
            ax.set_xlabel("steps")
            if i == 0:
                if e in [3, 4, 5]:
                    ax.set_ylabel("return", labelpad=5)
                else:
                    ax.set_ylabel("return", labelpad=8.5)

            for j, method in enumerate(methods):
                # Aggregate over seeds.
                mean = np.nanmean(tensor[i, j, :, :], 0)
                std = np.nanstd(tensor[i, j, :, :], 0)
                plotting.curve(
                    ax,
                    bins,
                    mean,
                    low=mean + std / 2,
                    high=mean - std / 2,
                    label=method,
                    order=j,
                    linestyle=modified_linestyles[method],
                    color=modified_colors[method]
                )
            if i != 0:
                ax.get_yaxis().set_ticklabels([])
        plotting.legend(fig, adjust=False, plotpad=0.3, ncol=5)
        fig.subplots_adjust(bottom=0.35, wspace=0.28)

        plotting.save(fig, f"figures/DMASH-Var-v0/{env_id}_{suffix}")


def save_expressiveness(
        env_ids, project_id, context_id="all", method_filter=None, task_filter=None, figsize=(2.4, 2.5), suffix="expressiveness", env_xlim=None, env_ylim=None, titles=None, env_max_steps=None, env_name_map=None, custom_method_fun=None
):
    modified_colors = {"DR": "k", "Amago": "k", "DMA": "k", "DMA-Pearl": "k", "DMA*": "k", "DMA*-SH": "r"}
    modified_linestyles = {"DMA*": "--", "DMA": ":", "DMA*-SH": "-"}
    for e, env_id in enumerate(env_ids):
        runs = prepare_runs(
            env_id, project_id, context_id, method_filter, task_filter, max_steps=env_max_steps[env_id], custom_method_fun=custom_method_fun, task_ids=["expressiveness_eval/eval_train_context_representation", "expressiveness_eval/eval_in_context_representation", "expressiveness_eval/eval_out_context_representation"],
        )
        xlim = (0, max(runs[0]["xs"])) if env_xlim is None else env_xlim[env_id]
        bins = np.linspace(*xlim, 10 + 1, endpoint=True)
        tasks = task_filter

        tensor, tasks, methods, seeds = plotting.tensor(runs, bins, tasks=tasks, methods=method_filter)

        fig, axes = plotting.plots(len(tasks), cols=len(tasks), size=figsize)

        if titles is not None:
            assert len(titles) == len(tasks)

        for i, task in enumerate(tasks if titles is None else titles):
            ax = axes[i]
            ax.set_title(f"{task}")

            ax.set_xlim(*xlim)
            ax.xaxis.set_major_formatter(plotting.smart_format)
            if env_ylim is not None:
                ax.set_ylim(*env_ylim[env_id])
            if i == 0:
                ax.set_ylabel("expressiveness", labelpad=9.8)

            for j, method in enumerate(methods):
                # Aggregate over seeds.
                mean = np.nanmean(tensor[i, j, :, :], 0)
                std = np.nanstd(tensor[i, j, :, :], 0)
                plotting.curve(
                    ax,
                    bins,
                    mean,
                    low=mean + std / 2,
                    high=mean - std / 2,
                    label=method,
                    order=j,
                    linestyle=modified_linestyles[method],
                    color=modified_colors[method]
                )
            if i != 0:
                ax.get_yaxis().set_ticklabels([])
            ax.get_xaxis().set_ticklabels([])
        fig.subplots_adjust(wspace=0.3)

        plotting.save(fig, f"figures/{project_id}/{env_id}_{suffix}")


def save_variability(
        env_ids, project_id, context_id="all", method_filter=None, task_filter=None, figsize=(2.4, 2.5), suffix="variability", env_xlim=None, env_ylim=None, titles=None, env_max_steps=None, env_name_map=None, custom_method_fun=None
):
    modified_colors = {"DR": "k", "Amago": "k", "DMA": "k", "DMA-Pearl": "k", "DMA*": "k", "DMA*-SH": "r"}
    modified_linestyles = {"DMA*": "--", "DMA": ":", "DMA*-SH": "-"}
    for e, env_id in enumerate(env_ids):
        runs = prepare_runs(
            env_id, project_id, context_id, method_filter, task_filter, max_steps=env_max_steps[env_id], custom_method_fun=custom_method_fun, task_ids=["context_variability_eval/eval_train_context_representation", "context_variability_eval/eval_in_context_representation", "context_variability_eval/eval_out_context_representation"],
        )
        xlim = (0, max(runs[0]["xs"])) if env_xlim is None else env_xlim[env_id]
        bins = np.linspace(*xlim, 10 + 1, endpoint=True)
        tasks = task_filter

        tensor, tasks, methods, seeds = plotting.tensor(runs, bins, tasks=tasks, methods=method_filter)

        fig, axes = plotting.plots(len(tasks), cols=len(tasks), size=figsize)

        if titles is not None:
            assert len(titles) == len(tasks)

        for i, task in enumerate(tasks if titles is None else titles):
            ax = axes[i]
            ax.set_xlim(*xlim)
            ax.xaxis.set_major_formatter(plotting.smart_format)
            if env_ylim is not None:
                ax.set_ylim(*env_ylim[env_id])
            if i == 0:
                ax.set_ylabel("variability")
            ax.set_yscale('log')

            for j, method in enumerate(methods):
                # Aggregate over seeds.
                mean = np.nanmean(tensor[i, j, :, :], 0)
                std = np.nanstd(tensor[i, j, :, :], 0)
                plotting.curve(
                    ax,
                    bins,
                    mean,
                    low=mean + std / 2,
                    high=mean - std / 2,
                    label=method,
                    order=j,
                    linestyle=modified_linestyles[method],
                    color=modified_colors[method]
                )
            if i != 0:
                ax.get_yaxis().set_ticklabels([])
            ax.get_xaxis().set_ticklabels([])
        fig.subplots_adjust(wspace=0.3)

        plotting.save(fig, f"figures/{project_id}/{env_id}_{suffix}")


def run_comparisons_Var_return(figsize=None):
    project_id = "DMASH-Main-v0"
    task_filter = ["Training", "Eval-in", "Eval-out"]
    method_filter = [
        "DMA*-SH",
        # "DMA-Pearl",
        "DMA*",
        "DMA",
        # "Amago",
        # "DR",
        # "Concat",
        # "DA",
    ]

    env_max_steps = {
        "DI-sparse-v0": 100000,
        "DI-friction-sparse-v0": 100000,
        "ODE-v0": 100000,
        "cartpole-balance-v0": 100000,
        "ball_in_cup-catch-v0": 200000,
        "walker-walk-v0": 200000,
    }
    env_name_map = {
        "DI-sparse-v0": "DI",
        "DI-friction-sparse-v0": "DI-friction",
        "ODE-v0": "ODE",
        "cartpole-balance-v0": "Cartpole",
        "ball_in_cup-catch-v0": "BallInCup",
        "walker-walk-v0": "Walker",
    }
    env_xlim = {
        "DI-sparse-v0": (0, 100000),
        "DI-friction-sparse-v0": (0, 100000),
        "ODE-v0": (0, 100000),
        "cartpole-balance-v0": (0, 100000),
        "ball_in_cup-catch-v0": (0, 200000),
        "walker-walk-v0": (0, 200000),
    }
    env_ylim = {
        "DI-sparse-v0": (0, 100),
        "DI-friction-sparse-v0": (0, 100),
        "ODE-v0": (0, 200),
        "cartpole-balance-v0": (0, 1000),
        "ball_in_cup-catch-v0": (0, 1000),
        "walker-walk-v0": (0, 1000),
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

    task_filter = ["Training", "Eval-in", "Eval-out"]
    figsize = figsize if figsize is not None else (1.2, 1.0)
    save_returns(
        env_max_steps.keys(), project_id, method_filter=method_filter,
        task_filter=task_filter, figsize=figsize,
        env_xlim=env_xlim, env_ylim=env_ylim, env_max_steps=env_max_steps, env_name_map=env_name_map, custom_method_fun=custom_method_fun
    )


def run_comparisons_Var_expressiveness(figsize=None):
    project_id = "DMASH-Main-v0"
    task_filter = ["Training", "Eval-in", "Eval-out"]
    method_filter = [
        "DMA*-SH",
        # "DMA-Pearl",
        "DMA*",
        "DMA",
        # "Amago",
        # "DR",
        # "Concat",
        # "DA",
    ]

    env_max_steps = {
        "DI-sparse-v0": 100000,
        "DI-friction-sparse-v0": 100000,
        "ODE-v0": 100000,
        "cartpole-balance-v0": 100000,
        "ball_in_cup-catch-v0": 200000,
        "walker-walk-v0": 200000,
    }
    env_name_map = {
        "DI-sparse-v0": "DI",
        "DI-friction-sparse-v0": "DI-friction",
        "ODE-v0": "ODE",
        "cartpole-balance-v0": "Cartpole",
        "ball_in_cup-catch-v0": "BallInCup",
        "walker-walk-v0": "Walker",
    }
    env_xlim = {
        "DI-sparse-v0": (0, 100000),
        "DI-friction-sparse-v0": (0, 100000),
        "ODE-v0": (0, 100000),
        "cartpole-balance-v0": (0, 100000),
        "ball_in_cup-catch-v0": (0, 200000),
        "walker-walk-v0": (0, 200000),
    }
    env_ylim = {
        "DI-sparse-v0": (0, 1.0),
        "DI-friction-sparse-v0": (0, 1.0),
        "ODE-v0": (0, 1.0),
        "cartpole-balance-v0": (0, 1.0),
        "ball_in_cup-catch-v0": (0, 1.0),
        "walker-walk-v0": (0, 1.0),
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

    task_filter = ["Training", "Eval-in", "Eval-out"]
    figsize = figsize if figsize is not None else (1.1, 0.9)
    save_expressiveness(
        env_max_steps.keys(), project_id, method_filter=method_filter,
        task_filter=task_filter, figsize=figsize,
        env_xlim=env_xlim, env_ylim=env_ylim, env_max_steps=env_max_steps, env_name_map=env_name_map, custom_method_fun=custom_method_fun
    )


def run_comparisons_Var_variability(figsize=None):
    project_id = "DMASH-Main-v0"
    task_filter = ["Training", "Eval-in", "Eval-out"]
    method_filter = [
        "DMA*-SH",
        # "DMA-Pearl",
        "DMA*",
        "DMA",
        # "Amago",
        # "DR",
        # "Concat",
        # "DA",
    ]

    env_max_steps = {
        "DI-sparse-v0": 100000,
        "DI-friction-sparse-v0": 100000,
        "ODE-v0": 100000,
        "cartpole-balance-v0": 100000,
        "ball_in_cup-catch-v0": 200000,
        "walker-walk-v0": 200000,
    }
    env_name_map = {
        "DI-sparse-v0": "DI",
        "DI-friction-sparse-v0": "DI-friction",
        "ODE-v0": "ODE",
        "cartpole-balance-v0": "Cartpole",
        "ball_in_cup-catch-v0": "BallInCup",
        "walker-walk-v0": "Walker",
    }
    env_xlim = {
        "DI-sparse-v0": (0, 100000),
        "DI-friction-sparse-v0": (0, 100000),
        "ODE-v0": (0, 100000),
        "cartpole-balance-v0": (0, 100000),
        "ball_in_cup-catch-v0": (0, 200000),
        "walker-walk-v0": (0, 200000),
    }
    env_ylim = {
        "DI-sparse-v0": (0.001, 0.5),
        "DI-friction-sparse-v0": (0.001, 0.5),
        "ODE-v0": (0.001, 0.5),
        "cartpole-balance-v0": (0.001, 0.5),
        "ball_in_cup-catch-v0": (0.001, 0.5),
        "walker-walk-v0": (0.001, 0.5),
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

    task_filter = ["Training", "Eval-in", "Eval-out"]
    figsize = figsize if figsize is not None else (1.15, 0.9)
    save_variability(
        env_max_steps.keys(), project_id, method_filter=method_filter,
        task_filter=task_filter, figsize=figsize,
        env_xlim=env_xlim, env_ylim=env_ylim, env_max_steps=env_max_steps, env_name_map=env_name_map, custom_method_fun=custom_method_fun
    )


if __name__ == "__main__":
    run_comparisons_Var_return(figsize=(1.0, 1.2))
    run_comparisons_Var_expressiveness(figsize=(1.0, 0.83))
    run_comparisons_Var_variability(figsize=(1.0, 0.83))

