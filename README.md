# DMA-SH

Code for "Shared Dynamic Model Aligned Hypernetworks for Contextual Reinforcement Learning".
It introduces DMA*-SH, an approach for contextual RL that does not assume explicit context information to be available. It extends dynamic model-aligned (DMA) context inference by an improved context representation and context utilization via shared hypernetworks.

## Setup

We use `uv` to setup the python environment. First, make sure `uv` is [installed](https://github.com/astral-sh/uv).
Then, from the current directory `./` run

```bash
uv sync
```

It will create the python environment, which has to be activated via

```bash
source ./.venv/bin/activate
```

## Basic usage

From within the `./dmash/contexts` directory, run

```bash
python sac.py
```

Configurations can be found and changed in `./dmash/contexts/config.py` and from the command line

```bash
python sac.py --wandb True
```

For common baselines, run one of the following presets

```bash
python sac.py --method unaware_dr                   # unaware, domain randomization, DR
python sac.py --method aware_concat                 # Aware, concat
python sac.py --method aware_hypernet               # Aware, Decision Adapter, DA
python sac.py --method inferred_plain_concat        # DMA
python sac.py --method inferred_plain_pearl         # DMA
python sac.py --method inferred_concat              # DMA*
python sac.py --method inferred_hypernet_shared     # DMA*-SH
```

## Compare methods

To reproduce the comparison of the different approaches across 6 environments and 10 seeds, consider running the slurm script from within `./dmash/scripts/DMASH-Main-v0`. Make sure that `DMASHDIR` is configured correctly inside `run.slurm`:

```bash 
bash run_all.sh
```

and then create visualization using the scripts inside `./dmash/scripts/visualization/`.

