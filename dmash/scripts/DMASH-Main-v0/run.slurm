#!/bin/bash -l

#SBATCH --ntasks=70
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=72:00:00
#SBATCH --mail-type=ERR,FAIL

export OMP_NUM_THREADS=1
export WANDB__SERVICE_WAIT=300

# slurm loading
. /etc/profile.d/module.sh
module purge




DMASHDIR=XXX  # change this!!




cd $DMASHDIR/dmash/contexts/

# uv
source $DMASHDIR/.venv/bin/activate

# experiment config
n_runs=10
start_seed=0
static_selector=0
while getopts s:n:a: flag; do
  case "${flag}" in
    s) start_seed=${OPTARG};;
    n) n_runs=${OPTARG};;
    a) static_selector=${OPTARG};;
  esac
done
project_name="DMASH-Main-v0"

case $static_selector in
  0)
    static_args="--env_id dm_control/walker-walk-v0 --context_size 128 --total_timesteps 200000 --num_train_envs 20 --num_eval_envs 20 --verbose False --write True --wandb True --wandb_offline True --wandb_project_name $project_name"
    ;;
  1)
    static_args="--env_id dm_control/ball_in_cup-catch-v0 --context_size 128 --total_timesteps 200000 --num_train_envs 20 --num_eval_envs 20 --verbose False --write True --wandb True --wandb_offline True --wandb_project_name $project_name"
    ;;
  2)
    static_args="--env_id dm_control/cartpole-balance-v0 --context_size 128 --total_timesteps 100000 --num_train_envs 20 --num_eval_envs 20 --verbose False --write True --wandb True --wandb_offline True --wandb_project_name $project_name"
    ;;
  3)
    static_args="--env_id dmash/ODE-v0 --context_size 24 --total_timesteps 100000 --num_train_envs 20 --num_eval_envs 20 --verbose False --write True --wandb True --wandb_offline True --wandb_project_name $project_name"
    ;;
  4)
    static_args="--env_id dmash/DI-sparse-v0 --context_size 24 --total_timesteps 100000 --num_train_envs 20 --num_eval_envs 20 --verbose False --write True --wandb True --wandb_offline True --wandb_project_name $project_name"
    ;;
  5)
    static_args="--env_id dmash/DI-friction-sparse-v0 --context_size 24 --total_timesteps 100000 --num_train_envs 20 --num_eval_envs 20 --verbose False --write True --wandb True --wandb_offline True --wandb_project_name $project_name"
    ;;
  *)
    exit 1
esac


changing_args=()

for method in unaware_dr aware_concat aware_hypernet inferred_plain_concat inferred_plain_pearl inferred_concat inferred_hypernet_shared; do
  changing_args+=("--method $method")
done

# run experiment
for ca in "${changing_args[@]}"
do
  for i in $(seq $n_runs)
  do
    seed=$((start_seed + i - 1))
    srun --exclusive -N 1 -n 1 python sac.py --seed=$seed $static_args $ca &
    sleep 2
  done
done
wait

