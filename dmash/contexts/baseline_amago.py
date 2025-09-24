import random
import time

import gymnasium as gym
import numpy as np
import torch
import tyro
import wandb

from dmash.contexts.context import setup_context_env
from dmash.contexts.config import Args, modify_groups

import amago
from amago.nets.tstep_encoders import FFTstepEncoder
from amago.nets.traj_encoders import FFTrajEncoder, TformerTrajEncoder, GRUTrajEncoder, MambaTrajEncoder
from amago.agent import MultiTaskAgent, Agent


def train():
    args = tyro.cli(Args)
    args = modify_groups(args)
    wandb_id = wandb.util.generate_id()
    wandb_additional_configs = vars(args)

    run_name = f"amago__{args.env_id.split('/')[-1]}__{args.seed}__{int(time.time())}__{wandb_id}"

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    envs, eval_train_envs, eval_in_envs, eval_out_envs, context_info = setup_context_env(args, run_name)

    class ReduceVectorSpace(gym.Wrapper):
        @property
        def observation_space(self):
            return self.get_wrapper_attr("single_observation_space")

        @property
        def action_space(self):
            return self.get_wrapper_attr("single_action_space")

    envs = ReduceVectorSpace(envs)
    eval_train_envs = ReduceVectorSpace(eval_train_envs)
    eval_in_envs = ReduceVectorSpace(eval_in_envs)
    eval_out_envs = ReduceVectorSpace(eval_out_envs)

    def make_train_env():
        return amago.envs.AMAGOEnv(env=envs, env_name=args.env_id, batched_envs=args.num_train_envs)

    def make_val_train_env():
        return amago.envs.AMAGOEnv(env=eval_train_envs, env_name=args.env_id, batched_envs=args.num_eval_envs)

    def make_val_in_env():
        return amago.envs.AMAGOEnv(env=eval_in_envs, env_name=args.env_id, batched_envs=args.num_eval_envs)

    def make_val_out_env():
        return amago.envs.AMAGOEnv(env=eval_out_envs, env_name=args.env_id, batched_envs=args.num_eval_envs)

    if args.traj_encoder == "ff":
        traj_encoder_type = FFTrajEncoder
    elif args.traj_encoder == "transformer":
        traj_encoder_type = TformerTrajEncoder
    elif args.traj_encoder == "gru":
        traj_encoder_type = GRUTrajEncoder
    elif args.traj_encoder == "mamba":
        traj_encoder_type = MambaTrajEncoder
    else:
        raise NotImplementedError()

    if args.multitask_agent:
        agent_type = MultiTaskAgent
    else:
        agent_type = Agent

    experiment = amago.Experiment(
        max_seq_len=args.context_size,
        traj_save_len=1024,
        make_train_env=make_train_env,
        make_val_train_env=make_val_train_env,
        make_val_in_env=make_val_in_env,
        make_val_out_env=make_val_out_env,
        parallel_actors=args.parallel_actors,  # match batch dim of environment
        env_mode="already_vectorized",  # prevents spawning multiple async instances
        tstep_encoder_type=FFTstepEncoder,
        traj_encoder_type=traj_encoder_type,
        agent_type=agent_type,
        log_to_wandb=args.wandb,
        wandb_project=args.wandb_project_name,
        run_name=run_name,
        dset_name=run_name,
        dset_root="amago_dset_root",
        verbose=args.verbose,
        epochs=args.total_timesteps // 1000,
        log_interval=5000,
        val_interval=5,
        wandb_additional_configs=wandb_additional_configs,
    )
    experiment.start()
    experiment.learn()


if __name__ == "__main__":
    train()
