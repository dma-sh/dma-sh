import random
import time
import pathlib
from collections import deque
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import tyro
import wandb

from dmash.contexts.context import setup_context_env
from dmash.contexts.dataset import MultiContextReplayBuffer
from dmash.contexts.encoder import (
    MLPContextEncoder,
    RNNContextEncoder,
    LSTMContextEncoder,
    TransformerContextEncoder,
    EnsembleContextEncoder,
    PMLPContextEncoder,
    PLSTMContextEncoder,
)
from dmash.contexts.model import (
    SoftQNetwork,
    Actor,
    ForwardModel,
    RewardModel,
    ForwardModelEnsemble,
    InverseModel,
    StateDecoderModel,
    ContextDecoderModel,
    RNDModel,
)

from dmash.contexts.util import (
    compute_context,
    compute_losses,
    anneal_beta,
    compute_expressiveness,
    compute_context_variability,
    get_tsne,
    prepare_contexts,
)
from dmash.contexts.config import Args, modify_groups
from dmash.common.logger import Logger, JSONLOutput, WandBOutput, TerminalOutput
from dmash.disentanglement import dci, mig, beta_vae, dcimig


def fill_context_window(
    args, actor, dce, rce, fm, im, envs, device, obs_stack, actions_stack, next_obs_stack, rewards_stack
):
    obs, _ = envs.reset()
    for _ in range(args.context_size):
        if args.context_warm_up_random:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            context_data = {
                "actions": np.array(actions_stack).swapaxes(0, 1),
                "observations": np.array(obs_stack).swapaxes(0, 1),
                "next_observations": np.array(next_obs_stack).swapaxes(0, 1),
                "rewards": np.array(rewards_stack, dtype=np.float32).swapaxes(0, 1),
            }
            dynamics_context, reward_context = compute_context(
                dce, rce, context_data, device, args, training=False
            )
            if args.policy_context_merge_type == "hypernet_shared":
                if fm is not None:
                    hnet_weights = fm.get_hnet_weights(
                        context=dynamics_context,
                        obs=torch.tensor(context_data["observations"]).to(device)[:, -1]
                    )
                elif im is not None:
                    hnet_weights = im.get_hnet_weights(
                        context=dynamics_context,
                        obs=torch.tensor(context_data["observations"]).to(device)[:, -1]
                    )
                else:
                    raise NotImplementedError()
            else:
                hnet_weights = None
            actions, _, _ = actor.get_action(
                torch.Tensor(obs).to(device), dynamics_context, hnet_weights, training=False
            )
            actions = actions.detach().cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        obs_stack.append(obs)
        next_obs_stack.append(next_obs)
        actions_stack.append(actions)
        rewards_stack.append(rewards)
        obs = next_obs


def make_evaluation_and_fill_ds(args, actor, dce, rce, fm, im, envs, rb, device):
    # Reset queues
    for i in range(envs.num_envs):
        envs.envs[i].return_queue = deque(maxlen=args.num_eval_episodes)
        envs.envs[i].length_queue = deque(maxlen=args.num_eval_episodes)
        envs.envs[i].is_success_queue = deque(maxlen=args.num_eval_episodes)
    for _ in range(args.num_eval_episodes):
        obs, _ = envs.reset()
        actions = np.array(
            [np.zeros(envs.single_action_space.shape, dtype=np.float32) for _ in range(envs.num_envs)]
        )
        rewards = np.array([0.0 for _ in range(envs.num_envs)])
        obs_stack = deque([obs] * args.context_size, args.context_size)
        next_obs_stack = deque([obs] * args.context_size, args.context_size)
        actions_stack = deque([actions] * args.context_size, args.context_size)
        rewards_stack = deque([rewards] * args.context_size, args.context_size)

        if args.context_warm_up:
            fill_context_window(
                args, actor, dce, rce, fm, im, envs, device, obs_stack, actions_stack, next_obs_stack, rewards_stack
            )

        obs, _ = envs.reset()
        done = False
        dynamics_context_eval, reward_context_eval = None, None
        while not done:
            context_data = {
                "actions": np.array(actions_stack).swapaxes(0, 1),
                "observations": np.array(obs_stack).swapaxes(0, 1),
                "next_observations": np.array(next_obs_stack).swapaxes(0, 1),
                "rewards": np.array(rewards_stack, dtype=np.float32).swapaxes(0, 1),
            }
            if dynamics_context_eval is None or not args.context_once:
                dynamics_context_eval, reward_context_eval = compute_context(
                    dce, rce, context_data, device, args, training=False
                )
            if args.policy_context_merge_type == "hypernet_shared":
                if fm is not None:
                    hnet_weights = fm.get_hnet_weights(
                        context=dynamics_context_eval,
                        obs=torch.tensor(context_data["observations"]).to(device)[:, -1]
                    )
                elif im is not None:
                    hnet_weights = im.get_hnet_weights(
                        context=dynamics_context_eval,
                        obs=torch.tensor(context_data["observations"]).to(device)[:, -1]
                    )
                else:
                    raise NotImplementedError()

            else:
                hnet_weights = None
            actions, _, _ = actor.get_action(
                torch.Tensor(obs).to(device), dynamics_context_eval, hnet_weights, training=False
            )
            actions = actions.detach().cpu().numpy()

            next_obs, rewards, terminations, truncations, infos = envs.step(actions)

            obs_stack.append(obs)
            next_obs_stack.append(next_obs)
            actions_stack.append(actions)
            rewards_stack.append(rewards)

            for i in range(envs.num_envs):
                rb.insert(
                    dict(
                        observations=np.array(obs_stack)[:, i, :],
                        next_observations=np.array(next_obs_stack)[:, i, :],
                        actions=np.array(actions_stack)[:, i, :],
                        rewards=np.array(rewards_stack)[:, i],
                        masks=np.logical_not(terminations)[i],
                        dones=np.logical_or(terminations, truncations)[i],
                    ),
                    dataset_index=i
                )

            obs = next_obs
            done = any(np.logical_or(terminations, truncations))
            if done and "is_success" in infos["episode"]:
                for i in range(envs.num_envs):
                    envs.envs[i].is_success_queue.append(infos["episode"]["is_success"][i])

    # track episodic returns for individual envs and averaged over all envs
    episodic_returns = []
    for i in range(envs.num_envs):
        if len(envs.envs[i].is_success_queue) == args.num_eval_episodes:
            episodic_return = np.mean(envs.envs[i].is_success_queue)
        else:
            episodic_return = np.mean(envs.envs[i].return_queue)
        episodic_returns.append(episodic_return)
    return episodic_returns


def train():
    args = tyro.cli(Args)
    args = modify_groups(args)
    wandb_id = wandb.util.generate_id()
    run_name = f"sac__{args.env_id.split('/')[-1]}__{args.seed}__{int(time.time())}__{wandb_id}"
    logger_types = []
    if args.wandb:
        logger_types.append(
            WandBOutput(
                project=args.wandb_project_name,
                config=vars(args),
                mode="offline" if args.wandb_offline else "online",
                id=wandb_id
            ),
        )
    if args.verbose:
        logger_types.append(
            TerminalOutput(
                pattern="rl_returns_train/|rl_returns_eval/|fps"
            ),
        )
    if args.write:
        metrics_filename = f"metrics/{args.wandb_project_name}/{run_name}/metrics.json"
        logger_types.append(
            JSONLOutput(filename=metrics_filename, config=vars(args)),
        )
    logger = Logger(0, logger_types)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs, eval_train_envs, eval_in_envs, eval_out_envs, context_info = setup_context_env(args, run_name)

    actor = Actor(envs, args).to(device)
    qf1 = SoftQNetwork(envs, args).to(device)
    qf2 = SoftQNetwork(envs, args).to(device)
    qf1_target = SoftQNetwork(envs, args).to(device)
    qf2_target = SoftQNetwork(envs, args).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    # Only initialize helper models if needed
    if args.context_mode in ["unaware", "aware"]:
        args.context_encoder_update = []

    # Make sure shared hypernet attempt is configured correctly
    if (
        args.q_context_merge_type == "hypernet_shared" or
        args.policy_context_merge_type == "hypernet_shared"
    ):
        if args.context_mode in ["inferred", "aware_inferred", "aware_inferred_reconstructed"]:
            assert args.helper_hidden_dim == args.policy_hidden_dim
            assert args.helper_hidden_dim == args.q_hidden_dim
            assert not args.helper_use_embedding
            assert "fm" in args.context_encoder_update or "im" in args.context_encoder_update
            args.helper_context_merge_type = "hypernet"
        elif args.context_mode in ["aware"]:
            assert args.q_context_merge_type == "hypernet_shared"  # Only consider that case
            args.policy_context_merge_type = "hypernet"
            assert args.q_hidden_dim == args.policy_hidden_dim
        else:
            raise NotImplementedError()

    if "fm" in args.context_encoder_update:
        if args.fm_ensemble:
            fm = ForwardModelEnsemble(envs, args).to(device)
        else:
            fm = ForwardModel(envs, args).to(device)
        fm_optimizer = optim.Adam(list(fm.parameters()), lr=args.fm_lr)
    else:
        fm = None
        fm_optimizer = None
    if "im" in args.context_encoder_update:
        im = InverseModel(envs, args).to(device)
        im_optimizer = optim.Adam(list(im.parameters()), lr=args.im_lr)
    else:
        im = None
        im_optimizer = None
    if "bm" in args.context_encoder_update:
        bm = ForwardModel(envs, args).to(device)
        bm_optimizer = optim.Adam(list(bm.parameters()), lr=args.bm_lr)
    else:
        bm = None
        bm_optimizer = None
    if "rm" in args.context_encoder_update:
        rm = RewardModel(envs, args).to(device)
        rm_optimizer = optim.Adam(list(rm.parameters()), lr=args.rm_lr)
    else:
        rm = None
        rm_optimizer = None
    if args.compute_reward_context:
        if "rm" not in args.context_encoder_update:
            args.context_encoder_update.append("rm")
        rc_rm = RewardModel(envs, args).to(device)
    else:
        rc_rm = None
    if "sdm" in args.context_encoder_update:
        assert args.reconstruction_lambda != 0.0
    if args.reconstruction_lambda != 0.0 and args.context_mode in ["inferred", "aware_inferred"]:
        if "sdm" not in args.context_encoder_update:
            args.context_encoder_update.append("sdm")
        sdm = StateDecoderModel(envs, args).to(device)
        sdm_optimizer = optim.Adam(list(sdm.parameters()), lr=args.sdm_lr)
    else:
        sdm = None
        sdm_optimizer = None
    if "cdm" in args.context_encoder_update:
        assert args.reconstruction_lambda != 0.0
    if args.reconstruction_lambda != 0.0 and args.context_mode in ["aware_inferred_reconstructed"]:
        if "cdm" not in args.context_encoder_update:
            args.context_encoder_update.append("cdm")
        cdm = ContextDecoderModel(envs, args).to(device)
        cdm_optimizer = optim.Adam(list(cdm.parameters()), lr=args.cdm_lr)
    else:
        cdm = None
        cdm_optimizer = None

    if args.context_encoder == "MLP":
        encoder_module = MLPContextEncoder
    elif args.context_encoder == "HopfieldTransformer":
        encoder_module = TransformerContextEncoder
        args.context_encoder_hopfield = True
    elif args.context_encoder == "RNN":
        encoder_module = RNNContextEncoder
    elif args.context_encoder == "LSTM":
        encoder_module = LSTMContextEncoder
    elif args.context_encoder == "Transformer":
        encoder_module = TransformerContextEncoder
    elif args.context_encoder == "mixed":
        encoder_module = "mixed"
    elif args.context_encoder == "PMLP":
        encoder_module = PMLPContextEncoder
    elif args.context_encoder == "PLSTM":
        encoder_module = PLSTMContextEncoder
    else:
        raise NotImplementedError()
    if args.context_encoder_ensemble:
        dce = EnsembleContextEncoder(
            n_encoders=args.context_encoder_num,
            encoder_module=encoder_module,
            sample=args.context_encoder_ensemble_sampling,
            env=envs,
            target_dim=np.array(envs.single_observation_space.shape).prod(),
            context_size=args.context_size,
            context_dim=args.context_dim,
            model_dim=args.context_encoder_model_dim,
            num_heads=args.context_encoder_num_heads,
            num_layers=args.context_encoder_num_layers,
            dropout=args.context_encoder_dropout,
            bias=args.context_encoder_bias,
            input_norm=args.context_encoder_input_norm,
            input_symlog=args.context_encoder_input_symlog,
            output_norm=args.context_encoder_output_norm,
            output_symlog=args.context_encoder_output_symlog,
            output_shuffle=args.context_encoder_output_shuffle,
            tfixup=args.context_encoder_tfixup,
            separate_input_embedding=args.context_encoder_separate_input_embedding,
            pos_enc=False,
            input_mask=args.context_encoder_input_mask,
            input_shuffle=args.context_encoder_input_shuffle,
            liu_input_mask=args.context_encoder_liu_input_mask,
            hopfield=args.context_encoder_hopfield,
            hf_update_steps_max=args.context_encoder_hf_update_steps_max,
            hf_scaling=args.context_encoder_hf_scaling,
            bidirectional=args.context_encoder_bidirectional,
            lstm_output_type=args.context_encoder_lstm_output_type,
            device=device,
            norm=args.context_encoder_norm,
            select_percentage=args.context_encoder_select_percentage,
            select_type=args.context_encoder_select_type,
            context_mask=args.context_mask,
            context_aware=args.context_mode in ["aware_inferred", "aware_inferred_reconstructed"],
            input_noise=args.context_encoder_input_noise,
            output_noise=args.context_encoder_output_noise,
            input_distractors=args.context_encoder_input_distractors,
        ).to(device)
        rce = EnsembleContextEncoder(
            n_encoders=args.context_encoder_num,
            encoder_module=encoder_module,
            sample=args.context_encoder_ensemble_sampling,
            env=envs,
            target_dim=1,
            context_size=args.context_size,
            context_dim=args.context_dim,
            model_dim=args.context_encoder_model_dim,
            num_heads=args.context_encoder_num_heads,
            num_layers=args.context_encoder_num_layers,
            dropout=args.context_encoder_dropout,
            bias=args.context_encoder_bias,
            input_norm=args.context_encoder_input_norm,
            input_symlog=args.context_encoder_input_symlog,
            output_norm=args.context_encoder_output_norm,
            output_symlog=args.context_encoder_output_symlog,
            output_shuffle=args.context_encoder_output_shuffle,
            tfixup=args.context_encoder_tfixup,
            separate_input_embedding=args.context_encoder_separate_input_embedding,
            pos_enc=False,
            input_mask=args.context_encoder_input_mask,
            input_shuffle=args.context_encoder_input_shuffle,
            liu_input_mask=args.context_encoder_liu_input_mask,
            hopfield=args.context_encoder_hopfield,
            hf_update_steps_max=args.context_encoder_hf_update_steps_max,
            hf_scaling=args.context_encoder_hf_scaling,
            bidirectional=args.context_encoder_bidirectional,
            lstm_output_type=args.context_encoder_lstm_output_type,
            device=device,
            norm=args.context_encoder_norm,
            topk_percentage=args.context_encoder_topk_percentage,
            select_percentage=args.context_encoder_select_percentage,
            select_type=args.context_encoder_select_type,
            context_mask=args.context_mask,
            context_aware=args.context_mode in ["aware_inferred", "aware_inferred_reconstructed"],
            input_noise=args.context_encoder_input_noise,
            output_noise=args.context_encoder_output_noise,
            input_distractors=args.context_encoder_input_distractors,
        ).to(device) if args.compute_reward_context else None
    else:
        assert encoder_module != "mixed"
        dce = encoder_module(
            env=envs,
            target_dim=np.array(envs.single_observation_space.shape).prod(),
            context_size=args.context_size,
            context_dim=args.context_dim,
            model_dim=args.context_encoder_model_dim,
            num_heads=args.context_encoder_num_heads,
            num_layers=args.context_encoder_num_layers,
            dropout=args.context_encoder_dropout,
            bias=args.context_encoder_bias,
            input_norm=args.context_encoder_input_norm,
            input_symlog=args.context_encoder_input_symlog,
            output_norm=args.context_encoder_output_norm,
            output_symlog=args.context_encoder_output_symlog,
            output_shuffle=args.context_encoder_output_shuffle,
            tfixup=args.context_encoder_tfixup,
            separate_input_embedding=args.context_encoder_separate_input_embedding,
            pos_enc=False,
            input_mask=args.context_encoder_input_mask,
            input_shuffle=args.context_encoder_input_shuffle,
            liu_input_mask=args.context_encoder_liu_input_mask,
            hopfield=args.context_encoder_hopfield,
            hf_update_steps_max=args.context_encoder_hf_update_steps_max,
            hf_scaling=args.context_encoder_hf_scaling,
            bidirectional=args.context_encoder_bidirectional,
            lstm_output_type=args.context_encoder_lstm_output_type,
            device=device,
            norm=args.context_encoder_norm,
            select_percentage=args.context_encoder_select_percentage,
            select_type=args.context_encoder_select_type,
            context_mask=args.context_mask,
            context_aware=args.context_mode in ["aware_inferred", "aware_inferred_reconstructed"],
            input_noise=args.context_encoder_input_noise,
            output_noise=args.context_encoder_output_noise,
            input_distractors=args.context_encoder_input_distractors,
        ).to(device)
        rce = encoder_module(
            env=envs,
            target_dim=1,
            context_size=args.context_size,
            context_dim=args.context_dim,
            model_dim=args.context_encoder_model_dim,
            num_heads=args.context_encoder_num_heads,
            num_layers=args.context_encoder_num_layers,
            dropout=args.context_encoder_dropout,
            bias=args.context_encoder_bias,
            input_norm=args.context_encoder_input_norm,
            input_symlog=args.context_encoder_input_symlog,
            output_norm=args.context_encoder_output_norm,
            output_symlog=args.context_encoder_output_symlog,
            output_shuffle=args.context_encoder_output_shuffle,
            tfixup=args.context_encoder_tfixup,
            separate_input_embedding=args.context_encoder_separate_input_embedding,
            pos_enc=False,
            input_mask=args.context_encoder_input_mask,
            input_shuffle=args.context_encoder_input_shuffle,
            liu_input_mask=args.context_encoder_liu_input_mask,
            hopfield=args.context_encoder_hopfield,
            hf_update_steps_max=args.context_encoder_hf_update_steps_max,
            hf_scaling=args.context_encoder_hf_scaling,
            bidirectional=args.context_encoder_bidirectional,
            lstm_output_type=args.context_encoder_lstm_output_type,
            device=device,
            norm=args.context_encoder_norm,
            select_percentage=args.context_encoder_select_percentage,
            select_type=args.context_encoder_select_type,
            context_mask=args.context_mask,
            context_aware=args.context_mode in ["aware_inferred", "aware_inferred_reconstructed"],
            input_noise=args.context_encoder_input_noise,
            output_noise=args.context_encoder_output_noise,
            input_distractors=args.context_encoder_input_distractors,
        ).to(device) if args.compute_reward_context else None
    if hasattr(dce, "yield_trainable_params"):
        dce_params = dce.yield_trainable_params()
    else:
        dce_params = list(dce.parameters())
    dce_optimizer = optim.Adam(dce_params, lr=args.ce_lr)
    if args.compute_reward_context:
        if hasattr(rce, "yield_trainable_params"):
            rce_params = rce.yield_trainable_params()
        else:
            rce_params = list(rce.parameters())
        rc_rm_optimizer = optim.Adam(list(rc_rm.parameters()), lr=args.rm_lr)
        rce_optimizer = optim.Adam(rce_params, lr=args.ce_lr)
    else:
        rc_rm_optimizer = None
        rce_optimizer = None

    if args.rnd_beta != 0.0:
        rnd = RNDModel(envs, args).to(device)
        rnd_optimizer = optim.Adam(list(rnd.predictor.parameters()), lr=args.fm_lr)
    else:
        rnd = None
        rnd_optimizer = None

    episode_length = envs.envs[0].get_wrapper_attr('_max_episode_steps')
    rb = MultiContextReplayBuffer(
        envs.single_observation_space,
        envs.single_action_space,
        min(args.buffer_size, args.total_timesteps),
        args.context_size,
        envs.num_envs,
        memory_efficient=args.memory_efficient_buffer,
        episode_length=episode_length,
    )
    rb.seed(args.seed)
    eval_train_rb = MultiContextReplayBuffer(
        eval_train_envs.single_observation_space,
        eval_train_envs.single_action_space,
        min(args.dataset_size, int(args.num_eval_episodes * episode_length)),
        args.context_size,
        eval_train_envs.num_envs,
        memory_efficient=args.memory_efficient_buffer,
        episode_length=episode_length,
    )
    eval_train_rb.seed(args.seed)
    eval_in_rb = MultiContextReplayBuffer(
        eval_in_envs.single_observation_space,
        eval_in_envs.single_action_space,
        min(args.dataset_size, int(args.num_eval_episodes * episode_length)),
        args.context_size,
        eval_in_envs.num_envs,
        memory_efficient=args.memory_efficient_buffer,
        episode_length=episode_length,
    )
    eval_in_rb.seed(args.seed)
    eval_out_rb = MultiContextReplayBuffer(
        eval_out_envs.single_observation_space,
        eval_out_envs.single_action_space,
        min(args.dataset_size, int(args.num_eval_episodes * episode_length)),
        args.context_size,
        eval_out_envs.num_envs,
        memory_efficient=args.memory_efficient_buffer,
        episode_length=episode_length,
    )
    eval_out_rb.seed(args.seed)

    done = True  # used for warm-up
    dynamics_context_collect, reward_context_collect = None, None
    obs, _ = envs.reset()
    actions = np.array([np.zeros(envs.single_action_space.shape, dtype=np.float32) for _ in range(envs.num_envs)])
    rewards = np.array([0.0 for _ in range(envs.num_envs)], dtype=np.float32)
    obs_stack = deque([obs] * args.context_size, args.context_size)
    next_obs_stack = deque([obs] * args.context_size, args.context_size)
    actions_stack = deque([actions] * args.context_size, args.context_size)
    rewards_stack = deque([rewards] * args.context_size, args.context_size)
    # obs_stack shape is (context_size, num_envs, feature_size)
    # while it should be (num_envs, context_size, feature_size) for context encoder,
    # hence axes are swaped.
    for step in np.arange(-args.learning_starts, args.total_timesteps + 1, dtype=int):
        if step < 0:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            if args.context_warm_up and done:
                fill_context_window(
                    args, actor, dce, rce, fm, im, envs, device, obs_stack, actions_stack, next_obs_stack, rewards_stack
                )
                obs, _ = envs.reset()
            context_data = {
                "actions": np.array(actions_stack).swapaxes(0, 1),
                "observations": np.array(obs_stack).swapaxes(0, 1),
                "next_observations": np.array(next_obs_stack).swapaxes(0, 1),
                "rewards": np.array(rewards_stack, dtype=np.float32).swapaxes(0, 1),
            }
            if dynamics_context_collect is None or not args.context_once:
                dynamics_context_collect, reward_context_collect = compute_context(
                    dce, rce, context_data, device, args, training=True
                )
            if args.context_mode != "aware" and args.policy_context_merge_type == "hypernet_shared":
                if fm is not None:
                    hnet_weights = fm.get_hnet_weights(
                        context=dynamics_context_collect,
                        obs=torch.tensor(context_data["observations"]).to(device)[:, -1]
                    )
                elif im is not None:
                    hnet_weights = im.get_hnet_weights(
                        context=dynamics_context_collect,
                        obs=torch.tensor(context_data["observations"]).to(device)[:, -1]
                    )
                else:
                    raise NotImplementedError()
            else:
                hnet_weights = None
            actions, _, _ = actor.get_action(
                torch.Tensor(obs).to(device), dynamics_context_collect, hnet_weights, training=True
            )
            actions = actions.detach().cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        obs_stack.append(obs)
        next_obs_stack.append(next_obs)
        actions_stack.append(actions)
        rewards_stack.append(rewards)

        # obs_stack shape is (context_size, num_envs, feature_size)
        for i in range(envs.num_envs):
            rb.insert(
                dict(
                    observations=np.array(obs_stack)[:, i, :],
                    next_observations=np.array(next_obs_stack)[:, i, :],
                    actions=np.array(actions_stack)[:, i, :],
                    rewards=np.array(rewards_stack)[:, i],
                    masks=np.logical_not(terminations)[i],
                    dones=np.logical_or(terminations, truncations)[i],
                ),
                dataset_index=i
            )

        obs = next_obs
        done = any(np.logical_or(terminations, truncations))
        if done:
            obs, _ = envs.reset()

        if step >= 0:
            data = rb.sample(args.batch_size)

            # forward (helper) model update
            if step % args.helper_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.helper_frequency
                ):  # compensate for the delay by doing 'helper_frequency' instead of 1
                    for _ in range(args.helper_update_per_step):
                        helper_losses_train, _ = compute_losses(
                            fm, im, bm, rm, sdm, cdm, rc_rm, dce, rce, data, device, args, training=True
                        )
                        total_loss = (
                            helper_losses_train["fm_l2_loss"] +
                            helper_losses_train["im_l2_loss"] +
                            helper_losses_train["bm_l2_loss"] +
                            helper_losses_train["rm_l2_loss"] +
                            helper_losses_train["sdm_l2_loss"] * args.reconstruction_lambda +
                            helper_losses_train["cdm_l2_loss"] * args.reconstruction_lambda
                        )

                        total_loss = total_loss * (1 + helper_losses_train["dce_uncertainty"] * args.context_encoder_uncertainty_beta)
                        total_loss = total_loss * (1 + helper_losses_train["dce_kl"] * args.context_encoder_kl_beta)

                        total_loss = total_loss * (1 + helper_losses_train["fm_hnet_l1_reg"] * args.hypernet_l1_reg_lambda)
                        total_loss = total_loss * (1 + helper_losses_train["fm_hnet_l2_reg"] * args.hypernet_l2_reg_lambda)
                        total_loss = total_loss * (1 + helper_losses_train["fm_hnet_mean_reg"] * args.hypernet_mean_reg_lambda)

                        fm_optimizer.zero_grad() if fm_optimizer is not None else None
                        im_optimizer.zero_grad() if im_optimizer is not None else None
                        bm_optimizer.zero_grad() if bm_optimizer is not None else None
                        rm_optimizer.zero_grad() if rm_optimizer is not None else None
                        sdm_optimizer.zero_grad() if sdm_optimizer is not None else None
                        cdm_optimizer.zero_grad() if cdm_optimizer is not None else None
                        if total_loss.requires_grad or total_loss.retains_grad:
                            dce_optimizer.zero_grad()
                            total_loss.backward()
                            dce_optimizer.step()
                        fm_optimizer.step() if fm_optimizer is not None else None
                        im_optimizer.step() if im_optimizer is not None else None
                        bm_optimizer.step() if bm_optimizer is not None else None
                        rm_optimizer.step() if rm_optimizer is not None else None
                        sdm_optimizer.step() if sdm_optimizer is not None else None
                        cdm_optimizer.step() if cdm_optimizer is not None else None

                        if args.compute_reward_context:
                            rc_rm_optimizer.zero_grad()
                            rce_optimizer.zero_grad()
                            helper_losses_train["rc_rm_l2_loss"].backward()
                            rc_rm_optimizer.step()
                            rce_optimizer.step()

            actions = torch.tensor(data["actions"]).to(device)[:, -1]
            observations = torch.tensor(data["observations"]).to(device)[:, -1]
            next_observations = torch.tensor(data["next_observations"]).to(device)[:, -1]
            dones = torch.tensor(data["dones"]).to(device)
            rewards = torch.tensor(data["rewards"]).to(device)[:, -1]

            # rnd update
            if args.rnd_beta != 0.0:
                rnd_loss, rnd_reward = rnd.compute_loss(observations)
                rewards += args.rnd_beta * rnd_reward
                rnd_optimizer.zero_grad()
                rnd_loss.backward()
                rnd_optimizer.step()

            # critic update
            if args.context_intrinsic_uncertainty_beta != 0.0:
                _, helper_info = compute_losses(
                    fm, im, bm, rm, sdm, cdm, rc_rm, dce, rce, data, device, args, training=True
                )
                dynamics_context, reward_context = helper_info["dynamics_context"], helper_info["reward_context"]
                beta = anneal_beta(
                    args.context_intrinsic_uncertainty_beta,
                    step,
                    args.total_timesteps,
                    args.context_intrinsic_annealing_speed
                ) if args.context_intrinsic_annealing else args.context_intrinsic_uncertainty_beta
                entropy = helper_info["dce_uncertainty_batch"]
                if args.context_intrinsic_uncertainty_scaled == "fixed":
                    min_entropy = -5
                    max_entropy = 1.5
                    entropy = (entropy - min_entropy) / (max_entropy - min_entropy)
                elif args.context_intrinsic_uncertainty_scaled == "batch":
                    min_entropy = entropy.min()
                    max_entropy = entropy.max()
                    entropy = (entropy - min_entropy) / (max_entropy - min_entropy)
                elif args.context_intrinsic_uncertainty_scaled == "independent":
                    min_entropy = -5
                    max_entropy = 1.5
                    entropy = -(max_entropy - min_entropy) / args.total_timesteps * step + max_entropy
                rewards += beta * entropy
            if args.context_intrinsic_kl_beta != 0.0:
                _, helper_info = compute_losses(
                    fm, im, bm, rm, sdm, cdm, rc_rm, dce, rce, data, device, args, training=True
                )
                dynamics_context, reward_context = helper_info["dynamics_context"], helper_info["reward_context"]
                beta = anneal_beta(
                    args.context_intrinsic_kl_beta,
                    step,
                    args.total_timesteps,
                    args.context_intrinsic_annealing_speed
                ) if args.context_intrinsic_annealing else args.context_intrinsic_kl_beta
                rewards += beta * helper_info["dce_kl_batch"]
            if args.context_intrinsic_error_beta != 0.0:
                _, helper_info = compute_losses(
                    fm, im, bm, rm, sdm, cdm, rc_rm, dce, rce, data, device, args, training=True
                )
                dynamics_context, reward_context = helper_info["dynamics_context"], helper_info["reward_context"]
                beta = anneal_beta(
                    args.context_intrinsic_error_beta,
                    step,
                    args.total_timesteps,
                    args.context_intrinsic_annealing_speed
                ) if args.context_intrinsic_annealing else args.context_intrinsic_error_beta
                rewards += beta * helper_info["fm_l2_loss_batch"]
            if args.context_intrinsic_uncertainty_beta == 0 and args.context_intrinsic_error_beta == 0 and args.context_intrinsic_kl_beta == 0:
                dynamics_context, reward_context = compute_context(
                    dce, rce, data, device, args, training=True
                )
            with torch.no_grad():
                if (
                    args.q_context_merge_type == "hypernet_shared" or
                    args.policy_context_merge_type == "hypernet_shared"
                ):
                    if args.context_mode == "aware":
                        hnet_weights = actor.get_hnet_weights(
                            context=dynamics_context,
                            obs=observations
                        )
                    elif fm is not None:
                        hnet_weights = fm.get_hnet_weights(
                            context=dynamics_context,
                            obs=observations
                        )
                    elif im is not None:
                        hnet_weights = im.get_hnet_weights(
                            context=dynamics_context,
                            obs=observations
                        )
                    else:
                        raise NotImplementedError()
                else:
                    hnet_weights = None
                next_state_actions, next_state_log_pi, _ = actor.get_action(
                    next_observations, dynamics_context, hnet_weights, training=True
                )
                qf1_next_target = qf1_target(next_observations, next_state_actions, dynamics_context, hnet_weights)
                qf2_next_target = qf2_target(next_observations, next_state_actions, dynamics_context, hnet_weights)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = rewards.flatten() + (1 - dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(observations, actions, dynamics_context, hnet_weights).view(-1)
            qf2_a_values = qf2(observations, actions, dynamics_context, hnet_weights).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            dce_optimizer.zero_grad()
            if args.compute_reward_context:
                rce_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()
            dce_optimizer.step()
            if args.compute_reward_context:
                rce_optimizer.step()

            # actor update
            if (step == 0) or (step % args.policy_frequency == 0):  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1

                    # encode contexts for each policy update in case encoder is updated with policy
                    dynamics_context, reward_context = compute_context(
                        dce, rce, data, device, args, training=True
                    )
                    with torch.no_grad():
                        if (
                            args.q_context_merge_type == "hypernet_shared" or
                            args.policy_context_merge_type == "hypernet_shared"
                        ):
                            if args.context_mode == "aware":
                                hnet_weights = actor.get_hnet_weights(
                                    context=dynamics_context,
                                    obs=observations
                                )
                            elif fm is not None:
                                hnet_weights = fm.get_hnet_weights(
                                    context=dynamics_context,
                                    obs=observations
                                )
                            elif im is not None:
                                hnet_weights = im.get_hnet_weights(
                                    context=dynamics_context,
                                    obs=observations
                                )
                            else:
                                raise NotImplementedError()
                        else:
                            hnet_weights = None
                    pi, log_pi, _ = actor.get_action(
                        observations, dynamics_context, hnet_weights, training=True
                    )
                    qf1_pi = qf1(observations, pi, dynamics_context, hnet_weights)
                    qf2_pi = qf2(observations, pi, dynamics_context, hnet_weights)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    dce_optimizer.zero_grad()
                    if args.compute_reward_context:
                        rce_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()
                    dce_optimizer.step()
                    if args.compute_reward_context:
                        rce_optimizer.step()

                    # alpha update
                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(
                                observations, dynamics_context, hnet_weights, training=True
                            )
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # target networks update
            if step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            # loss dicts for logging
            rl_losses_train = {
                "qf1_values": qf1_a_values.mean().item(),
                "qf2_values": qf2_a_values.mean().item(),
                "qf1_loss": qf1_loss.item(),
                "qf2_loss": qf2_loss.item(),
                "qf_loss": qf_loss.item() / 2,
                "actor_loss": actor_loss.item(),
                "alpha": alpha,
            }
            if args.autotune:
                rl_losses_train["alpha_loss"] = alpha_loss.item()

            # make evaluation and logging
            if step % 5000 == 0:
                eval_actor = deepcopy(actor)
                eval_qf1 = deepcopy(qf1)
                eval_dce = deepcopy(dce)
                eval_rce = deepcopy(rce)
                eval_fm = deepcopy(fm)
                eval_im = deepcopy(im)
                eval_bm = deepcopy(bm)
                eval_rm = deepcopy(rm)
                eval_sdm = deepcopy(sdm)
                eval_cdm = deepcopy(cdm)
                eval_rc_rm = deepcopy(rc_rm)

                helper_losses_eval = {}
                helper_losses_eval_i = {}
                rl_returns_eval = {}
                rl_returns_eval_i = {}
                disentanglement_eval = {}
                expressiveness_eval = {}
                context_variability_eval = {}
                for log_name, eval_rb, eval_envs in zip(
                    ["eval_train", "eval_in", "eval_out"],
                    [eval_train_rb, eval_in_rb, eval_out_rb],
                    [eval_train_envs, eval_in_envs, eval_out_envs]
                ):
                    eval_episodic_returns = make_evaluation_and_fill_ds(
                        args,
                        eval_actor,
                        eval_dce,
                        eval_rce,
                        eval_fm,
                        eval_im,
                        eval_envs,
                        eval_rb,
                        device,
                    )
                    for i, eval_episodic_return in enumerate(eval_episodic_returns):
                        rl_returns_eval_i[f"{log_name}_{i}_return"] = eval_episodic_return
                    rl_returns_eval[f"{log_name}_return"] = np.mean(eval_episodic_returns)
                    for di in [None, *range(eval_rb.num_datasets)]:
                        _data = eval_rb.sample(args.batch_size, dataset_index=di)
                        _losses, _losses_batch = compute_losses(
                            eval_fm, eval_im, eval_bm, eval_rm, eval_sdm, eval_cdm, eval_rc_rm,
                            eval_dce, eval_rce, _data, device, args, training=False
                        )
                        # track helper losses
                        if di is None:
                            helper_losses_eval[f"{log_name}_fm_l2_loss"] = _losses["fm_l2_loss"].item()
                            helper_losses_eval[f"{log_name}_fm_uncertainty"] = _losses["fm_uncertainty"].item()
                            helper_losses_eval[f"{log_name}_im_l2_loss"] = _losses["im_l2_loss"].item()
                            helper_losses_eval[f"{log_name}_bm_l2_loss"] = _losses["bm_l2_loss"].item()
                            helper_losses_eval[f"{log_name}_rm_l2_loss"] = _losses["rm_l2_loss"].item()
                            helper_losses_eval[f"{log_name}_sdm_l2_loss"] = _losses["sdm_l2_loss"].item()
                            helper_losses_eval[f"{log_name}_cdm_l2_loss"] = _losses["cdm_l2_loss"].item()
                            helper_losses_eval[f"{log_name}_rc_rm_l2_loss"] = _losses["rc_rm_l2_loss"].item()
                            helper_losses_eval[f"{log_name}_dce_uncertainty"] = _losses["dce_uncertainty"].item()
                            helper_losses_eval[f"{log_name}_dce_uncertainty_mean"] = _losses_batch["dce_uncertainty_batch"].mean().item()
                            helper_losses_eval[f"{log_name}_dce_uncertainty_min"] = _losses_batch["dce_uncertainty_batch"].min().item()
                            helper_losses_eval[f"{log_name}_dce_uncertainty_max"] = _losses_batch["dce_uncertainty_batch"].max().item()
                            helper_losses_eval[f"{log_name}_dce_uncertainty_var"] = _losses_batch["dce_uncertainty_batch"].var(correction=0).item()
                            helper_losses_eval[f"{log_name}_dce_kl"] = _losses["dce_kl"].item()
                            helper_losses_eval[f"{log_name}_fm_hnet_l1_reg"] = _losses["fm_hnet_l1_reg"].item()
                            helper_losses_eval[f"{log_name}_fm_hnet_l2_reg"] = _losses["fm_hnet_l2_reg"].item()
                            helper_losses_eval[f"{log_name}_fm_hnet_mean_reg"] = _losses["fm_hnet_mean_reg"].item()
                        else:
                            helper_losses_eval_i[f"{log_name}_{di}_fm_l2_loss"] = _losses["fm_l2_loss"].item()
                            helper_losses_eval_i[f"{log_name}_{di}_fm_uncertainty"] = _losses["fm_uncertainty"].item()
                            helper_losses_eval_i[f"{log_name}_{di}_im_l2_loss"] = _losses["im_l2_loss"].item()
                            helper_losses_eval_i[f"{log_name}_{di}_bm_l2_loss"] = _losses["bm_l2_loss"].item()
                            helper_losses_eval_i[f"{log_name}_{di}_rm_l2_loss"] = _losses["rm_l2_loss"].item()
                            helper_losses_eval_i[f"{log_name}_{di}_sdm_l2_loss"] = _losses["sdm_l2_loss"].item()
                            helper_losses_eval_i[f"{log_name}_{di}_cdm_l2_loss"] = _losses["cdm_l2_loss"].item()
                            helper_losses_eval_i[f"{log_name}_{di}_rc_rm_l2_loss"] = _losses["rc_rm_l2_loss"].item()
                            helper_losses_eval_i[f"{log_name}_{di}_dce_uncertainty"] = _losses["dce_uncertainty"].item()
                            helper_losses_eval_i[f"{log_name}_{di}_dce_kl"] = _losses["dce_kl"].item()

                        # track disentanglement
                        # only makes sense, if more than one context dim in true context
                        if len(context_info["train"][0].keys()) > 1:
                            if di is None:
                                _dynamics_context, _reward_context, _label, _true_context = prepare_contexts(
                                    eval_rb,
                                    eval_dce,
                                    eval_rce,
                                    device,
                                    args,
                                    training=False,
                                    context_info=context_info[log_name.split("_")[-1]]
                                )
                                dci_metrics = dci(_true_context, _dynamics_context)
                                mig_metric = mig(_true_context, _dynamics_context)
                                mig_kraskov_metric = mig(_true_context, _dynamics_context, kraskov=True)
                                beta_vae_metric = beta_vae(_true_context, _dynamics_context)
                                dcimig_metric = dcimig(_true_context, _dynamics_context)
                                dcimig_kraskov_metric = dcimig(_true_context, _dynamics_context, kraskov=True)
                                disentanglement_eval[f"{log_name}_dci_d"] = dci_metrics[0]
                                disentanglement_eval[f"{log_name}_dci_c"] = dci_metrics[1]
                                disentanglement_eval[f"{log_name}_dci_i"] = dci_metrics[2]
                                disentanglement_eval[f"{log_name}_mig"] = mig_metric
                                disentanglement_eval[f"{log_name}_mig_kraskov"] = mig_kraskov_metric
                                disentanglement_eval[f"{log_name}_beta_vae"] = beta_vae_metric
                                disentanglement_eval[f"{log_name}_dcimig"] = dcimig_metric
                                disentanglement_eval[f"{log_name}_dcimig_kraskov"] = dcimig_kraskov_metric

                        # track expressiveness
                        if di is None:
                            r2_expressiveness = compute_expressiveness(
                                eval_rb,
                                eval_dce,
                                eval_rce,
                                eval_actor,
                                eval_qf1,
                                eval_fm,
                                eval_im,
                                device,
                                args,
                                context_info=context_info[log_name.split("_")[-1]],
                                training=True
                            )
                            for k, v in r2_expressiveness.items():
                                expressiveness_eval[f"{log_name}_{k}"] = v

                        # track context variability
                        if di is None:
                            context_variability = compute_context_variability(
                                eval_rb,
                                eval_dce,
                                eval_rce,
                                eval_actor,
                                eval_qf1,
                                eval_fm,
                                eval_im,
                                device,
                                args,
                                context_info=context_info[log_name.split("_")[-1]],
                                training=True
                            )
                            for k, v in context_variability.items():
                                context_variability_eval[f"{log_name}_{k}"] = v

                    if args.wandb_images:
                        _dynamics_context, _reward_context, _label, _ = prepare_contexts(
                            eval_rb,
                            eval_dce,
                            eval_rce,
                            device,
                            args,
                            training=False,
                        )
                        tsne_fig = get_tsne(_dynamics_context, _label)
                        logger.add(
                            {f"{log_name}_tsne": tsne_fig},
                            prefix="context_representation_tsne_eval",
                            step=step
                        )

                # episodic returns
                mean_returns = []
                rl_returns_train_i = {}
                for i in range(envs.num_envs):
                    running_mean_return = np.mean(envs.envs[i].return_queue)
                    rl_returns_train_i[f"train_{i}_return"] = running_mean_return
                    mean_returns.append(running_mean_return)
                rl_returns_train = {"train_return": np.mean(mean_returns)}

                logger.add(
                    {k: v.item() for k, v in helper_losses_train.items()},
                    prefix="helper_losses_train",
                    step=step
                )
                logger.add(helper_losses_eval, prefix="helper_losses_eval", step=step)
                logger.add(helper_losses_eval_i, prefix="helper_losses_eval_i", step=step)
                logger.add(rl_losses_train, prefix="rl_losses_train", step=step)
                logger.add(rl_returns_train, prefix="rl_returns_train", step=step)
                logger.add(rl_returns_train_i, prefix="rl_returns_train_i", step=step)
                logger.add(rl_returns_eval, prefix="rl_returns_eval", step=step)
                logger.add(rl_returns_eval_i, prefix="rl_returns_eval_i", step=step)
                logger.add(disentanglement_eval, prefix="disentanglement_eval", step=step)
                logger.add(expressiveness_eval, prefix="expressiveness_eval", step=step)
                logger.add(context_variability_eval, prefix="context_variability_eval", step=step)
                logger.write(fps=True)

    if args.store:
        model_filename = f"models/{args.wandb_project_name}/{run_name}/model.pt"
        model_filename = pathlib.Path(model_filename)
        model_filename.parent.mkdir(parents=True, exist_ok=True)
        import dill
        torch.save(
            {
                "forward_model": fm,
                "inverse_model": im,
                "backward_model": bm,
                "reward_model": rm,
                "state_decoder_model": sdm,
                "context_decoder_model": cdm,
                "rce_reward_model": rc_rm,
                "dynamics_context_encoder": dce,
                "reward_context_encoder": rce,
                "train_ds": rb,
                "eval_train_ds": eval_train_rb,
                "eval_in_ds": eval_in_rb,
                "eval_out_ds": eval_out_rb,
                "metadata": vars(args),
                "contexts": context_info
            },
            model_filename,
            pickle_module=dill,  # dill needed to save hypernet, due to a hidden lambda
        )

    if args.wandb:
        wandb.run.finish()
    if args.wandb and args.wandb_offline:
        import subprocess
        import glob

        assert wandb_id is not None, "Provide wandb_id."
        offline_wandb_dir = glob.glob(f"./wandb/offline-run*{wandb_id}")
        assert len(offline_wandb_dir) == 1, "Offline wandb id is not unique."
        subprocess.run(["wandb", "sync", "--include-offline", offline_wandb_dir[0]])

    envs.close()


if __name__ == "__main__":
    train()
