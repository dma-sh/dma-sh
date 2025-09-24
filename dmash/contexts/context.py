import itertools
import numpy as np
import gymnasium as gym
from dmash.wrappers import (
    SinglePrecisionObservation,
    SinglePrecisionAction,
    ContextAwareObservation,
    NoisyObservation,
    DistractedObservation,
)

CONTEXT_DEFAULTS = {
    "dm_control/ball_in_cup-catch-v0": {
        "gravity": (9.81, [8.0, 12.0], [1.0, 20.0]),
        "distance": (0.3, [0.24, 0.36], [0.1, 0.5])
    },
    "dm_control/walker-walk-v0": {
        "gravity": (9.81, [4.91, 14.71], [0.98, 19.62]),
        "actuator_strength": (1.0, [0.5, 1.5], [0.1, 2.0])
    },
    "dm_control/cartpole-balance-v0": {
        "length": (0.5, [0.3, 0.85], [0.1, 2.0]),
        "action_factor": (1.0, [-1.0, 1.0], [-1.0, 1.0])
    },
    "dmash/DI-sparse-v0": {
        "mass": (1.0, [0.5, 1.5], [0.1, 2.0]),
        "action_factor": (1.0, [-1.0, 1.0], [-1.0, 1.0])
    },
    "dmash/DI-friction-sparse-v0": {
        "mass": (1.0, [0.5, 1.5], [0.1, 2.0]),
        "friction": (1.0, [0.5, 1.5], [0.1, 2.0])
    },
    "dmash/ODE-v0": {
        "parameter_0": (0.0, [-5.0, 5.0], [-10.0, 10.0]),
        "parameter_1": (0.0, [-5.0, 5.0], [-10.0, 10.0]),
    },
}
for i in range(1, 21):
    CONTEXT_DEFAULTS[f"dmash/ODE-{i}-v0"] = {
        f"parameter_{j}": (0.0, [-5.0, 5.0], [-10.0, 10.0]) for j in range(i)
    }


def make_env(
    env_id,
    seed,
    idx,
    capture_video,
    context_aware,
    context_aware_noisy,
    context_aware_varied,
    context_aware_onehot,
    observation_noisy,
    num_distractors,
    fixed_distractors,
    normalize_reward,
    run_name,
    num_envs=None,
    contexts={}
):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array", disable_env_checker=True, **contexts)
            env = gym.wrappers.RecordVideo(
                env, f"videos/{run_name}_{idx}", video_length=100, episode_trigger=lambda x: x % 10 == 0
            )
        else:
            env = gym.make(env_id, disable_env_checker=True, **contexts)
        if context_aware:
            env = ContextAwareObservation(env, context_aware_noisy, context_aware_varied, contexts)
        if observation_noisy > 0.0:
            env = NoisyObservation(env, observation_noisy)
        if num_distractors > 0:
            env = DistractedObservation(env, num_distractors, fixed_distractors)
        env = SinglePrecisionObservation(env)
        env = SinglePrecisionAction(env)
        # dict envs are not supported, they are flattened
        if isinstance(env.observation_space, gym.spaces.Dict):
            env = gym.wrappers.FlattenObservation(env)
        if normalize_reward:
            env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=20)
        _, _ = env.reset(seed=seed)
        env.action_space.seed(seed)
        # set max_episode_steps for dm_control
        if "dm_control" in env_id:
            env._max_episode_steps = int(env.unwrapped._env._step_limit)
        return env
    return thunk


def select_context_instances(contexts, n_instances, select_type="sample"):
    if len(contexts.keys()) <= 2:
        # create mesh of context combinations
        # that gets out of hand if number of contexts gets too large (> 2)
        mesh = [dict(zip(contexts.keys(), values)) for values in itertools.product(*contexts.values())]
        mesh = sorted(mesh, key=lambda k: [k[ks] for ks in k.keys()])
        if select_type == "sample":
            replace = len(mesh) < n_instances
            idx = np.sort(np.random.choice(np.arange(0, len(mesh), dtype=int), n_instances, replace=replace))
        elif select_type == "equidistant":
            if len(mesh) < n_instances:
                group_size = n_instances // len(mesh)
                remainder = n_instances % len(mesh)
                idx = np.repeat(np.arange(len(mesh)), group_size)
                if remainder > 0:
                    idx = np.concatenate([idx, np.arange(remainder)])
            else:
                idx = np.linspace(0, len(mesh) - 1, n_instances, dtype=int)
        else:
            raise NotImplementedError()
        mesh = np.array(mesh)[idx].tolist()
    else:
        mesh = []
        for _ in range(n_instances):
            instance = {c_name: np.random.choice(c_mesh) for c_name, c_mesh in contexts.items()}
            mesh.append(instance)
        mesh = sorted(mesh, key=lambda k: [k[ks] for ks in k.keys()])
    return mesh


def setup_context_env(args, run_name):
    # seeding, in case of context sampling
    np.random.seed(args.seed)

    assert args.env_id in CONTEXT_DEFAULTS.keys()

    if args.context_handcrafted:
        if "ball_in_cup-catch-v0" in args.env_id:
            train_env_contexts = [
                {"gravity": 9.81, "distance": 0.3},
                {"gravity": 9.81, "distance": 0.15},
                {"gravity": 4.91, "distance": 0.3},
                {"gravity": 4.91, "distance": 0.15},
            ]
            in_env_contexts = [
                {"gravity": 9.81, "distance": 0.2},
                {"gravity": 7, "distance": 0.2},
            ]
            out_env_contexts = [
                {"gravity": 9.81, "distance": 0.5},
                {"gravity": 12, "distance": 0.5},
            ]
        elif "walker-walk-v0" in args.env_id:
            train_env_contexts = [
                {"gravity": 9.81, "left_leg_length": 0.2},
                {"gravity": 9.81, "left_leg_length": 0.4},
            ]
            in_env_contexts = [
                {"gravity": 9.81, "left_leg_length": 0.2},
                {"gravity": 9.81, "left_leg_length": 0.4},
            ]
            out_env_contexts = [
                {"gravity": 9.81, "left_leg_length": 0.2},
                {"gravity": 9.81, "left_leg_length": 0.4},
            ]
        elif "dm_control/cartpole-balance-v0" in args.env_id:
            train_env_contexts = [
                {"length": 0.3, "action_factor": 1},
                {"length": 0.3, "action_factor": -1},
                {"length": 0.5, "action_factor": 1},
                {"length": 0.5, "action_factor": -1},
                {"length": 0.8, "action_factor": 1},
                {"length": 0.8, "action_factor": -1},
            ]
            in_env_contexts = [
                {"length": 0.2, "action_factor": 1},
                {"length": 0.2, "action_factor": -1},
                {"length": 0.5, "action_factor": 1},
                {"length": 0.5, "action_factor": -1},
                {"length": 0.7, "action_factor": 1},
                {"length": 0.7, "action_factor": -1},
                {"length": 1.0, "action_factor": 1},
                {"length": 1.0, "action_factor": -1},
            ]
            out_env_contexts = [
                {"length": 0.2, "action_factor": 1},
                {"length": 0.2, "action_factor": -1},
                {"length": 0.4, "action_factor": 1},
                {"length": 0.4, "action_factor": -1},
                {"length": 0.6, "action_factor": 1},
                {"length": 0.6, "action_factor": -1},
                {"length": 0.8, "action_factor": 1},
                {"length": 0.8, "action_factor": -1},
                {"length": 1.0, "action_factor": 1},
                {"length": 1.0, "action_factor": -1},
            ]
        elif "dmash/ODE-v0" in args.env_id:
            train_env_contexts = [
                {"parameter_0": -5.0, "parameter_1": 0.0},
                {"parameter_0": -1.0, "parameter_1": 0.0},
                {"parameter_0": 1.0, "parameter_1": 0.0},
                {"parameter_0": 5.0, "parameter_1": 0.0},
            ]
            in_env_contexts = [
                {"parameter_0": -4.5, "parameter_1": 0.0},
                {"parameter_0": -4.0, "parameter_1": 0.0},
                {"parameter_0": -3.5, "parameter_1": 0.0},
                {"parameter_0": -3.0, "parameter_1": 0.0},
                {"parameter_0": -2.5, "parameter_1": 0.0},
                {"parameter_0": -2.0, "parameter_1": 0.0},
                {"parameter_0": -1.5, "parameter_1": 0.0},
                {"parameter_0": -0.5, "parameter_1": 0.0},
                {"parameter_0": 0.5, "parameter_1": 0.0},
                {"parameter_0": 1.5, "parameter_1": 0.0},
                {"parameter_0": 2.0, "parameter_1": 0.0},
                {"parameter_0": 2.5, "parameter_1": 0.0},
                {"parameter_0": 3.0, "parameter_1": 0.0},
                {"parameter_0": 3.5, "parameter_1": 0.0},
                {"parameter_0": 4.0, "parameter_1": 0.0},
                {"parameter_0": 4.5, "parameter_1": 0.0},
            ]
            out_env_contexts = [
                {"parameter_0": -10.0, "parameter_1": 0.0},
                {"parameter_0": -9.5, "parameter_1": 0.0},
                {"parameter_0": -9.0, "parameter_1": 0.0},
                {"parameter_0": -8.5, "parameter_1": 0.0},
                {"parameter_0": -8.0, "parameter_1": 0.0},
                {"parameter_0": -7.5, "parameter_1": 0.0},
                {"parameter_0": -7.0, "parameter_1": 0.0},
                {"parameter_0": -6.5, "parameter_1": 0.0},
                {"parameter_0": -6.0, "parameter_1": 0.0},
                {"parameter_0": 6.0, "parameter_1": 0.0},
                {"parameter_0": 6.5, "parameter_1": 0.0},
                {"parameter_0": 7.0, "parameter_1": 0.0},
                {"parameter_0": 7.5, "parameter_1": 0.0},
                {"parameter_0": 8.0, "parameter_1": 0.0},
                {"parameter_0": 8.5, "parameter_1": 0.0},
                {"parameter_0": 9.0, "parameter_1": 0.0},
                {"parameter_0": 9.5, "parameter_1": 0.0},
                {"parameter_0": 10.0, "parameter_1": 0.0},
            ]
        elif "dmash/DI-sparse-v0" in args.env_id:
            train_env_contexts = [
                {"action_factor_0": 1, "action_factor_1": 1, "mass": 0.5},
                {"action_factor_0": 1, "action_factor_1": -1, "mass": 0.83},
                {"action_factor_0": -1, "action_factor_1": 1, "mass": 1.16},
                {"action_factor_0": -1, "action_factor_1": -1, "mass": 1.5},
            ]
            in_env_contexts = [
                {"action_factor_0": 1, "action_factor_1": 1, "mass": 1.4},
                {"action_factor_0": 1, "action_factor_1": -1, "mass": 1.2},
                {"action_factor_0": -1, "action_factor_1": 1, "mass": 0.9},
                {"action_factor_0": -1, "action_factor_1": -1, "mass": 0.6},
            ]
            out_env_contexts = [
                {"action_factor_0": 1, "action_factor_1": 1, "mass": 0.1},
                {"action_factor_0": 1, "action_factor_1": -1, "mass": 0.4},
                {"action_factor_0": -1, "action_factor_1": 1, "mass": 1.6},
                {"action_factor_0": -1, "action_factor_1": -1, "mass": 2.0},
                {"action_factor_0": 1, "action_factor_1": 1, "mass": 2.0},
                {"action_factor_0": 1, "action_factor_1": -1, "mass": 1.6},
                {"action_factor_0": -1, "action_factor_1": 1, "mass": 0.4},
                {"action_factor_0": -1, "action_factor_1": -1, "mass": 0.1},
            ]
        else:
            raise NotImplementedError()
    else:
        train_contexts = {}
        in_contexts = {}
        out_contexts = {}
        context_defaults = CONTEXT_DEFAULTS[args.env_id]
        if args.context_id == "single0":
            context_defaults = {list(context_defaults)[0]: context_defaults[list(context_defaults)[0]]}
        elif args.context_id == "single1":
            context_defaults = {list(context_defaults)[1]: context_defaults[list(context_defaults)[1]]}
        elif args.context_id == "single2":
            context_defaults = {list(context_defaults)[2]: context_defaults[list(context_defaults)[2]]}
        for c_name, (c_default, c_in_range, c_out_range) in context_defaults.items():
            if c_name not in ["action_factor", "length_factor", "paresis", "swap_lr", "swap_rf"]:
                # high difficulty makes in_range more narrow, vice versa
                c_in_range -= args.context_in_difficulty * (np.array(c_in_range) - c_default)
                c_out_range[1] += args.context_out_difficulty * (np.array(c_out_range) - c_default)[1]
                assert c_in_range[0] > c_out_range[0] and c_in_range[1] < c_out_range[1], (
                    "context_difficulty is negative, increase it st out_range is out of in_range"
                )
                in_margin = abs(c_in_range[1] - c_in_range[0]) * 0.01
                train_contexts[c_name] = np.linspace(*c_in_range, 50).round(4).tolist()
                in_contexts[c_name] = np.linspace(
                    c_in_range[0] + in_margin, c_in_range[1] - in_margin, 50
                ).round(4).tolist()
                out_contexts[c_name] = (
                    np.linspace(
                        c_out_range[0], c_in_range[0] - in_margin, 25
                    ).round(4).tolist() +
                    np.linspace(
                        c_in_range[1] + in_margin, c_out_range[1], 25
                    ).round(4).tolist()
                )
            else:
                train_contexts[c_name] = c_in_range
                in_contexts[c_name] = c_in_range
                out_contexts[c_name] = c_out_range

        train_env_contexts = select_context_instances(
            train_contexts, args.num_train_envs, args.context_select_type
        )
        if args.context_default:
            train_env_contexts = [{k: d for k, (d, _, _) in context_defaults.items()}]
        in_env_contexts = select_context_instances(in_contexts, args.num_eval_envs, "equidistant")
        out_env_contexts = select_context_instances(out_contexts, args.num_eval_envs, "equidistant")

    num_envs = len(train_env_contexts) + len(in_env_contexts) + len(out_env_contexts)
    envs = gym.vector.SyncVectorEnv([
        make_env(
            env_id=args.env_id,
            seed=args.seed,
            idx=i,
            capture_video=args.capture_video,
            context_aware=args.context_mode in ["aware", "aware_inferred", "aware_inferred_reconstructed"],
            context_aware_noisy=args.context_aware_noisy,
            context_aware_varied=args.context_aware_varied,
            context_aware_onehot=args.context_aware_onehot,
            observation_noisy=args.observation_noisy,
            num_distractors=args.num_distractors,
            fixed_distractors=args.fixed_distractors,
            normalize_reward=args.normalize_reward,
            run_name=run_name,
            num_envs=num_envs,
            contexts=contexts
        ) for i, contexts in enumerate(train_env_contexts)
    ], autoreset_mode=gym.vector.AutoresetMode.DISABLED)
    eval_train_envs = gym.vector.SyncVectorEnv([
        make_env(
            env_id=args.env_id,
            seed=args.seed,
            idx=i,
            capture_video=False,
            context_aware=args.context_mode in ["aware", "aware_inferred", "aware_inferred_reconstructed"],
            context_aware_noisy=args.context_aware_noisy,
            context_aware_varied=args.context_aware_varied,
            context_aware_onehot=args.context_aware_onehot,
            observation_noisy=args.observation_noisy,
            num_distractors=args.num_distractors,
            fixed_distractors=args.fixed_distractors,
            normalize_reward=False,
            run_name=run_name,
            num_envs=num_envs,
            contexts=contexts
        ) for i, contexts in enumerate(train_env_contexts)
    ], autoreset_mode=gym.vector.AutoresetMode.DISABLED)
    eval_in_envs = gym.vector.SyncVectorEnv([
        make_env(
            env_id=args.env_id,
            seed=args.seed,
            idx=i + len(train_env_contexts),
            capture_video=False,
            context_aware=args.context_mode in ["aware", "aware_inferred", "aware_inferred_reconstructed"],
            context_aware_noisy=args.context_aware_noisy,
            context_aware_varied=args.context_aware_varied,
            context_aware_onehot=args.context_aware_onehot,
            observation_noisy=args.observation_noisy,
            num_distractors=args.num_distractors,
            fixed_distractors=args.fixed_distractors,
            normalize_reward=False,
            run_name=run_name,
            num_envs=num_envs,
            contexts=contexts
        ) for i, contexts in enumerate(in_env_contexts)
    ], autoreset_mode=gym.vector.AutoresetMode.DISABLED)
    eval_out_envs = gym.vector.SyncVectorEnv([
        make_env(
            env_id=args.env_id,
            seed=args.seed,
            idx=i + len(train_env_contexts) + len(in_env_contexts),
            capture_video=False,
            context_aware=args.context_mode in ["aware", "aware_inferred", "aware_inferred_reconstructed"],
            context_aware_noisy=args.context_aware_noisy,
            context_aware_varied=args.context_aware_varied,
            context_aware_onehot=args.context_aware_onehot,
            observation_noisy=args.observation_noisy,
            num_distractors=args.num_distractors,
            fixed_distractors=args.fixed_distractors,
            normalize_reward=False,
            run_name=run_name,
            num_envs=num_envs,
            contexts=contexts
        ) for i, contexts in enumerate(out_env_contexts)
    ], autoreset_mode=gym.vector.AutoresetMode.DISABLED)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    context_info = {"train": train_env_contexts, "in": in_env_contexts, "out": out_env_contexts}
    return envs, eval_train_envs, eval_in_envs, eval_out_envs, context_info
