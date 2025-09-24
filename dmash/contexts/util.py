from collections import deque

import numpy as np
import torch


from dmash.contexts.dataset import MultiContextReplayBuffer
from dmash.common import plotting


def create_dataset(envs, seed, args) -> MultiContextReplayBuffer:
    episode_length = envs.envs[0].get_wrapper_attr('_max_episode_steps')
    rb = MultiContextReplayBuffer(
        envs.single_observation_space,
        envs.single_action_space,
        args.dataset_size,
        args.context_size,
        envs.num_envs,
        args.memory_efficient_buffer,
        episode_length=episode_length,
    )
    rb.seed(seed)

    obs, _ = envs.reset()
    actions = np.array([np.zeros(envs.single_action_space.shape) for _ in range(envs.num_envs)])
    rewards = np.array([0.0 for _ in range(envs.num_envs)])
    obs_stack = deque([obs] * args.context_size, args.context_size)
    next_obs_stack = deque([obs] * args.context_size, args.context_size)
    actions_stack = deque([actions] * args.context_size, args.context_size)
    rewards_stack = deque([rewards] * args.context_size, args.context_size)
    for step in range(args.dataset_size // envs.num_envs):
        actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]

        obs_stack.append(obs)
        next_obs_stack.append(real_next_obs)
        actions_stack.append(actions)
        rewards_stack.append(rewards)

        # obs_stack shape is (context_size, num_envs, feature_size) --> loop over num_envs
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
    return rb


def compute_context(dce, rce, data, device, args, training=True):
    actions = torch.tensor(data["actions"]).to(device)
    observations = torch.tensor(data["observations"]).to(device)
    next_observations = torch.tensor(data["next_observations"]).to(device)
    rewards = torch.tensor(data["rewards"]).to(device)[:, :, None]
    dc = dce(observations, actions, next_observations - observations, training)
    rc = rce(observations, actions, rewards, training) if args.compute_reward_context else None
    if args.context_propagate and hasattr(dce, "get_uncertainty"):
        dc_uncertainty = dce.get_uncertainty(observations, actions, next_observations - observations, training)
        dc = propagate_context(dc, dc_uncertainty)
    if args.context_constant:
        dc = torch.ones_like(dc)
        if rc is not None:
            rc = torch.ones_like(rc)
    return dc, rc


def compute_context_and_uncertainty(dce, rce, data, device, args, training=True):
    actions = torch.tensor(data["actions"]).to(device)
    observations = torch.tensor(data["observations"]).to(device)
    next_observations = torch.tensor(data["next_observations"]).to(device)
    rewards = torch.tensor(data["rewards"]).to(device)[:, :, None]
    dc = dce(observations, actions, next_observations - observations, training)
    rc = rce(observations, actions, rewards, training) if args.compute_reward_context else None
    dc_uncertainty = dce.get_uncertainty(observations, actions, next_observations - observations, training)
    rc_uncertainty = rce.get_uncertainty(observations, actions, rewards, training) if args.compute_reward_context else None
    if args.context_propagate:
        dc = propagate_context(dc, dc_uncertainty)
    if args.context_constant:
        dc = torch.ones_like(dc)
        if rc is not None:
            rc = torch.ones_like(rc)
    return dc, rc, dc_uncertainty, rc_uncertainty


def compute_context_and_kl(dce, rce, data, device, args, training=True):
    actions = torch.tensor(data["actions"]).to(device)
    observations = torch.tensor(data["observations"]).to(device)
    next_observations = torch.tensor(data["next_observations"]).to(device)
    rewards = torch.tensor(data["rewards"]).to(device)[:, :, None]
    dc = dce(observations, actions, next_observations - observations, training)
    rc = rce(observations, actions, rewards, training) if args.compute_reward_context else None
    dc_kl = dce.get_kl(observations, actions, next_observations - observations, training)
    rc_kl = rce.get_kl(observations, actions, rewards, training) if args.compute_reward_context else None
    if args.context_propagate:
        dc = propagate_context(dc, dc_kl)
    if args.context_constant:
        dc = torch.ones_like(dc)
        if rc is not None:
            rc = torch.ones_like(rc)
    return dc, rc, dc_kl, rc_kl


def compute_context_and_uncertainty_and_kl(dce, rce, data, device, args, training=True):
    actions = torch.tensor(data["actions"]).to(device)
    observations = torch.tensor(data["observations"]).to(device)
    next_observations = torch.tensor(data["next_observations"]).to(device)
    rewards = torch.tensor(data["rewards"]).to(device)[:, :, None]
    dc = dce(observations, actions, next_observations - observations, training)
    rc = rce(observations, actions, rewards, training) if args.compute_reward_context else None
    dc_uncertainty = dce.get_uncertainty(observations, actions, next_observations - observations, training)
    rc_uncertainty = rce.get_uncertainty(observations, actions, rewards, training) if args.compute_reward_context else None
    dc_kl = dce.get_kl(observations, actions, next_observations - observations, training)
    rc_kl = rce.get_kl(observations, actions, rewards, training) if args.compute_reward_context else None
    if args.context_propagate:
        dc = propagate_context(dc, dc_kl)
    if args.context_constant:
        dc = torch.ones_like(dc)
        if rc is not None:
            rc = torch.ones_like(rc)
    return dc, rc, dc_uncertainty, rc_uncertainty, dc_kl, rc_kl


def compute_losses(
        fm, im, bm, rm, sdm, cdm, rc_rm, dce, rce, data, device, args, training=True
):
    actions = torch.tensor(data["actions"]).to(device)[:, -1]
    observations = torch.tensor(data["observations"]).to(device)[:, -1]
    next_observations = torch.tensor(data["next_observations"]).to(device)[:, -1]
    rewards = torch.tensor(data["rewards"]).to(device)[:, -1]

    if hasattr(dce, "get_uncertainty") and hasattr(dce, "get_kl"):
        dynamics_context, reward_context, dce_uncertainty, rce_uncertainty, dce_kl, rce_kl = compute_context_and_uncertainty_and_kl(
            dce, rce, data, device, args, training=training
        )
        rce_uncertainty = rce_uncertainty if args.compute_reward_context else torch.tensor([[0.0]])
        rce_kl = rce_kl if args.compute_reward_context else torch.tensor([[0.0]])
    elif hasattr(dce, "get_uncertainty") and not hasattr(dce, "get_kl"):
        dynamics_context, reward_context, dce_uncertainty, rce_uncertainty = compute_context_and_uncertainty(
            dce, rce, data, device, args, training=training
        )
        rce_uncertainty = rce_uncertainty if args.compute_reward_context else torch.tensor([[0.0]])
        dce_kl, rce_kl = torch.tensor([[0.0]]), torch.tensor([[0.0]])
    elif hasattr(dce, "get_kl") and not hasattr(dce, "get_uncertainty"):
        dynamics_context, reward_context, dce_kl, rce_kl = compute_context_and_kl(
            dce, rce, data, device, args, training=training
        )
        rce_kl = rce_kl if args.compute_reward_context else torch.tensor([[0.0]])
        dce_uncertainty, rce_uncertainty = torch.tensor([[0.0]]), torch.tensor([[0.0]])
    else:
        dynamics_context, reward_context = compute_context(
            dce, rce, data, device, args, training=training
        )
        dce_uncertainty, rce_uncertainty = torch.tensor([[0.0]]), torch.tensor([[0.0]])
        dce_kl, rce_kl = torch.tensor([[0.0]]), torch.tensor([[0.0]])

    if hasattr(fm, "get_uncertainty"):
        fm_uncertainty = fm.get_uncertainty(observations, actions, dynamics_context)
        fm_uncertainty = fm_uncertainty
    else:
        fm_uncertainty = torch.tensor([[0.0]])

    fm_l2_loss, fm_hnet_weights = fm.compute_loss(
        observations, actions, next_observations, dynamics_context, return_hnet_weights=True
    ) if fm is not None else (torch.zeros_like(observations), None)
    im_l2_loss = im.compute_loss(
        observations, actions, next_observations, dynamics_context
    ) if im is not None else torch.tensor([[0.0]])
    bm_l2_loss = bm.compute_loss(
        next_observations, actions, observations, dynamics_context
    ) if bm is not None else torch.zeros_like(observations)
    rm_l2_loss = rm.compute_loss(
        observations, actions, rewards, dynamics_context
    ) if rm is not None else torch.tensor([[0.0]])
    sdm_l2_loss = sdm.compute_loss(
        observations, actions, next_observations, dynamics_context
    ) if sdm is not None else torch.tensor([[0.0]])
    cdm_l2_loss = cdm.compute_loss(
        observations, dynamics_context
    ) if cdm is not None else torch.tensor([[0.0]])
    rc_rm_l2_loss = rc_rm.compute_loss(
        observations, actions, rewards, reward_context
    ) if rc_rm is not None else torch.tensor([[0.0]])

    if fm_hnet_weights is not None:
        fm_hnet_l1_reg = torch.mean(torch.norm(fm_hnet_weights, dim=1, p=1))
        fm_hnet_l2_reg = torch.mean(torch.norm(fm_hnet_weights, dim=1, p=2))
        fm_hnet_mean_reg = torch.mean(torch.abs(fm_hnet_weights))
    else:
        fm_hnet_l1_reg = torch.tensor([[0.0]])
        fm_hnet_l2_reg = torch.tensor([[0.0]])
        fm_hnet_mean_reg = torch.tensor([[0.0]])

    return (
        {
            "fm_l2_loss": fm_l2_loss.mean(),
            "im_l2_loss": im_l2_loss,
            "bm_l2_loss": bm_l2_loss.mean(),
            "rm_l2_loss": rm_l2_loss,
            "sdm_l2_loss": sdm_l2_loss,
            "cdm_l2_loss": cdm_l2_loss,
            "rc_rm_l2_loss": rc_rm_l2_loss,
            "fm_uncertainty": fm_uncertainty.mean(),
            "dce_uncertainty": dce_uncertainty.mean(),
            "dce_kl": dce_kl.mean(),
            "fm_hnet_l1_reg": fm_hnet_l1_reg,
            "fm_hnet_l2_reg": fm_hnet_l2_reg,
            "fm_hnet_mean_reg": fm_hnet_mean_reg,
        },
        {
            "dce_uncertainty_batch": dce_uncertainty.mean(-1),
            "dce_kl_batch": dce_kl.mean(-1),
            "dynamics_context": dynamics_context,
            "reward_context": reward_context,
            "fm_l2_loss_batch": fm_l2_loss.mean(-1),
        }
    )


def anneal_beta(beta, step, max_steps, annealing_speed):
    beta_annealed = -beta + step / max_steps * 2 * beta * annealing_speed
    beta_annealed = min(max(beta_annealed, -abs(beta)), abs(beta))
    return beta_annealed


def compute_expressiveness(ds, dce, rce, actor, qf, fm, im, device, args, context_info, training=True):
    track_dynamics_context = []
    track_reward_context = []
    track_true_context = []
    track_qf_out = []
    track_qf_main = []
    track_qf_mnet = []
    track_qf_hnet = []
    track_actor_out = []
    track_actor_main = []
    track_actor_mnet = []
    track_actor_hnet = []
    track_fm_out = []
    track_fm_main = []
    track_fm_mnet = []
    track_fm_hnet = []
    track_im_out = []
    track_im_main = []
    track_im_mnet = []
    track_im_hnet = []

    with torch.no_grad():
        for i in range(ds.num_datasets):
            data = ds.sample(args.batch_size, dataset_index=i)
            dynamics_context, reward_context = compute_context(dce, rce, data, device, args, training)
            track_dynamics_context.append(dynamics_context.detach().cpu().numpy())
            track_reward_context.append(reward_context.detach().cpu().numpy() if args.compute_reward_context else None)
            track_true_context.append(np.array([list(context_info[i].values())] * args.batch_size, dtype=np.float32))

            # run models
            actions = torch.tensor(data["actions"]).to(device)[:, -1]
            observations = torch.tensor(data["observations"]).to(device)[:, -1]
            next_observations = torch.tensor(data["next_observations"]).to(device)[:, -1]

            if fm is not None:
                fm_out = fm(observations, actions, dynamics_context)
            if im is not None:
                im_out = im(observations, next_observations, dynamics_context)
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
            actor_out, _, _ = actor.get_action(observations, dynamics_context, hnet_weights)
            qf_out = qf(observations, actions, dynamics_context, hnet_weights)

            # track activities
            track_qf_out.append(qf_out.detach().cpu().numpy())
            track_qf_main.append(qf.activation_dict["main"].detach().cpu().numpy())
            track_qf_mnet.append(qf.activation_dict["mnet"].detach().cpu().numpy() if "mnet" in qf.activation_dict.keys() else None)
            track_qf_hnet.append(qf.activation_dict["hnet"].detach().cpu().numpy() if "hnet" in qf.activation_dict.keys() else None)
            track_actor_out.append(actor_out.detach().cpu().numpy())
            track_actor_main.append(actor.activation_dict["main"].detach().cpu().numpy())
            track_actor_mnet.append(actor.activation_dict["mnet"].detach().cpu().numpy() if "mnet" in actor.activation_dict.keys() else None)
            track_actor_hnet.append(actor.activation_dict["hnet"].detach().cpu().numpy() if "hnet" in actor.activation_dict.keys() else None)
            if fm is not None:
                track_fm_out.append(fm_out.detach().cpu().numpy())
                track_fm_main.append(fm.activation_dict["main"].detach().cpu().numpy())
                track_fm_mnet.append(fm.activation_dict["mnet"].detach().cpu().numpy() if "mnet" in fm.activation_dict.keys() else None)
                track_fm_hnet.append(fm.activation_dict["hnet"].detach().cpu().numpy() if "hnet" in fm.activation_dict.keys() else None)
            if im is not None:
                track_im_out.append(im_out.detach().cpu().numpy())
                track_im_main.append(im.activation_dict["main"].detach().cpu().numpy())
                track_im_mnet.append(im.activation_dict["mnet"].detach().cpu().numpy() if "mnet" in im.activation_dict.keys() else None)
                track_im_hnet.append(im.activation_dict["hnet"].detach().cpu().numpy() if "hnet" in im.activation_dict.keys() else None)

    track_dynamics_context = np.concatenate(track_dynamics_context, axis=0)
    track_reward_context = np.concatenate(track_reward_context, axis=0) if args.compute_reward_context else None
    track_true_context = np.concatenate(track_true_context, axis=0)

    track_qf_out = np.concatenate(track_qf_out, axis=0)
    track_qf_main = np.concatenate(track_qf_main, axis=0)
    track_qf_mnet = np.concatenate(track_qf_mnet, axis=0) if not track_qf_mnet[0] is None else None
    track_qf_hnet = np.concatenate(track_qf_hnet, axis=0) if not track_qf_hnet[0] is None else None
    track_actor_out = np.concatenate(track_actor_out, axis=0)
    track_actor_main = np.concatenate(track_actor_main, axis=0)
    track_actor_mnet = np.concatenate(track_actor_mnet, axis=0) if not track_actor_mnet[0] is None else None
    track_actor_hnet = np.concatenate(track_actor_hnet, axis=0) if not track_actor_hnet[0] is None else None

    if fm is not None:
        track_fm_out = np.concatenate(track_fm_out, axis=0)
        track_fm_main = np.concatenate(track_fm_main, axis=0)
        track_fm_mnet = np.concatenate(track_fm_mnet, axis=0) if not track_fm_mnet[0] is None else None
        track_fm_hnet = np.concatenate(track_fm_hnet, axis=0) if not track_fm_hnet[0] is None else None
    if im is not None:
        track_im_out = np.concatenate(track_im_out, axis=0)
        track_im_main = np.concatenate(track_im_main, axis=0)
        track_im_mnet = np.concatenate(track_im_mnet, axis=0) if not track_im_mnet[0] is None else None
        track_im_hnet = np.concatenate(track_im_hnet, axis=0) if not track_im_hnet[0] is None else None

    # fit a linear regression model and compute r2 score for the trainig set as expressiveness measure
    from sklearn.linear_model import LinearRegression
    from sklearn.decomposition import PCA

    y_train = track_true_context
    r2_expressiveness = {}
    for x_train, x_name in zip(
        [
            track_dynamics_context, track_qf_out, track_qf_main, track_qf_mnet, track_qf_hnet,
            track_actor_out, track_actor_main, track_actor_mnet, track_actor_hnet,
            track_fm_out, track_fm_main, track_fm_mnet, track_fm_hnet,
            track_im_out, track_im_main, track_im_mnet, track_im_hnet,
        ],
        [
            "context_representation", "qf_out", "qf_main", "qf_mnet", "qf_hnet",
            "actor_out", "actor_main", "actor_mnet", "actor_hnet",
            "fm_out", "fm_main", "fm_mnet", "fm_hnet",
            "im_out", "im_main", "im_mnet", "im_hnet",
        ]
    ):
        if x_train is None or len(x_train) == 0:
            continue
        if "hnet" in x_name:
            pca = PCA(args.helper_hidden_dim, random_state=args.seed).fit(x_train)
            x_train = pca.transform(x_train)
        reg = LinearRegression().fit(x_train, y_train)
        r2_expressiveness[x_name] = reg.score(x_train, y_train)
    return r2_expressiveness


def compute_context_variability(ds, dce, rce, actor, qf, fm, im, device, args, context_info, training=True):
    track_dynamics_context = []
    track_qf_main = []
    track_qf_mnet = []
    track_qf_hnet = []
    track_actor_main = []
    track_actor_mnet = []
    track_actor_hnet = []
    track_fm_main = []
    track_fm_mnet = []
    track_fm_hnet = []
    track_im_main = []
    track_im_mnet = []
    track_im_hnet = []

    with torch.no_grad():
        for i in range(ds.num_datasets):
            data = ds.sample(args.batch_size, dataset_index=i)
            dynamics_context, reward_context = compute_context(dce, rce, data, device, args, training)
            track_dynamics_context.append(dynamics_context.detach().cpu().numpy())

            # run models
            observations = torch.tensor(data["observations"]).to(device)[:, -1]

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
            actor_out, _, _ = actor.get_action(observations, dynamics_context, hnet_weights)

            # track activities
            track_qf_main.append(qf.activation_dict["main"].detach().cpu().numpy())
            track_qf_mnet.append(qf.activation_dict["mnet"].detach().cpu().numpy() if "mnet" in qf.activation_dict.keys() else None)
            track_qf_hnet.append(qf.activation_dict["hnet"].detach().cpu().numpy() if "hnet" in qf.activation_dict.keys() else None)
            track_actor_main.append(actor.activation_dict["main"].detach().cpu().numpy())
            track_actor_mnet.append(actor.activation_dict["mnet"].detach().cpu().numpy() if "mnet" in actor.activation_dict.keys() else None)
            track_actor_hnet.append(actor.activation_dict["hnet"].detach().cpu().numpy() if "hnet" in actor.activation_dict.keys() else None)
            if fm is not None:
                track_fm_main.append(fm.activation_dict["main"].detach().cpu().numpy())
                track_fm_mnet.append(fm.activation_dict["mnet"].detach().cpu().numpy() if "mnet" in fm.activation_dict.keys() else None)
                track_fm_hnet.append(fm.activation_dict["hnet"].detach().cpu().numpy() if "hnet" in fm.activation_dict.keys() else None)
            if im is not None:
                track_im_main.append(im.activation_dict["main"].detach().cpu().numpy())
                track_im_mnet.append(im.activation_dict["mnet"].detach().cpu().numpy() if "mnet" in im.activation_dict.keys() else None)
                track_im_hnet.append(im.activation_dict["hnet"].detach().cpu().numpy() if "hnet" in im.activation_dict.keys() else None)

    track_dynamics_context = np.concatenate(track_dynamics_context, axis=0)

    track_qf_main = np.concatenate(track_qf_main, axis=0)
    track_qf_mnet = np.concatenate(track_qf_mnet, axis=0) if not track_qf_mnet[0] is None else None
    track_qf_hnet = np.concatenate(track_qf_hnet, axis=0) if not track_qf_hnet[0] is None else None
    track_actor_main = np.concatenate(track_actor_main, axis=0)
    track_actor_mnet = np.concatenate(track_actor_mnet, axis=0) if not track_actor_mnet[0] is None else None
    track_actor_hnet = np.concatenate(track_actor_hnet, axis=0) if not track_actor_hnet[0] is None else None

    if fm is not None:
        track_fm_main = np.concatenate(track_fm_main, axis=0)
        track_fm_mnet = np.concatenate(track_fm_mnet, axis=0) if not track_fm_mnet[0] is None else None
        track_fm_hnet = np.concatenate(track_fm_hnet, axis=0) if not track_fm_hnet[0] is None else None
    if im is not None:
        track_im_main = np.concatenate(track_im_main, axis=0)
        track_im_mnet = np.concatenate(track_im_mnet, axis=0) if not track_im_mnet[0] is None else None
        track_im_hnet = np.concatenate(track_im_hnet, axis=0) if not track_im_hnet[0] is None else None

    from sklearn.decomposition import PCA

    context_variability = {}
    for x_train, x_name in zip(
        [
            track_dynamics_context, track_qf_main, track_qf_mnet, track_qf_hnet,
            track_actor_main, track_actor_mnet, track_actor_hnet,
            track_fm_main, track_fm_mnet, track_fm_hnet,
            track_im_main, track_im_mnet, track_im_hnet,
        ],
        [
            "context_representation", "qf_main", "qf_mnet", "qf_hnet",
            "actor_main", "actor_mnet", "actor_hnet",
            "fm_main", "fm_mnet", "fm_hnet",
            "im_main", "im_mnet", "im_hnet",
        ]
    ):
        if x_train is None or len(x_train) == 0:
            continue
        if "hnet" in x_name:
            pca = PCA(args.helper_hidden_dim, random_state=args.seed).fit(x_train)
            x_train = pca.transform(x_train)
        context_variability[x_name] = x_train.var(axis=0).mean().item()
    return context_variability


def propagate_context(context, uncertainty):
    from sklearn.neighbors import kneighbors_graph
    knn_context = context.clone().detach()
    connectivities = kneighbors_graph(knn_context.detach().numpy(), 5, include_self=False)
    connectivities = torch.tensor(connectivities.todense(), dtype=torch.float32)

    for i in np.arange(20):
        certainty_threshold_pct = 0.50 * np.exp(-0.1 * i)
        certain = -uncertainty > torch.quantile(-uncertainty, q=certainty_threshold_pct)
        knn_context[(certain is False).squeeze()] = 0.000001
        knn_context = torch.matmul(connectivities, knn_context)
    return knn_context


def get_tsne(context, label):
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=0)
    context_low = tsne.fit_transform(context)

    fig = plotting.representation_fig(context_low, label, "TSNE")
    return fig


def prepare_contexts(ds, dce, rce, device, args, training=True, context_info=None):
    dynamics_contexts = []
    reward_contexts = []
    labels = []
    true_contexts = []

    for i in range(ds.num_datasets):
        data = ds.sample(args.batch_size, dataset_index=i)
        dynamics_context, reward_context = compute_context(dce, rce, data, device, args, training)
        dynamics_contexts.append(dynamics_context.detach().cpu().numpy())
        reward_contexts.append(reward_context.detach().cpu().numpy() if args.compute_reward_context else None)
        labels.append(np.ones((dynamics_contexts[0].shape[0], 1), dtype=int) * i)
        if context_info is not None:
            true_contexts.append(np.array([list(context_info[i].values())] * args.batch_size, dtype=np.float32))

    dynamics_context = np.concatenate(dynamics_contexts, axis=0)
    reward_context = np.concatenate(reward_contexts, axis=0) if args.compute_reward_context else None
    label = np.concatenate(labels, axis=0)
    if context_info is not None:
        true_contexts = np.concatenate(true_contexts, axis=0)
    return dynamics_context, reward_context, label, true_contexts
