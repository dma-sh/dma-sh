import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hypnettorch import mnets, hnets


class BaseModel(nn.Module):
    def __init__(self, env, args):
        super().__init__()

        # obs contains real context if context mode in aware or aware_inferred
        obs_dim = np.prod(env.single_observation_space.shape)
        action_dim = np.prod(env.single_action_space.shape)
        if args.context_mode == "unaware":
            real_context_dim = 0
            inferred_context_dim = 0
            self.context_dim = real_context_dim + inferred_context_dim
        elif args.context_mode == "inferred":
            real_context_dim = 0
            inferred_context_dim = args.context_dim
            self.context_dim = real_context_dim + inferred_context_dim
        elif args.context_mode == "aware":
            real_context_dim = env.envs[0].get_wrapper_attr("context").size
            obs_dim -= real_context_dim
            inferred_context_dim = 0
            self.context_dim = real_context_dim + inferred_context_dim
        elif args.context_mode == "aware_inferred":
            real_context_dim = env.envs[0].get_wrapper_attr("context").size
            obs_dim -= real_context_dim
            inferred_context_dim = args.context_dim
            self.context_dim = real_context_dim + inferred_context_dim
        elif args.context_mode == "aware_inferred_reconstructed":
            real_context_dim = env.envs[0].get_wrapper_attr("context").size
            obs_dim -= real_context_dim
            inferred_context_dim = args.context_dim
            self.context_dim = inferred_context_dim
        else:
            raise NotImplementedError()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.real_context_dim = real_context_dim
        self.inferred_context_dim = inferred_context_dim

        self.model_embedding_dim = args.model_embedding_dim
        self.model_embedding_num_hidden_layer = args.model_embedding_num_hidden_layer

        self.use_additional_layer = args.use_additional_layer

        self.context_mode = args.context_mode
        self.hypernet_skip_connection = args.hypernet_skip_connection
        self.hypernet_chunked = args.hypernet_chunked
        self.hypernet_pre_activation = args.hypernet_pre_activation
        self.hypernet_post_activation = args.hypernet_post_activation
        self.hypernet_hidden_dim = args.hypernet_hidden_dim
        self.hypernet_num_hidden_layer = args.hypernet_num_hidden_layer
        self.hypernet_bottleneck = args.hypernet_bottleneck
        self.hypernet_bias = args.hypernet_bias
        self.hypernet_hnet_dropout = args.hypernet_hnet_dropout
        self.hypernet_mnet_dropout = args.hypernet_mnet_dropout
        self.hypernet_hyperfan_init = args.hypernet_hyperfan_init
        self.hypernet_no_update = args.hypernet_no_update
        self.embedding_post_activation = args.model_embedding_post_activation

        self.context_merge_type = None  # initialized later, as it depends on specific model config

        self.activation_dict = {}

    def forward(self):
        raise NotImplementedError()

    def _forward(self, x, context, hnet_weights=None):
        if self.context_mode in ["inferred", "aware", "aware_inferred", "aware_inferred_reconstructed"]:
            if self.context_merge_type == "concat":
                x = torch.cat([x, context], 1)
                x = self.fcs(x)
                self.activation_dict["main"] = x.detach()
            elif self.context_merge_type in ["hypernet", "hypernet_shared"]:
                x = self.fcs(x)
                self.activation_dict["main"] = x.detach()
                B, D = x.shape
                if self.context_merge_type == "hypernet_shared":
                    assert hnet_weights is not None
                else:
                    hnet_weights = self.fc_hnet(cond_input=context)
                    # hnet_weights are not detached in activation_dict,
                    # as we might need them for regularization, detached later
                    self.activation_dict["hnet"] = torch.cat([torch.cat(  # flatten hyperweights
                        [
                            p.detach().flatten() for p in hnet_weight
                        ])[None] for hnet_weight in hnet_weights], axis=0)
                x_ = torch.cat(
                    [self.fc_mnet(x[i], weights=hnet_weights[i])[None] for i in range(B)],
                    dim=0
                )
                if self.hypernet_skip_connection:
                    x = x + x_
                else:
                    x = x_
                if self.hypernet_post_activation:
                    x = F.relu(x)
                self.activation_dict["mnet"] = x.detach()
            elif self.context_merge_type == "mul":
                x = x * context
                x = self.fcs(x)
                self.activation_dict["main"] = x.detach()
            elif self.context_merge_type == "none":
                x = self.fcs(x)
                self.activation_dict["main"] = x.detach()
            else:
                raise NotImplementedError()
        elif self.context_mode == "unaware":
            x = self.fcs(x)
            self.activation_dict["main"] = x.detach()
        else:
            raise NotImplementedError()

        if self.fcs_post is not None:
            x = self.fcs_post(x)

        if hnet_weights is not None:
            hnet_weights = torch.cat([torch.cat(  # flatten hyperweights
                [
                    p.flatten() for p in hnet_weight
                ])[None] for hnet_weight in hnet_weights], axis=0)
        return x, hnet_weights

    def embed(self, input, context):
        if self.context_stop_gradient:
            context = context.detach()

        if self.context_mode == "unaware":
            x = torch.cat(list(input.values()), 1)
            context = None
            if self.input_embedding is not None:
                x = self.input_embedding(x)
        elif self.context_mode == "inferred":
            x = torch.cat(list(input.values()), 1)
            if self.input_embedding is not None:
                x = self.input_embedding(x)
            if self.context_embedding is not None:
                context = self.context_embedding(context)
        elif self.context_mode == "aware":
            context = input["obs"][:, -self.real_context_dim:]
            input["obs"] = input["obs"][:, :-self.real_context_dim]
            if "next_obs" in input.keys():
                input["next_obs"] = input["next_obs"][:, :-self.real_context_dim]
            x = torch.cat(list(input.values()), 1)
            if self.input_embedding is not None:
                x = self.input_embedding(x)
            if self.context_embedding is not None:
                context = self.context_embedding(context)
        elif self.context_mode == "aware_inferred":
            context = torch.cat([input["obs"][:, -self.real_context_dim:], context], 1)
            input["obs"] = input["obs"][:, :-self.real_context_dim]
            if "next_obs" in input.keys():
                input["next_obs"] = input["next_obs"][:, :-self.real_context_dim]
            x = torch.cat(list(input.values()), 1)
            if self.input_embedding is not None:
                x = self.input_embedding(x)
            if self.context_embedding is not None:
                context = self.context_embedding(context)
        elif self.context_mode == "aware_inferred_reconstructed":
            input["obs"] = input["obs"][:, :-self.real_context_dim]
            if "next_obs" in input.keys():
                input["next_obs"] = input["next_obs"][:, :-self.real_context_dim]
            x = torch.cat(list(input.values()), 1)
            if self.input_embedding is not None:
                x = self.input_embedding(x)
            if self.context_embedding is not None:
                context = self.context_embedding(context)
        else:
            raise NotImplementedError()

        return x, context

    def get_hnet_weights(self, context, obs):
        if getattr(self, "fc_hnet", None) is not None:
            x, context = self.embed({"obs": obs}, context)
            hnet_weights = self.fc_hnet(cond_input=context)
        else:
            hnet_weights = None
        return hnet_weights

    def _init_corpus(
        self,
        input_dim,
        use_embedding,
        context_merge_type,
        model_hidden_dim,
        model_num_hidden_layer,
    ):
        # set merge type as it is specific to the model
        self.context_merge_type = context_merge_type

        # embedding layers
        if use_embedding:
            self.input_embedding = nn.Sequential(
                nn.Linear(input_dim, self.model_embedding_dim),
                nn.ReLU()
            )
            for _ in range(self.model_embedding_num_hidden_layer - 1):
                self.input_embedding.append(
                    nn.Linear(self.model_embedding_dim, self.model_embedding_dim)
                )
                self.input_embedding.append(nn.ReLU())
            if not self.embedding_post_activation:
                self.input_embedding = self.input_embedding[:-1]
            if self.context_mode == "unaware":
                self.context_embedding = None
                embedding_dim = self.model_embedding_dim
            else:
                self.context_embedding = nn.Sequential(
                    nn.Linear(self.context_dim, self.model_embedding_dim),
                    nn.ReLU()
                )
                for _ in range(self.model_embedding_num_hidden_layer - 1):
                    self.context_embedding.append(
                        nn.Linear(self.model_embedding_dim, self.model_embedding_dim)
                    )
                    self.context_embedding.append(nn.ReLU())
                if not self.embedding_post_activation:
                    self.context_embedding = self.context_embedding[:-1]
                self.context_dim = self.model_embedding_dim
                if self.context_merge_type == "concat":
                    embedding_dim = 2 * self.model_embedding_dim
                elif self.context_merge_type == "mul":
                    embedding_dim = self.model_embedding_dim
                elif self.context_merge_type == "hypernet":
                    embedding_dim = self.model_embedding_dim
                elif self.context_merge_type == "none":
                    embedding_dim = self.model_embedding_dim
                else:
                    raise NotImplementedError()
        else:
            self.input_embedding = None
            self.context_embedding = None
            if self.context_merge_type == "concat":
                embedding_dim = input_dim + self.context_dim
            elif self.context_merge_type == "mul":
                assert use_embedding, "mul can only be combined with the use of embeddings!"
            elif self.context_merge_type in ["hypernet", "hypernet_shared"]:
                embedding_dim = input_dim
            elif self.context_merge_type == "none":
                embedding_dim = input_dim
            else:
                raise NotImplementedError()

        # main layers
        self.fcs = nn.Sequential(nn.Linear(embedding_dim, model_hidden_dim), nn.ReLU())
        for _ in range(model_num_hidden_layer - 1):
            self.fcs.append(nn.Linear(model_hidden_dim, model_hidden_dim))
            self.fcs.append(nn.ReLU())

        # hypernet layers
        if self.context_merge_type in ["hypernet", "hypernet_shared"]:
            if not self.hypernet_pre_activation:
                self.fcs = self.fcs[:-1]

            self.fc_mnet = mnets.MLP(
                n_in=model_hidden_dim,
                n_out=model_hidden_dim,
                hidden_layers=(model_hidden_dim // 8 if self.hypernet_bottleneck else model_hidden_dim,),
                no_weights=True,
                use_bias=self.hypernet_bias,
                dropout_rate=self.hypernet_mnet_dropout if self.hypernet_mnet_dropout != 0.0 else -1,
                verbose=False,
            )
            self.fc_hnet = None  # shared: it is assumed hnet weights are coming from somewhere else
            if self.context_merge_type == "hypernet":
                if self.hypernet_chunked:
                    self.fc_hnet = hnets.ChunkedHMLP(
                        self.fc_mnet.param_shapes,
                        chunk_size=512,
                        cond_in_size=self.context_dim,
                        layers=[self.hypernet_hidden_dim for _ in range(self.hypernet_num_hidden_layer)],
                        use_bias=self.hypernet_bias,
                        dropout_rate=self.hypernet_hnet_dropout if self.hypernet_hnet_dropout != 0.0 else -1,
                        verbose=False
                    )
                    if self.hypernet_hyperfan_init:
                        self.fc_hnet.apply_chunked_hyperfan_init(mnet=self.fc_mnet)
                else:
                    self.fc_hnet = hnets.HMLP(
                        self.fc_mnet.param_shapes,
                        cond_in_size=self.context_dim,
                        layers=[self.hypernet_hidden_dim for _ in range(self.hypernet_num_hidden_layer)],
                        use_bias=self.hypernet_bias,
                        dropout_rate=self.hypernet_hnet_dropout if self.hypernet_hnet_dropout != 0.0 else -1,
                        verbose=False
                    )
                    if self.hypernet_hyperfan_init:
                        self.fc_hnet.apply_hyperfan_init(mnet=self.fc_mnet)
                if self.hypernet_no_update:
                    for p in self.fc_hnet.parameters():
                        p.requires_grad_(False)

        else:
            self.fc_mnet = None
            self.fc_hnet = None

        if self.use_additional_layer:
            self.fcs_post = nn.Sequential(nn.Linear(model_hidden_dim, model_hidden_dim), nn.ReLU())
        else:
            self.fcs_post = None


class SoftQNetwork(BaseModel):
    def __init__(self, env, args):
        super().__init__(env, args)

        self._init_corpus(
            input_dim=self.obs_dim + self.action_dim,
            use_embedding=args.q_use_embedding,
            context_merge_type=args.q_context_merge_type,
            model_hidden_dim=args.q_hidden_dim,
            model_num_hidden_layer=args.q_num_hidden_layer,
        )

        self.fc_head = nn.Linear(args.q_hidden_dim, 1)

        self.context_stop_gradient = "critic" not in args.context_encoder_update

    def forward(self, obs, action, context, hnet_weights=None):
        x, context = self.embed({"obs": obs, "action": action}, context)
        x, _ = self._forward(x, context, hnet_weights)
        x = self.fc_head(x)
        return x


class Actor(BaseModel):
    def __init__(self, env, args):
        super().__init__(env, args)

        self._init_corpus(
            input_dim=self.obs_dim,
            use_embedding=args.policy_use_embedding,
            context_merge_type=args.policy_context_merge_type,
            model_hidden_dim=args.policy_hidden_dim,
            model_num_hidden_layer=args.policy_num_hidden_layer,
        )

        self.fc_mean = nn.Linear(args.policy_hidden_dim, self.action_dim)
        self.fc_logstd = nn.Linear(args.policy_hidden_dim, self.action_dim)

        self.context_stop_gradient = "actor" not in args.context_encoder_update

        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32
            )
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32
            )
        )

        self.log_std_max = 2
        self.log_std_min = -5

    def forward(self, obs, context, hnet_weights=None):
        x, context = self.embed({"obs": obs}, context)
        x, _ = self._forward(x, context, hnet_weights)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        return mean, log_std

    def get_action(self, obs, context, hnet_weights=None, training=True):
        mean, log_std = self(obs, context, hnet_weights)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample() if training else mean  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class ForwardModel(BaseModel):
    def __init__(self, env, args):
        super().__init__(env, args)

        self._init_corpus(
            input_dim=self.obs_dim + self.action_dim,
            use_embedding=args.helper_use_embedding,
            context_merge_type=args.helper_context_merge_type,
            model_hidden_dim=args.helper_hidden_dim,
            model_num_hidden_layer=args.helper_num_hidden_layer,
        )

        self.fc_head = nn.Linear(args.helper_hidden_dim, self.obs_dim)

        self.context_stop_gradient = "fm" not in args.context_encoder_update
        self.loss_type = args.fm_loss_type

    def forward(self, obs, action, context, return_hnet_weights=False):
        x, context = self.embed({"obs": obs, "action": action}, context)
        x, hnet_weights = self._forward(x, context)
        x = self.fc_head(x)
        if return_hnet_weights:
            return x, hnet_weights
        else:
            return x

    def compute_loss(self, obs, action, next_obs, context, return_hnet_weights=False):
        next_observations_diff_pred, hnet_weights = self(
            obs, action, context, return_hnet_weights=True
        )
        if self.context_mode in ["aware", "aware_inferred", "aware_inferred_reconstructed"]:
            obs = obs[:, :-self.real_context_dim]
            next_obs = next_obs[:, :-self.real_context_dim]
        if self.loss_type == 0:
            loss = F.mse_loss(next_observations_diff_pred, next_obs - obs, reduction="none")
        elif self.loss_type == 1:
            def symlog(x):
                return torch.sign(x) * torch.log(torch.abs(x) + 1.0)
            loss = F.mse_loss(symlog(next_observations_diff_pred), symlog(next_obs - obs), reduction="none")
        elif self.loss_type == 2:
            loss = torch.norm(next_observations_diff_pred - (next_obs - obs), p=2, dim=-1, keepdim=True)
        elif self.loss_type == 3:
            loss = F.l1_loss(next_observations_diff_pred, next_obs - obs, reduction="none")
        elif self.loss_type == 4:
            loss = F.huber_loss(next_observations_diff_pred, next_obs - obs, reduction="none")
        else:
            raise NotImplementedError()
        if return_hnet_weights:
            return loss, hnet_weights
        else:
            return loss


class ForwardModelEnsemble(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self._models = nn.ModuleList(
            [ForwardModel(env, args) for _ in range(args.context_encoder_num)]
        )

    def forward(self, obs, action, context):
        x = torch.stack([model(obs, action, context) for model in self._models])
        return torch.mean(x, dim=0)

    def get_uncertainty(self, obs, action, context):
        x = torch.stack([model(obs, action, context) for model in self._models])
        variance = torch.var(x, dim=0, correction=0)
        return variance.mean(dim=-1, keepdim=True)

    def compute_loss(self, obs, action, next_obs, context):
        next_observations_diff_pred = self(obs, action, context)
        if self.context_mode in ["aware", "aware_inferred", "aware_inferred_reconstructed"]:
            obs = obs[:, :-self.real_context_dim]
            next_obs = next_obs[:, :-self.real_context_dim]
        if self.loss_type == 0:
            loss = F.mse_loss(next_observations_diff_pred, next_obs - obs, reduction="none")
        elif self.loss_type == 1:
            def symlog(x):
                return torch.sign(x) * torch.log(torch.abs(x) + 1.0)
            loss = F.mse_loss(symlog(next_observations_diff_pred), symlog(next_obs - obs), reduction="none")
        elif self.loss_type == 2:
            loss = torch.norm(next_observations_diff_pred - (next_obs - obs), p=2, dim=-1, keepdim=True)
        elif self.loss_type == 3:
            loss = F.l1_loss(next_observations_diff_pred, next_obs - obs, reduction="none")
        elif self.loss_type == 4:
            loss = F.huber_loss(next_observations_diff_pred, next_obs - obs, reduction="none")
        else:
            raise NotImplementedError()
        return loss


class InverseModel(BaseModel):
    def __init__(self, env, args):
        super().__init__(env, args)

        self._init_corpus(
            input_dim=self.obs_dim + self.obs_dim,
            use_embedding=args.helper_use_embedding,
            context_merge_type=args.helper_context_merge_type,
            model_hidden_dim=args.helper_hidden_dim,
            model_num_hidden_layer=args.helper_num_hidden_layer,
        )

        self.fc_head = nn.Linear(args.helper_hidden_dim, self.action_dim)

        self.context_stop_gradient = "im" not in args.context_encoder_update

        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32
            )
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32
            )
        )

    def forward(self, obs, next_obs, context):
        x, context = self.embed({"obs": obs, "next_obs": next_obs}, context)
        x = self._forward(x, context)
        x = self.fc_head(x)
        x = torch.tanh(x) * self.action_scale + self.action_bias
        return x

    def compute_loss(self, obs, action, next_obs, context):
        action_pred = self(obs, next_obs - obs, context)
        loss = F.mse_loss(action_pred, action)
        return loss


class RewardModel(BaseModel):
    def __init__(self, env, args):
        super().__init__(env, args)

        self._init_corpus(
            input_dim=self.obs_dim + self.action_dim,
            use_embedding=args.helper_use_embedding,
            context_merge_type=args.helper_context_merge_type,
            model_hidden_dim=args.helper_hidden_dim,
            model_num_hidden_layer=args.helper_num_hidden_layer,
        )

        self.fc_head = nn.Linear(args.helper_hidden_dim, 1)

        self.context_stop_gradient = "rm" not in args.context_encoder_update

    def forward(self, obs, action, context):
        x, context = self.embed({"obs": obs, "action": action}, context)
        x = self._forward(x, context)
        x = self.fc_head(x)
        return x.squeeze(1)

    def compute_loss(self, obs, action, reward, context):
        reward_pred = self(obs, action, context)
        loss = F.mse_loss(reward_pred, reward)
        return loss


class StateDecoderModel(BaseModel):
    def __init__(self, env, args):
        super().__init__(env, args)

        self.input_embedding = None
        self.context_embedding = None

        input_dim = args.context_dim

        self.fcs = nn.Sequential(nn.Linear(input_dim, args.helper_hidden_dim), nn.ReLU())
        for _ in range(args.helper_num_hidden_layer - 1):
            self.fcs.append(nn.Linear(args.helper_hidden_dim, args.helper_hidden_dim))
            self.fcs.append(nn.ReLU())
        self.fc_head = nn.Linear(args.helper_hidden_dim, self.obs_dim)

        self.context_stop_gradient = "sdm" not in args.context_encoder_update

    def forward(self, context):
        if self.context_stop_gradient:
            context = context.detach()
        x = torch.cat([context], 1)
        x = self.fcs(x)
        x = self.fc_head(x)
        return x

    def compute_loss(self, obs, action, next_obs, context):
        reconstruction_pred = self(context)
        if self.context_mode in ["aware", "aware_inferred", "aware_inferred_reconstructed"]:
            obs = obs[:, :-self.real_context_dim]
            next_obs = next_obs[:, :-self.real_context_dim]
        loss = F.mse_loss(reconstruction_pred, next_obs - obs)
        return loss


class ContextDecoderModel(BaseModel):
    def __init__(self, env, args):
        super().__init__(env, args)

        self.input_embedding = None
        self.context_embedding = None

        input_dim = args.context_dim

        self.fcs = nn.Sequential(nn.Linear(input_dim, args.helper_hidden_dim), nn.ReLU())
        for _ in range(args.helper_num_hidden_layer - 1):
            self.fcs.append(nn.Linear(args.helper_hidden_dim, args.helper_hidden_dim))
            self.fcs.append(nn.ReLU())
        self.fc_head = nn.Linear(args.helper_hidden_dim, self.real_context_dim)

        self.context_stop_gradient = "cdm" not in args.context_encoder_update

    def forward(self, context):
        if self.context_stop_gradient:
            context = context.detach()
        x = torch.cat([context], 1)
        x = self.fcs(x)
        x = self.fc_head(x)
        return x

    def compute_loss(self, obs, context):
        reconstruction_pred = self(context)
        assert self.context_mode in ["aware", "aware_inferred", "aware_inferred_reconstructed"]
        real_context = obs[:, -self.real_context_dim:]
        loss = F.mse_loss(reconstruction_pred, real_context)
        return loss


class RNDModel(BaseModel):
    def __init__(self, env, args):
        super().__init__(env, args)

        self.input_embedding = None
        self.context_embedding = None

        input_dim = self.obs_dim

        self.predictor = nn.Sequential(nn.Linear(input_dim, args.helper_hidden_dim), nn.ReLU())
        for _ in range(args.helper_num_hidden_layer - 1):
            self.predictor.append(nn.Linear(args.helper_hidden_dim, args.helper_hidden_dim))
            self.predictor.append(nn.ReLU())
        self.predictor.append(nn.Linear(args.helper_hidden_dim, args.helper_hidden_dim))

        self.target = nn.Sequential(nn.Linear(input_dim, args.helper_hidden_dim), nn.ReLU())
        for _ in range(args.helper_num_hidden_layer - 1):
            self.target.append(nn.Linear(args.helper_hidden_dim, args.helper_hidden_dim))
            self.target.append(nn.ReLU())
        self.target.append(nn.Linear(args.helper_hidden_dim, args.helper_hidden_dim))

        # freeze the network parameters
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, obs):
        x = torch.cat([obs], 1)
        return self.predictor(x), self.target(x)

    def compute_loss(self, obs):
        prediction, target = self(obs)
        dist = F.mse_loss(prediction, target, reduction="none")
        loss = dist.mean()
        dist = dist.detach().mean(dim=1)
        dist = (dist - dist.min()) / (dist.max() - dist.min() + 1e-11)
        return loss, dist

