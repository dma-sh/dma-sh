from dataclasses import dataclass, field
from tyro.conf import FlagConversionOff


@dataclass
class Args:
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: FlagConversionOff[bool] = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: FlagConversionOff[bool] = True
    """if toggled, cuda will be enabled by default"""
    verbose: FlagConversionOff[bool] = True
    """if toggled, returns are printed"""
    write: FlagConversionOff[bool] = False
    """if toggled, metrics are stored as json"""
    wandb: FlagConversionOff[bool] = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "dmash"
    """the wandb's project name"""
    wandb_offline: FlagConversionOff[bool] = False
    """if toggled, the wandb tracking will be synced offline"""
    wandb_images: FlagConversionOff[bool] = False
    """if toggled, images are potentially tracked via wandb"""
    capture_video: FlagConversionOff[bool] = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "dmash/DI-sparse-v0"
    """the environment id of the task"""
    num_train_envs: int = 10
    """the number of environments used for training"""
    num_eval_envs: int = 10
    """the number of environments used for evaluation"""
    num_eval_episodes: int = 5
    """the number of episodes used for evaluation"""
    total_timesteps: int = 40000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e5)
    """the replay memory buffer size"""
    memory_efficient_buffer: FlagConversionOff[bool] = True
    """if toggled, a more memory efficient buffer is used for multi context case."""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    policy_hidden_dim: int = 256
    """the hidden dimension of the policy network"""
    policy_num_hidden_layer: int = 2
    """the number of hidden layers of the policy network"""
    policy_use_embedding: FlagConversionOff[bool] = False
    """if toggled, separate embedding encoders for context and state/action are used"""
    policy_context_merge_type: str = "concat"
    """the kind of how context is processed, either concat, mul, hypernet, none"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    q_hidden_dim: int = 256
    """the hidden dimension of the q network"""
    q_num_hidden_layer: int = 2
    """the number of hidden layers of the q network"""
    q_use_embedding: FlagConversionOff[bool] = False
    """if toggled, separate embedding encoders for context and state/action are used"""
    q_context_merge_type: str = "concat"
    """the kind of how context is processed, either concat, mul, hypernet, none"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target networks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: FlagConversionOff[bool] = True
    """automatic tuning of the entropy coefficient"""
    observation_noisy: float = 0.0
    """if larger 0, the observation is noisy"""
    num_distractors: int = 0
    """the number of distractor variables in the observation"""
    fixed_distractors: FlagConversionOff[bool] = True
    """if toggled the distractor variables in the observation are fixed over time"""
    rnn_mode: str = "batch_last"
    use_rnn: FlagConversionOff[bool] = False
    rnd_beta: float = 0.0
    """if larger 0, rnd exploration is used as intrinsic reward"""
    normalize_reward: FlagConversionOff[bool] = False
    use_additional_layer: FlagConversionOff[bool] = False

    # Context
    context_id: str = "all"
    """if single0 or single1, just the first or second context component is applied."""
    context_default: FlagConversionOff[bool] = False
    """if toggled, it's only trained on the default context"""
    context_mode: str = "unaware"
    """the context mode, either unaware, aware, inferred, aware_inferred"""
    context_aware_noisy: float = 0.0
    """if larger 0, the context aware information is noisy"""
    context_aware_varied: FlagConversionOff[bool] = True
    """if toggled, only the contexts that actually varied are used for the aware case"""
    context_aware_onehot: FlagConversionOff[bool] = False
    """if toggled, provide the actual context information as onehot instead of the inferred one"""
    context_constant: FlagConversionOff[bool] = False
    """if toggled, provide only constant ones as context, making it unaware"""
    context_handcrafted: FlagConversionOff[bool] = False
    """if toggled, the context selection is handcrafted, not sampled"""
    context_in_difficulty: float = 0.0
    """if larger 0, the context selection's difficulty increases with this argument"""
    context_out_difficulty: float = 0.0
    """if larger 0, the context selection's difficulty increases with this argument"""
    context_size: int = 24
    """the amount of past (s, a, s') used for context inference"""
    context_dim: int = 8
    """the number of hidden context dimensions"""
    context_warm_up: FlagConversionOff[bool] = True
    """if toggled, warm up steps are performed to fill context window"""
    context_warm_up_random: FlagConversionOff[bool] = False
    """if toggled, use random actions for warm up steps"""
    context_once: FlagConversionOff[bool] = False
    """if toggled, context is computed only once per episode"""
    context_intrinsic_uncertainty_beta: float = 0.0
    """the amount of intrinsic reward based on encoder uncertainty"""
    context_intrinsic_kl_beta: float = 0.0
    """the amount of intrinsic reward based on encoder kl div to gaussian prior"""
    context_intrinsic_error_beta: float = 0.0
    """the amount of intrinsic reward based on helper error"""
    context_intrinsic_annealing: FlagConversionOff[bool] = False
    """if toggled, intrinsic beta is changed over time with provided beta being the final value"""
    context_intrinsic_uncertainty_scaled: str = "none"
    """if fixed or batch, uncertainty is scaled before being used as intrinsic reward"""
    context_intrinsic_annealing_speed: float = 1.0
    """the speed how fast target beta is achieved"""
    compute_reward_context: FlagConversionOff[bool] = False
    """if toggled, a reward context is computed"""
    store: FlagConversionOff[bool] = False
    """if toggled, the trained context encoder and metadata is stored"""
    context_propagate: FlagConversionOff[bool] = False
    """if toggled, context propagated across neighboorhood"""
    context_select_type: str = "equidistant"
    """if 'sample', contexts are randomly sampled, if 'equidistant' contexts are equidistant."""
    context_mask: FlagConversionOff[bool] = False

    # Context encoder
    ce_lr: float = 3e-4
    """the learning rate of the context encoder network optimizer"""
    context_encoder: str = "MLP"
    """the context encoder type, this can be 'MLP', 'RNN', 'LSTM' or 'Transformer'"""
    context_encoder_model_dim: int = 32
    """the number of hidden model dimensions in context encoders"""
    context_encoder_num_heads: int = 1
    """the number of attention heads for Transformer context encoder"""
    context_encoder_num_layers: int = 1
    """the number of sub-encoder layers for Transformer context encoder"""
    context_encoder_tfixup: FlagConversionOff[bool] = True
    """if toggled, tfixup is applied for Transformer context encoder"""
    context_encoder_separate_input_embedding: FlagConversionOff[bool] = False
    """if toggled, inputs are embedded separately for Transformer context encoder"""
    context_encoder_input_mask: float = 0.2
    """the rate of input masking"""
    context_encoder_input_shuffle: FlagConversionOff[bool] = True
    """if toggled, the past (s, a, s') are shuffled in recurrent context encoders"""
    context_encoder_liu_input_mask: float = 0.0
    """if toggled, an alternative masking approach is applied (based on Liu et al. 2022)"""
    context_encoder_output_shuffle: FlagConversionOff[bool] = True
    """if toggled, the output of the encoder is flattened and shuffled, else it is average pooled"""
    context_encoder_dropout: float = 0.1
    """the dropout rate applied to the Transformer context encoder"""
    context_encoder_bias: FlagConversionOff[bool] = False
    """if toggled, bias is added to the input and output networks in the context encoder"""
    context_encoder_input_norm: str = "avgl1norm"
    """if toggled, the context input embedding is normalized, layer, window, avgl1norm, simnorm, none"""
    context_encoder_input_symlog: FlagConversionOff[bool] = False
    """if toggled, the context input embedding is first symlogged"""
    context_encoder_output_norm: str = "simnorm"
    """if toggled, the context output embedding is normalized, layer, window, avgl1norm, simnorm, none"""
    context_encoder_output_symlog: FlagConversionOff[bool] = False
    """if toggled, the context embedding is symlogged"""
    context_encoder_update: list[str] = field(default_factory=lambda: ["fm"])
    """the context encoder update type, this can be multiple from 'critic', 'actor', 'fm', 'im'"""
    context_encoder_hopfield: FlagConversionOff[bool] = False
    """if toggled, the Transformer context encoder is Hopfield-based"""
    context_encoder_ensemble: FlagConversionOff[bool] = False
    """if toggled, an ensemble is used for the context encoder to compute uncertainty"""
    context_encoder_num: int = 2
    """the number of encoders if ensemble is used"""
    context_encoder_ensemble_sampling: FlagConversionOff[bool] = False
    """if toggled, contexts are sampled from the ensemble if it is used"""
    context_encoder_hf_update_steps_max: int = 1
    context_encoder_hf_scaling: float = 1.0
    context_encoder_bidirectional: FlagConversionOff[bool] = False
    context_encoder_lstm_output_type: str = "h"
    context_encoder_norm: FlagConversionOff[bool] = False
    context_encoder_select_percentage: float = 0.2
    context_encoder_select_type: str = "random_pre"
    context_encoder_kl_beta: float = 0.0
    context_encoder_uncertainty_beta: float = 0.0
    """the impact of uncertainty in the loss"""
    context_encoder_input_noise: float = 0.0
    context_encoder_output_noise: float = 0.0
    context_encoder_input_distractors: int = 0

    # Helper
    fm_lr: float = 3e-4
    """the learning rate of the forward model network optimizer"""
    im_lr: float = 3e-4
    """the learning rate of the inverse model network optimizer"""
    bm_lr: float = 3e-4
    """the learning rate of the backward model network optimizer"""
    rm_lr: float = 3e-4
    """the learning rate of the reward model network optimizer"""
    sdm_lr: float = 3e-4
    """the learning rate of the decoder model network optimizer"""
    cdm_lr: float = 3e-4
    """the learning rate of the decoder model network optimizer"""
    helper_hidden_dim: int = 256
    """the hidden dimension of the helper network"""
    helper_num_hidden_layer: int = 2
    """the number of hidden layers of the helper network"""
    helper_use_embedding: FlagConversionOff[bool] = False
    """if toggled, separate embedding encoders for context and state/action are used"""
    helper_context_merge_type: str = "concat"
    """the kind of how context is processed, either concat, mul, hypernet, none"""
    reconstruction_lambda: float = 0.0
    """the weight for the reconstruction auxilliary loss"""
    fm_ensemble: FlagConversionOff[bool] = False
    """if toggled, an ensemble is used for the fm to compute uncertainty"""
    fm_loss_type: int = 0
    helper_update_per_step: int = 1
    """the number of updates for the helper networks"""
    helper_frequency: int = 1
    """the frequency of training helper (delayed)"""
    dataset_size: int = 10000
    """the size of the dataset for pretraining"""

    # additions
    hypernet_skip_connection: FlagConversionOff[bool] = True
    hypernet_chunked: FlagConversionOff[bool] = False
    hypernet_pre_activation: FlagConversionOff[bool] = False
    hypernet_post_activation: FlagConversionOff[bool] = True
    hypernet_hidden_dim: int = 64
    hypernet_num_hidden_layer: int = 2
    hypernet_bottleneck: FlagConversionOff[bool] = True
    hypernet_bias: FlagConversionOff[bool] = True
    hypernet_hnet_dropout: float = 0.0
    hypernet_mnet_dropout: float = 0.0
    hypernet_hyperfan_init: FlagConversionOff[bool] = False
    hypernet_no_update: FlagConversionOff[bool] = False
    hypernet_l1_reg_lambda: float = 0.0
    hypernet_l2_reg_lambda: float = 0.0
    hypernet_mean_reg_lambda: float = 0.0
    model_embedding_post_activation: FlagConversionOff[bool] = False
    model_embedding_dim: int = 32
    model_embedding_num_hidden_layer: int = 2

    # amago
    multitask_agent: FlagConversionOff[bool] = False
    parallel_actors: int = 10
    traj_encoder: str = "ff"
    """ff, transformer, gru, mamba"""

    # default config croups
    method: str = "none"


def modify_groups(args):
    if args.method != "none":

        # MAIN COMPARISON
        if args.method == "unaware_dr":
            args.context_mode = "unaware"
        elif args.method == "unaware_default":
            args.context_mode = "unaware"
            args.context_default = True

        elif args.method == "aware_concat":
            args.context_mode = "aware"
            args.policy_context_merge_type = "concat"
            args.q_context_merge_type = "concat"
        elif args.method == "aware_hypernet":
            args.context_mode = "aware"
            args.policy_context_merge_type = "hypernet"
            args.q_context_merge_type = "hypernet"
        elif args.method == "aware_hypernet_shared":
            args.context_mode = "aware"
            args.policy_context_merge_type = "hypernet"
            args.q_context_merge_type = "hypernet_shared"

        elif args.method == "inferred_concat":
            args.context_encoder = "LSTM"
            args.context_mode = "inferred"
        elif args.method == "inferred_plain_concat":
            args.context_encoder = "LSTM"
            args.context_mode = "inferred"
            args.context_encoder_separate_input_embedding = False
            args.context_encoder_input_mask = 0.0
            args.context_encoder_output_norm = "none"
            args.context_encoder_output_symlog = False
            args.context_encoder_input_norm = "none"
            args.context_encoder_input_symlog = False
        elif args.method == "inferred_hypernet":
            args.context_encoder = "LSTM"
            args.context_mode = "inferred"
            args.helper_context_merge_type = "hypernet"
            args.policy_context_merge_type = "concat"
            args.q_context_merge_type = "concat"
            args.context_encoder_input_mask = 0.4
        elif args.method == "inferred_hypernet_all":
            args.context_encoder = "LSTM"
            args.context_mode = "inferred"
            args.helper_context_merge_type = "hypernet"
            args.policy_context_merge_type = "hypernet"
            args.q_context_merge_type = "hypernet"
            args.context_encoder_input_mask = 0.4
        elif args.method == "inferred_hypernet_shared":
            args.context_encoder = "LSTM"
            args.context_mode = "inferred"
            args.helper_context_merge_type = "hypernet"
            args.policy_context_merge_type = "hypernet_shared"
            args.q_context_merge_type = "hypernet_shared"
            args.context_encoder_input_mask = 0.4
        elif args.method == "inferred_plain_hypernet_shared":
            args.context_encoder = "LSTM"
            args.context_mode = "inferred"
            args.helper_context_merge_type = "hypernet"
            args.policy_context_merge_type = "hypernet_shared"
            args.q_context_merge_type = "hypernet_shared"
            args.context_encoder_separate_input_embedding = False
            args.context_encoder_input_mask = 0.0
            args.context_encoder_output_norm = "none"
            args.context_encoder_output_symlog = False
            args.context_encoder_input_norm = "none"
            args.context_encoder_input_symlog = False

        elif args.method == "inferred_plain_pearl":
            args.context_encoder = "PLSTM"
            args.context_mode = "inferred"
            if args.context_encoder_kl_beta == 0.0:
                args.context_encoder_kl_beta = 0.4
            args.context_encoder_separate_input_embedding = False
            args.context_encoder_input_mask = 0.0
            args.context_encoder_output_norm = "none"
            args.context_encoder_output_symlog = False
            args.context_encoder_input_norm = "none"
            args.context_encoder_input_symlog = False

        elif args.method == "inferred_plain_pearl_q":
            args.context_encoder = "PLSTM"
            args.context_mode = "inferred"
            if args.context_encoder_kl_beta == 0.0:
                args.context_encoder_kl_beta = 0.4
            args.context_encoder_update = ["critic"]
            args.context_encoder_separate_input_embedding = False
            args.context_encoder_input_mask = 0.0
            args.context_encoder_output_norm = "none"
            args.context_encoder_output_symlog = False
            args.context_encoder_input_norm = "none"
            args.context_encoder_input_symlog = False
        elif args.method == "inferred_pearl":
            args.context_encoder = "PLSTM"
            args.context_mode = "inferred"
            if args.context_encoder_kl_beta == 0.0:
                args.context_encoder_kl_beta = 0.4
        elif args.method == "inferred_pearl_hypernet_shared":
            args.context_encoder = "PLSTM"
            args.context_mode = "inferred"
            if args.context_encoder_kl_beta == 0.0:
                args.context_encoder_kl_beta = 0.4
            args.helper_context_merge_type = "hypernet"
            args.policy_context_merge_type = "hypernet_shared"
            args.q_context_merge_type = "hypernet_shared"
            args.context_encoder_input_mask = 0.4

        else:
            raise NotImplementedError()

    return args
