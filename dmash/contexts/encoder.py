import numpy as np
import torch
import torch.nn as nn

from dmash.common import (
    WindowNorm,
    SimNorm,
    SymLog,
    AvgL1Norm,
    TransformerEncoderLayer,
    TransformerEncoder,
    PositionalEncoding
)


class MLPContextEncoder(nn.Module):
    def __init__(
        self,
        env,
        target_dim,
        context_size,
        context_dim,
        model_dim=16,
        num_heads=1,
        num_layers=1,
        dropout=0.0,
        bias=False,
        input_norm="none",
        input_symlog=False,
        output_norm="none",
        output_symlog=False,
        output_shuffle=True,
        tfixup=False,
        separate_input_embedding=True,
        pos_enc=False,
        input_mask=0.0,
        input_shuffle=True,
        liu_input_mask=False,
        hopfield=True,
        hf_update_steps_max=1,
        hf_scaling=1.0,
        bidirectional=False,
        lstm_output_type="hc",
        device="cpu",
        norm=False,
        select_percentage=1.0,
        select_type=None,
        context_mask=False,
        context_aware=False,
        input_noise=0.0,
        output_noise=0.0,
        input_distractors=0,
    ):
        super().__init__()
        self.select_type = select_type
        self.select_k = int(context_size * select_percentage) if select_type in [
            "random_pre"
        ] else context_size

        self.mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim),
            nn.ReLU()
        )
        self.input_embedding = InputEmbedding(
            env,
            target_dim,
            context_dim,
            model_dim,
            bias,
            input_norm,
            input_symlog,
            separate_input_embedding,
            input_mask,
            liu_input_mask,
            input_shuffle,
            context_aware,
            input_noise,
            input_distractors,
        )
        self.output_summarizer = OutputSummarizer(
            self.select_k,
            context_dim,
            model_dim,
            bias,
            output_shuffle,
            output_norm,
            output_symlog,
            output_noise,
        )
        self.context_mask = context_mask

    def forward(self, x, a, target, training):
        x = self.input_embedding(x, a, target, training)
        if self.select_type == "random_pre":
            x = select_context(x, self.select_type, self.select_k, training=training)
        x = self.mlp(x)
        context = self.output_summarizer(x, training)  # batch_size, context_size, f_dim

        if training and self.context_mask:
            N, D = context.shape
            idx = torch.randint(low=0, high=D - 1, size=(N, 1), device=context.device)
            context = torch.scatter(context, 1, idx, 0.0)

        return context


class RNNContextEncoder(nn.Module):
    def __init__(
        self,
        env,
        target_dim,
        context_size,
        context_dim,
        model_dim=16,
        num_heads=1,
        num_layers=1,
        dropout=0.0,
        bias=False,
        input_norm="none",
        input_symlog=False,
        output_norm="none",
        output_symlog=False,
        output_shuffle=True,
        tfixup=False,
        separate_input_embedding=True,
        pos_enc=False,
        input_mask=0.0,
        input_shuffle=True,
        liu_input_mask=False,
        hopfield=True,
        hf_update_steps_max=1,
        hf_scaling=1.0,
        bidirectional=False,
        lstm_output_type="hc",
        device="cpu",
        norm=False,
        select_percentage=1.0,
        select_type=None,
        context_mask=False,
        context_aware=False,
        input_noise=0.0,
        output_noise=0.0,
        input_distractors=0,
    ):
        super().__init__()
        self.select_type = select_type
        self.select_k = int(context_size * select_percentage) if select_type in [
            "random_pre"
        ] else context_size

        self.rnn = nn.RNN(model_dim, model_dim, batch_first=True, bidirectional=bidirectional)
        self.input_embedding = InputEmbedding(
            env,
            target_dim,
            context_dim,
            model_dim,
            bias,
            input_norm,
            input_symlog,
            separate_input_embedding,
            input_mask,
            liu_input_mask,
            input_shuffle,
            context_aware,
            input_noise,
            input_distractors,
        )
        self.output_summarizer = OutputSummarizer(
            1 * num_layers * 2 if bidirectional else 1 * num_layers,  # there is no context size anymore, but h is put in that dim, hence 1
            context_dim,
            model_dim,
            bias,
            output_shuffle,
            output_norm,
            output_symlog,
            output_noise,
        )
        self.context_mask = context_mask

    def forward(self, x, a, target, training):
        x = self.input_embedding(x, a, target, training)

        if self.select_type == "random_pre":
            x = select_context(x, self.select_type, self.select_k, training=training)
        x, h = self.rnn(x)
        h = h.permute(1, 0, 2)
        context = self.output_summarizer(h, training)  # batch_size, 1, f_dim

        if training and self.context_mask:
            N, D = context.shape
            idx = torch.randint(low=0, high=D - 1, size=(N, 1), device=context.device)
            context = torch.scatter(context, 1, idx, 0.0)

        return context


class LSTMContextEncoder(nn.Module):
    def __init__(
        self,
        env,
        target_dim,
        context_size,
        context_dim,
        model_dim=16,
        num_heads=1,
        num_layers=1,
        dropout=0.0,
        bias=False,
        input_norm="none",
        input_symlog=False,
        output_norm="none",
        output_symlog=False,
        output_shuffle=True,
        tfixup=False,
        separate_input_embedding=True,
        pos_enc=False,
        input_mask=0.0,
        input_shuffle=True,
        liu_input_mask=False,
        hopfield=True,
        hf_update_steps_max=1,
        hf_scaling=1.0,
        bidirectional=True,
        lstm_output_type="hc",
        device="cpu",
        norm=False,
        select_percentage=1.0,
        select_type=None,
        context_mask=False,
        context_aware=False,
        input_noise=0.0,
        output_noise=0.0,
        input_distractors=0,
    ):
        super().__init__()
        self.select_type = select_type
        self.select_k = int(context_size * select_percentage) if select_type in [
            "random_pre"
        ] else context_size

        self.lstm = nn.LSTM(model_dim, model_dim, batch_first=True, num_layers=1, bidirectional=bidirectional)
        self.input_embedding = InputEmbedding(
            env,
            target_dim,
            context_dim,
            model_dim,
            bias,
            input_norm,
            input_symlog,
            separate_input_embedding,
            input_mask,
            liu_input_mask,
            input_shuffle,
            context_aware,
            input_noise,
            input_distractors,
        )
        summarizer_size = 0
        if "x" in lstm_output_type:
            summarizer_size += self.select_k
        if "h" in lstm_output_type:
            summarizer_size += 2 * num_layers if bidirectional else 1 * num_layers
        if "c" in lstm_output_type:
            summarizer_size += 2 * num_layers if bidirectional else 1 * num_layers
        assert summarizer_size > 0

        self.output_summarizer = OutputSummarizer(
            summarizer_size,
            context_dim,
            model_dim,
            bias,
            output_shuffle,
            output_norm,
            output_symlog,
            output_noise,
        )
        self.context_mask = context_mask
        self.lstm_output_type = lstm_output_type

    def forward(self, x, a, target, training):
        x = self.input_embedding(x, a, target, training)

        if self.select_type == "random_pre":
            x = select_context(x, self.select_type, self.select_k, training=training)
        x, (h, c) = self.lstm(x)
        h = h.permute(1, 0, 2)
        c = c.permute(1, 0, 2)

        summarizer_input = []
        if "x" in self.lstm_output_type:
            summarizer_input.append(x)
        if "h" in self.lstm_output_type:
            summarizer_input.append(h)
        if "c" in self.lstm_output_type:
            summarizer_input.append(c)
        assert len(summarizer_input) > 0

        context = self.output_summarizer(torch.cat(summarizer_input, 1), training)

        if training and self.context_mask:
            N, D = context.shape
            idx = torch.randint(low=0, high=D - 1, size=(N, 1), device=context.device)
            context = torch.scatter(context, 1, idx, 0.0)

        return context


class TransformerContextEncoder(nn.Module):
    def __init__(
        self,
        env,
        target_dim,
        context_size,
        context_dim,
        model_dim=16,
        num_heads=1,
        num_layers=1,
        dropout=0.0,
        bias=False,
        input_norm="none",
        input_symlog=False,
        output_norm="none",
        output_symlog=False,
        output_shuffle=True,
        tfixup=False,
        separate_input_embedding=True,
        pos_enc=False,
        input_mask=0.0,
        input_shuffle=True,
        liu_input_mask=False,
        hopfield=True,
        hf_update_steps_max=1,
        hf_scaling=1.0,
        bidirectional=False,
        lstm_output_type="hc",
        device="cpu",
        norm=False,
        select_percentage=1.0,
        select_type=None,
        context_mask=False,
        context_aware=False,
        input_noise=0.0,
        output_noise=0.0,
        input_distractors=0,
    ):
        super().__init__()

        self.select_type = select_type
        self.select_k = int(context_size * select_percentage) if select_type in [
            "topk", "random_post", "random_pre"
        ] else context_size

        if liu_input_mask:
            context_size = int(context_size * 3 * (1 - input_mask))

        if hopfield:
            from hflayers import Hopfield
            from dmash.common import TransformerEncoderLayer_hopfield
            hopfield = Hopfield(input_size=model_dim, num_heads=num_heads)
            transformer_encoder_layer = TransformerEncoderLayer_hopfield(
                hopfield_association=hopfield,
                dim_feedforward=model_dim,
                dropout=dropout,
            )
            assert select_type == "random_pre"
        else:
            tel_module = TransformerEncoderLayer
            transformer_encoder_layer = tel_module(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=model_dim,
                dropout=dropout,
                batch_first=True,
                norm=norm,
            )

        self.transformer_encoder = TransformerEncoder(
            encoder_layer=transformer_encoder_layer,
            num_layers=num_layers,
        )
        self.input_embedding = InputEmbedding(
            env,
            target_dim,
            context_dim,
            model_dim,
            bias,
            input_norm,
            input_symlog,
            separate_input_embedding,
            input_mask,
            liu_input_mask,
            input_shuffle,
            context_aware,
            input_noise,
            input_distractors,
        )

        self.output_summarizer = OutputSummarizer(
            self.select_k,
            context_dim,
            model_dim,
            bias,
            output_shuffle,
            output_norm,
            output_symlog,
            output_noise,
        )
        self.pos_encoder = PositionalEncoding(
            model_dim,
            dropout,
            batch_first=True
        ) if pos_enc else None

        self.src_mask = torch.log(torch.tril(torch.ones(context_size, context_size)))
        self.context_mask = context_mask
        self.hopfield = hopfield

        # weight init
        # https://github.com/luckeciano/transformers-metarl/blob/trmrl-torch/src/garage/torch/policies/gaussian_transformer_encoder_policy.py#L152
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        if tfixup:
            temp_state_dic = {}
            for name, param in self.transformer_encoder.named_parameters():
                if any(s in name for s in [
                    "linear1.weight",
                    "linear2.weight",
                    "self_attn.out_proj.weight"
                ]):
                    temp_state_dic[name] = (0.67 * (num_layers) ** (- 1. / 4.)) * param
                elif "self_attn.in_proj_weight" in name:
                    temp_state_dic[name] = (0.67 * (num_layers) ** (- 1. / 4.)) * (param * (2**0.5))

            for name in self.transformer_encoder.state_dict():
                if name not in temp_state_dic:
                    temp_state_dic[name] = self.transformer_encoder.state_dict()[name]
            self.transformer_encoder.load_state_dict(temp_state_dic)

    def forward(self, x, a, target, training=True):
        x = self.input_embedding(x, a, target, training)
        # [batch_size, context_size, input_dim]
        x = self.pos_encoder(x) if self.pos_encoder else x
        if self.select_type == "random_pre":
            x = select_context(x, self.select_type, self.select_k, training=training)
            x, _ = self.transformer_encoder(x, need_weights=self.select_type == "topk")
        else:
            x, x_weights = self.transformer_encoder(x, need_weights=self.select_type == "topk")
            x = select_context(
                x, self.select_type, self.select_k,
                attention_weights=x_weights if self.select_type == "topk" else None,
                training=training
            )
        context = self.output_summarizer(x, training)

        if training and self.context_mask:
            N, D = context.shape
            idx = torch.randint(low=0, high=D - 1, size=(N, 1), device=context.device)
            context = torch.scatter(context, 1, idx, 0.0)

        return context


def select_context(x, select_type, select_k, attention_weights=None, training=False):
    if select_type in ["random_pre", "random_post"]:
        # similar to liu input masking
        N, K, D = x.shape
        if training:
            noise = torch.rand(N, K, device=x.device)
            shuffle_ind = torch.argsort(noise, dim=1)
            select_ind = shuffle_ind[:, :select_k]
        else:
            select_ind = torch.linspace(0, K - 1, select_k, dtype=int, device=x.device).repeat(N, 1)
    elif select_type == "topk":
        assert attention_weights is not None
        topk_source = attention_weights.sum(dim=2)
        _, select_ind = topk_source.topk(select_k, sorted=False, dim=1)
    else:
        return x
    # select 3D based on 2D tensor
    select_x = torch.take_along_dim(x, indices=select_ind.unsqueeze(-1), dim=1)
    return select_x


class EnsembleContextEncoder(nn.Module):
    def __init__(
        self,
        n_encoders,
        encoder_module,
        sample,
        env,
        target_dim,
        context_size,
        context_dim,
        model_dim=16,
        num_heads=1,
        num_layers=1,
        dropout=0.0,
        bias=False,
        input_norm="none",
        input_symlog=False,
        output_norm="none",
        output_symlog=False,
        output_shuffle=True,
        tfixup=False,
        separate_input_embedding=True,
        pos_enc=False,
        input_mask=0.0,
        input_shuffle=True,
        liu_input_mask=False,
        hopfield=True,
        hf_update_steps_max=1,
        hf_scaling=1.0,
        bidirectional=True,
        lstm_output_type="hc",
        device="cpu",
        norm=False,
        select_percentage=1.0,
        select_type=None,
        context_mask=False,
        context_aware=False,
        input_noise=0.0,
        output_noise=0.0,
        input_distractors=0,
    ):
        super().__init__()
        if encoder_module == "mixed":
            self._models = nn.ModuleList(
                [
                    TransformerContextEncoder(
                        env,
                        target_dim,
                        context_size,
                        context_dim,
                        model_dim,
                        num_heads,
                        num_layers,
                        dropout,
                        bias,
                        input_norm,
                        input_symlog,
                        output_norm,
                        output_symlog,
                        output_shuffle,
                        tfixup,
                        separate_input_embedding,
                        pos_enc,
                        input_mask,
                        input_shuffle,
                        liu_input_mask,
                        hopfield,
                        hf_update_steps_max,
                        hf_scaling,
                        bidirectional,
                        lstm_output_type,
                        device,
                        norm,
                        select_percentage,
                        select_type,
                        context_mask,
                        context_aware,
                        input_noise,
                        output_noise,
                        input_distractors,
                    ) for _ in range(int(n_encoders / 2))
                ] +
                [
                    LSTMContextEncoder(
                        env,
                        target_dim,
                        context_size,
                        context_dim,
                        model_dim,
                        num_heads,
                        num_layers,
                        dropout,
                        bias,
                        input_norm,
                        input_symlog,
                        output_norm,
                        output_symlog,
                        output_shuffle,
                        tfixup,
                        separate_input_embedding,
                        pos_enc,
                        input_mask,
                        input_shuffle,
                        liu_input_mask,
                        hopfield,
                        hf_update_steps_max,
                        hf_scaling,
                        bidirectional,
                        lstm_output_type,
                        device,
                        norm,
                        select_percentage,
                        select_type,
                        context_mask,
                        context_aware,
                        input_noise,
                        output_noise,
                        input_distractors,
                    ) for _ in range(int(n_encoders / 2))
                ]
            )
        else:
            self._models = nn.ModuleList(
                [encoder_module(
                    env,
                    target_dim,
                    context_size,
                    context_dim,
                    model_dim,
                    num_heads,
                    num_layers,
                    dropout,
                    bias,
                    input_norm,
                    input_symlog,
                    output_norm,
                    output_symlog,
                    output_shuffle,
                    tfixup,
                    separate_input_embedding,
                    pos_enc,
                    input_mask,
                    input_shuffle,
                    liu_input_mask,
                    hopfield,
                    hf_update_steps_max,
                    hf_scaling,
                    bidirectional,
                    lstm_output_type,
                    device,
                    norm,
                    select_percentage,
                    select_type,
                    context_mask,
                    context_aware,
                    input_noise,
                    output_noise,
                    input_distractors,
                ) for _ in range(n_encoders)]
            )
        self._sample = sample

    def forward(self, x, a, target, training=True):
        x = torch.stack([model(x, a, target, training=True) for model in self._models])
        if self._sample and training:
            E, N, D = x.shape
            idx = torch.randint(E, size=(1,))
            return x[idx].squeeze()
        else:
            return torch.mean(x, dim=0)

    def get_uncertainty(self, x, a, target, training=True):
        x = torch.stack([model(x, a, target, training=True) for model in self._models])
        variance = torch.var(x, dim=0, correction=0)
        return variance


class PMLPContextEncoder(nn.Module):
    def __init__(
        self,
        env,
        target_dim,
        context_size,
        context_dim,
        model_dim=16,
        num_heads=1,
        num_layers=1,
        dropout=0.0,
        bias=False,
        input_norm="none",
        input_symlog=False,
        output_norm="none",
        output_symlog=False,
        output_shuffle=True,
        tfixup=False,
        separate_input_embedding=True,
        pos_enc=False,
        input_mask=0.0,
        input_shuffle=True,
        liu_input_mask=False,
        hopfield=True,
        hf_update_steps_max=1,
        hf_scaling=1.0,
        bidirectional=False,
        lstm_output_type="hc",
        device="cpu",
        norm=False,
        select_percentage=1.0,
        select_type=None,
        context_mask=False,
        context_aware=False,
        input_noise=0.0,
        output_noise=0.0,
        input_distractors=0,
    ):
        super().__init__()
        self.select_type = select_type
        self.select_k = int(context_size * select_percentage) if select_type in [
            "random_pre"
        ] else context_size

        self.mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim),
            nn.ReLU()
        )
        self.input_embedding = InputEmbedding(
            env,
            target_dim,
            context_dim,
            model_dim,
            bias,
            input_norm,
            input_symlog,
            separate_input_embedding,
            input_mask,
            liu_input_mask,
            input_shuffle,
            context_aware,
            input_noise,
            input_distractors,
        )
        self.output_summarizer = OutputSummarizer(
            self.select_k,
            context_dim,
            model_dim,
            bias,
            output_shuffle,
            output_norm,
            output_symlog,
            output_noise,
        )
        self.output_mu = nn.Sequential(
            nn.Linear(model_dim, context_dim, bias=bias),
        )
        if output_norm:
            self.output_mu.append(nn.LayerNorm(context_dim))
        self.output_var = nn.Sequential(
            nn.Linear(model_dim, context_dim, bias=bias),
        )
        if output_norm:
            self.output_var.append(nn.LayerNorm(context_dim))
        self.output_var.append(nn.Softplus())
        self.context_mask = context_mask

        self.output_noise = output_noise

    def forward(self, x, a, target, training):
        x = self.input_embedding(x, a, target, training)
        if self.select_type == "random_pre":
            x = select_context(x, self.select_type, self.select_k, training=training)

        x = self.mlp(x)
        mus = self.output_mu(x)  # batch_size, context_size, context_dim
        vars = self.output_var(x)  # batch_size, context_size, context_dim

        # product of gaussians
        vars = torch.clamp(vars, min=1e-7)
        var = 1. / torch.sum(torch.reciprocal(vars), dim=1)
        mu = var * torch.sum(mus / vars, dim=1)

        # sample
        posteriors = torch.distributions.Normal(mu, torch.sqrt(var))
        if training:
            context = posteriors.rsample()
        else:
            context = mu

        if training and self.context_mask:
            N, D = context.shape
            idx = torch.randint(low=0, high=D - 1, size=(N, 1), device=context.device)
            context = torch.scatter(context, 1, idx, 0.0)

        if self.output_noise != 0.0:
            n_context = 0.0 + (torch.rand(context.shape, device=context.device) * 2 - 1) * self.output_noise
            context = context + n_context

        return context

    def get_kl(self, x, a, target, training):
        x = self.input_embedding(x, a, target, training)
        if self.select_type == "random_pre":
            x = select_context(x, self.select_type, self.select_k, training=training)
        else:
            N, K, D = x.shape  # batch_size, context_size, feature_dim
            idx = torch.randperm(K)
            x = x[:, idx]

        x = self.mlp(x)
        mus = self.output_mu(x)  # batch_size, context_size, context_dim
        vars = self.output_var(x)  # batch_size, context_size, context_dim

        # product of gaussians
        vars = torch.clamp(vars, min=1e-7)
        var = 1. / torch.sum(torch.reciprocal(vars), dim=1)
        mu = var * torch.sum(mus / vars, dim=1)

        # KL
        posteriors = torch.distributions.Normal(mu, torch.sqrt(var))
        priors = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(var))
        kl_divs = torch.distributions.kl.kl_divergence(posteriors, priors)

        return kl_divs

    def get_uncertainty(self, x, a, target, training):
        x = self.input_embedding(x, a, target, training)
        if self.select_type == "random_pre":
            x = select_context(x, self.select_type, self.select_k, training=training)
        else:
            N, K, D = x.shape  # batch_size, context_size, feature_dim
            idx = torch.randperm(K)
            x = x[:, idx]

        x = self.mlp(x)
        mus = self.output_mu(x)  # batch_size, context_size, context_dim
        vars = self.output_var(x)  # batch_size, context_size, context_dim

        # product of gaussians
        vars = torch.clamp(vars, min=1e-7)
        var = 1. / torch.sum(torch.reciprocal(vars), dim=1)
        mu = var * torch.sum(mus / vars, dim=1)

        # entropy
        posteriors = torch.distributions.Normal(mu, torch.sqrt(var))
        uncertainties = posteriors.entropy()

        return uncertainties


class PLSTMContextEncoder(nn.Module):
    def __init__(
        self,
        env,
        target_dim,
        context_size,
        context_dim,
        model_dim=16,
        num_heads=1,
        num_layers=1,
        dropout=0.0,
        bias=False,
        input_norm="none",
        input_symlog=False,
        output_norm="none",
        output_symlog=False,
        output_shuffle=True,
        tfixup=False,
        separate_input_embedding=True,
        pos_enc=False,
        input_mask=0.0,
        input_shuffle=True,
        liu_input_mask=False,
        hopfield=True,
        hf_update_steps_max=1,
        hf_scaling=1.0,
        bidirectional=True,
        lstm_output_type="hc",
        device="cpu",
        norm=False,
        select_percentage=1.0,
        select_type=None,
        context_mask=False,
        context_aware=False,
        input_noise=0.0,
        output_noise=0.0,
        input_distractors=0,
    ):
        super().__init__()
        self.select_type = select_type
        self.select_k = int(context_size * select_percentage) if select_type in [
            "random_pre"
        ] else context_size

        self.lstm = nn.LSTM(model_dim, model_dim, batch_first=True, num_layers=1, bidirectional=bidirectional)
        self.input_embedding = InputEmbedding(
            env,
            target_dim,
            context_dim,
            model_dim,
            bias,
            input_norm,
            input_symlog,
            separate_input_embedding,
            input_mask,
            liu_input_mask,
            input_shuffle,
            context_aware,
            input_noise,
            input_distractors,
        )
        summarizer_size = 0
        if "x" in lstm_output_type:
            summarizer_size += self.select_k
        if "h" in lstm_output_type:
            summarizer_size += 2 * num_layers if bidirectional else 1 * num_layers
        if "c" in lstm_output_type:
            summarizer_size += 2 * num_layers if bidirectional else 1 * num_layers
        assert summarizer_size > 0

        self.output_mu = nn.Sequential(
            nn.Linear(model_dim * summarizer_size, context_dim, bias=bias),
        )
        if output_norm:
            self.output_mu.append(nn.LayerNorm(context_dim))
        self.output_var = nn.Sequential(
            nn.Linear(model_dim * summarizer_size, context_dim, bias=bias),
        )
        if output_norm:
            self.output_var.append(nn.LayerNorm(context_dim))
        self.output_var.append(nn.Softplus())
        self.context_mask = context_mask
        self.lstm_output_type = lstm_output_type

        self.output_noise = output_noise

    def forward(self, x, a, target, training):
        x = self.input_embedding(x, a, target, training)

        if self.select_type == "random_pre":
            x = select_context(x, self.select_type, self.select_k, training=training)
        x, (h, c) = self.lstm(x)
        h = h.permute(1, 0, 2)
        c = c.permute(1, 0, 2)

        summarizer_input = []
        if "x" in self.lstm_output_type:
            summarizer_input.append(x)
        if "h" in self.lstm_output_type:
            summarizer_input.append(h)
        if "c" in self.lstm_output_type:
            summarizer_input.append(c)
        assert len(summarizer_input) > 0
        x = torch.cat(summarizer_input, 1).flatten(1)

        mu = self.output_mu(x)  # batch_size, context_dim
        var = self.output_var(x)  # batch_size, context_dim

        # no product of gaussians in recurrent case

        # sample
        posteriors = torch.distributions.Normal(mu, torch.sqrt(var))
        if training:
            context = posteriors.rsample()
        else:
            context = mu

        if training and self.context_mask:
            N, D = context.shape
            idx = torch.randint(low=0, high=D - 1, size=(N, 1), device=context.device)
            context = torch.scatter(context, 1, idx, 0.0)

        if self.output_noise != 0.0:
            n_context = 0.0 + (torch.rand(context.shape, device=context.device) * 2 - 1) * self.output_noise
            context = context + n_context

        return context

    def get_kl(self, x, a, target, training):
        x = self.input_embedding(x, a, target, training)

        if self.select_type == "random_pre":
            x = select_context(x, self.select_type, self.select_k, training=training)
        x, (h, c) = self.lstm(x)
        h = h.permute(1, 0, 2)
        c = c.permute(1, 0, 2)

        summarizer_input = []
        if "x" in self.lstm_output_type:
            summarizer_input.append(x)
        if "h" in self.lstm_output_type:
            summarizer_input.append(h)
        if "c" in self.lstm_output_type:
            summarizer_input.append(c)
        assert len(summarizer_input) > 0
        x = torch.cat(summarizer_input, 1).flatten(1)

        mu = self.output_mu(x)  # batch_size, context_dim
        var = self.output_var(x)  # batch_size, context_dim

        # no product of gaussians in recurrent case

        # KL
        posteriors = torch.distributions.Normal(mu, torch.sqrt(var))
        priors = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(var))
        kl_divs = torch.distributions.kl.kl_divergence(posteriors, priors)

        return kl_divs

    def get_uncertainty(self, x, a, target, training):
        x = self.input_embedding(x, a, target, training)

        if self.select_type == "random_pre":
            x = select_context(x, self.select_type, self.select_k, training=training)
        x, (h, c) = self.lstm(x)
        h = h.permute(1, 0, 2)
        c = c.permute(1, 0, 2)

        summarizer_input = []
        if "x" in self.lstm_output_type:
            summarizer_input.append(x)
        if "h" in self.lstm_output_type:
            summarizer_input.append(h)
        if "c" in self.lstm_output_type:
            summarizer_input.append(c)
        assert len(summarizer_input) > 0
        x = torch.cat(summarizer_input, 1).flatten(1)

        mu = self.output_mu(x)  # batch_size, context_dim
        var = self.output_var(x)  # batch_size, context_dim

        # no product of gaussians in recurrent case

        # entropy
        posteriors = torch.distributions.Normal(mu, torch.sqrt(var))
        uncertainties = posteriors.entropy()

        return uncertainties


class InputEmbedding(nn.Module):
    def __init__(
        self,
        env,
        target_dim,
        context_dim,
        model_dim=64,
        bias=False,
        input_norm=None,
        input_symlog=False,
        separate_input_embedding=False,
        input_mask=0.0,
        liu_input_mask=False,
        input_shuffle=True,
        context_aware=False,
        input_noise=0.0,
        input_distractors=0,
    ):
        super().__init__()

        obs_dim = np.array(env.single_observation_space.shape).prod()
        action_dim = np.array(env.single_action_space.shape).prod()

        if context_aware:
            self.explicit_context_dim = env.envs[0].get_wrapper_attr("context").size
            if obs_dim == target_dim:
                target_dim -= self.explicit_context_dim
            obs_dim -= self.explicit_context_dim

        if input_distractors > 0:
            obs_dim += input_distractors
            action_dim += input_distractors
            target_dim += input_distractors

        self.context_aware = context_aware

        if liu_input_mask > 0.0:
            separate_input_embedding = True
            "Warning: set separate_input_embedding True for liu input masking."

        if separate_input_embedding:
            self.input_embedding_x = nn.Sequential()
            self.input_embedding_a = nn.Sequential()
            self.input_embedding_target = nn.Sequential()
            if input_symlog:
                self.input_embedding_x.append(SymLog())
                self.input_embedding_a.append(SymLog())
                self.input_embedding_target.append(SymLog())
            self.input_embedding_x.append(nn.Linear(obs_dim, model_dim, bias=bias))
            self.input_embedding_a.append(nn.Linear(action_dim, model_dim, bias=bias))
            self.input_embedding_target.append(nn.Linear(target_dim, model_dim, bias=bias))

            self.input_embedding_all = nn.Sequential()
            self.input_embedding_all.append(
                nn.Linear(model_dim if liu_input_mask else 3 * model_dim, model_dim, bias=bias)
            )
            if input_norm == "layer":
                self.input_embedding_all.append(nn.LayerNorm(model_dim))
            elif input_norm == "window":
                self.input_embedding_all.append(WindowNorm(model_dim))
            elif input_norm == "simnorm":
                self.input_embedding_all.append(SimNorm(model_dim))
            elif input_norm == "avgl1norm":
                self.input_embedding_all.append(AvgL1Norm())
        else:
            self.input_embedding_x = None
            self.input_embedding_a = None
            self.input_embedding_target = None
            self.input_embedding_all = nn.Sequential()
            if input_symlog:
                self.input_embedding_all.append(SymLog())
            self.input_embedding_all.append(nn.Linear(obs_dim + action_dim + target_dim, model_dim, bias=bias))
            if input_norm == "layer":
                self.input_embedding_all.append(nn.LayerNorm(model_dim))
            elif input_norm == "window":
                self.input_embedding_all.append(WindowNorm(model_dim))
            elif input_norm == "simnorm":
                self.input_embedding_all.append(SimNorm(model_dim // 2))
            elif input_norm == "avgl1norm":
                self.input_embedding_all.append(AvgL1Norm())

        assert 0.0 <= input_mask <= 1.0
        assert 0.0 <= liu_input_mask <= 1.0
        self.input_mask = input_mask
        self.liu_input_mask = liu_input_mask
        self.input_shuffle = input_shuffle

        self.input_noise = input_noise
        self.input_distractors = input_distractors

    def forward(self, x, a, target, training):
        if self.context_aware:
            if x.shape == target.shape:
                target = target[:, :, :-self.explicit_context_dim]
            x = x[:, :, :-self.explicit_context_dim]

        if self.input_noise != 0.0:
            n_x = 0.0 + (torch.rand(x.shape, device=x.device) * 2 - 1) * self.input_noise
            n_a = 0.0 + (torch.rand(a.shape, device=a.device) * 2 - 1) * self.input_noise
            n_target = 0.0 + (torch.rand(target.shape, device=target.device) * 2 - 1) * self.input_noise
            x = x + n_x
            a = a + n_a
            target = target + n_target

        if self.input_distractors > 0:
            d_x = torch.rand((x.shape[0], x.shape[1], self.input_distractors), device=x.device)
            d_a = torch.rand((x.shape[0], x.shape[1], self.input_distractors), device=a.device)
            d_target = torch.rand((x.shape[0], x.shape[1], self.input_distractors), device=target.device)
            x = torch.concat([x, d_x], dim=2)
            a = torch.concat([a, d_a], dim=2)
            target = torch.concat([target, d_target], dim=2)

        if self.input_shuffle and training:
            N, K, D = x.shape  # batch_size, context_size, feature_dim
            # shuffle to break temporal correlation
            idx = torch.randperm(K)
            x = x[:, idx]
            a = a[:, idx]
            target = target[:, idx]

        x = self.input_embedding_x(x) if self.input_embedding_x else x
        a = self.input_embedding_a(a) if self.input_embedding_a else a
        target = self.input_embedding_target(target) if self.input_embedding_target else target
        if self.input_mask > 0.0 and training:
            N, K, D = x.shape  # batch_size, context_size, feature_dim
            m_x = torch.rand(N, K, 1, device=x.device) > self.input_mask
            m_a = torch.rand(N, K, 1, device=a.device) > self.input_mask
            m_target = torch.rand(N, K, 1, device=target.device) > self.input_mask
            x = x * m_x
            a = a * m_a
            target = target * m_target
        if self.liu_input_mask > 0.0:
            N, K, D = x.shape
            x = torch.stack([x, a, target], dim=1).permute(0, 2, 1, 3).reshape(N, 3 * K, D)
            N, K, D = x.shape
            masked_context_size = int(K * (1 - self.liu_input_mask))
            if training:
                noise = torch.rand(N, K)
                ids_shuffle = torch.argsort(noise, dim=1)
                ids_keep = ids_shuffle[:, :masked_context_size]
            else:
                # not checked
                ids_keep = torch.linspace(0, K - 1, masked_context_size, dtype=int, device=x.device).repeat(N, 1)
            x = torch.take_along_dim(x, indices=ids_keep.unsqueeze(-1), dim=1)
        else:
            x = torch.cat([x, a, target], -1)
        x = self.input_embedding_all(x)
        return x


class OutputSummarizer(nn.Module):
    def __init__(
        self,
        context_size,
        context_dim,
        model_dim=64,
        bias=False,
        shuffle=False,
        norm=True,
        symlog=False,
        output_noise=0.0,
    ):
        super().__init__()

        self.summarizer = nn.Sequential()
        if symlog:
            self.summarizer.append(SymLog())
        self.summarizer.append(
            nn.Linear(model_dim * context_size if shuffle else model_dim, context_dim, bias=bias)
        )
        if norm == "layer":
            self.summarizer.append(nn.LayerNorm(context_dim))
        elif norm == "simnorm":
            self.summarizer.append(SimNorm(context_dim // 2))
        elif norm == "avgl1norm":
            self.summarizer.append(AvgL1Norm())

        self.shuffle = shuffle

        self.output_noise = output_noise

    def forward(self, x, training):
        N, K, D = x.shape  # batch_size, context_size, feature_dim
        if self.shuffle and training:
            idx = torch.randperm(K)
            x = x[:, idx]
            x = x.flatten(1)
        elif self.shuffle and not training:
            x = x.flatten(1)
        else:
            x = x.mean(1)
        x = self.summarizer(x)

        if self.output_noise != 0.0:
            n_x = 0.0 + (torch.rand(x.shape, device=x.device) * 2 - 1) * self.output_noise
            x = x + n_x
        return x

