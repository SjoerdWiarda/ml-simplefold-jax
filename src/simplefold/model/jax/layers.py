#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import math

import jax
from einops import rearrange
from flax import nnx


def modulate(x, shift: jax.Array, scale: jax.Array) -> jax.Array:
    return x * (1 + jax.numpy.expand_dims(scale, axis=1)) + jax.numpy.expand_dims(
        shift, axis=1
    )


#################################################################################
#                            Attention Layers                                  #
#################################################################################


# class MLP(nnx.Module):
#     def __init__(self, din: int, dmid: int, dout: int, *, rngs: nnx.Rngs):
#         self.linear1 = nnx.Linear(din, dmid, rngs=rngs)
#         self.dropout = nnx.Dropout(rate=0.1, rngs=rngs)
#         self.bn = nnx.BatchNorm(dmid, rngs=rngs)
#         self.linear2 = nnx.Linear(dmid, dout, rngs=rngs)

#     def __call__(self, x):
#         x = nnx.gelu(self.dropout(self.bn(self.linear1(x))))
#         return self.linear2(x)


class SelfAttentionLayer(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        rngs: nnx.Rngs = nnx.Rngs(0),
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale=None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_bias: bool = True,
        qk_norm: bool = True,
        pos_embedder=None,
        linear_target: type[nnx.Linear] = nnx.Linear,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = linear_target(
            in_features=hidden_size,
            out_features=hidden_size * 3,
            use_bias=qkv_bias,
            rngs=rngs,
        )
        self.attn_drop = nnx.Dropout(attn_drop)
        self.proj = linear_target(
            hidden_size, hidden_size, use_bias=use_bias, rngs=rngs
        )
        self.proj_drop = nnx.Dropout(proj_drop)

        self.q_norm = RMSNorm(head_dim) if qk_norm else nnx.identity
        self.k_norm = RMSNorm(head_dim) if qk_norm else nnx.identity

        self.pos_embedder = pos_embedder

    def __call__(self, x: jax.Array, **kwargs) -> jax.Array:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        pos = kwargs.get("pos")

        qkv = rearrange(qkv, "b n t h c -> t b h n c")
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.pos_embedder and pos is not None:
            q, k = self.pos_embedder(q, k, pos)

        attn = (q @ k.swapaxes(-2, -1)) * self.scale
        attn = nnx.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).swapaxes(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EfficientSelfAttentionLayer(SelfAttentionLayer):
    """Started from https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/attention.py"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def __call__(self, x: jax.Array, **kwargs) -> jax.Array:
        B, N, C = x.shape
        attn_mask = kwargs.get("attention_mask")
        pos = kwargs.get("pos")

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = rearrange(qkv, "b n t h c -> t b h n c")
        q, k, v = (qkv[0, :], qkv[1, :], qkv[2, :])

        # TODO: fix device managment
        # if attn_mask is not None:
        #    attn_mask = attn_mask.to(dtype=q.dtype)

        if self.pos_embedder and pos is not None:
            q, k = self.pos_embedder(q, k, pos)

        q, k = self.q_norm(q), self.k_norm(k)

        arrange_str = "b h n c -> b n h c"
        q = rearrange(q, arrange_str)
        k = rearrange(k, arrange_str)
        v = rearrange(v, arrange_str)
        x = nnx.dot_product_attention(q, k, v, mask=attn_mask)

        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


#################################################################################
#                              FeedForward Layer                                #
#################################################################################


class SwiGLUFeedForward(nnx.Module):
    def __init__(self, dim, hidden_dim, rngs: nnx.Rngs = nnx.Rngs(0), multiple_of=256):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nnx.Linear(dim, hidden_dim, use_bias=False, rngs=rngs)
        self.w2 = nnx.Linear(hidden_dim, dim, use_bias=True, rngs=rngs)
        self.w3 = nnx.Linear(dim, hidden_dim, use_bias=False, rngs=rngs)

        # self.reset_parameters()

    # TODO: Initialization
    # def reset_parameters(self):
    #     torch.nnx.init.xavier_uniform_(self.w1.weight)
    #     torch.nnx.init.xavier_uniform_(self.w2.weight)
    #     torch.nnx.init.xavier_uniform_(self.w3.weight)
    #     if self.w1.bias is not None:
    #         torch.nnx.init.constant_(self.w1.bias, 0)
    #     if self.w2.bias is not None:
    #         torch.nnx.init.constant_(self.w2.bias, 0)
    #     if self.w3.bias is not None:
    #         torch.nnx.init.constant_(self.w3.bias, 0)

    def __call__(self, x):
        return self.w2(nnx.silu(self.w1(x)) * self.w3(x))


#################################################################################
#                               Utility Layers                                  #
#################################################################################


class TimestepEmbedder(nnx.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(
        self,
        hidden_size: int,
        rngs: nnx.Rngs = nnx.Rngs(0),
        frequency_embedding_size: int = 256,
    ) -> None:
        super().__init__()
        self.mlp = nnx.Sequential(
            nnx.Linear(
                in_features=frequency_embedding_size,
                out_features=hidden_size,
                use_bias=True,
                rngs=rngs,
            ),
            nnx.silu,
            nnx.Linear(
                in_features=hidden_size,
                out_features=hidden_size,
                use_bias=True,
                rngs=rngs,
            ),
        )
        self.frequency_embedding_size = frequency_embedding_size
        # self.initialize_weights()

    # TODO: get initialization correct
    # def initialize_weights(self):
    #    nnx.init.normal_(self.mlp[0].weight, std=0.02)
    #    nnx.init.normal_(self.mlp[2].weight, std=0.02)

    @staticmethod
    def timestep_embedding(
        t: jax.Array, dim: int, max_period: int = 10000
    ) -> jax.Array:
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nnx.py
        half = dim // 2
        # TODO: fix device
        freqs = jax.numpy.exp(
            -math.log(max_period)
            * jax.numpy.arange(start=0, stop=half, dtype=jax.numpy.float64)
            / half
        )  # .to(device=t.device)
        args = t[:, None].astype(jax.numpy.float64) * freqs[None]
        embedding = jax.numpy.concatenate(
            [jax.numpy.cos(args), jax.numpy.sin(args)], axis=-1
        )
        if dim % 2:
            embedding = jax.numpy.concatenate(
                [embedding, jax.numpy.zeros_like(embedding[:, :1])], axis=-1
            )
        return embedding

    def __call__(self, t: jax.Array) -> jax.Array:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ConditionEmbedder(nnx.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        dropout_prob: float,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        super().__init__()
        self.proj = nnx.Sequential(
            nnx.Linear(input_dim, hidden_size, rngs=rngs),
            nnx.LayerNorm(
                hidden_size, rngs=rngs, epsilon=1e-5
            ),  # Adjust epsilon for constistency with torch
            nnx.silu,
        )
        self.dropout_prob = dropout_prob
        self.null_token = nnx.Param(
            nnx.initializers.normal(input_dim), requires_grad=True
        )

    def token_drop(self, cond, force_drop_ids=None):
        """
        cond: (B, N, D)
        Drops conditions to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = (
                jax.random.normal(key=jax.random.key(0), shape=cond.shape[0])
                < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids
        cond[drop_ids] = self.null_token[None, None, :]
        return cond

    def __call__(self, cond, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            cond = self.token_drop(cond, force_drop_ids)
        embeddings = self.proj(cond)
        return embeddings


class FinalLayer(nnx.Module):
    """
    The final layer of DiT.
    """

    def __init__(
        self,
        hidden_size: int,
        out_channels: int,
        rngs: nnx.Rngs = nnx.Rngs(0),
        c_dim=None,
    ) -> None:
        super().__init__()
        self.norm_final = nnx.LayerNorm(
            hidden_size, use_scale=False, use_bias=False, epsilon=1e-6, rngs=rngs
        )
        self.linear = nnx.Linear(hidden_size, out_channels, use_bias=True, rngs=rngs)
        self.adaLN_modulation = nnx.Sequential(
            nnx.silu, nnx.Linear(c_dim, 2 * hidden_size, use_bias=True, rngs=rngs)
        )

    # TODO: Initialization
    #     self.initialize_weights()

    # def initialize_weights(self):
    #     # Initialize transformer layers:
    #     def _basic_init(module):
    #         if isinstance(module, nnx.Linear):
    #             torch.nnx.init.xavier_uniform_(module.weight)
    #             if module.bias is not None:
    #                 nnx.init.constant_(module.bias, 0)

    #     self.apply(_basic_init)

    #     # Zero-out output layers:
    #     nnx.init.constant_(self.adaLN_modulation[-1].weight, 0)
    #     nnx.init.constant_(self.adaLN_modulation[-1].bias, 0)
    #     nnx.init.constant_(self.linear.weight, 0)
    #     nnx.init.constant_(self.linear.bias, 0)

    def __call__(self, x: jax.Array, c) -> jax.Array:
        shift, scale = jax.numpy.split(self.adaLN_modulation(c), 2, axis=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class RMSNorm(nnx.Module):
    def __init__(self, d, p=-1.0, epsilon=1e-8, bias=False) -> None:
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param epsilon:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.epsilon = epsilon
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nnx.Param(jax.numpy.ones(d))

        if self.bias:
            self.offset = nnx.Param(jax.numpy.zeros(d))

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.p < 0.0 or self.p > 1.0:
            norm_x = jax.numpy.linalg.norm(x, ord=2, axis=-1, keepdims=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = jax.numpy.split(
                x, [partial_size, self.d - partial_size], axis=-1
            )

            norm_x = jax.numpy.linalg.norm(x=partial_x, ord=2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1.0 / 2)
        x_normed = x / (rms_x + self.epsilon)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed
