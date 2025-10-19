#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

# Started from https://github.com/facebookresearch/esm/tree/main,
# licensed under MIT License, Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Optional

import jax
from flax import nnx
from flax.nnx import LayerNorm as ESM1bLayerNorm

from simplefold.model.jax.esm_multihead_attention import MultiheadAttention  # noqa


def symmetrize(x):
    "Make layer symmetric in final two dimensions, used for contact prediction."
    return x + jax.numpy.swapaxes(x, axis1=-1, axis2=-2)


def apc(x):
    "Perform average product correct, used for contact prediction."
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)

    avg = a1 * a2
    avg = jax.numpy.divide(avg, a12)
    normalized = x - avg
    return normalized


class ESM1LayerNorm(nnx.Module):
    def __init__(
        self, hidden_size, eps=1e-12, affine=True, rngs: nnx.Rngs = nnx.Rngs(0)
    ):
        """Construct a layernorm layer in the TF style (eps inside the sqrt)."""
        super().__init__()
        self.hidden_size = (
            (hidden_size,) if isinstance(hidden_size, int) else tuple(hidden_size)
        )
        self.eps = eps
        self.affine = bool(affine)
        if self.affine:
            self.weight = nnx.Param(jax.numpy.array(jax.numpy.ones(hidden_size)))
            self.bias = nnx.Param(jax.numpy.array(jax.numpy.zeros(hidden_size)))
        else:
            self.weight, self.bias = None, None

    def __call__(self, x):
        dims = tuple(-(i + 1) for i in range(len(self.hidden_size)))
        means = jax.numpy.mean(x, axis=dims, keepdims=True)
        x_zeromean = x - means
        variances = jax.numpy.mean(
            jax.numpy.pow(x_zeromean, 2), axis=dims, keepdims=True
        )
        x = x_zeromean / jax.numpy.sqrt(variances + self.eps)
        if self.affine:
            x = (self.weight * x) + self.bias
        return x


class TransformerLayer(nnx.Module):

    def __init__(
        self,
        embed_dim: int,
        ffn_embed_dim: int,
        attention_heads: int,
        add_bias_kv: bool = True,
        use_esm1b_layer_norm: bool = False,  # This is true in the implementation
        use_rotary_embeddings: bool = False,  # This is true in the implementation
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.use_rotary_embeddings = use_rotary_embeddings
        self._init_submodules(add_bias_kv, use_esm1b_layer_norm, rngs)

    def _init_submodules(
        self,
        add_bias_kv: bool,
        use_esm1b_layer_norm: bool,
        rngs: nnx.Rngs,
    ):
        BertLayerNorm = ESM1bLayerNorm if use_esm1b_layer_norm else ESM1LayerNorm

        self.self_attn = MultiheadAttention(
            self.embed_dim,
            self.attention_heads,
            add_bias_kv=add_bias_kv,
            add_zero_attn=False,
            use_rotary_embeddings=self.use_rotary_embeddings,
            rngs=rngs,
        )
        self.self_attn_layer_norm = BertLayerNorm(self.embed_dim, rngs=rngs)

        self.fc1 = nnx.Linear(self.embed_dim, self.ffn_embed_dim, rngs=rngs)
        self.fc2 = nnx.Linear(self.ffn_embed_dim, self.embed_dim, rngs=rngs)

        self.final_layer_norm = BertLayerNorm(self.embed_dim, rngs=rngs)

    def __call__(
        self,
        x,
        self_attn_mask=None,
        self_attn_padding_mask=None,
        need_head_weights=False,
    ):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=True,
            need_head_weights=need_head_weights,
            attn_mask=self_attn_mask,
        )
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = self.fc1(x)
        x = nnx.gelu(
            x, approximate=False
        )  # use approximate = False, otherwise we get upt to 1% error w.r.t the output of the torch ESM module
        x = self.fc2(x)
        x = residual + x

        return x, attn


class RobertaLMHead(nnx.Module):
    """Head for masked language modeling."""

    def __init__(
        self,
        embed_dim: int,
        output_dim: int,
        weight: jax.Array,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ) -> None:
        super().__init__()
        self.dense = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        self.layer_norm = ESM1bLayerNorm(embed_dim, rngs=rngs)
        self.weight = nnx.Param(
            weight
        )  # Made a variable twice, in the torch dict it is saved twice
        self.bias = nnx.Param(jax.numpy.array(jax.numpy.zeros(output_dim)))

    def __call__(self, features: jax.Array) -> jax.Array:
        x = self.dense(features)
        x = nnx.gelu(
            x, approximate=False
        )  # use approximate = False, otherwise we get upt to 1% error w.r.t the output of the torch ESM module
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = (
            jax.numpy.matmul(x, jax.numpy.swapaxes(self.weight, axis1=0, axis2=1))
            + self.bias
        )
        return x


class ContactPredictionHead(nnx.Module):
    """Performs symmetrization, apc, and computes a logistic regression on the output features"""

    def __init__(
        self,
        in_features: int,
        prepend_bos: bool,
        append_eos: bool,
        bias=True,
        eos_idx: Optional[int] = None,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        super().__init__()
        self.in_features = in_features
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        if append_eos and eos_idx is None:
            raise ValueError(
                "Using an alphabet with eos token, but no eos token was passed in."
            )
        self.eos_idx = eos_idx
        self.regression = nnx.Linear(in_features, 1, use_bias=bias, rngs=rngs)
        self.activation = nnx.sigmoid

    def __call__(self, tokens, attentions):
        # remove eos token attentions
        if self.append_eos:
            eos_mask = tokens.ne(self.eos_idx)
            eos_mask = eos_mask[:, None, ...] * eos_mask[:, :, None, ...]
            attentions = attentions * eos_mask[:, None, None, :, :]
            attentions = attentions[..., :-1, :-1]
        # remove cls token attentions
        if self.prepend_bos:
            attentions = attentions[..., 1:, 1:]
        batch_size, layers, heads, seqlen, _ = attentions.shape
        attentions = attentions.reshape(batch_size, layers * heads, seqlen, seqlen)

        # features: B x C x T x T
        attentions = apc(symmetrize(attentions))
        attentions = attentions.transpose(0, 2, 3, 1)
        return self.activation(self.regression(attentions).squeeze(3))
