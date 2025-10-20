#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

# Started from https://github.com/facebookresearch/esm/tree/main,
# licensed under MIT License, Copyright (c) Meta Platforms, Inc. and affiliates.

import jax
from flax import nnx


def rotate_half(x: jax.Array) -> jax.Array:
    x1, x2 = jax.numpy.split(x, 2, axis=-1)
    return jax.numpy.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    cos = cos[:, : x.shape[-2], :]
    sin = sin[:, : x.shape[-2], :]

    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(nnx.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.
    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration
    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox
    .. warning: Please note that this embedding is not registered on purpose, as it is transformative
        (it does not create the embedding dimension) and will likely be picked up (imported) on a ad-hoc basis
    """

    def __init__(self, dim: int, *_, **__):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        self.inv_freq = nnx.Variable(
            jax.numpy.array(
                1.0
                / (
                    10000
                    ** (jax.numpy.arange(0, dim, 2).astype(jax.numpy.float32) / dim)
                )
            )
        )

        self._seq_len_cached = None
        self._cos_cached = nnx.data(None)
        self._sin_cached = nnx.data(None)

    def _update_cos_sin_tables(self, x: jax.Array, seq_dimension: int = 1):
        seq_len = x.shape[seq_dimension]

        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if seq_len != self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = jax.numpy.arange(x.shape[seq_dimension]).astype(self.inv_freq.dtype)
            freqs = jax.numpy.einsum("i,j->ij", t, self.inv_freq)
            emb = jax.numpy.concatenate([freqs, freqs], axis=-1)

            self._cos_cached = jax.numpy.cos(emb)[None, :, :]
            self._sin_cached = jax.numpy.sin(emb)[None, :, :]

        return self._cos_cached, self._sin_cached

    def __call__(self, q: jax.Array, k: jax.Array):
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(
            k, seq_dimension=-2
        )

        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
        )
