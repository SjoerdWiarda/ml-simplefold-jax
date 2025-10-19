#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import jax
from flax import nnx

from simplefold.model.jax.layers import SwiGLUFeedForward, modulate


class DiTBlock(nnx.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        self_attention_layer,
        hidden_size: int,
        rngs: nnx.Rngs = nnx.Rngs(0),
        mlp_ratio: float = 4.0,
        use_swiglu: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = nnx.LayerNorm(
            hidden_size, use_scale=False, use_bias=False, epsilon=1e-6, rngs=rngs
        )
        self.attn = self_attention_layer()
        self.norm2 = nnx.LayerNorm(
            hidden_size, use_scale=False, use_bias=False, epsilon=1e-6, rngs=rngs
        )
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        assert use_swiglu, "Need use_swiglu=True for JAX"

        self.mlp = SwiGLUFeedForward(hidden_size, mlp_hidden_dim)

        self.adaLN_modulation = nnx.Sequential(
            nnx.silu, nnx.Linear(hidden_size, 6 * hidden_size, use_bias=True, rngs=rngs)
        )
        # self.initialize_weights()

    # def initialize_weights(self):
    # Initialize transformer layers:

    # TODO: update initialization
    # def _basic_init(module):
    #     if isinstance(module, nnx.Linear):
    #         torch.nnx.init.xavier_uniform_(module.weight)
    #         if module.bias is not None:
    #             nnx.init.constant_(module.bias, 0)

    # self.apply(_basic_init)

    # Zero-out adaLN modulation layers in DiT encoder blocks:
    # nnx.init.constant_(self.adaLN_modulation[-1].weight, 0)
    # nnx.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def __call__(
        self,
        latents: jax.Array,
        c: jax.Array,
        **kwargs,
    ) -> jax.Array:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            jax.numpy.split(self.adaLN_modulation(c), 6, axis=1)
        )
        _latents = self.attn(
            modulate(self.norm1(latents), shift_msa, scale_msa), **kwargs
        )
        latents = latents + jax.numpy.expand_dims(gate_msa, axis=1) * _latents
        latents = latents + jax.numpy.expand_dims(gate_mlp, axis=1) * self.mlp(
            modulate(self.norm2(latents), shift_mlp, scale_mlp)
        )
        return latents


class TransformerBlock(nnx.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        self_attention_layer,
        hidden_size,
        rngs: nnx.Rngs = nnx.Rngs(0),
        mlp_ratio=4.0,
        use_swiglu=False,
    ) -> None:
        super().__init__()
        self.norm1 = nnx.LayerNorm(
            hidden_size, use_scale=False, use_bias=False, epsilon=1e-6, rngs=rngs
        )
        self.attn = self_attention_layer()
        self.norm2 = nnx.LayerNorm(
            hidden_size, use_scale=False, use_bias=False, epsilon=1e-6, rngs=rngs
        )
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        assert use_swiglu, "Need use_swiglu=True for MLX"
        self.mlp = SwiGLUFeedForward(hidden_size, mlp_hidden_dim)

    def __call__(
        self,
        latents: jax.Array,
        **kwargs,
    ) -> jax.Array:
        _latents = self.attn(self.norm1(latents), **kwargs)
        latents = latents + _latents
        latents = latents + self.mlp(self.norm2(latents))
        return latents


class HomogenTrunk(nnx.Module):
    def __init__(self, block, depth: int) -> None:
        super().__init__()
        self.blocks = nnx.List([block() for _ in range(depth)])

    def __call__(self, latents: jax.Array, c: jax.Array, **kwargs) -> jax.Array:
        for i, block in enumerate(self.blocks):
            kwargs["layer_idx"] = i
            latents = block(latents=latents, c=c, **kwargs)
        return latents
