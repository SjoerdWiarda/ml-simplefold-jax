#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import math

import jax
from flax import nnx


class AbsolutePositionEncoding(nnx.Module):
    def __init__(
        self, in_dim: int, embed_dim: int, include_input: bool = False
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = embed_dim
        self.include_input = include_input
        assert embed_dim % in_dim == 0, "embed_dim must be divisible by in_dim"
        self.embed_dim = embed_dim + in_dim if include_input else embed_dim

    def __call__(self, pos: jax.Array) -> jax.Array:
        pos_embs = []
        for i in range(self.in_dim):
            pe = self.get_1d_pos_embed(pos[..., i])
            pos_embs.append(pe)
        if self.include_input:
            pos_embs.append(pos)
        pos_embs = jax.numpy.concatenate(pos_embs, axis=-1)
        return pos_embs

    def get_1d_pos_embed(self, pos: jax.Array) -> jax.Array:
        """
        https://github.com/facebookresearch/DiT/blob/main/models.py#L303
        """
        embed_dim = self.hidden_dim // (self.in_dim * 2)
        # TODO: fix device
        omega = 2 ** jax.numpy.linspace(0, math.log(224, 2) - 1, embed_dim)
        # .to(
        #    pos.device
        # )
        omega *= jax.numpy.pi

        if len(pos.shape) == 1:
            out = jax.numpy.einsum("m,d->md", pos, omega)  # (M, D/2), outer product
        elif len(pos.shape) == 2:
            out = jax.numpy.einsum("nm,d->nmd", pos, omega)

        emb_sin = jax.numpy.sin(out)  # (*, M, D/2)
        emb_cos = jax.numpy.cos(out)  # (*, M, D/2)
        emb = jax.numpy.concatenate([emb_sin, emb_cos], axis=-1)  # (*, M, D)
        return emb


class FourierPositionEncoding(nnx.Module):
    def __init__(
        self,
        in_dim: int,
        include_input: bool = False,
        min_freq_log2: float = 0,
        max_freq_log2: float = 12,
        num_freqs: int = 32,
        log_sampling: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.include_input = include_input
        self.min_freq_log2 = min_freq_log2
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        self.log_sampling = log_sampling
        self.create_embedding_fn()

    def create_embedding_fn(self) -> None:
        d = self.in_dim
        dim_out = 0
        if self.include_input:
            dim_out += d

        min_freq = self.min_freq_log2
        max_freq = self.max_freq_log2
        N_freqs = self.num_freqs

        if self.log_sampling:
            freq_bands = 2.0 ** jax.numpy.linspace(
                min_freq, max_freq, num=N_freqs
            )  # (nf,)
        else:
            freq_bands = jax.numpy.linspace(
                2.0**min_freq, 2.0**max_freq, num=N_freqs
            )  # (nf,)

        assert jax.numpy.isfinite(
            freq_bands
        ).all(), f"nan: {jax.numpy.isnan(freq_bands).any()} inf: {jax.numpy.isfinite(freq_bands).any()}"

        self.freq_bands = nnx.Variable(freq_bands)  # (nf,)
        self.embed_dim = dim_out + d * self.freq_bands.size * 2

    def __call__(
        self,
        pos: jax.Array,
    ) -> jax.Array:
        """
        Get the positional encoding for each coordinate.
        Args:
            pos:
                (*, in_dim)
        Returns:
            out:
                (*, in_dimitional_encoding)
        """

        out = []
        if self.include_input:
            out = [pos]  # (B, in_dim, 3)
        pos = (
            jax.numpy.expand_dims(pos, axis=-1) * self.freq_bands
        )  # (B, in_dim, 3, nf)

        out += [
            jax.numpy.sin(pos).reshape(pos.shape[:2] + (-1,)),  # (B, in_dim, 3*nf)
            jax.numpy.cos(pos).reshape(pos.shape[:2] + (-1,)),  # (B, in_dim, 3*nf)
        ]

        out = jax.numpy.concatenate(out, axis=-1)  # (B, in_dim, (2 * nf + 1 ) * 3 )
        return out


def compute_axial_cis(
    ts: jax.Array,
    in_dim: int,
    dim: int,
    theta: float = 100.0,
) -> jax.Array:
    B, N, D = ts.shape
    freqs_all = []
    interval = 2 * in_dim
    for i in range(in_dim):
        # TODO: fix device
        freq = 1.0 / (
            theta
            ** (
                jax.numpy.arange(0, dim, interval)[: (dim // interval)].astype(ts.dtype)
                / dim
            )
        )  # .to(ts.device)
        t = ts[..., i].flatten()
        freq_i = jax.numpy.outer(t, freq)
        freq_cis_i = jax.numpy.exp(1j * freq_i)
        freq_cis_i = freq_cis_i.reshape(B, N, -1)
        freqs_all.append(freq_cis_i)
    freqs_cis = jax.numpy.concatenate(freqs_all, axis=-1)
    return freqs_cis


def apply_rotary_emb(
    xq: jax.Array, xk: jax.Array, freqs_cis: jax.Array
) -> tuple[jax.Array, jax.Array]:

    xq_real, xq_imag = jax.numpy.split(xq.reshape(*xq.shape[:-1], -1, 2), 2, axis=-1)
    xq_ = xq_real + 1j * xq_imag

    xk_real, xk_imag = jax.numpy.split(xk.reshape(*xk.shape[:-1], -1, 2), 2, axis=-1)
    xk_ = xk_real + 1j * xk_imag

    modulated_xq = xq_.reshape(xq_.shape[:-1]) * freqs_cis
    modulated_xk = xk_.reshape(xk_.shape[:-1]) * freqs_cis
    xq_out = jax.numpy.stack([modulated_xq.real, modulated_xq.imag], axis=-1).reshape(
        modulated_xq.shape[:3] + (-1,)
    )
    xk_out = jax.numpy.stack([modulated_xk.real, modulated_xk.imag], axis=-1).reshape(
        modulated_xk.shape[:3] + (-1,)
    )
    # TODO: fix device
    return xq_out, xk_out


class AxialRotaryPositionEncoding(nnx.Module):
    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        num_heads: int,
        base: float = 100.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.embed_dim = embed_dim // num_heads
        self.base = base

    def __call__(
        self, xq: jax.Array, xk: jax.Array, pos: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """
        xq: [B, H, N, D]
        xk: [B, H, N, D]
        pos: [B, N, in_dim]
        """
        if pos.ndim == 2:
            pos = jax.numpy.expand_dims(pos, axis=-1)
        freqs_cis = compute_axial_cis(pos, self.in_dim, self.embed_dim, self.base)
        freqs_cis = jax.numpy.expand_dims(freqs_cis, axis=1)
        # TODO: device management
        return apply_rotary_emb(xq, xk, freqs_cis)
