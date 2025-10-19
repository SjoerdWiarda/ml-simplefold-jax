#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import jax
from flax import nnx


def compute_aggregated_metric(logits: jax.Array, end=1.0) -> jax.Array:
    """Compute the metric from the logits.

    Parameters
    ----------
    logits : jax.Array
        The logits of the metric
    end : float
        Max value of the metric, by default 1.0

    Returns
    -------
    Tensor
        The metric value

    """
    num_bins = logits.shape[-1]
    bin_width = end / num_bins
    bounds = jax.numpy.arange(
        start=0.5 * bin_width, stop=end, step=bin_width, device=logits.device
    )
    probs = nnx.softmax(logits, axis=-1)
    plddt = jax.numpy.sum(
        probs * bounds.view(*((1,) * len(probs.shape[:-1])), *bounds.shape),
        axis=-1,
    )
    return plddt


class ConfidenceModule(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        transformer_blocks,
        rngs: nnx.Rngs = nnx.Rngs(0),
        num_plddt_bins: int = 50,
    ):
        super().__init__()
        self.transformer_blocks = transformer_blocks
        self.to_plddt_logits = nnx.Sequential(
            nnx.Linear(hidden_size, hidden_size, rngs=rngs),
            nnx.LayerNorm(
                hidden_size, rngs=rngs, epsilon=1e-5
            ),  # Adjust epsilon for constistency with torch
            nnx.silu,
            nnx.Linear(hidden_size, num_plddt_bins, rngs=rngs),
        )

    def __call__(
        self,
        latent,
        feats,
    ):
        token_pe_pos = jax.numpy.concatenate(
            [
                jax.numpy.expand_dims(feats["residue_index"], axis=-1).astype(
                    latent.dtype
                ),  # (B, M, 1)
                jax.numpy.expand_dims(feats["entity_id"], axis=-1).astype(
                    latent.dtype
                ),  # (B, M, 1)
                jax.numpy.expand_dims(feats["asym_id"], axis=-1).astype(
                    latent.dtype
                ),  # (B, M, 1)
                jax.numpy.expand_dims(feats["sym_id"], axis=-1).astype(
                    latent.dtype
                ),  # (B, M, 1)
            ],
            axis=-1,
        )

        latent = self.transformer_blocks(
            latents=latent,
            c=None,
            pos=token_pe_pos,
        )

        # Compute the pLDDT
        plddt_logits = self.to_plddt_logits(latent)

        # Compute the aggregated pLDDT
        plddt = compute_aggregated_metric(plddt_logits)

        return dict(
            plddt=plddt,
            plddt_logits=plddt_logits,
        )
