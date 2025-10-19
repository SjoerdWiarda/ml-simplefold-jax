#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

from typing import TYPE_CHECKING

import jax
from einops import repeat
from tqdm import tqdm

from simplefold.model.flow import LinearPath
from simplefold.model.jax.architecture import FoldingDiT
from simplefold.utils.jax_boltz_utils import center_random_augmentation


class EMSampler:
    """
    A Euler-Maruyama solver for SDEs.
    """

    def __init__(
        self,
        num_timesteps: int = 500,
        t_start: float = 1e-4,
        tau: float = 0.3,
        log_timesteps: bool = False,
        w_cutoff: float = 0.99,
    ):
        self.num_timesteps = num_timesteps
        self.log_timesteps = log_timesteps
        self.t_start = t_start
        self.tau = tau
        self.w_cutoff = w_cutoff

        if self.log_timesteps:
            t = 1.0 - jax.numpy.logspace(-2, 0, self.num_timesteps + 1)[::-1, ...]
            t = t - jax.numpy.min(t)
            t = t / jax.numpy.max(t)
            self.steps = jax.numpy.clip(t, min=self.t_start, max=1.0)
        else:
            self.steps = jax.numpy.linspace(
                start=self.t_start, stop=1.0, num=self.num_timesteps + 1
            )

    def diffusion_coefficient(
        self, t: jax.Array, epsilon: float = 0.01
    ) -> float | jax.Array:
        # determine diffusion coefficient
        w = (1.0 - t) / (t + epsilon)
        if t >= self.w_cutoff:
            w = 0.0
        return w

    def euler_maruyama_step(
        self,
        model_fn: FoldingDiT,
        flow: LinearPath,
        y: jax.Array,
        t: jax.Array,
        t_next: jax.Array,
        batch,
        rng_key: jax.Array,
    ) -> jax.Array:
        dt = t_next - t

        eps_key, aug_key = jax.random.split(rng_key, num=2)
        epsilon = jax.random.normal(key=eps_key, shape=y.shape)

        y = center_random_augmentation(
            y,
            batch["atom_pad_mask"],
            augmentation=False,
            centering=True,
            rng_key=aug_key,
        )

        batched_t = repeat(t, " -> b", b=y.shape[0])
        velocity = model_fn(
            noised_pos=y,
            t=batched_t,
            feats=batch,
        )["predict_velocity"]
        score = flow.compute_score_from_velocity(velocity, y, t)

        diff_coeff = self.diffusion_coefficient(t)
        drift = velocity + diff_coeff * score
        mean_y = y + drift * dt
        y_sample = mean_y + jax.numpy.sqrt(2.0 * dt * diff_coeff * self.tau) * epsilon

        return y_sample

    def sample(
        self, model_fn: FoldingDiT, flow: LinearPath, noise: jax.Array, batch
    ) -> dict[str, jax.Array]:
        sampling_timesteps = self.num_timesteps
        # TODO: fix device management
        steps = self.steps
        y_sampled = noise
        feats = batch

        rng_key = jax.random.split(jax.random.key(0), sampling_timesteps)
        for i in tqdm(
            range(sampling_timesteps),
            desc="Sampling",
            total=sampling_timesteps,
        ):
            t = steps[i]
            t_next = steps[i + 1]

            y_sampled = self.euler_maruyama_step(
                model_fn, flow, y_sampled, t, t_next, feats, rng_key=rng_key[i]
            )

        return {"denoised_coords": y_sampled}
