#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

# Started from https://github.com/jwohlwend/boltz,
# licensed under MIT License, Copyright (c) 2024 Jeremy Wohlwend, Gabriele Corso, Saro Passaro.

# Started from code from https://github.com/lucidrains/alphafold3-pytorch,
# licensed under MIT License, Copyright (c) 2024 Phil Wang


from typing import Optional

import jax


def exists(v):
    return v is not None


def randomly_rotate(
    coords: jax.Array,
    return_second_coords: bool = False,
    second_coords: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array | None] | jax.Array:
    R = random_rotations(len(coords), coords.dtype)  # , coords.device)

    if return_second_coords:
        return jax.numpy.einsum("bmd,bds->bms", coords, R), (
            jax.numpy.einsum("bmd,bds->bms", second_coords, R)
            if second_coords is not None
            else None
        )

    return jax.numpy.einsum("bmd,bds->bms", coords, R)


def center_random_augmentation(
    atom_coords: jax.Array,
    atom_mask: jax.Array,
    rng_key: jax.Array,
    s_trans: float = 1.0,
    augmentation: bool | None = True,
    centering: bool | None = True,
    return_second_coords: bool | None = False,
    second_coords: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array | None] | jax.Array:
    """Center and randomly augment the input coordinates.

    Parameters
    ----------
    atom_coords : jax.Array
        The atom coordinates.
    atom_mask : jax.Array
        The atom mask.
    s_trans : float, optional
        The translation factor, by default 1.0
    augmentation : bool, optional
        Whether to add rotational and translational augmentation the input, by default True
    centering : bool, optional
        Whether to center the input, by default True
    return_second_coords : bool, optional
    second_coords : bool, optional

    Returns
    -------
    Tensor
        The augmented atom coordinates.

    """
    if centering:
        atom_mean = jax.numpy.sum(
            atom_coords * atom_mask[:, :, None], axis=1, keepdims=True
        ) / jax.numpy.sum(atom_mask[:, :, None], axis=1, keepdims=True)
        atom_coords = atom_coords - atom_mean

        if second_coords is not None:
            # apply same transformation also to this input
            second_coords = second_coords - atom_mean

    if augmentation:
        atom_coords, second_coords = randomly_rotate(
            atom_coords, return_second_coords=True, second_coords=second_coords
        )
        # TODO: fix key
        random_trans = (
            jax.random.normal(key=rng_key, shape=atom_coords[:, 0:1, :].shape) * s_trans
        )
        atom_coords = atom_coords + random_trans

        if second_coords is not None:
            second_coords = second_coords + random_trans

    if return_second_coords:
        return atom_coords, second_coords

    return atom_coords


# the following is copied from Torch3D, BSD License, Copyright (c) Meta Platforms, Inc. and affiliates.


def _copysign(a: jax.Array, b: jax.Array) -> jax.Array:
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return jax.numpy.where(signs_differ, -a, a)


def quaternion_to_matrix(quaternions: jax.Array) -> jax.Array:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = jax.numpy.split(quaternions, 4, axis=-1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = jax.numpy.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


# TODO: fix device
def random_quaternions(
    n: int,
    dtype: Optional[jax.numpy.dtype] = None,  # device: Optional[Device] = None
) -> jax.Array:
    """
    Generate random quaternions representing rotations,
    i.e. versors with nonnegative real part.

    Args:
        n: Number of quaternions in a batch to return.
        dtype: Type to return.
        device: Desired device of returned tensor. Default:
            uses the current device for the default tensor type.

    Returns:
        Quaternions as tensor of shape (N, 4).
    """
    # if isinstance(device, str):
    #    device = torch.device(device)
    # TODO: fix key
    o = jax.random.normal(
        key=jax.random.key(0), shape=(n, 4), dtype=dtype
    )  # , device=device)
    s = (o * o).sum(1)
    o = o / _copysign(jax.numpy.sqrt(s), o[:, 0])[:, None]
    return o


# TODO: fix device
def random_rotations(
    n: int,
    dtype: Optional[jax.numpy.dtype] = None,  # device: Optional[Device] = None
) -> jax.Array:
    """
    Generate random rotations as 3x3 rotation matrices.

    Args:
        n: Number of rotation matrices in a batch to return.
        dtype: Type to return.
        device: Device of returned tensor. Default: if None,
            uses the current device for the default tensor type.

    Returns:
        Rotation matrices as tensor of shape (n, 3, 3).
    """
    quaternions = random_quaternions(n, dtype=dtype)  # device=device)
    return quaternion_to_matrix(quaternions)
