from collections import defaultdict
from typing import Any

import jax
import torch


# We redefine the centering function here using jax primitives
def center_random_augmentation(
    atom_coords: jax.Array,
    atom_mask: jax.Array,
    s_trans: float = 1.0,
    augmentation: bool = True,
    centering: bool = True,
    return_second_coords: bool = False,
    second_coords: bool = None,
) -> jax.Array:
    """Center and randomly augment the input coordinates.

    Parameters
    ----------
    atom_coords : Tensor
        The atom coordinates.
    atom_mask : Tensor
        The atom mask.
    s_trans : float, optional
        The translation factor, by default 1.0
    augmentation : bool, optional
        Whether to add rotational and translational augmentation the input, by default True
    centering : bool, optional
        Whether to center the input, by default True

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

    return atom_coords


def try_convert_int(x):
    try:
        return int(x)
    except ValueError:
        return x


def unflatten_state_dict(flat_dict):
    """Convert {'a.b.c': x} into {'a': {'b': {'c': x}}}."""
    nested = {}
    for key, value in flat_dict.items():
        parts = key.split(".")
        d = nested
        for p in parts[:-1]:
            if p.isdigit():
                p = int(p)
            d = d.setdefault(p, {})
        d[parts[-1]] = value
    return nested


def convert_to_jax_dict(torch_dict: dict) -> dict[str, Any]:

    new_dict = {}

    def tensor_to_jax_array(x: Any) -> Any:
        if isinstance(x, torch.Tensor):
            return jax.numpy.asarray(x.numpy())
        else:
            return x

    for key, value in torch_dict.items():
        new_dict[key] = tensor_to_jax_array(value)

    return new_dict


def replace_by_torch_dict(
    state,
    pure_dict: dict[str, Any],
) -> dict[str, Any]:

    new_dict = {}
    for kp, v in pure_dict.items():
        if isinstance(v, dict):
            if kp == "embed_tokens":
                assert kp in state.keys()
                new_dict[kp] = jax.numpy.asarray(v["weight"].numpy())
            elif kp in state:
                new_dict[kp] = replace_by_torch_dict(state[kp], v)
            elif try_convert_int(kp) in state:
                new_dict[try_convert_int(kp)] = replace_by_torch_dict(
                    state[try_convert_int(kp)], v
                )
            elif "layers" in state:
                new_dict["layers"] = replace_by_torch_dict(state["layers"], pure_dict)
            else:
                raise ValueError(
                    f"Following key is not recognized in state: {kp}, available keys in state: {set(state.keys())}"
                )
        else:
            assert isinstance(v, torch.Tensor)
            if kp in state:
                new_dict[kp] = jax.numpy.asarray(v.numpy())
            elif try_convert_int(kp) in state:
                new_dict[try_convert_int(kp)] = jax.numpy.asarray(v.numpy())
            elif "kernel" in state:
                new_dict["kernel"] = jax.numpy.asarray(v.numpy()).T
            elif "scale" in state:
                new_dict["scale"] = jax.numpy.asarray(v.numpy())
            else:
                raise ValueError(
                    f"Following key is not recognized in state: {kp}, available keys in state: {set(state.keys())}"
                )

    return new_dict
