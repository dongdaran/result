"""Add a specified direction at the output of one decoder block.

The vector sign is never guessed or inverted. ``coefficient > 0`` adds the
supplied direction; a negative coefficient subtracts it. Raw vectors preserve
activation units; ``normalize=True`` instead makes the coefficient the added
L2 magnitude. Hooks affect every token in prompt and autoregressive forwards.
"""

from contextlib import contextmanager
from math import isfinite


def resolve_layer_index(index: int, count: int) -> int:
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("Layer index must be an integer.")
    resolved = count + index if index < 0 else index
    if not 0 <= resolved < count:
        raise ValueError(f"Layer {index} is outside a model with {count} decoder blocks.")
    return resolved


def block_hidden(output):
    """Return the residual stream from a decoder block's public forward output."""
    import torch

    hidden = output[0] if isinstance(output, tuple) else output
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
        raise TypeError(
            "Expected decoder block output Tensor[B, T, H], or a tuple beginning with it."
        )
    return hidden


@contextmanager
def steering_hook(layers, layer_index: int, vector, coefficient: float, *, normalize: bool = False):
    """Temporarily add ``coefficient * vector`` to a block output.

    Extraction uses this same output site, before the next block or final norm.
    The vector must be finite, one-dimensional, and match the hidden dimension.
    Hooks are always removed, including when model forward raises an exception.
    """
    import torch

    layer_index = resolve_layer_index(layer_index, len(layers))
    if not isinstance(vector, torch.Tensor) or vector.ndim != 1 or vector.numel() == 0:
        raise ValueError("Steering vector must be a nonempty one-dimensional torch.Tensor.")
    if not torch.isfinite(vector).all().item():
        raise ValueError("Steering vector contains NaN or infinity.")
    coefficient = float(coefficient)
    if not isfinite(coefficient):
        raise ValueError("Steering coefficient must be finite.")
    direction = vector.detach().to(device="cpu", dtype=torch.float32)
    if normalize:
        norm = torch.linalg.vector_norm(direction)
        if not torch.isfinite(norm).item() or norm.item() == 0:
            raise ValueError("Cannot normalize a zero or nonfinite steering vector.")
        direction = direction / norm

    def apply(_module, _inputs, output):
        hidden = block_hidden(output)
        if hidden.shape[-1] != direction.numel():
            raise ValueError(
                f"Vector width {direction.numel()} does not match block width {hidden.shape[-1]}."
            )
        delta = direction.to(device=hidden.device, dtype=hidden.dtype) * coefficient
        if not torch.isfinite(delta).all().item():
            raise ValueError("Steering delta overflowed the model dtype.")
        updated = hidden + delta
        return (updated, *output[1:]) if isinstance(output, tuple) else updated

    # Do not perform an addition for a zero intervention, preserving bitwise output.
    handle = None if coefficient == 0 else layers[layer_index].register_forward_hook(apply)
    try:
        yield
    finally:
        if handle is not None:
            handle.remove()
