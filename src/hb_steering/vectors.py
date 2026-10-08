"""Compute level-specific harmful-minus-benign decoder output directions."""

import os
import tempfile
from pathlib import Path

from .steering import resolve_layer_index

SCHEMA_VERSION = 1
ACTIVATION_SITE = "decoder_block_output:last_nonpadding_prompt_token"


def compute_vectors(
    runner,
    level_prompts: dict[int, list[str]],
    benign_prompts: list[str],
    layers: list[int],
    *,
    normalize: bool = False,
) -> dict:
    """Subtract one shared benign mean from each level's harmful mean.

    Every prompt has equal weight. A positive intervention moves along the
    harmful-minus-benign direction; a negative coefficient subtracts it.
    This function makes no claim that either sign improves ASR. Layer and
    coefficient are explicit experiment configuration choices.
    """
    import torch

    if not isinstance(normalize, bool):
        raise ValueError("normalize must be a boolean.")
    if not benign_prompts or not level_prompts:
        raise ValueError("Both benign prompts and harmful level prompts are required.")
    selected = [resolve_layer_index(index, len(runner.layers)) for index in layers]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Select at least one unique resolved layer.")
    for level, prompts in level_prompts.items():
        if type(level) is not int or level not in {1, 2, 3} or not prompts:
            raise ValueError("Harmful levels must be 1, 2, or 3, each with at least one prompt.")
    benign = runner.activation_means(benign_prompts, selected)
    vectors = {}
    for level in sorted(level_prompts):
        harmful = runner.activation_means(level_prompts[level], selected)
        vectors[level] = {}
        for layer in selected:
            for name, means in (("benign", benign), ("harmful", harmful)):
                mean = means.get(layer)
                if not isinstance(mean, torch.Tensor) or mean.shape != (runner.hidden_size,):
                    raise ValueError(f"Invalid {name} mean at layer {layer}.")
                if not torch.isfinite(mean).all().item():
                    raise ValueError(f"Nonfinite {name} mean at layer {layer}.")
            vector = harmful[layer].to(device="cpu", dtype=torch.float32) - benign[layer].to(
                device="cpu", dtype=torch.float32
            )
            if normalize:
                norm = torch.linalg.vector_norm(vector)
                if not torch.isfinite(norm).item() or norm.item() == 0:
                    raise ValueError(
                        f"Cannot normalize zero or nonfinite difference at level {level}, layer {layer}."
                    )
                vector = vector / norm
            vectors[level][layer] = vector.contiguous()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "direction": "harmful_minus_benign",
        "activation_site": ACTIVATION_SITE,
        "normalized": normalize,
        "model": {
            "hf_id": runner.hf_id,
            "revision": runner.revision,
            "hidden_size": runner.hidden_size,
        },
        "layers": selected,
        "counts": {
            "benign": len(benign_prompts),
            "levels": {level: len(prompts) for level, prompts in level_prompts.items()},
        },
        "vectors": vectors,
    }
    validate_vector_payload(payload)
    return payload


def validate_vector_payload(payload: dict, *, expected_model: dict | None = None):
    """Validate serialized vectors; optionally require exact model identity."""
    import torch

    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unrecognized vector file schema.")
    if (
        payload.get("direction") != "harmful_minus_benign"
        or payload.get("activation_site") != ACTIVATION_SITE
    ):
        raise ValueError(
            "Vector direction or activation extraction site does not match this implementation."
        )
    if not isinstance(payload.get("normalized"), bool):
        raise ValueError("Vector metadata must declare whether it is normalized.")
    model = payload.get("model", {})
    width = model.get("hidden_size")
    if type(width) is not int or width <= 0 or not model.get("hf_id") or not model.get("revision"):
        raise ValueError("Vector model identity is incomplete.")
    if expected_model is not None:
        for key in ("hf_id", "revision", "hidden_size"):
            if key in expected_model and model.get(key) != expected_model[key]:
                raise ValueError(
                    f"Vector model {key} does not match requested model: {model.get(key)!r} != {expected_model[key]!r}."
                )
    layers = payload.get("layers", [])
    if (
        not layers
        or any(type(layer) is not int or layer < 0 for layer in layers)
        or len(layers) != len(set(layers))
    ):
        raise ValueError("Vector file must contain unique resolved nonnegative layer indices.")
    counts = payload.get("counts", {})
    if type(counts.get("benign")) is not int or counts["benign"] <= 0:
        raise ValueError("Vector file must record a positive benign prompt count.")
    vectors = payload.get("vectors", {})
    level_counts = counts.get("levels", {})
    if not vectors or set(vectors) != set(level_counts):
        raise ValueError("Vector levels and prompt-count levels disagree.")
    for level, layer_vectors in vectors.items():
        if type(level) is not int or level not in {1, 2, 3}:
            raise ValueError("Unexpected severity level in vector file.")
        if type(level_counts[level]) is not int or level_counts[level] <= 0:
            raise ValueError("Each vector level needs a positive prompt count.")
        if set(layer_vectors) != set(layers):
            raise ValueError(f"Vector layers disagree for level {level}.")
        for layer, vector in layer_vectors.items():
            if not isinstance(vector, torch.Tensor) or vector.shape != (width,):
                raise ValueError(f"Invalid vector shape at level {level}, layer {layer}.")
            if vector.dtype != torch.float32 or vector.device.type != "cpu":
                raise ValueError("Stored vectors must be CPU float32 tensors.")
            if not torch.isfinite(vector).all().item():
                raise ValueError(f"Nonfinite vector at level {level}, layer {layer}.")
            if (
                payload["normalized"]
                and not torch.isclose(
                    torch.linalg.vector_norm(vector), torch.tensor(1.0), atol=1e-5
                ).item()
            ):
                raise ValueError("Normalized vector metadata disagrees with its L2 norm.")
    return payload


def save_vectors(payload: dict, path):
    """Atomically save a validated tensor-and-primitive-only file."""
    import torch

    validate_vector_payload(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_vectors(path, *, expected_model: dict | None = None) -> dict:
    """Never unpickle arbitrary classes when reading tensor artifacts."""
    import torch

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    return validate_vector_payload(payload, expected_model=expected_model)
