"""Public API of the generator layer.

The pipelines import only the names below, so the internals (loaders, normalizers,
parsers, the registry table) can be reorganized later without touching pipeline code.

    from ..generators import AVAILABLE_MODELS, get_model_spec, load_generator, generate

Typical use:

    model, processor, spec = load_generator(model_id)
    prediction = generate(model, processor, conversation, images, spec)
"""

from .generators import generate
from .registry import AVAILABLE_MODELS, MODELS

__all__ = ["AVAILABLE_MODELS", "get_model_spec", "load_generator", "generate"]


def get_model_spec(model_id):
    """Return the registry entry for ``model_id`` without loading any weights.

    Lets a pipeline validate a configuration (for example a prompt mode the checkpoint
    cannot support) before paying the cost of loading a multi-billion-parameter model.
    """
    if model_id not in MODELS:
        raise ValueError(
            f"Invalid model id '{model_id}': not a supported model. "
            f"Supported models: {', '.join(AVAILABLE_MODELS)}"
        )
    return MODELS[model_id]


def load_generator(model_id):
    """Load a generator checkpoint and return ``(model, processor, spec)``.

    ``spec`` is the registry entry and must be passed back to ``generate``.
    """
    spec = get_model_spec(model_id)

    model, processor = spec["loader"](model_id)
    model.eval()
    print(f"generator model dtype: {next(model.parameters()).dtype}")
    print(f"generator device map: {model.hf_device_map}")

    return model, processor, spec
