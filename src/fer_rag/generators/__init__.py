"""Public API of the generator layer.

The pipelines import only the names below, so the internals (loaders, normalizers,
parsers, the registry table) can be reorganized later without touching pipeline code.

    from ..generators import AVAILABLE_MODELS, get_model_spec, load_generator, generate_prediction

Typical use:

    model, processor, spec = load_generator(model_id)
    prediction, stats = generate_prediction(model, processor, conversation, images, spec)
"""

from .core import generate_prediction
from .registry import AVAILABLE_MODELS, get_model_spec, load_generator

__all__ = ["AVAILABLE_MODELS", "get_model_spec", "load_generator", "generate_prediction"]
