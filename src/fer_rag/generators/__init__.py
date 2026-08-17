"""Public API of the generator layer.

The pipelines import only the names below, so the internals (loaders, normalizers,
the registry table) can be reorganized later without touching pipeline code.

    from ..generators import AVAILABLE_MODELS, get_model_spec, load_generator, generate_prediction

Typical use:

    model, processor, spec = load_generator(model_id)
    prediction, thinking, stats = generate_prediction(model, processor, conversation, images, spec,
                                                      enable_thinking=False)

``thinking`` is None unless the run actually reasoned. Two functions cover everything a
pipeline needs to know about reasoning, and neither requires it to read a registry key:

    validate_thinking_request(model_id, spec, flag)  may this run reason?  raises if not
    resolve_thinking(spec, flag)                     will it?              returns a bool

``thinking_models(mode)`` names the checkpoints of one kind ("optional" or "always"), for
help text.
"""

from .core import generate_prediction, resolve_thinking
from .registry import (
    AVAILABLE_MODELS,
    get_model_spec,
    load_generator,
    thinking_models,
    validate_thinking_request,
    validate_prompt_request,
)

__all__ = [
    "AVAILABLE_MODELS",
    "get_model_spec",
    "load_generator",
    "generate_prediction",
    "resolve_thinking",
    "thinking_models",
    "validate_thinking_request",
    "validate_prompt_request",
]
