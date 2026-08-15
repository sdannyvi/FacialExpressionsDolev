"""The generator registry: one entry per supported checkpoint, plus the lookup over it.

This file answers two questions: *which generators exist* (the ``MODELS`` table) and
*how do I get one* (``get_model_spec`` / ``load_generator``).

It is the single place to edit when adding a generator. Add an entry here (and a
loader in ``core.py`` if the family is new) and the model becomes available to
both pipelines automatically, including in their ``--llava_model_id`` choices.

Keys per entry:

    loader           callable(model_id) -> (model, processor)
    style            "two_step"  - apply_chat_template(tokenize=False), images passed
                                   separately to the processor (LLaVA family)
                     "one_step"  - apply_chat_template(tokenize=True, return_dict=True)
                                   with the PIL image inline on the content item
    supports_system  whether the chat template accepts a system role. When False the
                     system text is folded into the first user message.
    max_new_tokens   generation budget. Optional; defaults to DEFAULT_MAX_NEW_TOKENS.
    parse            optional callable(str) -> str applied to the decoded output.
    supports_consecutive_user
                     whether the template tolerates several user messages in a row.
                     Optional, defaults True. False rules out the RAG pipeline's
                     "multi-user-message" prompt mode.
"""

from .core import (
    load_gemma3,
    load_gemma4,
    load_llava_next,
    load_llava_onevision,
    load_qwen3_vl,
    parse_thinking,
)

MODELS = {
    "llava-hf/llava-v1.6-34b-hf": {
        "loader": load_llava_next,
        "style": "two_step",
        "supports_system": True,
        "max_new_tokens": 20,
    },
    "llava-hf/llava-v1.6-mistral-7b-hf": {
        "loader": load_llava_next,
        "style": "two_step",
        # The Mistral chat template has no system role and requires user/assistant to
        # alternate, which also rules out the multi-user-message prompt mode.
        "supports_system": False,
        "supports_consecutive_user": False,
        "max_new_tokens": 20,
    },
    "llava-hf/llava-onevision-qwen2-7b-ov-hf": {
        "loader": load_llava_onevision,
        "style": "two_step",
        "supports_system": True,
        "max_new_tokens": 20,
    },
    "google/gemma-3-27b-it": {
        "loader": load_gemma3,
        "style": "one_step",
        "supports_system": True,
        "max_new_tokens": 20,
    },
    "google/gemma-4-31B-it": {
        "loader": load_gemma4,
        "style": "one_step",
        "supports_system": True,
        "max_new_tokens": 20,
    },
    "Qwen/Qwen3-VL-32B-Instruct": {
        "loader": load_qwen3_vl,
        "style": "one_step",
        "supports_system": True,
        "max_new_tokens": 20,
    },
    "Qwen/Qwen3-VL-32B-Thinking": {
        "loader": load_qwen3_vl,
        "style": "one_step",
        "supports_system": True,
        # Reasoning is emitted before the answer, so the 20-token budget used by the
        # other checkpoints would return truncated reasoning instead of a label.
        "max_new_tokens": 1024,
        "parse": parse_thinking,
    },
}

AVAILABLE_MODELS = sorted(MODELS)


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

    ``spec`` is the registry entry and must be passed back to ``generate_prediction``.

    Deliberately silent: this is library code, so the calling pipeline decides what to
    report. Both pipelines print the loaded model's dtype and device map right after this
    call, tagged ``[generators.registry.load_generator]``. See
    ``docs/use_logging_recommendation.md`` for turning those prints into proper logging.
    """
    spec = get_model_spec(model_id)

    model, processor = spec["loader"](model_id)
    model.eval()

    return model, processor, spec
