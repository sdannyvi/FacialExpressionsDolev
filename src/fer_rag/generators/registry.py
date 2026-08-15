"""The generator registry: one entry per supported checkpoint, plus the lookup over it.

This file answers two questions: *which generators exist* (the ``MODELS`` table) and
*how do I get one* (``get_model_spec`` / ``load_generator``).

It is the single place to edit when adding a generator: add an entry here and the model
becomes available to both pipelines automatically, including in their ``--llava_model_id``
choices. A loader in ``core.py`` is only needed for a checkpoint the Auto classes cannot
resolve.

Keys per entry:

    loader           optional callable(model_id) -> (model, processor). Defaults to
                     ``load_multimodal_lm``, which resolves the model and processor classes
                     from the checkpoint's config. Set it only for a checkpoint that needs
                     non-default loading (``trust_remote_code``, a non-Auto class, ...).
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

from .core import load_multimodal_lm, parse_thinking

MODELS = {
    "llava-hf/llava-v1.6-34b-hf": {
        "style": "two_step",
        "supports_system": True,
        "max_new_tokens": 20,
    },
    "llava-hf/llava-v1.6-mistral-7b-hf": {
        "style": "two_step",
        # The Mistral chat template has no system role and requires user/assistant to
        # alternate, which also rules out the multi-user-message prompt mode.
        "supports_system": False,
        "supports_consecutive_user": False,
        "max_new_tokens": 20,
    },
    "llava-hf/llava-onevision-qwen2-7b-ov-hf": {
        "style": "two_step",
        "supports_system": True,
        "max_new_tokens": 20,
    },
    "google/gemma-3-27b-it": {
        "style": "one_step",
        "supports_system": True,
        "max_new_tokens": 20,
    },
    "google/gemma-4-31B-it": {
        "style": "one_step",
        "supports_system": True,
        "max_new_tokens": 20,
    },
    "Qwen/Qwen3-VL-32B-Instruct": {
        "style": "one_step",
        "supports_system": True,
        "max_new_tokens": 20,
    },
    "Qwen/Qwen3-VL-32B-Thinking": {
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

    model, processor = spec.get("loader", load_multimodal_lm)(model_id)
    model.eval()

    return model, processor, spec
