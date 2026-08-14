"""The generator registry: one entry per supported checkpoint.

This is the single place to edit when adding a generator. Add an entry here (and a
loader in ``generators.py`` if the family is new) and the model becomes available to
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

from .generators import (
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
