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
    supports_consecutive_user
                     whether the template tolerates several user messages in a row.
                     Optional, defaults True. False rules out the RAG pipeline's
                     "multi-user-message" prompt mode.

Reasoning ("thinking") is one key, in the same spirit as ``style``: a checkpoint states
which of the three kinds it is, so it cannot claim two at once.

    thinking         "optional" - the chat template takes an ``enable_thinking`` kwarg,
                                  so each run chooses.
                     "always"   - the checkpoint reasons unconditionally, with no
                                  argument to turn it off.
                     omitted    - the checkpoint never reasons, whatever a run asks for.
    max_new_tokens_thinking
                     generation budget used only when reasoning is on, where the
                     reasoning and the answer share it. Optional; falls back to
                     max_new_tokens.
    thinking_end_token
                     the token that closes the reasoning. Needed only by a checkpoint
                     whose tokenizer ships no ``response_template``; see
                     ``decode_generation``.

Nothing outside this package reads those keys. A pipeline asks ``resolve_thinking``
whether a run will reason and ``validate_thinking_request`` whether it may, so the key
names stay an implementation detail of the table below.
"""

from .core import load_multimodal_lm

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
        # the model card's quickstart budget, which is its enable_thinking=False example
        "max_new_tokens": 1024,
        # the only checkpoint here whose thinking is a per-run choice. The template
        # defaults enable_thinking to false, so omitting it already means no thinking;
        # the pipelines pass it explicitly anyway, so a template revision cannot silently
        # flip a run. Its output is split by the response_template the checkpoint ships,
        # so no thinking_end_token is needed.
        "thinking": "optional",
        # the card publishes no thinking budget: reasoning and answer must share this one
        "max_new_tokens_thinking": 4096,
    },
    "Qwen/Qwen3-VL-32B-Instruct": {
        "style": "one_step",
        "supports_system": True,
        "max_new_tokens": 128,
    },
    "Qwen/Qwen3-VL-32B-Thinking": {
        # does not support reasoning=False.
        "style": "one_step",
        "supports_system": True,
        # the model card's vision-language recommendation (out_seq_length=40960)
        "max_new_tokens": 40960,
        "thinking": "always",
        # its tokenizer ships no response_template, so the reasoning is split on this
        # token instead; </think> is a single token here
        "thinking_end_token": "</think>",
    },
    "OpenGVLab/InternVL3-38B-hf": {
        # the transformers-format conversion, whose config declares model_type "internvl";
        # AutoModelForMultimodalLM resolves it, so no loader and no trust_remote_code
        "style": "one_step",
        # the chat template is a bare ChatML loop that emits <|im_start|>{role} for any
        # role it is handed, with no allowlist and no alternating-role check, so the
        # system role is kept and consecutive user messages are allowed
        "supports_system": True,
        # the model card's "Inference on a single image" example, the one that matches
        # how the pipelines call this checkpoint
        "max_new_tokens": 50,
    },
    "OpenGVLab/InternVL3_5-38B-HF": {
        # same template, byte for byte, as InternVL3-38B-hf
        "style": "one_step",
        "supports_system": True,
        # no transformers quickstart of its own: the card defers to the official InternVL
        # docs, whose single-image example uses this budget
        "max_new_tokens": 50,
        # deliberately no "thinking" key, though this checkpoint does reason. Its card
        # turns thinking on by setting the system prompt to R1_SYSTEM_PROMPT, not by an
        # enable_thinking argument the template reads, so it is neither "optional" (there
        # is nothing to pass) nor "always" (it does not reason unprompted). The card also
        # pairs that mode with do_sample=True and temperature=0.6, which contradicts the
        # deterministic GENERATION_ARGS in constants.py.
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


def thinking_models(mode):
    """Return the checkpoints whose ``thinking`` key is ``mode``, for help and error text.

    Derived from ``MODELS`` on demand rather than frozen into module-level constants, so
    registering a reasoning checkpoint updates every message that names one by editing the
    table above and nothing else.
    """
    return sorted(model_id for model_id, spec in MODELS.items() if spec.get("thinking") == mode)

def validate_thinking_request(model_id, spec, enable_thinking):
    """Reject a thinking request the checkpoint cannot honour, before any weights load.

    Only an "optional" checkpoint can be switched on by the caller: an "always" one needs
    no flag, and one with no ``thinking`` key cannot be talked into reasoning at all. Both
    groups are named, because a run that asked for thinking wants to know where to find it.

    Lives here, next to the table it reports on, so that every pipeline raises the same
    message and no pipeline has to know how the table spells its keys.
    """
    # if thinking is on and the model does not have an optional thinking argument, raise an error.
    if enable_thinking and spec.get("thinking") != "optional":
        raise ValueError(
            f"--enable_thinking was requested, but the model checkpoint '{model_id}' does not take "
            f"an enable_thinking argument, so its thinking mode cannot be switched on.\n"
            f"Model checkpoints that accept --enable_thinking: {', '.join(thinking_models('optional'))}.\n"
            f"Model checkpoints that always think, with no flag needed: {', '.join(thinking_models('always'))}.\n"
            f"Drop --enable_thinking to run '{model_id}' without thinking."
        )


def validate_prompt_request(model_id, spec, prompt):
    """Reject a prompt structure the checkpoint cannot honour, before any weights load.

    A template that requires user and assistant to alternate cannot take the RAG
    pipeline's "multi-user-message" mode, which sends several user messages in a row.

    Lives here, next to the table it reports on, so no pipeline has to know how the table
    spells its keys.
    """
    # if the prompt sends consecutive user messages but the template needs alternating roles, raise
    if prompt == "multi-user-message" and not spec.get("supports_consecutive_user", True):
        raise ValueError(
            f"'{model_id}' requires its chat roles to alternate, so it cannot be used with "
            f"--prompt multi-user-message. Use --prompt single-user-message instead."
        )




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
