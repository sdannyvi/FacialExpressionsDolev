"""Generator (VLM) implementations shared by the RAG and zero-shot pipelines.

This module holds everything that differs between generator checkpoints:

* loader           - how a checkpoint is instantiated
* normalizers      - how the pipeline's neutral conversation is reshaped for a checkpoint
* generate_prediction         - the shared inference call (template -> processor -> generate -> decode)
* decode_generation - how generated ids become ``(thinking, prediction)``


The decoding defaults these functions apply (``GENERATION_ARGS``,
``DEFAULT_MAX_NEW_TOKENS``) live in ``constants.py``.

The pipelines never import from here directly; they go through the package's public
API in ``__init__.py``. The registry in ``registry.py`` maps a checkpoint id onto the
functions below.

The functions are deliberately plain functions rather than classes. Each entry in the
registry maps one-to-one onto a future class, so moving to classes later is mechanical.

Assumes ``transformers>=5``, where the Auto classes are the recommended way to instantiate a
checkpoint. Resolving the concrete model and processor classes is therefore delegated to
``AutoModelForMultimodalLM`` / ``AutoProcessor`` rather than hard-coded per family; see
``load_multimodal_lm``.
"""

import copy
import gc
import re

import torch

from .constants import (DEFAULT_MAX_NEW_TOKENS, GENERATION_ARGS, OOM_RETRY_CACHE,
                        QUANTIZATION)


# --------------------------------------------------------------------------------------
# Loader. Returns (model, processor) and leaves .eval() to load_generator().
# --------------------------------------------------------------------------------------

def load_multimodal_lm(model_id):
    """Default loader: the concrete classes come from the checkpoint's own config.

    ``AutoModelForMultimodalLM`` and ``AutoProcessor`` resolve to exactly the classes the
    per-family loaders used to import — verified against the transformers>=5 mapping for
    LLaVA-NeXT, LLaVA-OneVision, Gemma 3/4 and Qwen3-VL — so every registered checkpoint
    loads the same objects it always did.

    Two cases fall outside the mapping and need an explicit ``loader`` in the registry entry:
    a checkpoint requiring ``trust_remote_code=True``, and a text-only Gemma 3
    (``model_type: gemma3_text`` -> ``Gemma3TextConfig``), which is not a multimodal config.
    """
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    model = AutoModelForMultimodalLM.from_pretrained(
        pretrained_model_name_or_path=model_id,
        dtype=torch.bfloat16,
        quantization_config=QUANTIZATION,
        device_map={"": 0},   # everything on GPU 0
    )
    processor = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path=model_id,
        use_fast=True
    )
    return model, processor


# --------------------------------------------------------------------------------------
# Conversation normalization.
# --------------------------------------------------------------------------------------

def _fold_system_into_first_user(conversation):
    """Move the system message's text into the first user message.

    Used for checkpoints whose chat template has no system role (llava-v1.6-mistral).
    The system text is prepended to the first text item of the first user message, so
    the instruction still reaches the model, just as user text instead of system text.
    System messages are always dropped, including ones that carry no text.
    """
    # iterate over the conversation and collect the system text as a list of strings
    system_texts = []
    for message in conversation:
        if message.get("role") == "system":
            for item in message.get("content", []):
                if item.get("type") == "text":
                    text = item.get("text", "")
                    if text:
                        system_texts.append(text)

    # remove the system messages from the conversation
    conversation = [m for m in conversation if m.get("role") != "system"]

    # nothing to fold; any system messages have already been dropped
    if not system_texts:
        return conversation

    # join the system messages into a single string
    system_text = "\n".join(system_texts)

    # iterate over the conversation and add the system text to the first user message
    for message in conversation:
        if message.get("role") != "user":
            continue
        # iterate over the content of the user message, and add the system text to the first text item
        for item in message.get("content", []):
            if item.get("type") == "text":
                item["text"] = f"{system_text}\n{item.get('text', '')}"
                return conversation
        # user message with images but no text item: add one carrying the system text
        message["content"].append({"type": "text", "text": system_text})
        return conversation

    raise ValueError("Cannot fold the system message: the conversation has no user message.")


def _attach_images_inline(conversation, images):
    """Attach each PIL image to its placeholder, in order.

    One-step templates (Gemma 3/4, Qwen3-VL) expect the image on the content item
    itself rather than as a separate ``images=`` argument to the processor.
    """
    image_index = 0
    for message in conversation:
        for item in message.get("content", []):
            if item.get("type") != "image":
                continue
            if image_index >= len(images):
                raise ValueError(
                    f"conversation has more image placeholders than the {len(images)} "
                    f"images supplied."
                )
            item["image"] = images[image_index]
            image_index += 1

    if image_index != len(images):
        raise ValueError(
            f"{len(images)} images were supplied but the conversation has only "
            f"{image_index} image placeholders."
        )
    return conversation


def normalize_conversation(conversation, images, spec):
    """Reshape the pipeline's neutral conversation for one checkpoint.

    The pipelines always build the same neutral structure:
    ``[{"role": ..., "content": [{"type": "image"}, {"type": "text", "text": ...}]}]``.
    Everything checkpoint-specific about that structure is applied here, so prompt code
    stays model-agnostic.

    Returns a normalized deep copy; the caller's conversation is never mutated.
    """
    conversation = copy.deepcopy(conversation)

    supports_system = spec.get("supports_system", True)
    if supports_system is False:
        conversation = _fold_system_into_first_user(conversation)

    if spec["style"] == "one_step":
        conversation = _attach_images_inline(conversation, images)

    return conversation


# --------------------------------------------------------------------------------------
# Thinking mode.
# --------------------------------------------------------------------------------------

def resolve_thinking(spec, enable_thinking):
    """Return whether this run will actually produce reasoning text.

    The registry's ``thinking`` key says which kind of checkpoint this is, and the request
    only gets a say for the "optional" kind: an "always" checkpoint reasons whatever the
    caller asks, and one that declares no kind never does. Callers therefore never have to
    know which kind they are holding, nor how the registry spells it.

    This is the only place that turns those two inputs into a yes or no. Everything that
    depends on the answer - the generation budget, the decode strategy, whether the
    results file grows a thinking column - asks here rather than re-deriving it.
    """
    mode = spec.get("thinking")
    if mode == "always":
        return True
    return mode == "optional" and bool(enable_thinking)


# --------------------------------------------------------------------------------------
# Output decoding and parsing.
# --------------------------------------------------------------------------------------

def decode_generation(processor, prompt_ids, gen_ids, spec, thinking_on):
    """Turn the generated ids into ``(thinking, prediction)``.

    ``thinking`` is ``None`` for a run that produces no reasoning, so a caller can tell
    "this model does not reason" apart from "it reasoned and said nothing".

    Splitting happens on **token ids, before decoding**, never on the decoded string,
    because the two reasoning checkpoints delimit their reasoning differently in kind and
    not merely in spelling: Qwen3-VL marks ``</think>`` as ``special: false`` (it survives
    ``skip_special_tokens=True``) while Gemma 4 marks ``<channel|>`` as ``special: true``
    (it is erased). A string split would therefore work for one and silently fail for the
    other.

    Three strategies, in order of how generic they are:

    1. the checkpoint ships a machine-readable response schema (``response_template``) ->
       hand the ids to ``parse_response`` and let the checkpoint's own schema do the
       split. Names no model-specific token at all and covers both thinking modes with
       one call. ``prefix`` is required because the template pre-writes part of the
       assistant message that the parser has to see.
    2. no schema, but the registry declares the token that closes the reasoning -> split
       the ids at it and decode the two halves separately.
    3. neither -> a plain decode, exactly as before thinking existed.
    """
    tokenizer = getattr(processor, "tokenizer", None)

    # 1. the checkpoint describes its own output format
    if getattr(tokenizer, "response_template", None) is not None:
        message = processor.parse_response(gen_ids, prefix=prompt_ids)
        # 'content' is absent when generation stopped before the reasoning closed
        return message.get("thinking"), message.get("content", "")

    # 2. older template: split on the closing token the registry declares
    end_token = spec.get("thinking_end_token")
    if thinking_on and end_token is not None:
        end_id = tokenizer.convert_tokens_to_ids(end_token)
        if end_id is None or end_id == getattr(tokenizer, "unk_token_id", None):
            raise ValueError(
                f"thinking_end_token '{end_token}' is not a token of this checkpoint's "
                f"tokenizer, so the reasoning cannot be split off from the answer."
            )
        positions = (gen_ids == end_id).nonzero()
        # if end token never appeared in generation, model wasn't finish to generate reasoning. 
        if positions.numel() == 0:
            # the reasoning never closed: generation hit max_new_tokens mid-thought, so
            # everything generated is reasoning and no answer was reached
            return processor.decode(gen_ids, skip_special_tokens=True).strip(), ""
        cut = int(positions[0])
        thinking = processor.decode(gen_ids[:cut], skip_special_tokens=True)
        prediction = processor.decode(gen_ids[cut + 1:], skip_special_tokens=True)
        return thinking.strip(), prediction

    # 3. the checkpoint does not reason
    return None, processor.decode(gen_ids, skip_special_tokens=True)


# --------------------------------------------------------------------------------------
# Inference.
# --------------------------------------------------------------------------------------

def _stop_token_ids(model):
    """Return the ids that end a generation, as a set.

    ``generation_config.eos_token_id`` is what ``generate`` itself stops on, so it is read
    from there rather than from the tokenizer. It is an int for some checkpoints
    (llava-v1.6-34b, llava-v1.6-mistral, llava-onevision) and a list for others (Gemma 3/4
    and Qwen3-VL, whose chat templates close a turn with a token of their own). Both
    spellings are normalized here, so a caller never has to ask which kind it is holding.
    """
    stop_ids = set()
    eos_token_id = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(eos_token_id, int):
        stop_ids.add(eos_token_id)
    elif eos_token_id is not None:
        stop_ids.update(eos_token_id)
    return stop_ids


def _collapse_repeated_tokens(text):
    """Collapse a run of one repeated token into ``<token>xN``, for a readable dump.

    The processor expands every image placeholder into one token per patch, so a decoded
    prompt carries thousands of identical tokens in a row. They are collapsed to their
    count, which is the part worth reading: it says how many tokens each image cost.
    """
    return re.sub(r"(<[^<>]+>)\1+",
                  lambda match: f"{match.group(1)}x{match.group(0).count(match.group(1))}",
                  text)

def get_context_window(model):
    """
    model: a loaded generator model
    returns: the context window in tokens, or None if the config does not declare one
    """
    for config in (getattr(model.config, "text_config", None), model.config):
        context_window = getattr(config, "max_position_embeddings", None)
        if context_window:
            return context_window
    return None
    
def generate_prediction(model, processor, conversation, images, spec, enable_thinking=False,
                        debug=False, check_truncation=True):
    """Run one generation and return ``(prediction, thinking, stats)``.

    ``conversation`` is the pipeline's neutral message list and ``images`` the PIL images
    in placeholder order. Normalization, templating, generation, slicing and decoding all
    happen here, so the pipelines never touch model-specific details.

    ``enable_thinking`` asks the checkpoint to reason before answering. It defaults to
    False, so a caller that says nothing gets no thinking, and it only reaches templates
    that declare they read it. ``thinking`` comes back ``None`` whenever the run produced
    no reasoning.

    ``stats`` holds the per-call diagnostics (device and token counts, and why generation
    stopped: ``finish_reason`` is "stop" when the model ended the turn itself, "length" when
    max_new_tokens cut it off, and "unknown" for a checkpoint that declares no stop token,
    which cannot be told apart). ``cache_mode`` is "gpu" for a sample that generated within
    the device's memory and "offloaded" for one that only finished because its KV cache was
    moved to host RAM, so a run can report how many samples needed that. They are returned
    rather than printed: this is library code, so the calling pipeline decides whether and
    how often to report them.

    ``check_truncation`` can be switched off by a pipeline that has already learned what
    it needs, which is the case once a run has been told the checkpoint declares no stop
    token: that cannot change later, so there is nothing left to find out. ``finish_reason``
    is then None, meaning "not checked" rather than "nothing wrong".

    ``debug`` adds a ``stats["debug"]`` entry describing this one call: the prompt as the
    model receives it, the shapes of the tensors it is given, and the raw generation before
    it is split into thinking and prediction. It is off by default because collecting it
    means decoding a multi-thousand-token prompt, which is wasted work on every sample of a
    run but exactly what is wanted on the first one.
    """

    # system role truncated if support system is false, and imaged are attached if one-step style
    conversation = normalize_conversation(conversation, images, spec)
    thinking_on = resolve_thinking(spec, enable_thinking)

    # set enable thinking if model accepts enable thinking arg 
    template_kwargs = {}
    if spec.get("thinking") == "optional":
        template_kwargs["enable_thinking"] = enable_thinking

    if spec["style"] == "two_step":
        # Images are passed separately to the processor; placeholders stay bare.
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False,
                                                    **template_kwargs)
        inputs = processor(images=images, text=text_prompt, padding=True, return_tensors="pt")
    else:
        # One-step templates process the inline images themselves.
        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            return_tensors="pt",
            **template_kwargs,
        )

    debug_stats = {}
    if debug:
        # validate the prompt structure as the generator see it, in cpu 
        debug_stats["prompt_text"] = _collapse_repeated_tokens(
            processor.decode(inputs["input_ids"][0], skip_special_tokens=False)
        )
        # name -> shape of every tensor handed to the model: pixel_values missing or shorter than len(images) means dropped images
        debug_stats["input_keys"] = {
            key: tuple(value.shape) if hasattr(value, "shape") else type(value).__name__
            for key, value in inputs.items()
        }

    # move inputs to the model device
    device_type = next(iter(model.parameters())).device
    inputs = inputs.to(device_type)

    # reasoning and answer share one budget, so a thinking run gets its own larger one
    if thinking_on:
        max_new_tokens = spec.get("max_new_tokens_thinking",
                                  spec.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS))
    else:
        max_new_tokens = spec.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)

    # a reasoning run whose chain of thought grows far past the usual length can fill the
    # device with KV cache alone, so a generation that runs out of memory is retried with
    # the cache held in host RAM instead of on the card. Only the storage location of the
    # cache changes, so the retry produces the same tokens the first attempt was heading
    # for, at the cost of moving it back and forth once per layer per token.
    cache_mode = "gpu"
    out_of_memory = False
    with torch.no_grad():
        try:
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, **GENERATION_ARGS)
        except torch.cuda.OutOfMemoryError:
            out_of_memory = True

        # the retry happens outside the except block on purpose: while the exception is
        # being handled its traceback still references generate()'s frames, and those
        # reference the cache that filled the device, so freeing it has to wait until the
        # handler has been left.
        if out_of_memory:
            gc.collect()
            torch.cuda.empty_cache()
            output = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                    cache_implementation=OOM_RETRY_CACHE, **GENERATION_ARGS)
            cache_mode = OOM_RETRY_CACHE

    prompt_len = inputs["input_ids"].shape[-1]
    gen_ids = output[0][prompt_len:]

    # validate whether generation stopped early 
    finish_reason = None
    if check_truncation:
        stop_ids = _stop_token_ids(model)
        # if no eos finish generation token was found, validation cannot be applied 
        if not stop_ids:
            finish_reason = "unknown"
        # if eos token not in generated ids, generation stopped early 
        elif int(gen_ids[-1]) not in stop_ids:
            finish_reason = "length"
        # generation completed 
        else:
            finish_reason = "stop"

    stats = {
        "device": str(device_type),
        "prompt_token_len": prompt_len,
        "generated_token_len": gen_ids.shape[-1],
        "total_token_len": output[0].shape[-1],
        "thinking_on": thinking_on,
        "max_new_tokens": max_new_tokens,
        "finish_reason": finish_reason,
        "cache_mode": cache_mode,
        "generation_args": GENERATION_ARGS,
    }

    if debug:
        # validate decoder function extracts thinking and prediction 
        debug_stats["raw_generated"] = processor.decode(gen_ids, skip_special_tokens=False)
        stats["debug"] = debug_stats

    prompt_tokens = inputs["input_ids"][0]

    # if thinking is off then thinking is None
    thinking, prediction = decode_generation(processor, prompt_tokens, gen_ids, spec, thinking_on)

    return prediction.strip().lower(), thinking, stats
