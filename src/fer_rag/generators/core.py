"""Generator (VLM) implementations shared by the RAG and zero-shot pipelines.

This module holds everything that differs between generator checkpoints:

* loader           - how a checkpoint is instantiated
* normalizers      - how the pipeline's neutral conversation is reshaped for a checkpoint
* generate_prediction         - the shared inference call (template -> processor -> generate -> decode)
* parsers          - how a raw decoded string becomes a prediction

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
``load_multimodal_lm``. transformers is imported inside the loader so that importing this
module stays cheap and does not depend on the library being installed.
"""

import copy

import torch

from .constants import DEFAULT_MAX_NEW_TOKENS, GENERATION_ARGS


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
        torch_dtype=torch.float16,
        device_map="balanced",
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
# Output parsing.
# --------------------------------------------------------------------------------------

def parse_thinking(text):
    """Strip the reasoning block from a thinking model's output.

    Qwen3-VL Thinking emits its reasoning before the answer. Keep whatever follows the
    closing tag; if the tag is absent (for instance because generation hit the token
    limit mid-reasoning) fall back to the last non-empty line, which is where the answer
    lands when the model finishes normally.
    """
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        return lines[-1]
    return text


# --------------------------------------------------------------------------------------
# Inference.
# --------------------------------------------------------------------------------------

def generate_prediction(model, processor, conversation, images, spec):
    """Run one generation and return ``(prediction, stats)``.

    ``conversation`` is the pipeline's neutral message list and ``images`` the PIL images
    in placeholder order. Normalization, templating, generation, slicing and decoding all
    happen here, so the pipelines never touch model-specific details.

    ``stats`` holds the per-call diagnostics (device and token counts). They are returned
    rather than printed: this is library code, so the calling pipeline decides whether and
    how often to report them.
    """

    # system role truncated if support system is false, and imaged are attached if one-step style 
    conversation = normalize_conversation(conversation, images, spec)

    if spec["style"] == "two_step":
        # Images are passed separately to the processor; placeholders stay bare.
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(images=images, text=text_prompt, padding=True, return_tensors="pt")
    else:
        # One-step templates process the inline images themselves.
        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    # move inputs to the model device
    device_type = next(iter(model.parameters())).device
    inputs = inputs.to(device_type)

    max_new_tokens = spec.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, **GENERATION_ARGS)

    prompt_len = inputs["input_ids"].shape[-1]
    gen_ids = output[0][prompt_len:]
    stats = {
        "device": str(device_type),
        "prompt_toeken_len": prompt_len,
        "generated_token_len": gen_ids.shape[-1],
        "total_token_len": output[0].shape[-1],
    }

    prediction = processor.decode(gen_ids, skip_special_tokens=True)

    parse = spec.get("parse")
    if parse is not None:
        prediction = parse(prediction)

    return prediction.strip().lower(), stats
