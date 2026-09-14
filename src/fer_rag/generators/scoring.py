"""Generation with token log-probabilities, and teacher-forced label scoring, for the frameworks.

Written for the LLaVA "two_step" flow of ``llava-hf/llava-v1.6-34b-hf``, the only generator the
frameworks run on: the chat template is rendered to text and the images are passed separately to
the processor, as ``generate_prediction`` does for that checkpoint. ``generate_prediction`` itself is
untouched, so the original pipelines do not depend on anything here.

Two functions:

* generate_with_logprobs - greedy generation, plus the log-probability of every generated token
* score_labels           - the log-probability of every token of each given label, with the label
                           teacher forced right after the prompt

Both accept ``continue_final_message``: the conversation then ends with an assistant message that
the model continues (a reasoning trigger, or reasoning followed by an answer trigger) instead of a
new assistant turn.
"""
import math

import torch

from .constants import GENERATION_ARGS
from .core import _stop_token_ids, decode_generation, get_context_window, normalize_conversation


def build_two_step_inputs(model, processor, conversation, images, spec, continue_final_message=False):
    """Return ``(prompt_text, inputs)``, with the inputs on the model device.

    Builds the inputs the way ``generate_prediction`` does for a two_step checkpoint, with the one
    addition of ``continue_final_message``.
    """
    if spec["style"] != "two_step":
        raise ValueError("the framework scoring is written for two_step checkpoints "
                         "(llava-hf/llava-v1.6-34b-hf) only.")
    conversation = normalize_conversation(conversation, images, spec)
    prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=not continue_final_message,
                                                continue_final_message=continue_final_message, tokenize=False)
    inputs = processor(images=images, text=prompt_text, padding=True, return_tensors="pt")
    device = next(model.parameters()).device
    return prompt_text, inputs.to(device)


def generate_with_logprobs(model, processor, conversation, images, spec, max_new_tokens,
                           continue_final_message=False):
    """Greedy generation that also returns the log-probability of every generated token.

    Decoding uses the shared ``GENERATION_ARGS``, so the generated tokens are the ones
    ``generate_prediction`` would produce for the same prompt; asking for the logits does not change
    them.

    returns a dict:
        prompt_text     the rendered prompt, before the processor expands the image placeholders
        text            the decoded generation, stop token excluded
        prediction      ``text`` as generate_prediction returns it: stripped, lower case
        token_ids       generated ids before the first stop token
        token_logprobs  log-probability of each of those ids
        finish_reason   "stop" when the model ended its turn, "length" when max_new_tokens cut it off
        context_tokens  the context this call used: prompt tokens (images expanded) + generated tokens
        context_window  the model's context window, or None if its config declares none
        exceeds_context True when context_tokens is larger than the context window, None when the
                        window is unknown. Returned rather than printed: the caller decides how to report it.
    """
    prompt_text, inputs = build_two_step_inputs(model, processor, conversation, images, spec,
                                                continue_final_message)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, output_logits=True,
                                return_dict_in_generate=True, **GENERATION_ARGS)

    prompt_len = inputs["input_ids"].shape[-1]
    gen_ids = output.sequences[0][prompt_len:]

    # the answer is everything before the first stop token
    stop_ids = _stop_token_ids(model)
    token_ids = gen_ids.tolist()
    finish_reason = "length"
    for position, token_id in enumerate(token_ids):
        if token_id in stop_ids:
            token_ids = token_ids[:position]
            finish_reason = "stop"
            break

    # output.logits[i] are the raw logits that chose the i-th generated token
    token_logprobs = [torch.log_softmax(output.logits[position][0].float(), dim=-1)[token_id].item()
                      for position, token_id in enumerate(token_ids)]

    # the tokens the model actually worked with, not the max_new_tokens budget: a reasoning that stops
    # early does not count the budget it did not use
    context_tokens = prompt_len + gen_ids.shape[-1]
    context_window = get_context_window(model)
    exceeds_context = None if context_window is None else context_tokens > context_window

    _, text = decode_generation(processor, inputs["input_ids"][0], gen_ids, spec, thinking_on=False)
    return {
        "prompt_text": prompt_text,
        "text": text,
        "prediction": text.strip().lower(),
        "token_ids": token_ids,
        "token_logprobs": token_logprobs,
        "finish_reason": finish_reason,
        "context_tokens": context_tokens,
        "context_window": context_window,
        "exceeds_context": exceeds_context,
    }


def _label_token_ids(tokenizer, prompt_text, label):
    """The ids ``label`` adds after ``prompt_text``.

    The label is tokenized together with the prompt rather than on its own, because a tokenizer
    splits a word differently depending on what precedes it (LLaVA-34B: "Sad" is "S" + "ad" at the
    start of the answer, but a single "▁Sad" after "is").
    """
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    full_ids = tokenizer(prompt_text + label, add_special_tokens=False).input_ids
    if full_ids[:len(prompt_ids)] != prompt_ids or len(full_ids) == len(prompt_ids):
        raise ValueError(f"the label {label!r} merges with the end of the prompt, so its tokens cannot "
                         f"be separated from the prompt's tokens.")
    return full_ids[len(prompt_ids):]


def score_labels(model, processor, conversation, images, spec, labels, continue_final_message=False,
                 check_full_forward=False):
    """Teacher force every label right after the prompt.

    labels: the exact strings to score, including a leading space where the model would write one.
    check_full_forward: also score the longest label with one plain forward pass over prompt + label
        and print the largest difference from the cached scores. Meant for the first call of a run.
    returns: {label: [log-probability of each label token]}

    The prompt runs once and its KV cache is kept. Each label then feeds only its own tokens on top of
    that cache, and the cache is cropped back to the prompt before the next label. The first label
    token is scored by the prompt's last logits, every later token by the logits of the token before it.
    """
    prompt_text, inputs = build_two_step_inputs(model, processor, conversation, images, spec,
                                                continue_final_message)
    tokenizer = processor.tokenizer
    device = inputs["input_ids"].device
    prompt_len = inputs["input_ids"].shape[-1]
    label_ids = {label: _label_token_ids(tokenizer, prompt_text, label) for label in labels}

    # the label ids were cut from the text tokenization of the prompt, which has one token per image
    # placeholder, while the model receives the processor's ids with every image expanded. The two must
    # end the same way, or the labels would be scored after a different prompt ending.
    text_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    tail = min(16, len(text_ids))
    if inputs["input_ids"][0, -tail:].tolist() != text_ids[-tail:]:
        raise ValueError("the processor's prompt ids and the tokenizer's prompt ids end differently, so "
                         "the label tokens cannot be placed after the prompt.")

    logprobs = {}
    with torch.no_grad():
        prompt_out = model(**inputs, use_cache=True, logits_to_keep=1)
        first_logprobs = torch.log_softmax(prompt_out.logits[0, -1].float(), dim=-1)
        cache = prompt_out.past_key_values
        for label, ids in label_ids.items():
            token_logprobs = [first_logprobs[ids[0]].item()]
            if len(ids) > 1:
                n_new = len(ids) - 1
                out = model(input_ids=torch.tensor([ids[:-1]], device=device),
                            attention_mask=torch.ones(1, prompt_len + n_new,
                                                      dtype=inputs["attention_mask"].dtype, device=device),
                            cache_position=torch.arange(prompt_len, prompt_len + n_new, device=device),
                            past_key_values=cache, use_cache=True)
                step_logprobs = torch.log_softmax(out.logits[0].float(), dim=-1)
                token_logprobs += [step_logprobs[i, token_id].item() for i, token_id in enumerate(ids[1:])]
                # a negative value removes that many tokens, back to the prompt
                cache.crop(-n_new)
            logprobs[label] = token_logprobs
        del cache, prompt_out

        if check_full_forward:
            label = max(label_ids, key=lambda name: len(label_ids[name]))
            ids = label_ids[label]
            full_inputs = dict(inputs)
            full_inputs["input_ids"] = torch.cat([inputs["input_ids"], torch.tensor([ids[:-1]], device=device)],
                                                 dim=1)
            full_inputs["attention_mask"] = torch.ones_like(full_inputs["input_ids"])
            out = model(**full_inputs, logits_to_keep=len(ids))
            full_logprobs = torch.log_softmax(out.logits[0].float(), dim=-1)
            max_diff = max(abs(full_logprobs[i, token_id].item() - logprobs[label][i])
                           for i, token_id in enumerate(ids))
            print(f"[generators.scoring.score_labels] cached vs full forward log-probs for {label!r} "
                  f"({len(ids)} tokens): max abs diff {max_diff:.2e}")
            if not math.isfinite(max_diff) or max_diff > 0.05:
                print("[WARNING] generators.scoring: the cached label scores do not match a full forward "
                      "pass, so the class scores of this run are not reliable.")
    return logprobs
