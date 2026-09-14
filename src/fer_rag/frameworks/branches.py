"""One branch of a framework: build its prompt, generate its answer, and score every class.

A branch is one view of the query: zero-shot, or RAG with some of the retrieved examples. Every
branch returns the same flat result (keys without the branch prefix; ``registry.py`` adds it):

    prediction        the answer the branch generated, stripped and lower case
    confidence        exp(mean log-prob of the answer tokens); reasoning and explanation tokens excluded
    n_answer_tokens   how many tokens the answer has, i.e. the length the mean divides by
    finish_reason     "stop" or "length" for the step that can run out of tokens: the answer in "direct",
                      the reasoning in "cot_*", the answer with its explanation in "answer_explain"
    context_tokens    the largest context among the branch's generation calls: prompt + generated tokens.
                      A two-stage branch makes two calls; the larger one is kept
    exceeds_context   True when any of the branch's generation calls went past the model's context window,
                      so the branch's result is not reliable; None when the model declares no window
    reasoning         the generated reasoning ("cot_*" formats only)
    explanation       the generated explanation ("answer_explain" only)
    conf__<class>     exp(mean log-prob of the class label's tokens), teacher forced where the answer starts
    prob__<class>     conf__<class> divided by the sum of conf over all classes

Class labels are scored as ``class_name.capitalize()``, the case LLaVA-34B writes its answer in.
The class names themselves (column names, prompt text) always come from ``classes_list``.
"""
import numpy as np

from ..generators import (DEFAULT_MAX_NEW_TOKENS, REASONING_MAX_NEW_TOKENS, generate_with_logprobs,
                          score_labels)
from ..pipelines.prompts import (ANSWER_FORMATS, ANSWER_TRIGGER, build_rag_conversation,
                                 build_rag_framework_conversation, build_zero_shot_conversation,
                                 build_zero_shot_framework_conversation)

# positions in the retrieved top-k list of the examples each branch puts in its prompt; None = zero-shot
BRANCH_EXAMPLES = {
    "zs": None,
    "rag_top12": [0, 1],
    "rag_top1": [0],
    "rag_top2": [1],
}

# the text column an answer format adds to its branches' results
TEXT_COLUMN = {
    "cot_generic": "reasoning",
    "cot_task": "reasoning",
    "answer_explain": "explanation",
}


def branch_columns(text_column, classes_list):
    """
    text_column: "reasoning", "explanation" or None.
    returns: the result keys of one branch, in column order.
    """
    columns = ["prediction", "confidence", "n_answer_tokens", "finish_reason", "context_tokens", "exceeds_context"]
    if text_column is not None:
        columns.append(text_column)
    columns += [f"conf__{class_name}" for class_name in classes_list]
    columns += [f"prob__{class_name}" for class_name in classes_list]
    return columns


def _assistant(text):
    """An assistant message for the model to continue."""
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def _generate(name, conversation, images, max_new_tokens, ctx, continue_final_message=False):
    """generate_with_logprobs with the run's model, printing prompt and output for the first sample."""
    generation = generate_with_logprobs(ctx["model"], ctx["processor"], conversation, images, ctx["spec"],
                                        max_new_tokens, continue_final_message)
    if ctx["debug"]:
        print(f"[framework debug] {name} | prompt the model sees (image placeholders not expanded):\n"
              f"{generation['prompt_text']}")
        print(f"[framework debug] {name} | generation: {generation['text']!r} "
              f"| finish_reason: {generation['finish_reason']}")
    if generation["finish_reason"] == "length" and max_new_tokens == REASONING_MAX_NEW_TOKENS:
        ctx["truncated_count"] += 1
    if generation["exceeds_context"]:
        ctx["context_exceeded_count"] += 1
        ctx["largest_context_tokens"] = max(ctx["largest_context_tokens"], generation["context_tokens"])
    return generation


def _context(generations):
    """context_tokens and exceeds_context of a branch, from every generation call it made.

    The branch is unreliable if any one call went past the context window, so a two-stage branch is
    judged by both of its calls, and the largest context is the one saved.
    """
    flags = [generation["exceeds_context"] for generation in generations]
    return {"context_tokens": max(generation["context_tokens"] for generation in generations),
            "exceeds_context": None if None in flags else any(flags)}


def _confidence(token_logprobs):
    """confidence and n_answer_tokens of an answer's token log-probs; nan confidence for no answer."""
    n_tokens = len(token_logprobs)
    confidence = float(np.exp(np.mean(token_logprobs))) if n_tokens else np.nan
    return {"confidence": confidence, "n_answer_tokens": n_tokens}


def _class_scores(conversation, images, ctx, continue_final_message, label_prefix):
    """conf__<class> and prob__<class> for every class, teacher forced after the conversation.

    label_prefix: text the model writes before the label at this position (" " after the answer trigger).
    """
    classes_list = ctx["classes_list"]
    labels = {class_name: label_prefix + class_name.capitalize() for class_name in classes_list}
    logprobs = score_labels(ctx["model"], ctx["processor"], conversation, images, ctx["spec"],
                            list(labels.values()), continue_final_message,
                            check_full_forward=not ctx["scoring_checked"])
    ctx["scoring_checked"] = True

    conf = {class_name: float(np.exp(np.mean(logprobs[labels[class_name]]))) for class_name in classes_list}
    total = sum(conf.values())
    scores = {f"conf__{class_name}": conf[class_name] for class_name in classes_list}
    scores.update({f"prob__{class_name}": conf[class_name] / total for class_name in classes_list})
    return scores


def _leading_word(generation, processor):
    """Split a generation into its leading word, which is the answer, and the text after it.

    The answer tokens are the generated tokens up to the first one that adds anything other than
    letters, so "▁Happy" or "Ang" + "ry" are kept and "." ends the word. Tokens that are only
    whitespace before the word (e.g. "\\n") are not answer tokens and do not enter the confidence.
    returns: (prediction in lower case, log-probs of the answer tokens, the rest of the text)
    """
    token_ids = generation["token_ids"]
    decode = lambda ids: processor.decode(ids, skip_special_tokens=True)
    # skip whitespace-only tokens before the word
    start = 0
    while start < len(token_ids) and not decode(token_ids[:start + 1]).strip():
        start += 1
    end = start
    for position in range(start, len(token_ids)):
        if not decode(token_ids[start:position + 1]).strip().isalpha():
            break
        end = position + 1
    word = decode(token_ids[start:end]).strip()
    rest = decode(token_ids[end:]).strip().lstrip(".,:;-").strip()
    return word.lower(), generation["token_logprobs"][start:end], rest


def reason_then_answer(name, conversation, images, reasoning_trigger, ctx):
    """Two-stage chain of thought (zero-shot reasoners): reason first, then answer after a trigger.

    Stage 1 opens the assistant's answer with ``reasoning_trigger`` and generates the reasoning.
    Stage 2 gives the model its own reasoning followed by ANSWER_TRIGGER; the leading word it writes
    next is the answer, and the class labels are teacher forced at that same position. The reasoning
    is context in stage 2 and never enters a score.
    """
    spec = ctx["spec"]
    stage_1 = conversation + [_assistant(reasoning_trigger)]
    reasoning_generation = _generate(f"{name} reasoning", stage_1, images, REASONING_MAX_NEW_TOKENS, ctx,
                                     continue_final_message=True)
    reasoning = reasoning_generation["text"].strip()

    stage_2 = conversation + [_assistant(f"{reasoning_trigger} {reasoning}".rstrip() + f"\n{ANSWER_TRIGGER}")]
    answer_generation = _generate(f"{name} answer", stage_2, images,
                                  spec.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS), ctx,
                                  continue_final_message=True)
    prediction, answer_logprobs, _ = _leading_word(answer_generation, ctx["processor"])

    result = {"prediction": prediction, **_confidence(answer_logprobs),
              "finish_reason": reasoning_generation["finish_reason"],
              **_context([reasoning_generation, answer_generation]), "reasoning": reasoning}
    result.update(_class_scores(stage_2, images, ctx, continue_final_message=True, label_prefix=" "))
    return result


def _answer_directly(name, conversation, images, ctx):
    """The original prompt: the whole generation is the answer, exactly as the original RAG predicts."""
    spec = ctx["spec"]
    generation = _generate(f"{name} answer", conversation, images,
                           spec.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS), ctx)
    result = {"prediction": generation["prediction"], **_confidence(generation["token_logprobs"]),
              "finish_reason": generation["finish_reason"], **_context([generation])}
    result.update(_class_scores(conversation, images, ctx, continue_final_message=False, label_prefix=""))
    return result


def _answer_then_explain(name, conversation, images, ctx):
    """The label first, then a brief explanation; the classes are scored before the explanation."""
    generation = _generate(f"{name} answer and explanation", conversation, images, REASONING_MAX_NEW_TOKENS, ctx)
    prediction, answer_logprobs, explanation = _leading_word(generation, ctx["processor"])
    result = {"prediction": prediction, **_confidence(answer_logprobs),
              "finish_reason": generation["finish_reason"], **_context([generation]), "explanation": explanation}
    result.update(_class_scores(conversation, images, ctx, continue_final_message=False, label_prefix=""))
    return result


def run_branch(branch, answer_format, query_image, top_examples, ctx):
    """
    branch: a key of BRANCH_EXAMPLES.
    answer_format: "direct" or a key of ANSWER_FORMATS.
    query_image: the query PIL image.
    top_examples: knowledge base rows of the retrieved examples, most similar first.
    ctx: the run's framework context (see run_framework in registry.py).
    returns: the branch's flat result, keys as in the module docstring.
    """
    classes_list = ctx["classes_list"]
    positions = BRANCH_EXAMPLES[branch]
    instruction = None if answer_format == "direct" else ANSWER_FORMATS[answer_format]["instruction"]

    if positions is None:
        conversation = (build_zero_shot_conversation(classes_list) if instruction is None
                        else build_zero_shot_framework_conversation(classes_list, instruction))
        images = [query_image]
    else:
        examples = top_examples.iloc[positions]
        if instruction is None:
            conversation, images = build_rag_conversation(ctx["prompt"], classes_list, examples, query_image)
        else:
            conversation, images = build_rag_framework_conversation(ctx["prompt"], classes_list, examples,
                                                                    query_image, instruction)

    if answer_format == "direct":
        return _answer_directly(branch, conversation, images, ctx)
    if answer_format == "answer_explain":
        return _answer_then_explain(branch, conversation, images, ctx)
    return reason_then_answer(branch, conversation, images, ANSWER_FORMATS[answer_format]["reasoning_trigger"], ctx)
