"""Prompt text of the pipelines and the frameworks.

Two groups, kept apart on purpose:

* original prompts - ``build_rag_conversation`` (single-user-message and multi-user-message) and
  ``build_zero_shot_conversation``. Their text is moved here unchanged from ``rag.py`` and
  ``zero_shot.py`` at tag ``rag-baseline`` and must stay exactly as it is: the original RAG and the
  frameworks' "direct" branches use them. The one adaptation: with a single retrieved example the
  RAG prompt speaks of it in the singular (see ``_example_wording``); with two or more examples, as
  in the original RAG runs, the text is exactly the original.
* framework prompts - the same structure with the one-word output instruction replaced by an answer
  format's own instruction, the reasoning and answer triggers, and the aggregator. These are free to
  change per method.

Every builder returns the pipelines' neutral conversation:
``[{"role": ..., "content": [{"type": "image"}, {"type": "text", "text": ...}]}]``.
"""
from PIL import Image

from config import resolve_path


# --------------------------------------------------------------------------------------
# Original prompts. Do not edit the text.
# --------------------------------------------------------------------------------------

def _example_wording(top_k):
    """The words of the RAG prompts that depend on the number of examples.

    Plural, the original text, unless there is exactly one example (the frameworks' rag_top1 and
    rag_top2 branches), where "1 example images ... their labels" would misdescribe the prompt.
    """
    if top_k == 1:
        return {"example_images": "example image with its corresponding emotion label",
                "labeled_examples": "labeled example",
                "the_examples": "the example"}
    return {"example_images": "example images with their corresponding emotion labels",
            "labeled_examples": "labeled examples",
            "the_examples": "the examples"}


def build_rag_conversation(prompt, classes_list, top_examples, query_image):
    """The original RAG prompt.

    prompt: "single-user-message" or "multi-user-message".
    top_examples: knowledge base rows of the retrieved examples, most similar first. The prompt says
        how many examples it holds, so a run with top_k=2 gives the original text; a single example is
        described in the singular.
    returns: (conversation, images), the images in placeholder order: the examples, then the query.
    """
    top_k = len(top_examples)
    wording = _example_wording(top_k)
    # initialize vars for inference
    conversation = []
    images = []
    # if prompt is multiple user message
    if prompt == "multi-user-message":
        # loop through examples
        for _, example_row in top_examples.iterrows():
            example_label = example_row['true_label']
            example_image = Image.open(resolve_path(example_row['file_path'])).convert('RGB')

            example_prompt_text = (f"Example: This image shows a person expressing the emotion: "
                                   f"'{example_label}'.")
            conversation.append(
                {"role": "user",
                 "content": [
                     {"type": "image"},
                     {"type": "text", "text": example_prompt_text}
                 ]}
            )
            images.append(example_image)

        # query image prompt
        query_prompt_text = (f"This image also shows a person expressing an emotion."
                             f"Based on {wording['the_examples']} provided, please analyze the emotion in this image and "
                             f"select the best match from"
                             f"the following options: {', '.join(classes_list)}."
                             f"Respond with only one word: the emotion name.")

        conversation.append(
            {"role": "user",
             "content": [
                 {"type": "image"},
                 {"type": "text", "text": query_prompt_text}
             ]}
        )
        images.append(query_image)
    elif prompt == "single-user-message":
        # add a system message
        conversation.append({
            "role": "system",
            "content": [
                {"type": "text",
                 "text": f"You are an expert in classifying emotions from facial expressions in images.\n"
                         f"You are given {top_k} {wording['example_images']}, followed by a query image.\n"
                         f"Based on {wording['the_examples']} provided, analyze the facial expression in the query "
                         f"image and classify the emotion. Follow the user's requested output format."
                 }
            ]
        })

        content = []
        user_text = [f"You are given {top_k} {wording['labeled_examples']} and one query image, in the following order:"]

        for i, example_row in enumerate(top_examples.itertuples(), start=1):
            # add images to image list
            example_label = example_row.true_label
            example_image = Image.open(resolve_path(example_row.file_path)).convert('RGB')
            images.append(example_image)

            # add user text
            content.append({"type": "image"})
            user_text.append(f"Image {i} label: {example_row.true_label}")

        # add query image
        images.append(query_image)
        content.append({"type": "image"})
        user_text.append(f"Image {top_k + 1} is the query. Based on {wording['the_examples']}, classify the emotion shown in this image into one of the following emotions: {', '.join(classes_list)}.")
        user_text.append("Respond with only one word: the emotion label.")
        # concat user text
        user_text = "\n".join(user_text)
        content.append({"type": "text", "text": user_text})
        conversation.append({"role": "user", "content": content})

    return conversation, images


def build_zero_shot_conversation(classes_list):
    """The original zero-shot prompt. Its one image placeholder is the query."""
    conversation = []
    # system role
    conversation.append({
        "role": "system",
        "content": [
            {"type": "text",
             "text": f"You are an expert in classifying emotions from facial expressions in images.\n"
                     f"You are given a query image. Analyze the facial expression in the query image and classify the emotion.\n"
                     f"Follow the user's requested output format."
             }
        ]
    })

    # user role
    conversation.append({"role": "user",
                         "content": [{
                             "type": "image"},
                             {"type": "text", "text": f"Classify the emotion shown in this image into one of the following emotions: {', '.join(classes_list)}.\n"
                                                      f"Respond with only one word: the emotion label."}]})
    return conversation


# --------------------------------------------------------------------------------------
# Framework prompts.
# --------------------------------------------------------------------------------------

# Answer formats other than "direct" (which is the original prompt as it is).
#   instruction        replaces the original "Respond with only one word" line
#   reasoning_trigger  opens the assistant's answer before the reasoning is generated, as in
#                      zero-shot chain of thought ("Let's think step by step.")
ANSWER_FORMATS = {
    "cot_generic": {
        "instruction": "Briefly reason step by step, then give the emotion label.",
        "reasoning_trigger": "Let's think step by step.",
    },
    "cot_task": {
        # write the task-specific chain-of-thought text here; a run refuses to start while these are None
        "instruction": None,
        "reasoning_trigger": None,
    },
    "answer_explain": {
        "instruction": "Respond with the emotion label first, then briefly explain why you chose it.",
    },
    # the label is read from the "Answer:" field of the text, so this format has no class scores;
    # the aggregator answers in it too (see build_aggregator_responses_conversation)
    "answer_explain_format": {
        "instruction": "Provide your answer and a step-by-step reasoning explanation.\n"
                       "Please follow the format: Answer: {}. Explanation: {}.",
    },
}

# continues the reasoning in the second stage, so the label the model writes next is its answer
ANSWER_TRIGGER = "Therefore, the emotion label is"

# opens the aggregator's evaluation of the analyses
AGGREGATOR_REASONING_TRIGGER = "Let's think step by step."


def build_rag_framework_conversation(prompt, classes_list, top_examples, query_image, instruction):
    """The original RAG prompt with its output instruction replaced by ``instruction``.

    returns: (conversation, images), the images in placeholder order: the examples, then the query.
    """
    top_k = len(top_examples)
    wording = _example_wording(top_k)
    conversation = []
    images = []
    if prompt == "multi-user-message":
        for _, example_row in top_examples.iterrows():
            example_label = example_row['true_label']
            example_image = Image.open(resolve_path(example_row['file_path'])).convert('RGB')

            example_prompt_text = (f"Example: This image shows a person expressing the emotion: "
                                   f"'{example_label}'.")
            conversation.append(
                {"role": "user",
                 "content": [
                     {"type": "image"},
                     {"type": "text", "text": example_prompt_text}
                 ]}
            )
            images.append(example_image)

        query_prompt_text = (f"This image also shows a person expressing an emotion."
                             f"Based on {wording['the_examples']} provided, please analyze the emotion in this image and "
                             f"select the best match from"
                             f"the following options: {', '.join(classes_list)}."
                             f"{instruction}")

        conversation.append(
            {"role": "user",
             "content": [
                 {"type": "image"},
                 {"type": "text", "text": query_prompt_text}
             ]}
        )
        images.append(query_image)
    elif prompt == "single-user-message":
        conversation.append({
            "role": "system",
            "content": [
                {"type": "text",
                 "text": f"You are an expert in classifying emotions from facial expressions in images.\n"
                         f"You are given {top_k} {wording['example_images']}, followed by a query image.\n"
                         f"Based on {wording['the_examples']} provided, analyze the facial expression in the query "
                         f"image and classify the emotion. Follow the user's requested output format."
                 }
            ]
        })

        content = []
        user_text = [f"You are given {top_k} {wording['labeled_examples']} and one query image, in the following order:"]

        for i, example_row in enumerate(top_examples.itertuples(), start=1):
            example_image = Image.open(resolve_path(example_row.file_path)).convert('RGB')
            images.append(example_image)

            content.append({"type": "image"})
            user_text.append(f"Image {i} label: {example_row.true_label}")

        images.append(query_image)
        content.append({"type": "image"})
        user_text.append(f"Image {top_k + 1} is the query. Based on {wording['the_examples']}, classify the emotion shown in this image into one of the following emotions: {', '.join(classes_list)}.")
        user_text.append(instruction)
        user_text = "\n".join(user_text)
        content.append({"type": "text", "text": user_text})
        conversation.append({"role": "user", "content": content})

    return conversation, images


def build_zero_shot_framework_conversation(classes_list, instruction):
    """The original zero-shot prompt with its output instruction replaced by ``instruction``."""
    return [
        {"role": "system",
         "content": [
             {"type": "text",
              "text": f"You are an expert in classifying emotions from facial expressions in images.\n"
                      f"You are given a query image. Analyze the facial expression in the query image and classify the emotion.\n"
                      f"Follow the user's requested output format."
              }
         ]},
        {"role": "user",
         "content": [{"type": "image"},
                     {"type": "text", "text": f"Classify the emotion shown in this image into one of the following emotions: {', '.join(classes_list)}.\n"
                                              f"{instruction}"}]},
    ]


def build_aggregator_conversation(classes_list, analyses, text_name):
    """The aggregator's prompt: the query image and the branches' analyses, without their examples.

    analyses: [(text, label)] per branch, in branch order. The branches are not named, so the
        aggregator judges the analyses by their content.
    text_name: what the analysis text is, "reasoning" or "explanation".
    """
    blocks = []
    for number, (text, label) in enumerate(analyses, start=1):
        text = text if isinstance(text, str) and text else "(none)"
        label = label if isinstance(label, str) and label else "(none)"
        blocks.append(f"Analysis {number}:\n{text_name.capitalize()}: {text}\nEmotion label: {label}")

    return [
        {"role": "system",
         "content": [
             {"type": "text",
              "text": f"You are an expert in classifying emotions from facial expressions in images.\n"
                      f"You are given a query image and {len(analyses)} independent analyses of it. Each analysis "
                      f"gives a {text_name} and an emotion label. The analyses may disagree, and any of them may be wrong.\n"
                      f"Evaluate the analyses against the facial expression in the query image and classify the emotion. "
                      f"Follow the user's requested output format."
              }
         ]},
        {"role": "user",
         "content": [{"type": "image"},
                     {"type": "text", "text": f"Classify the emotion shown in this image into one of the following emotions: {', '.join(classes_list)}.\n\n"
                                              + "\n\n".join(blocks)
                                              + "\n\nBriefly evaluate each analysis against the image, then give the emotion label."}]},
    ]


def build_aggregator_responses_conversation(classes_list, responses, instruction):
    """The aggregator's prompt that reads the agents' responses verbatim, without their examples.

    responses: the raw text each agent generated ("Answer: ... Explanation: ..."), in branch order.
    instruction: the output instruction the agents were given, so the aggregator answers in the same format.
    """
    agent_lines = []
    for number, response in enumerate(responses, start=1):
        response = response if isinstance(response, str) and response else "(none)"
        agent_lines.append(f"Agent {number}: {response}")

    return [
        {"role": "system",
         "content": [
             {"type": "text",
              "text": "You are an aggregator for facial expression recognition. You are given a query image and "
                      "responses from multiple agents.\n"
                      "Each agent's Answer is its predicted emotion label for the query image, and its Explanation "
                      "describes the reasoning supporting that prediction.\n"
                      "Compare the agents' predictions and explanations, evaluate them in relation to the query image, "
                      "resolve any disagreement, and determine the final emotion label that best matches the query "
                      "image. Follow the user's requested output format."
              }
         ]},
        {"role": "user",
         "content": [{"type": "image"},
                     {"type": "text", "text": "Agent responses:\n"
                                              + "\n".join(agent_lines)
                                              + "\nBased on the query image and the agent responses, classify the "
                                                f"emotion shown in the query image into one of the following emotions: "
                                                f"{', '.join(classes_list)}.\n"
                                              + instruction}]},
    ]
