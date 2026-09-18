"""Frameworks: what runs on a sample whose top-1 and top-2 retrieved labels differ.

``rag.py --framework <name>`` sends every sample the gate (``needs_new_framework`` below) lets through
to the framework, and every other sample to the original RAG. A framework is one entry of ``FRAMEWORKS``:

    branches        the views of the query that each produce an answer (see branches.BRANCH_EXAMPLES)
    answer_format   how every branch answers: "direct" is the original prompt; the others are defined
                    in pipelines/prompts.py, ANSWER_FORMATS
    combine         "rules": the branches' answers are combined by the decision rules
                    "aggregator": the VLM reads the branches' analyses and gives the final answer. With an
                    answer format of branches.UNSCORED_FORMATS it reads the branches' responses verbatim
                    and answers in that same format; otherwise it reasons step by step (two stages)
    decision_rule   ("rules" only) the rule whose answer fills the ``prediction`` column. Every rule is
                    stored in its own ``rule__<rule>`` column either way.

Adding a framework is adding an entry here. Frameworks run on llava-hf/llava-v1.6-34b-hf with top_k=2.

Results columns of a framework sample: ``<branch>__<key>`` per branch (keys in branches.py),
``rule__<rule>`` for "rules", ``agg__<key>`` for "aggregator", and ``route``. The separator is a double
underscore because branch names contain single ones.
"""
from ..pipelines.prompts import (AGGREGATOR_REASONING_TRIGGER, ANSWER_FORMATS, build_aggregator_conversation,
                                 build_aggregator_responses_conversation)
from .branches import (TEXT_COLUMN, UNSCORED_FORMATS, answer_in_format, branch_columns, reason_then_answer,
                       run_branch)
from .decision_rules import RULES, predict


def needs_new_framework(top_labels):
    """The inference-time gate.

    top_labels: labels of the retrieved examples, most similar first.
    returns: True when the top-1 and top-2 labels differ, so the sample goes to the framework;
             False when they agree, so the sample keeps the original RAG.
    """
    if len(top_labels) < 2:
        raise ValueError(f"the gate compares the top-1 and top-2 labels, but only {len(top_labels)} "
                         f"retrieved labels were given.")
    return top_labels[0] != top_labels[1]


FRAMEWORK_GENERATOR = "llava-hf/llava-v1.6-34b-hf"

# the column prefix of the aggregator's results
AGGREGATOR = "agg"

FRAMEWORKS = {
    "rules_direct_zs_rag12": {
        "branches": ["zs", "rag_top12"],
        "answer_format": "direct",
        "combine": "rules",
        "decision_rule": "sum_class_prob",
    },
    "rules_direct_zs_rag1_rag2": {
        "branches": ["zs", "rag_top1", "rag_top2"],
        "answer_format": "direct",
        "combine": "rules",
        "decision_rule": "sum_class_prob",
    },
    "rules_cot_generic": {
        "branches": ["zs", "rag_top1", "rag_top2"],
        "answer_format": "cot_generic",
        "combine": "rules",
        "decision_rule": "sum_class_prob",
    },
    "rules_cot_task": {
        "branches": ["zs", "rag_top1", "rag_top2"],
        "answer_format": "cot_task",
        "combine": "rules",
        "decision_rule": "sum_class_prob",
    },
    "aggregator_cot": {
        "branches": ["zs", "rag_top1", "rag_top2"],
        # set to "cot_task" if rules_cot_task turns out better than rules_cot_generic
        "answer_format": "cot_generic",
        "combine": "aggregator",
    },
    "aggregator_answer_explain": {
        "branches": ["zs", "rag_top1", "rag_top2"],
        "answer_format": "answer_explain",
        "combine": "aggregator",
    },
    "aggregator_answer_explain_format": {
        "branches": ["zs", "rag_top1", "rag_top2"],
        "answer_format": "answer_explain_format",
        "combine": "aggregator",
    },
}


def validate_framework_request(framework, generator_id, top_k, enable_thinking):
    """Reject a framework run the code cannot honour, before any weights load."""
    config = FRAMEWORKS[framework]
    if generator_id != FRAMEWORK_GENERATOR:
        raise ValueError(f"--framework {framework} runs on '{FRAMEWORK_GENERATOR}' only, got '{generator_id}'.")
    if top_k != 2:
        raise ValueError(f"--framework {framework} needs --top_k 2: the gate and the branches use the top-1 and "
                         f"top-2 examples, got --top_k {top_k}.")
    if enable_thinking:
        raise ValueError(f"--framework {framework} does not support --enable_thinking.")

    answer_format = config["answer_format"]
    if answer_format != "direct" and any(text is None for text in ANSWER_FORMATS[answer_format].values()):
        raise ValueError(f"--framework {framework} uses the answer format '{answer_format}', whose prompt text "
                         f"is not written yet. Set it in ANSWER_FORMATS in pipelines/prompts.py.")
    if config["combine"] == "aggregator" and answer_format not in TEXT_COLUMN:
        raise ValueError(f"framework '{framework}' combines with the aggregator, which reads the branches' "
                         f"reasoning or explanation, but its answer format '{answer_format}' writes neither.")
    if config["combine"] == "rules" and answer_format in UNSCORED_FORMATS:
        raise ValueError(f"framework '{framework}' combines with the decision rules, which need the branches' "
                         f"confidence and class scores, but its answer format '{answer_format}' has none.")


def framework_columns(framework, classes_list):
    """The results columns a framework adds, in order (``route`` is added by the pipeline)."""
    config = FRAMEWORKS[framework]
    text_column = TEXT_COLUMN.get(config["answer_format"])
    scored = config["answer_format"] not in UNSCORED_FORMATS
    columns = []
    for branch in config["branches"]:
        columns += [f"{branch}__{key}" for key in branch_columns(text_column, classes_list, scored)]
    if config["combine"] == "rules":
        columns += [f"rule__{rule}" for rule in RULES]
    elif scored:
        columns += [f"{AGGREGATOR}__{key}" for key in branch_columns("reasoning", classes_list)]
    else:
        columns += [f"{AGGREGATOR}__{key}" for key in branch_columns(text_column, classes_list, scored=False)]
    return columns


def run_framework(framework, query_image, top_examples, ctx):
    """
    framework: a key of FRAMEWORKS.
    query_image: the query PIL image.
    top_examples: knowledge base rows of the top-2 retrieved examples, most similar first.
    ctx: a plain dict the pipeline builds once per run:
        model, processor, spec   the loaded generator
        classes_list             the classes, as prompted and as written in true_label
        prompt                   the RAG prompt structure (--prompt)
        debug                    print the prompts and generations; switched off after the first sample
        scoring_checked          whether the label scoring was compared with a full forward pass
        truncated_count          reasoning or explanation generations cut off by max_new_tokens
        context_exceeded_count   generation calls whose context went past the model's context window
        largest_context_tokens   the largest context among those calls
    returns: a flat dict: "route", "prediction" and every framework column.
    """
    config = FRAMEWORKS[framework]
    classes_list = ctx["classes_list"]
    row = {"route": "framework"}

    branch_results = {}
    for branch in config["branches"]:
        result = run_branch(branch, config["answer_format"], query_image, top_examples, ctx)
        branch_results[branch] = result
        row.update({f"{branch}__{key}": value for key, value in result.items()})

    if config["combine"] == "rules":
        for rule in RULES:
            row[f"rule__{rule}"] = predict(row, rule, config["branches"], classes_list)
        row["prediction"] = row[f"rule__{config['decision_rule']}"]
    elif config["answer_format"] in UNSCORED_FORMATS:
        responses = [result["response"] for result in branch_results.values()]
        instruction = ANSWER_FORMATS[config["answer_format"]]["instruction"]
        conversation = build_aggregator_responses_conversation(classes_list, responses, instruction)
        result = answer_in_format(AGGREGATOR, conversation, [query_image], ctx)
        row.update({f"{AGGREGATOR}__{key}": value for key, value in result.items()})
        row["prediction"] = result["prediction"]
    else:
        text_column = TEXT_COLUMN[config["answer_format"]]
        analyses = [(result[text_column], result["prediction"]) for result in branch_results.values()]
        conversation = build_aggregator_conversation(classes_list, analyses, text_column)
        result = reason_then_answer(AGGREGATOR, conversation, [query_image], AGGREGATOR_REASONING_TRIGGER, ctx)
        row.update({f"{AGGREGATOR}__{key}": value for key, value in result.items()})
        row["prediction"] = result["prediction"]

    ctx["debug"] = False
    return row
