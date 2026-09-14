"""Decision rules that combine a framework's branches into one prediction.

Pure functions over one results row (a dict built by the pipeline, or a row of the results csv), so
the pipeline and offline analysis compute every rule with the same code. Passing a subset of a
framework's branches gives an ablation without running the model again:

    apply_rule(df, "sum_class_prob", ["rag_top1", "rag_top2"], classes_list)

Column names follow the results csv: ``<branch>__prediction``, ``<branch>__confidence``,
``<branch>__conf__<class>`` and ``<branch>__prob__<class>``.

    max_conf        the answer of the branch whose generated answer is the most confident
    sum_conf        each branch adds its confidence to its own answer; the answer with the highest sum
    sum_class_conf  the class with the highest sum over branches of conf__<class>
    sum_class_prob  the class with the highest sum over branches of prob__<class>

Ties go to the branch listed first, and in the class rules to the class listed first in
classes_list. A branch with no answer (empty prediction, so no confidence) does not vote in
max_conf and sum_conf. A rule that has nothing to decide on returns None.

Read the results csv with ``pd.read_csv(path, float_precision="round_trip")`` to get back the
stored values exactly.
"""
import math

import pandas as pd

RULES = ["max_conf", "sum_conf", "sum_class_conf", "sum_class_prob"]


def _answers(row, branches):
    """(answer, confidence) of every branch that produced an answer, in branch order."""
    answers = []
    for branch in branches:
        prediction = row[f"{branch}__prediction"]
        confidence = row[f"{branch}__confidence"]
        if isinstance(prediction, str) and prediction and pd.notna(confidence):
            answers.append((prediction, float(confidence)))
    return answers


def predict(row, rule, branches, classes_list):
    """
    row: one results row, a dict or a pandas Series.
    rule: one of RULES.
    branches: the branches that take part, e.g. ["zs", "rag_top1", "rag_top2"].
    classes_list: the classes, in the order used to break ties.
    returns: the predicted label, or None when the rule has nothing to decide on.
    """
    if rule == "max_conf":
        answers = _answers(row, branches)
        if not answers:
            return None
        # max keeps the first of equal confidences, so ties go to the earlier branch
        return max(answers, key=lambda answer: answer[1])[0]

    if rule == "sum_conf":
        totals = {}
        for prediction, confidence in _answers(row, branches):
            totals[prediction] = totals.get(prediction, 0.0) + confidence
        if not totals:
            return None
        return max(totals, key=totals.get)

    if rule in ("sum_class_conf", "sum_class_prob"):
        score = "conf" if rule == "sum_class_conf" else "prob"
        totals = {}
        for class_name in classes_list:
            values = [float(row[f"{branch}__{score}__{class_name}"]) for branch in branches]
            # a row the gate sent to the original RAG has no class scores
            if any(math.isnan(value) for value in values):
                return None
            totals[class_name] = sum(values)
        return max(totals, key=totals.get)

    raise ValueError(f"unknown decision rule '{rule}'. Available rules: {', '.join(RULES)}")


def apply_rule(df, rule, branches, classes_list):
    """
    df: results dataframe.
    returns: a Series with the rule's prediction for every row (None for rows it cannot decide).
    """
    return df.apply(lambda row: predict(row, rule, branches, classes_list), axis=1)
