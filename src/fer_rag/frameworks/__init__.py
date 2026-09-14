"""Framework layer: what runs on a sample whose top-1 and top-2 retrieved labels differ.

``registry``        the gate (``needs_new_framework``), the ``FRAMEWORKS`` table, and ``run_framework``
``branches``        one branch of a framework: its prompt, its answer, and its class scores
``decision_rules``  pure functions that combine the branches' results into one prediction

Nothing is imported here on purpose: importing ``decision_rules`` for offline analysis must not
load torch and the generator code that ``registry`` and ``branches`` need.
"""
