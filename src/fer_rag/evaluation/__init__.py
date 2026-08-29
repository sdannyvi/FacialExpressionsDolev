"""Evaluation layer: two files, two roles.

``vis_results``  generic, dataset-agnostic plotting and reporting primitives
                 (classification reports, confusion matrices, label distributions).
                 Working copy of LLaVa/llava_rag/vis_results.py.

``analysis``     FER-specific analysis semantics built on those primitives
                 (retriever vs RAG, retrieval-case error analysis, RAG vs zero-shot
                 on retrieval failures, retrieval tie statistics).

The dependency runs one way: ``analysis`` imports ``vis_results``, never the reverse.
Experiment scripts import from the two modules directly - the public surface is not
settled yet, so nothing is re-exported here.
"""
