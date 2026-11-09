"""Evaluation: extraction and discovery scored against hand-checked labels (P2).

`extraction.py` is the scorer and does no I/O beyond reading the gold set. `runner.py` finds the
labelled pages' text and calls the model, through `llm/extract.py` like `cli extract` does.
"""
