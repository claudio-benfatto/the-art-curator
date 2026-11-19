"""Evaluation: extraction and discovery scored against hand-checked labels (P2), and the curator
scored over scripted conversations (P3).

`extraction.py` and `curation.py` are the scorers and do no I/O beyond reading their gold files.
`runner.py` finds the labelled pages' text and calls the model, through `llm/extract.py` like
`cli extract` does; `curation_runner.py` runs each scenario through the curator over the live
database and checks what its replies name in SQL.
"""
