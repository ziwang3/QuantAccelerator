"""Which QuantAccelerator build runs, from $QA_BUILD: "v1" (default) or "prototype".

The prototype is v1 without the verification fixes made after the first batch of runs: no
grounding check on code meanings, no dataset-specific PIT evidence checks, and plain page retrieval in read_doc (no
section boost, no whole code-list sections, no cross-reference following). Prompts, tools, model and the loop's
robustness fixes are shared, so a prototype-vs-v1 comparison isolates what the checks buy.
"""
import os

BUILD = os.environ.get("QA_BUILD", "v1")
if BUILD not in ("v1", "prototype"):
    raise ValueError(f"QA_BUILD must be 'v1' or 'prototype', not {BUILD!r}")


def prototype() -> bool:
    return BUILD == "prototype"
