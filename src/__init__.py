"""Application subpackages (`src.datasets`, `src.experiments`, …).

Avoid eager imports here: scripts and tests import `src.*` modules directly; pulling
optional research deps (e.g. DSPy) via this file breaks lightweight entrypoints.
"""
