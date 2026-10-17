"""Evaluation harness for registered agents.

Drives any :class:`~genai.agents.base.Agent` through a declarative YAML suite,
runs it live (real LLM + real sandbox), scores the resulting workspace, and
writes JSONL + Markdown artifacts under ``workspaces/_evals/<run_id>/``.
"""

from .cases import EvalCase, EvalSuite, FunctionalCheck
from .harness import CaseResult, SuiteResult, run_case, run_suite
from .suite import load_suite

__all__ = [
    "CaseResult",
    "EvalCase",
    "EvalSuite",
    "FunctionalCheck",
    "SuiteResult",
    "load_suite",
    "run_case",
    "run_suite",
]
