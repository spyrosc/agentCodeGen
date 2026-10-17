"""Reflection memory: lessons learnt that survive across runs.

Centred on a JSONL :class:`~genai.reflection.store.LessonStore` and a small
:func:`~genai.reflection.injection.prepend_lessons` helper agents can use
to surface relevant past failures in their system prompts.
"""

from .injection import prepend_lessons  # noqa: F401
from .seed_from_evals import seed_from_run  # noqa: F401
from .store import Lesson, LessonStore, sanitise_text  # noqa: F401
