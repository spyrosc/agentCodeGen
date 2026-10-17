"""Full-stack app generator (multi-agent).

A LangGraph ``StateGraph`` with specialist nodes:

    planner -> scaffolder -> (backend ∥ frontend) -> integrator -> reviewer
                                                                      |
                                                                  (loop or submit)

Generates a complete project: FastAPI + SQLAlchemy + Alembic backend, React +
Vite + TS + Tailwind frontend, PostgreSQL via per-project ``docker-compose.yml``.
"""

from .agent import FullStackAgent  # noqa: F401  -- registers the agent
