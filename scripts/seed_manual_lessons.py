"""Seed the LessonStore with a curated set of gotchas discovered during
manual end-to-end runs of the fullstack agent.

These are injected (when ``REFLECT_LESSONS_ENABLED=true``) into the
backend / frontend / planner specialist prompts so the model avoids
re-introducing already-fixed failure modes.

Usage:
    python scripts/seed_manual_lessons.py
"""

from __future__ import annotations

import sys

# Make ``src/`` importable when run directly.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genai.reflection.store import Lesson, LessonStore  # noqa: E402

CURATED: list[Lesson] = [
    Lesson(
        category="fullstack-backend",
        problem=(
            "Backend container crashes at import with "
            "`ImportError: email-validator is not installed` when any "
            "schema uses `pydantic.EmailStr`."
        ),
        fix=(
            "Make sure `pydantic[email]` (or `email-validator>=2`) is in "
            "backend/pyproject.toml dependencies. The scaffold already lists "
            "it; do NOT remove it when editing pyproject.toml."
        ),
        tags=["fullstack", "backend", "pydantic", "email-validator"],
        source="manual",
    ),
    Lesson(
        category="fullstack-backend",
        problem=(
            "`run_python_tests` returns exit=127 because the agent tried to "
            "invoke `pytest`/`pip` from a directory that does not exist, or "
            "shelled out to a tool that is not on PATH."
        ),
        fix=(
            "Use the provided `run_python_tests` tool — it already runs "
            "`pip install -e .` then `pytest` from the correct cwd. Do NOT "
            "write your own `run_tests.py` or shell wrappers; do NOT try to "
            "invoke `docker` from inside the sandbox."
        ),
        tags=["fullstack", "backend", "pytest", "tooling"],
        source="manual",
    ),
    Lesson(
        category="fullstack-backend",
        problem=(
            "Backend service in docker-compose fails to start because the "
            "Alembic migration is missing or broken, so `alembic upgrade "
            "head` exits non-zero before uvicorn can launch."
        ),
        fix=(
            "The scaffold ships a `backend/start.sh` that falls back to "
            "`Base.metadata.create_all(engine)` when alembic fails. Keep "
            "`backend/Dockerfile` CMD as `[\"sh\", \"/app/start.sh\"]` and do "
            "NOT override it. Migrations are still preferred, but the "
            "fallback keeps the stack runnable."
        ),
        tags=["fullstack", "backend", "alembic", "docker-compose"],
        source="manual",
    ),
    Lesson(
        category="fullstack-frontend",
        problem=(
            "TypeScript build fails with `Property 'env' does not exist on "
            "type 'ImportMeta'` when code reads `import.meta.env.VITE_*`."
        ),
        fix=(
            "The scaffold ships `frontend/src/vite-env.d.ts` with the "
            "`/// <reference types=\"vite/client\" />` triple-slash. Do NOT "
            "delete it. If you need extra typed env vars, extend the "
            "`ImportMetaEnv` interface there."
        ),
        tags=["fullstack", "frontend", "vite", "typescript"],
        source="manual",
    ),
    Lesson(
        category="fullstack-frontend",
        problem=(
            "TS build fails with `Property 'isLoading' does not exist on "
            "type 'UseMutationResult<...>'` when calling `useMutation`."
        ),
        fix=(
            "React Query v5 renamed `isLoading` to `isPending` on "
            "`useMutation` results. Use `mutation.isPending` (not "
            "`isLoading`). `useQuery` still has `isLoading`, but prefer "
            "`isPending` consistently."
        ),
        tags=["fullstack", "frontend", "react-query", "tanstack", "typescript"],
        source="manual",
    ),
    Lesson(
        category="fullstack-frontend",
        problem=(
            "Frontend specialist gets stuck looping `npm run build` to "
            "check TypeScript when the React-Query v5 API has changed."
        ),
        fix=(
            "Call `run_frontend_check` at most a handful of times. If the "
            "same TS error class repeats twice in a row, the API changed: "
            "verify against React Query v5 docs (single options object, "
            "`isPending`, `queryKey`/`queryFn`) before re-running."
        ),
        tags=["fullstack", "frontend", "react-query", "iteration"],
        source="manual",
    ),
    Lesson(
        category="fullstack-plan",
        problem=(
            "Planner produces a spec with more entities/pages than the user "
            "asked for, which inflates the rest of the pipeline and makes "
            "the specialists run out of iterations."
        ),
        fix=(
            "When the prompt says 'tiny', 'minimal', or names exactly one "
            "entity, produce exactly one entity and at most one page. Do "
            "NOT invent admin dashboards, settings pages, or extra "
            "relationships the user did not request."
        ),
        tags=["fullstack", "plan", "scope"],
        source="manual",
    ),
    Lesson(
        category="fullstack-backend",
        problem=(
            "POST /auth/register returns 500 with "
            "`AttributeError: module 'bcrypt' has no attribute '__about__'` "
            "because `passlib` 1.7.x introspects `bcrypt.__about__` which "
            "was removed in bcrypt 4.x."
        ),
        fix=(
            "Pin `bcrypt<4` in backend/pyproject.toml alongside "
            "`passlib[bcrypt]>=1.7`. Without the pin, pip resolves bcrypt 4.x "
            "and password hashing crashes at first /auth/register call."
        ),
        tags=["fullstack", "backend", "auth", "passlib", "bcrypt"],
        source="manual",
    ),
    Lesson(
        category="fullstack-backend",
        problem=(
            "GET/POST endpoints that return a model with a `datetime` field "
            "raise `PydanticUserError: TypeAdapter[...BookmarkOut...] is not "
            "fully defined` at runtime because the appended entity schema "
            "uses `datetime` without importing it (the file has "
            "`from __future__ import annotations` so the missing import is "
            "only caught at response-validation time)."
        ),
        fix=(
            "The scaffold's `app/schemas.py` already imports `datetime`, "
            "`date`, and `Optional` at the top. When you append entity "
            "schemas under `# >>> AUTOGEN SCHEMAS <<<`, use those names "
            "directly — do NOT re-import them and do NOT use any other "
            "type without first adding its import above the autogen marker."
        ),
        tags=["fullstack", "backend", "pydantic", "schemas", "datetime"],
        source="manual",
    ),
]


def main() -> int:
    store = LessonStore.from_settings()
    appended = 0
    for lesson in CURATED:
        if store.append(lesson):
            appended += 1
    total = len(store.all())
    print(f"appended {appended} new lessons (skipped {len(CURATED) - appended}); store now has {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
