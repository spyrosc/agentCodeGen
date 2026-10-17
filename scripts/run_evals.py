"""Run an eval suite end-to-end.

Usage:
    python scripts/run_evals.py --suite evals/suites/codegen_smoke.yaml
    python scripts/run_evals.py --suite evals/suites/fullstack_smoke.yaml \\
        --enable-frontend-build --case todo_app
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from genai.core.config import reload_settings
from genai.evals import load_suite, run_suite
from genai.evals.report import render_table, write_markdown


def main() -> int:
    load_dotenv(".env")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, help="Path to a suite YAML file")
    parser.add_argument(
        "--case", action="append", default=None, help="Run only the named case (repeatable)"
    )
    parser.add_argument("--model", default=None, help="Override the LLM model name")
    parser.add_argument(
        "--enable-frontend-build",
        action="store_true",
        help="Run npm_build functional checks (slow; off by default)",
    )
    parser.add_argument(
        "--no-functional",
        action="store_true",
        help="Skip all functional checks (structural-only fast pass)",
    )
    parser.add_argument(
        "--base-dir",
        default=None,
        help="Override the eval run base directory (default: <workspace_root>/_evals)",
    )
    args = parser.parse_args()

    if args.model:
        os.environ["LLM"] = args.model
    reload_settings()

    suite = load_suite(args.suite)
    suite_result = run_suite(
        suite,
        base_dir=Path(args.base_dir) if args.base_dir else None,
        only=args.case,
        enable_frontend_build=args.enable_frontend_build,
        skip_functional=args.no_functional,
    )

    # Locate the run dir from the JSONL writer (mirrors harness convention).
    from genai.core.config import get_settings

    base = Path(args.base_dir) if args.base_dir else Path(get_settings().sandbox.workspace_root) / "_evals"
    run_dir = base / suite_result.run_id
    write_markdown(run_dir, suite_result)

    print(render_table(suite_result))
    print(f"\nArtifacts: {run_dir}")
    return 0 if suite_result.failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
