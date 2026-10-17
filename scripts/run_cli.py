"""Run an agent from the command line, no Streamlit required.

Usage:
    python scripts/run_cli.py codegen "Build a Fibonacci CLI with pytest tests"
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from genai.agents import all_agents
from genai.core.config import reload_settings


def main() -> int:
    load_dotenv(".env")
    reload_settings()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("agent", choices=sorted(all_agents()), help="Registered agent name")
    parser.add_argument("task", help="Task / requirements text (or '-' to read from stdin)")
    parser.add_argument("--max-iterations", type=int, default=None)
    args = parser.parse_args()

    task = sys.stdin.read() if args.task == "-" else args.task

    agent_cls = all_agents()[args.agent]
    agent = agent_cls()

    for ev in agent.stream(task, max_iterations=args.max_iterations):
        print(f"[{ev.kind}] {ev.data}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
