"""Markdown report rendering for an eval suite run."""

from __future__ import annotations

from pathlib import Path

from .harness import CaseResult, SuiteResult


def write_markdown(run_dir: Path, suite: SuiteResult) -> Path:
    out = run_dir / "report.md"
    out.write_text(render_markdown(suite), encoding="utf-8")
    return out


def render_markdown(suite: SuiteResult) -> str:
    lines: list[str] = []
    lines.append(f"# Eval run `{suite.run_id}`")
    lines.append("")
    lines.append(f"- **Suite**: {suite.suite_name}")
    lines.append(f"- **Model**: `{suite.model}`")
    lines.append(f"- **Started**: {suite.started_at}")
    lines.append(f"- **Finished**: {suite.finished_at}")
    lines.append(f"- **Wall time**: {suite.wall_seconds}s")
    lines.append(
        f"- **Results**: {suite.passed_count} passed, {suite.failed_count} failed, "
        f"{suite.skipped_count} skipped"
    )

    total_calls = sum(c.telemetry.get("llm_calls", 0) for c in suite.cases)
    total_tokens = sum(c.telemetry.get("total_tokens", 0) for c in suite.cases)
    estimated = any(c.telemetry.get("tokens_estimated") for c in suite.cases)
    tok_label = f"{total_tokens}{'~est' if estimated else ''}"
    lines.append(f"- **LLM**: {total_calls} calls, {tok_label} tokens")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Case | Agent | Pass | Wall (s) | LLM calls | Tokens | Notes |")
    lines.append("|------|-------|------|----------|-----------|--------|-------|")
    for c in suite.cases:
        verdict = "skip" if c.skipped else ("pass" if c.passed else "FAIL")
        tokens = c.telemetry.get("total_tokens", 0)
        if c.telemetry.get("tokens_estimated"):
            tokens_str = f"{tokens}~est"
        else:
            tokens_str = str(tokens)
        notes = c.skip_reason or (c.error.splitlines()[0] if c.error else "")
        notes = notes.replace("|", "\\|")[:80]
        lines.append(
            f"| {c.case_id} | {c.agent} | {verdict} | {c.wall_seconds} | "
            f"{c.telemetry.get('llm_calls', 0)} | {tokens_str} | {notes} |"
        )
    lines.append("")

    for c in suite.cases:
        lines.append(f"## Case `{c.case_id}`")
        lines.append("")
        if c.skipped:
            lines.append(f"_Skipped: {c.skip_reason}_")
            lines.append("")
            continue
        lines.extend(_render_case(c))

    return "\n".join(lines) + "\n"


def _render_case(c: CaseResult) -> list[str]:
    lines: list[str] = []
    lines.append(f"- Agent: `{c.agent}`")
    lines.append(f"- Project: `{c.project_id}`")
    lines.append(f"- Submitted: `{c.submitted}`  ·  Pass: `{c.passed}`")
    lines.append(f"- Wall: {c.wall_seconds}s")
    if c.error:
        lines.append("")
        lines.append("**Error**")
        lines.append("```")
        lines.append(c.error.strip())
        lines.append("```")
    if c.summary:
        lines.append("")
        lines.append(f"_Summary_: {c.summary}")

    s = c.structural or {}
    if s:
        lines.append("")
        lines.append("**Structural**")
        if s.get("expected_files_present"):
            lines.append(f"- Files present ({len(s['expected_files_present'])}): "
                         + ", ".join(f"`{p}`" for p in s["expected_files_present"]))
        if s.get("expected_files_missing"):
            lines.append(f"- Files missing ({len(s['expected_files_missing'])}): "
                         + ", ".join(f"`{p}`" for p in s["expected_files_missing"]))
        if s.get("integrator_ok") is not None:
            lines.append(f"- Integrator: `{s['integrator_ok']}`")
            if s.get("integrator_missing"):
                lines.append(f"  - missing: {', '.join(s['integrator_missing'])}")
            if s.get("integrator_issues"):
                lines.append(f"  - issues: {', '.join(s['integrator_issues'])}")
        if s.get("syntax_ok") is not None:
            lines.append(f"- Syntax (compileall): `{s['syntax_ok']}`")
            if not s["syntax_ok"] and s.get("syntax_output"):
                lines.append("```")
                lines.append(s["syntax_output"])
                lines.append("```")

    f = c.functional or {}
    checks = f.get("checks") or []
    if checks:
        lines.append("")
        lines.append("**Functional**")
        for ch in checks:
            verdict = "pass" if ch["passed"] else "FAIL"
            req = "" if ch["required"] else " (optional)"
            lines.append(
                f"- `{ch['name']}` ({ch['type']}){req}: {verdict} "
                f"— exit={ch['exit_code']}, {ch['seconds']}s"
            )
            if not ch["passed"] and ch.get("output_tail"):
                lines.append("```")
                lines.append(ch["output_tail"])
                lines.append("```")

    t = c.telemetry or {}
    if t:
        lines.append("")
        lines.append("**Telemetry**")
        tok = t.get("total_tokens", 0)
        tok_label = f"{tok}{'~est' if t.get('tokens_estimated') else ''}"
        lines.append(
            f"- LLM: {t.get('llm_calls', 0)} calls, "
            f"{t.get('llm_seconds', 0)}s, {tok_label} tokens "
            f"(prompt={t.get('prompt_tokens', 0)}, completion={t.get('completion_tokens', 0)})"
        )
        kinds = t.get("event_kinds") or {}
        if kinds:
            top = sorted(kinds.items(), key=lambda kv: -kv[1])[:10]
            lines.append("- Top events: " + ", ".join(f"`{k}`={v}" for k, v in top))
    lines.append("")
    return lines


def render_table(suite: SuiteResult) -> str:
    """Plain-text version for stdout."""
    rows = [
        f"{'CASE':<28} {'AGENT':<10} {'VERDICT':<7} {'WALL':>7} {'CALLS':>6} {'TOKENS':>10}"
    ]
    for c in suite.cases:
        verdict = "skip" if c.skipped else ("pass" if c.passed else "FAIL")
        tok = c.telemetry.get("total_tokens", 0)
        tok_str = f"{tok}{'~' if c.telemetry.get('tokens_estimated') else ''}"
        rows.append(
            f"{c.case_id[:28]:<28} {c.agent[:10]:<10} {verdict:<7} "
            f"{c.wall_seconds:>7.1f} {c.telemetry.get('llm_calls', 0):>6} {tok_str:>10}"
        )
    rows.append(
        f"-> {suite.passed_count} passed / {suite.failed_count} failed / "
        f"{suite.skipped_count} skipped  in {suite.wall_seconds}s"
    )
    return "\n".join(rows)
