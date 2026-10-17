"""Streamlit front-end for the agent platform.

Discovers agents through :mod:`genai.agents.registry` so adding a new agent
requires no UI changes — pick it from the sidebar dropdown.
"""

from __future__ import annotations

import traceback
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from genai.agents import all_agents
from genai.agents.base import AgentEvent
from genai.core.config import get_settings, reload_settings
from genai.core.logging import get_logger
from genai.sandbox import Workspace, new_project_id

load_dotenv(".env")
reload_settings()  # pick up .env values

log = get_logger("ui")
SETTINGS = get_settings()

# Registered agents not offered in the UI (still usable from the CLI, evals and tests).
_HIDDEN_AGENTS = frozenset({"reflect"})


def _ui_agents():
    return {name: cls for name, cls in all_agents().items() if name not in _HIDDEN_AGENTS}


# ----------------------------------------------------------- session state
def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("agent_name", next(iter(_ui_agents()), "codegen"))
    ss.setdefault("workspace", None)
    ss.setdefault("requirements", "")
    ss.setdefault("events", [])  # list[AgentEvent]
    ss.setdefault("messages", [])
    ss.setdefault("running", False)
    ss.setdefault("submitted", False)
    ss.setdefault("summary", None)
    ss.setdefault("error", None)


def _reset() -> None:
    for k in ("workspace", "events", "messages", "submitted", "summary", "error"):
        st.session_state[k] = [] if k in {"events", "messages"} else None
    st.session_state.submitted = False
    st.session_state.running = False


# ------------------------------------------------------------- formatting
_ICONS = {
    "workspace_created": "📁",
    "list_files": "📂",
    "read_file": "📖",
    "write_file": "📝",
    "delete_file": "🗑️",
    "run_in_sandbox": "⚙️",
    "run_in_sandbox_done": "  ↳",
    "run_tests": "🧪",
    "run_tests_done": "  ↳",
    "run_python_tests": "🐍",
    "run_python_tests_done": "  ↳",
    "run_frontend_check": "🌐",
    "run_frontend_check_done": "  ↳",
    "submit_project": "🏁",
    "submit_backend": "🎒",
    "submit_frontend": "🎨",
    "message": "💬",
    "specialist_message": "🗣️",
    "tool_result": "🔧",
    "phase_started": "✨",
    "phase_completed": "🏷️",
    "spec_drafted": "📐",
    "scaffolded": "🏗️",
    "compose_validated": "🐳",
    "review": "🔍",
    "clone_started": "📥",
    "clone_ready": "📦",
    "lesson_recorded": "🎓",
    "eval_gate_started": "🚦",
    "eval_gate_done": "🛡️",
    "patch_proposed": "📤",
    "patch_rejected": "🚫",
    "write_denied": "⛔",
    "run_lint": "🧹",
    "run_lint_done": "  ↳",
    "tool_warning": "⚠️",
    "budget_exhausted": "⏳",
    "error": "⚠️",
    "done": "✅",
}


def _render_event(ev: AgentEvent) -> str:
    icon = _ICONS.get(ev.kind, "•")
    d = ev.data
    scope = f" [{d.get('scope')}]" if d.get("scope") else ""
    if ev.kind == "workspace_created":
        return f"{icon} workspace `{d['project_id']}`"
    if ev.kind in {"list_files", "read_file", "delete_file"}:
        return f"{icon}{scope} {ev.kind} `{d.get('path')}`"
    if ev.kind == "write_file":
        return f"{icon}{scope} write_file `{d['path']}` ({d['bytes']} bytes)"
    if ev.kind == "run_in_sandbox":
        return f"{icon} run_in_sandbox: `{d['command']}`"
    if ev.kind == "run_in_sandbox_done":
        return f"{icon} exit={d['exit']}"
    if ev.kind in {"run_tests", "run_python_tests"}:
        return f"{icon}{scope} {ev.kind}"
    if ev.kind in {"run_tests_done", "run_python_tests_done"}:
        ok = "✅" if d["exit"] == 0 else "❌"
        return f"{icon}{scope} {ok} exit={d['exit']}"
    if ev.kind == "run_frontend_check":
        return f"{icon} run_frontend_check (timeout={d.get('timeout')}s)"
    if ev.kind == "run_frontend_check_done":
        ok = "✅" if d["exit"] == 0 else "❌"
        return f"{icon} {ok} exit={d['exit']}"
    if ev.kind == "phase_started":
        return f"{icon} **{d['phase']}** started"
    if ev.kind == "phase_completed":
        extras = " ".join(f"{k}={v}" for k, v in d.items() if k != "phase")
        return f"{icon} **{d['phase']}** completed {extras}".rstrip()
    if ev.kind == "spec_drafted":
        return f"{icon} spec drafted: {len(d.get('entities', []))} entities ({', '.join(d.get('entities', []))})"
    if ev.kind == "scaffolded":
        return f"{icon} scaffold wrote {d.get('count')} files"
    if ev.kind == "compose_validated":
        flag = "✅" if d.get("ok") else "❌"
        return f"{icon} compose {flag} (missing={len(d.get('missing', []))}, issues={len(d.get('issues', []))})"
    if ev.kind == "review":
        return f"{icon} review pass {d.get('pass')}: {d.get('status')} — {d.get('notes', '')}"
    if ev.kind in {"submit_project", "submit_backend", "submit_frontend"}:
        return f"{icon} {ev.kind}: {d.get('summary', '')}"
    if ev.kind == "specialist_message":
        return f"{icon}{scope} {d.get('content', '')[:160]}"
    if ev.kind == "tool_result":
        return f"{icon} [{d['name']}] {d['preview']}"
    if ev.kind == "tool_warning":
        return f"{icon} [{d.get('name')}] {d.get('preview')}"
    if ev.kind == "clone_started":
        return f"{icon} cloning source from `{d.get('mount')}`"
    if ev.kind == "clone_ready":
        sha = (d.get('base_sha') or '')[:8] or 'no-git'
        return f"{icon} clone ready at `{d.get('path')}` (base={sha})"
    if ev.kind == "lesson_recorded":
        return f"{icon} lesson recorded ({d.get('category')}): {(d.get('problem') or '')[:80]}"
    if ev.kind == "eval_gate_started":
        return f"{icon} eval gate started (cases={d.get('case_ids') or 'all'})"
    if ev.kind == "eval_gate_done":
        regs = d.get('regressions') or []
        fixed = d.get('fixed_cases') or []
        ok = "✅" if not regs else "❌"
        return f"{icon} {ok} eval gate: regressions={len(regs)} fixed={len(fixed)}"
    if ev.kind == "patch_proposed":
        return f"{icon} patch proposed on branch `{d.get('branch')}` ({d.get('files_changed')} files, {d.get('patch_lines')} lines)"
    if ev.kind == "patch_rejected":
        return f"{icon} patch rejected: {d.get('reason')}"
    if ev.kind == "write_denied":
        return f"{icon} write denied `{d.get('path')}`: {d.get('reason')}"
    if ev.kind == "run_lint":
        return f"{icon} ruff check"
    if ev.kind == "run_lint_done":
        ok = "✅" if d.get('exit') == 0 else "❌"
        return f"{icon} {ok} ruff exit={d.get('exit')}"
    if ev.kind == "budget_exhausted":
        return f"{icon}{scope} step budget exhausted ({d.get('limit')} steps)"
    if ev.kind == "error":
        return f"{icon} {d.get('message', '')}"
    if ev.kind == "done":
        return f"{icon} done (submitted={d['submitted']})"
    return f"• {ev.kind}"


def _lang_for(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    name = Path(filename).name.lower()
    if name == "dockerfile":
        return "dockerfile"
    return {
        ".py": "python", ".md": "markdown", ".json": "json",
        ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
        ".sh": "bash", ".txt": "text", ".cfg": "ini", ".ini": "ini",
        ".ts": "typescript", ".tsx": "tsx",
        ".js": "javascript", ".jsx": "jsx",
        ".css": "css", ".html": "html", ".sql": "sql",
        ".mako": "mako", ".env": "bash",
    }.get(suffix, "text")


# ------------------------------------------------------------- agent run
@st.cache_resource(show_spinner=False)
def _get_agent(name: str):
    return all_agents()[name]()


def _start_run(max_iter: int, *, eval_run_id: str | None = None) -> None:
    _reset()
    st.session_state.running = True

    project_id = new_project_id()
    workspace = Workspace.create(project_id)
    st.session_state.workspace = workspace

    try:
        with st.spinner(f"Agent `{st.session_state.agent_name}` working in `{project_id}`…"):
            agent = _get_agent(st.session_state.agent_name)
            stream_kwargs = {
                "workspace": workspace,
                "max_iterations": max_iter,
            }
            if eval_run_id:
                stream_kwargs["eval_run_id"] = eval_run_id
            for ev in agent.stream(
                st.session_state.requirements,
                **stream_kwargs,
            ):
                st.session_state.events.append(ev)
                if ev.kind == "message":
                    st.session_state.messages.append(ev.data)
                if ev.kind in {"submit_project", "patch_proposed"}:
                    st.session_state.submitted = True
                    st.session_state.summary = ev.data.get("summary")
    except Exception as exc:  # noqa: BLE001
        st.session_state.error = f"{type(exc).__name__}: {exc}"
        log.error("agent run failed:\n%s", traceback.format_exc())
    finally:
        st.session_state.running = False
        st.rerun()


# ------------------------------------------------------------------ main
def main() -> None:
    st.set_page_config(page_title="GenAI Agent Platform", page_icon="🤖", layout="wide")
    _init_state()

    st.title("🤖 GenAI Agent Platform")
    st.caption(
        f"model `{SETTINGS.llm.model}` · sandbox `{SETTINGS.sandbox.container_name}` · "
        f"workspace `{SETTINGS.sandbox.workspace_root}`"
    )

    with st.sidebar:
        st.header("Run settings")
        agents = _ui_agents()
        st.session_state.agent_name = st.selectbox(
            "Agent",
            list(agents),
            index=list(agents).index(st.session_state.agent_name)
            if st.session_state.agent_name in agents else 0,
            help="Agents are auto-discovered from the registry.",
        )
        st.caption(getattr(agents[st.session_state.agent_name], "description", ""))
        max_iter = st.slider(
            "Max agent steps", 5, 120, value=SETTINGS.agent.max_iterations, step=5
        )
        if st.session_state.agent_name == "reflect":
            st.divider()
            st.caption("Reflect agent")
            evals_dir = Path(SETTINGS.sandbox.workspace_root) / "_evals"
            run_ids: list[str] = []
            if evals_dir.exists():
                run_ids = sorted(
                    (p.name for p in evals_dir.iterdir() if p.is_dir()),
                    reverse=True,
                )[:30]
            options = ["(none)"] + run_ids
            chosen = st.selectbox(
                "Failing eval run id",
                options,
                help="Optional: focus reflection on a past eval run.",
            )
            st.session_state["reflect_eval_run_id"] = (
                None if chosen == "(none)" else chosen
            )
        st.divider()
        if st.button("New project", use_container_width=True):
            _reset()
            st.rerun()
        ws = st.session_state.workspace
        if ws is not None:
            st.caption(f"Project id: `{ws.project_id}`")

    req_col, log_col = st.columns([1, 1], gap="large")

    with req_col:
        st.subheader("Task / requirements")
        st.session_state.requirements = st.text_area(
            "Describe what you want built",
            value=st.session_state.requirements,
            height=320,
            placeholder=(
                "Example:\n"
                "Build a CLI tool that takes a CSV file and prints summary "
                "statistics (count, mean, min, max) for every numeric column. "
                "Include pytest tests covering an empty file and a file with "
                "mixed types."
            ),
            disabled=st.session_state.running,
        )
        if st.button(
            "▶  Run agent",
            type="primary",
            disabled=st.session_state.running or not st.session_state.requirements.strip(),
        ):
            eval_run_id = st.session_state.get("reflect_eval_run_id") or None
            _start_run(max_iter, eval_run_id=eval_run_id)

        if st.session_state.error:
            st.error(st.session_state.error)
        if st.session_state.submitted:
            st.success("Project submitted by the agent.")
            if st.session_state.summary:
                st.info(st.session_state.summary)

        # Reflect-agent patch viewer.
        patch_event = next(
            (e for e in st.session_state.events if e.kind == "patch_proposed"),
            None,
        )
        if patch_event is not None:
            with st.expander("📤 Proposed patch", expanded=True):
                d = patch_event.data
                st.markdown(
                    f"**Branch:** `{d.get('branch')}`  \n"
                    f"**Files changed:** {d.get('files_changed')}  \n"
                    f"**Lines:** {d.get('patch_lines')}  \n"
                    f"**Base SHA:** `{(d.get('base_sha') or 'n/a')[:12]}`"
                )
                patch_path = d.get("patch_path")
                if patch_path:
                    p = Path(patch_path)
                    if p.exists():
                        diff_text = p.read_text(encoding="utf-8", errors="replace")
                        st.code(diff_text, language="diff")
                        st.download_button(
                            "⬇  Download patch.diff",
                            data=diff_text,
                            file_name="patch.diff",
                            mime="text/x-diff",
                        )

        ws = st.session_state.workspace
        if ws is not None and Path(ws.root).exists():
            st.subheader("Project files")
            entries = ws.list_tree()
            if entries:
                st.code("\n".join(entries), language="text")
                files = [e for e in entries if not e.endswith("/") and not e.startswith("...")]
                if files:
                    chosen = st.selectbox("Preview file", files)
                    if chosen:
                        try:
                            content = ws.resolve(chosen).read_text(
                                encoding="utf-8", errors="replace"
                            )
                        except Exception as exc:  # noqa: BLE001
                            content = f"<could not read: {exc}>"
                        st.code(content, language=_lang_for(chosen))
                try:
                    st.download_button(
                        "⬇  Download project (.zip)",
                        data=ws.zip_bytes(),
                        file_name=f"{ws.project_id}.zip",
                        mime="application/zip",
                    )
                except Exception as exc:  # noqa: BLE001
                    st.warning(f"Zip failed: {exc}")
            else:
                st.caption("Workspace is empty.")

    with log_col:
        st.subheader("Activity")
        if not st.session_state.events:
            st.caption("Submit a task to start the agent.")
        for ev in st.session_state.events:
            st.markdown(_render_event(ev))
        if st.session_state.messages:
            st.divider()
            st.subheader("Agent messages")
            for m in st.session_state.messages[-30:]:
                with st.chat_message(m.get("role", "assistant")):
                    st.markdown(m.get("content", ""))


main()
