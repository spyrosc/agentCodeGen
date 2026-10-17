# GenAI Agent Platform

A small, **pluggable platform for building autonomous LLM agents** that can
read free-text instructions, plan, and produce real artifacts. Two agents
ship with the platform:

- **`codegen`** — a LangGraph ReAct agent that turns functional requirements
  into a working multi-file Python project, validating it inside a sandboxed
  Docker container.
- **`fullstack`** — a multi-agent (LangGraph `StateGraph`) orchestrator that
  produces a complete app from a brief: **FastAPI + SQLAlchemy + Alembic**
  backend, **React + Vite + TypeScript + TailwindCSS** frontend, **PostgreSQL**
  via a per-project `docker-compose.yml`. Bundled JWT auth, OpenAPI docs and
  CORS configured for the frontend.

The platform is designed to grow. New agents register themselves and appear
in the UI dropdown automatically. New sandbox transports plug into a single
`Sandbox` protocol. New LLM providers slot into one `load_llm` dispatcher.

```
┌─────────── Streamlit UI (port 8506) ───────────┐
│   agent dropdown · task input · live activity  │
└──────────────────────┬─────────────────────────┘
                       │
              ┌────────▼────────┐    registers
              │ agents.registry │◀───────────────  CodegenAgent, …
              └────────┬────────┘
                       │
        ┌──────────────▼──────────────┐
        │ LangGraph ReAct (codegen)   │
        │  tools: write_file, run_…   │
        └────────┬────────┬───────────┘
                 │        │
       writes    │        │ exec
                 ▼        ▼
        ┌─────────────┐  ┌─────────────────┐
        │  Workspace  │  │ Sandbox (docker │
        │ /workspaces │  │  exec transport)│
        │    /<id>    │  │  no net access  │
        └─────────────┘  └─────────────────┘
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full architecture
with mermaid diagrams, [docs/adding-an-agent.md](docs/adding-an-agent.md)
for extension how-to, [docs/SECURITY.md](docs/SECURITY.md) for the
threat model, and [docs/ROADMAP.md](docs/ROADMAP.md) for what's next.

## Quickstart

```bash
git clone <this-repo> genai-agent-platform
cd genai-agent-platform
cp .env.example .env          # edit LLM, OLLAMA_BASE_URL, API keys, …
docker compose up --build     # builds agent + sandbox, starts UI on :8506
```

Open <http://localhost:8506>, paste a requirements brief, click **Run agent**.
When the agent calls `submit_project` you can preview every generated file
and download the project as a zip.

### Generating a full app (`fullstack` agent)

Pick **`fullstack`** in the sidebar dropdown and submit a brief like:

> Build a TODO list with users, JWT login, and CRUD tasks owned by the user.
> Tasks have a title, an optional due date, and a done flag.

The agent runs `plan → scaffold → backend → frontend → integrate → review` in
sequence; the activity panel shows each phase. Download the resulting zip,
unpack it, and on your host run:

```bash
cp .env.example .env
docker compose up --build
# frontend: http://localhost:5173
# backend OpenAPI: http://localhost:8000/docs
```

The sandbox container needs Node.js as well as Python (it now bundles both)
so the frontend specialist can run `tsc --noEmit && vite build` for
validation. The first frontend check pays a one-time `npm install` cost.

### Local development without Docker

```bash
make install-dev      # editable install + pytest, ruff
make ui               # streamlit on :8506 (sandbox container still required)
make test             # run the test suite (no LLM/Docker needed)
make lint             # ruff
```

The Streamlit app needs the `sandbox` container running so the agent has
somewhere to execute generated code:

```bash
docker compose up sandbox
```

## Configuration

All configuration lives in environment variables (loaded from `.env`).
See [.env.example](.env.example) for the full list. Highlights:

| Variable                  | Default                         | Purpose                                   |
|---------------------------|---------------------------------|-------------------------------------------|
| `LLM`                     | `llama2`                        | Model id; provider inferred from name     |
| `LLM_TIMEOUT_S`           | `600`                           | Per-request LLM timeout (`0` disables)    |
| `LLM_MAX_RETRIES`         | `2`                             | Provider retries on transient errors (not Ollama) |
| `OLLAMA_BASE_URL`         | `http://host.docker.internal:11434` | Used when LLM is an Ollama model      |
| `ANTHROPIC_API_KEY`       | —                               | Required when `LLM=claude-*` (Anthropic API) |
| `ANTHROPIC_MAX_TOKENS`    | `8192`                          | Max output tokens per Claude response     |
| `WORKSPACE_HOST_DIR`      | `./workspaces`                  | Host dir mounted into both containers     |
| `WORKSPACE_ROOT`          | `/workspaces`                   | Mount target inside both containers       |
| `SANDBOX_CONTAINER`       | `genai-sandbox`                 | Container the agent shells into           |
| `SANDBOX_TIMEOUT_S`       | `180`                           | Per-command sandbox timeout               |
| `SANDBOX_MAX_FILE_BYTES`  | `64000`                         | `read_file` truncation shown to the LLM   |
| `SANDBOX_MAX_OUTPUT_CHARS`| `8000`                          | Command output truncation shown to the LLM |
| `AGENT_MAX_ITERATIONS`    | `40`                            | LangGraph recursion limit per run         |
| `FULLSTACK_POSTGRES_IMAGE`| `postgres:16-alpine`            | DB image in generated `docker-compose.yml` |

## Project layout

```
genAIv2/
├── docker-compose.yml          two services: agent + sandbox
├── docker/                     Dockerfiles
├── scripts/run_cli.py          run any agent without the UI
├── src/genai/
│   ├── core/                   config + logging
│   ├── llm/                    one load_llm() entrypoint per provider
│   ├── sandbox/                Sandbox protocol + workspaces + transports
│   ├── agents/                 base contracts, registry, built-in agents
│   │   └── codegen/            the LangGraph ReAct codegen agent
│   └── ui/                     Streamlit front-end (agent-agnostic)
├── tests/                      pytest, no Docker/LLM required
├── workspaces/                 generated projects land here (gitignored)
├── pyproject.toml
├── Makefile
└── requirements*.txt
```

