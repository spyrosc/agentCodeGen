# Sandbox container: where generated code runs.
# - No exposed ports.
# - Not attached to the application network in docker-compose.yml.
# - The agent shells in via `docker exec`.
#
# Multi-language image:
#   * Python 3.11 + pytest (codegen agent)
#   * Node.js 20 + npm (fullstack agent — frontend builds)
#   * sqlite3 CLI (fullstack backend tests use SQLite)
#   * Pre-installed FastAPI/SQLAlchemy/Alembic/etc. so backend tests skip pip install per run.
#   * Docker CLI + Compose v2 plugin — used by the fullstack agent to bring up
#     the generated `docker-compose.yml` via the host docker socket (mounted
#     into this container in docker-compose.yml). Effectively docker-out-of-
#     docker; local development only (see SECURITY note in docker-compose.yml).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    NODE_MAJOR=20

# System packages: Node.js 20 (NodeSource), Docker CLI + Compose plugin,
# sqlite3, build essentials for native deps.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg sqlite3 build-essential git lsb-release \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
        | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_${NODE_MAJOR}.x nodistro main" \
        > /etc/apt/sources.list.d/nodesource.list \
    && curl -fsSL https://download.docker.com/linux/debian/gpg \
        | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
        > /etc/apt/sources.list.d/docker.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        nodejs docker-ce-cli docker-compose-plugin \
    && rm -rf /var/lib/apt/lists/*

# Python baseline: codegen test runner + fullstack backend stack.
RUN pip install --no-cache-dir \
        pytest==8.* \
        pytest-asyncio==0.23.* \
        httpx==0.27.* \
        fastapi==0.115.* \
        "uvicorn[standard]==0.30.*" \
        sqlalchemy==2.0.* \
        alembic==1.13.* \
        pydantic==2.* \
        pydantic-settings==2.* \
        email-validator==2.* \
        "python-jose[cryptography]==3.3.*" \
        "passlib[bcrypt]==1.7.*" \
        "bcrypt<4" \
        python-multipart==0.0.* \
        ruff==0.6.*

WORKDIR /workspaces

# Stay alive so `docker exec` can attach.
CMD ["sh", "-c", "tail -f /dev/null"]
