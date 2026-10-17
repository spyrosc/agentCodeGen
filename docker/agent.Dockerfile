# Agent container: runs the Streamlit UI and the LangGraph agents.
# It shells into the sandbox container via the host Docker socket.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# build-essential for any wheels that need compiling; docker-ce-cli for
# `docker exec` into the sandbox via the mounted host socket.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        gnupg \
        git \
        lsb-release \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg \
        | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
        > /etc/apt/sources.list.d/docker.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends docker-ce-cli \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts

EXPOSE 8506

HEALTHCHECK --interval=10s --timeout=5s --retries=5 \
    CMD curl --fail http://localhost:8506/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "src/genai/ui/streamlit_app.py", \
            "--server.port=8506", "--server.address=0.0.0.0", \
            "--client.toolbarMode=viewer"]
