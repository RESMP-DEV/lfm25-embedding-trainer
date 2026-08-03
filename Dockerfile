# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:0.11.32@sha256:df4cae8f3a96d175e2e5f992e597550000edbe78fdc2594d5cd8de1a217f504c AS uv
FROM python:3.13-slim-bookworm@sha256:9d7f287598e1a5a978c015ee176d8216435aaf335ed69ac3c38dd1bbb10e8d64

COPY --from=uv /uv /uvx /bin/

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

ENV HF_HOME=/home/trainer/.cache/huggingface \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

ARG TRAINER_UID=1000
ARG TRAINER_GID=1000

RUN groupadd --non-unique --gid "${TRAINER_GID}" trainer \
    && useradd --non-unique --uid "${TRAINER_UID}" --gid trainer --create-home trainer \
    && mkdir -p /workspace "$HF_HOME" \
    && chown -R trainer:trainer /workspace /home/trainer

WORKDIR /workspace

COPY --chown=trainer:trainer pyproject.toml uv.lock README.md ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra train --no-install-project

COPY --chown=trainer:trainer src ./src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra train

USER trainer

ENTRYPOINT ["lfm25-embed"]
CMD ["--help"]
