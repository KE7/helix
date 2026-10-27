# Mutation-agent image for examples/circle_packing: Claude Code with the
# experiment's Python environment (circle_packing/requirements.txt, Python
# 3.11), so code the agent tries runs as it will in the evaluator.
# Build context: examples/.
FROM node:22-bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        git \
        procps \
        python3 \
        python3-pip \
        python3-venv \
        ripgrep \
    && rm -rf /var/lib/apt/lists/*

COPY circle_packing/requirements.txt /opt/cpenv/requirements.txt
RUN python3 -m pip install --break-system-packages --no-cache-dir \
        -r /opt/cpenv/requirements.txt \
    && ln -s /usr/bin/python3 /usr/local/bin/python

ARG CLAUDE_CODE_VERSION=2.1.274
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION} \
    && npm cache clean --force

WORKDIR /workspace
