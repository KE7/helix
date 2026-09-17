# Shared base for the agent and evaluator images of the leak-free
# circle-packing run: HELIX's runner base (docker/base.Dockerfile) plus pinned
# numpy/scipy, so the agent's local Python matches the evaluator's exactly.
ARG NODE_IMAGE=node:22-bookworm-slim
FROM ${NODE_IMAGE}

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

RUN python3 -m pip install --break-system-packages --no-cache-dir \
        numpy==2.3.3 \
        scipy==1.16.2 \
    && ln -s /usr/bin/python3 /usr/local/bin/python

LABEL helix-circle-clean="1" \
      helix-circle-clean.role="base"

WORKDIR /workspace
