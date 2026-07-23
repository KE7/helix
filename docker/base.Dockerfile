ARG NODE_BASE=node:22-bookworm-slim@sha256:6c74791e557ce11fc957704f6d4fe134a7bc8d6f5ca4403205b2966bd488f6b3
FROM ${NODE_BASE}

ARG NODE_BASE
ENV DEBIAN_FRONTEND=noninteractive
ARG UV_VERSION=0.11.7

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        git \
        python3 \
        python3-pip \
        python3-venv \
        ripgrep \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --break-system-packages --no-cache-dir "uv==${UV_VERSION}" \
    && ln -s /usr/bin/python3 /usr/local/bin/python

LABEL org.opencontainers.image.source="https://github.com/KE7/helix" \
      org.opencontainers.image.description="HELIX mutation-agent runner base" \
      io.helix.runner.node-base="${NODE_BASE}" \
      io.helix.runner.uv-version="${UV_VERSION}"

WORKDIR /workspace
