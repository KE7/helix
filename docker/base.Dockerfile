FROM node:22-bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive

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

RUN python3 -m pip install --break-system-packages --no-cache-dir uv \
    && ln -s /usr/bin/python3 /usr/local/bin/python

WORKDIR /workspace
