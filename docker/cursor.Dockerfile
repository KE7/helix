ARG BASE_IMAGE=helix-runner-base:latest
FROM ${BASE_IMAGE}

RUN curl https://cursor.com/install -fsS | bash \
    && cp -a /root/.local/share/cursor-agent /opt/cursor-agent \
    && ln -s "$(readlink -f /opt/cursor-agent/versions/*/cursor-agent)" /usr/local/bin/cursor-agent \
    && printf '%s\n' \
        '#!/bin/sh' \
        'if [ "$1" = "agent" ]; then shift; fi' \
        'exec cursor-agent "$@"' \
        > /usr/local/bin/cursor \
    && chmod +x /usr/local/bin/cursor

WORKDIR /workspace
