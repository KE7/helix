ARG BASE_IMAGE=helix-runner-base:latest
FROM ${BASE_IMAGE}

# The installer places the binary under /root/.local/bin, a directory only
# root can traverse. HELIX always runs the agent container as the unprivileged
# `node` user (see `_docker_args` in src/helix/sandbox.py), so a symlink
# straight into /root would resolve to nothing for that user -- confirmed by
# actually running this image as `node`, not just as root. Copy the binary out
# to /opt (world-traversable, the same fix cursor.Dockerfile already applies to
# its own root-owned install) before exposing it on PATH.
RUN curl -fsSL https://antigravity.google/cli/install.sh | bash \
    && cp /root/.local/bin/agy /opt/agy \
    && chmod 755 /opt/agy \
    && ln -s /opt/agy /usr/local/bin/agy

WORKDIR /workspace
