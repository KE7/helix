ARG BASE_IMAGE=helix-runner-base:latest
FROM ${BASE_IMAGE}

# The installer places the binary under /root/.local/bin, a directory only
# root can traverse. HELIX always runs the agent as uid 1000 (see
# _RUNNER_UID_GID in sandbox.py), so a symlink straight into /root would
# resolve to nothing for that user -- confirmed by actually running this
# image as uid 1000, not just as root. Copy the binary out to /opt (world-
# traversable, same fix cursor.Dockerfile already applies to its own
# root-owned install) before exposing it on PATH.
RUN curl -fsSL https://antigravity.google/cli/install.sh | bash \
    && cp /root/.local/bin/agy /opt/agy \
    && chmod 755 /opt/agy \
    && ln -s /opt/agy /usr/local/bin/agy

WORKDIR /workspace
