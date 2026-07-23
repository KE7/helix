ARG BASE_IMAGE=helix-runner-base:latest
FROM ${BASE_IMAGE}

ARG TARGETARCH
ARG CLI_VERSION=0.52.0
ARG CLI_TARBALL=https://registry.npmjs.org/@google/gemini-cli/-/gemini-cli-0.52.0.tgz
ARG CLI_SHA512=ffa16f7ef95cb0e26d9f7340813a79fdc6bdc52ea799ab78a2d43e3445684fa968be62363798c7c6d06a1ad8cb528f9cbc17c682d4765a69023f1c691e3afad5
ARG CLI_SHARED_TARBALL=https://registry.npmjs.org/@lydell/node-pty/-/node-pty-1.1.0.tgz
ARG CLI_SHARED_SHA512=5430fc2ed94c4ceacf296317500701f7e2d392dceeba7aab3309184750cc441912e8b42b0adfb4fd6b35a36acc9a5fe7de0b9e3e94bb73184717b366e4ae2233
ARG CLI_AMD64_TARBALL=https://registry.npmjs.org/@lydell/node-pty-linux-x64/-/node-pty-linux-x64-1.1.0.tgz
ARG CLI_AMD64_SHA512=35c36a4530f5e104febd7704baa492be6598fb4f965019f6b9113c10dd332ad0e9204afd77e622163d7a52a76ce907dc2c21df6660be2eced8cf04daa839da9c
ARG CLI_ARM64_TARBALL=https://registry.npmjs.org/@lydell/node-pty-linux-arm64/-/node-pty-linux-arm64-1.1.0.tgz
ARG CLI_ARM64_SHA512=cb20c199a9427c7a4b890313db3c8b72a2f615acb85f2eeb22cf061f876a28b9c4be232f3c638aecb003564280b1bb325c148848bdcbb759b532dc613c7e9c92

# The bundled CLI prefers @lydell/node-pty and falls back to child_process if
# optional keychain/legacy PTY modules are absent. Install only the exact
# content-pinned selector and native PTY package; never invoke npm.
RUN case "${TARGETARCH}" in \
        amd64) platform_name=node-pty-linux-x64; platform_url="${CLI_AMD64_TARBALL}"; platform_sha="${CLI_AMD64_SHA512}" ;; \
        arm64) platform_name=node-pty-linux-arm64; platform_url="${CLI_ARM64_TARBALL}"; platform_sha="${CLI_ARM64_SHA512}" ;; \
        *) echo "unsupported TARGETARCH: ${TARGETARCH}" >&2; exit 2 ;; \
    esac \
    && curl -fsSL "${CLI_TARBALL}" -o /tmp/gemini-cli.tgz \
    && curl -fsSL "${CLI_SHARED_TARBALL}" -o /tmp/gemini-pty.tgz \
    && curl -fsSL "${platform_url}" -o /tmp/gemini-pty-platform.tgz \
    && echo "${CLI_SHA512}  /tmp/gemini-cli.tgz" | sha512sum --check --strict \
    && echo "${CLI_SHARED_SHA512}  /tmp/gemini-pty.tgz" | sha512sum --check --strict \
    && echo "${platform_sha}  /tmp/gemini-pty-platform.tgz" | sha512sum --check --strict \
    && mkdir -p /opt/gemini \
        /opt/gemini/node_modules/@lydell/node-pty \
        "/opt/gemini/node_modules/@lydell/${platform_name}" \
    && tar -xzf /tmp/gemini-cli.tgz --strip-components=1 -C /opt/gemini \
    && tar -xzf /tmp/gemini-pty.tgz --strip-components=1 \
        -C /opt/gemini/node_modules/@lydell/node-pty \
    && tar -xzf /tmp/gemini-pty-platform.tgz --strip-components=1 \
        -C "/opt/gemini/node_modules/@lydell/${platform_name}" \
    && chmod 0755 /opt/gemini/bundle/gemini.js \
    && ln -s /opt/gemini/bundle/gemini.js /usr/local/bin/gemini \
    && rm -f /tmp/gemini-cli.tgz /tmp/gemini-pty.tgz \
        /tmp/gemini-pty-platform.tgz

LABEL org.opencontainers.image.source="https://github.com/KE7/helix" \
      io.helix.runner.backend="gemini" \
      io.helix.runner.cli-version="${CLI_VERSION}" \
      io.helix.runner.cli-tarball="${CLI_TARBALL}" \
      io.helix.runner.cli-sha512="${CLI_SHA512}" \
      io.helix.runner.cli-shared-sha512="${CLI_SHARED_SHA512}" \
      io.helix.runner.cli-amd64-sha512="${CLI_AMD64_SHA512}" \
      io.helix.runner.cli-arm64-sha512="${CLI_ARM64_SHA512}"

WORKDIR /workspace
