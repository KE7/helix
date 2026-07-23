ARG BASE_IMAGE=helix-runner-base:latest
FROM ${BASE_IMAGE}

ARG TARGETARCH
ARG CLI_VERSION=2.1.218
ARG CLI_TARBALL=https://registry.npmjs.org/@anthropic-ai/claude-code/-/claude-code-2.1.218.tgz
ARG CLI_SHA512=04757de75aee21ae905da6450c5d70461c3100e90089a7c1d803961bac064542786a9689f66973a750452c08467c6d12927c00caa05e9ea793ea78ade9ab78b9
ARG CLI_AMD64_TARBALL=https://registry.npmjs.org/@anthropic-ai/claude-code-linux-x64/-/claude-code-linux-x64-2.1.218.tgz
ARG CLI_AMD64_SHA512=7b5df6900e1d562a6454858aeba8e90a771b26f2eabee8369c3df35c401d99a8afb97356bdf41b31a5e5e82d13d92bac122980a038ab673e7b32a28819555570
ARG CLI_ARM64_TARBALL=https://registry.npmjs.org/@anthropic-ai/claude-code-linux-arm64/-/claude-code-linux-arm64-2.1.218.tgz
ARG CLI_ARM64_SHA512=09c6d5402cd777d127964b4210fae412585d059baa2168647a29d11b152e65ae9a6a5e21e89fbc0dba3e3a77dc84113377599a8510d02bc05c906040628cefda

# Extract the exact launcher and the one exact glibc native package selected
# for TARGETARCH. No npm resolution occurs in the image build.
RUN case "${TARGETARCH}" in \
        amd64) platform_name=claude-code-linux-x64; platform_url="${CLI_AMD64_TARBALL}"; platform_sha="${CLI_AMD64_SHA512}" ;; \
        arm64) platform_name=claude-code-linux-arm64; platform_url="${CLI_ARM64_TARBALL}"; platform_sha="${CLI_ARM64_SHA512}" ;; \
        *) echo "unsupported TARGETARCH: ${TARGETARCH}" >&2; exit 2 ;; \
    esac \
    && curl -fsSL "${CLI_TARBALL}" -o /tmp/claude-code.tgz \
    && curl -fsSL "${platform_url}" -o /tmp/claude-platform.tgz \
    && echo "${CLI_SHA512}  /tmp/claude-code.tgz" | sha512sum --check --strict \
    && echo "${platform_sha}  /tmp/claude-platform.tgz" | sha512sum --check --strict \
    && mkdir -p /opt/claude \
        "/opt/claude/node_modules/@anthropic-ai/${platform_name}" \
    && tar -xzf /tmp/claude-code.tgz --strip-components=1 -C /opt/claude \
    && tar -xzf /tmp/claude-platform.tgz --strip-components=1 \
        -C "/opt/claude/node_modules/@anthropic-ai/${platform_name}" \
    && chmod 0755 /opt/claude/cli-wrapper.cjs \
    && ln -s /opt/claude/cli-wrapper.cjs /usr/local/bin/claude \
    && rm -f /tmp/claude-code.tgz /tmp/claude-platform.tgz

LABEL org.opencontainers.image.source="https://github.com/KE7/helix" \
      io.helix.runner.backend="claude" \
      io.helix.runner.cli-version="${CLI_VERSION}" \
      io.helix.runner.cli-tarball="${CLI_TARBALL}" \
      io.helix.runner.cli-sha512="${CLI_SHA512}" \
      io.helix.runner.cli-amd64-sha512="${CLI_AMD64_SHA512}" \
      io.helix.runner.cli-arm64-sha512="${CLI_ARM64_SHA512}"

WORKDIR /workspace
