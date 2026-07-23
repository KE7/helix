ARG BASE_IMAGE=helix-runner-base:latest
FROM ${BASE_IMAGE}

ARG TARGETARCH
ARG CLI_VERSION=1.18.4
ARG CLI_TARBALL=https://registry.npmjs.org/opencode-ai/-/opencode-ai-1.18.4.tgz
ARG CLI_SHA512=07ca4502cd60d79f1953e0ba792889cffe59846ed5af6987ed9eebb981072c4940957f6d7a1ad14e7cd34e08d2c47636bf067fe55a65c11de874553e0da8bc8e
ARG CLI_AMD64_TARBALL=https://registry.npmjs.org/opencode-linux-x64/-/opencode-linux-x64-1.18.4.tgz
ARG CLI_AMD64_SHA512=982899b8a401a911d586da76d181437cd7f9714c13b7afc8f7c3052500c97434d68209e40153a78ad413992b76710c283c9f7856b4700e1672637a48637f8242
ARG CLI_AMD64_FALLBACK_TARBALL=https://registry.npmjs.org/opencode-linux-x64-baseline/-/opencode-linux-x64-baseline-1.18.4.tgz
ARG CLI_AMD64_FALLBACK_SHA512=2e5548bf9e20a4cf08e902550285b9baf71a54ee33fb2e43b307d40acc60683d025a08c46d42da880019212fcbf8de6b470fe31dea9c1d25e22a29c566dd31c3
ARG CLI_ARM64_TARBALL=https://registry.npmjs.org/opencode-linux-arm64/-/opencode-linux-arm64-1.18.4.tgz
ARG CLI_ARM64_SHA512=de0070f70784efa8e5a0d3c698fbf397718947c6fab1cc1200437150a45d277582902eea723266e3d285c9fa858cee7efb51f54368db9aaab1020b79ba9aa831

# Pin both AVX2 and baseline x64 binaries so the same amd64 image remains
# portable across runner CPUs. The runtime wrapper chooses without networking.
RUN case "${TARGETARCH}" in \
        amd64) primary_name=opencode-linux-x64; primary_url="${CLI_AMD64_TARBALL}"; primary_sha="${CLI_AMD64_SHA512}"; fallback_name=opencode-linux-x64-baseline; fallback_url="${CLI_AMD64_FALLBACK_TARBALL}"; fallback_sha="${CLI_AMD64_FALLBACK_SHA512}" ;; \
        arm64) primary_name=opencode-linux-arm64; primary_url="${CLI_ARM64_TARBALL}"; primary_sha="${CLI_ARM64_SHA512}"; fallback_name=; fallback_url=; fallback_sha= ;; \
        *) echo "unsupported TARGETARCH: ${TARGETARCH}" >&2; exit 2 ;; \
    esac \
    && curl -fsSL "${CLI_TARBALL}" -o /tmp/opencode.tgz \
    && curl -fsSL "${primary_url}" -o /tmp/opencode-primary.tgz \
    && echo "${CLI_SHA512}  /tmp/opencode.tgz" | sha512sum --check --strict \
    && echo "${primary_sha}  /tmp/opencode-primary.tgz" | sha512sum --check --strict \
    && mkdir -p /opt/opencode "/opt/opencode/node_modules/${primary_name}" \
    && tar -xzf /tmp/opencode.tgz --strip-components=1 -C /opt/opencode \
    && tar -xzf /tmp/opencode-primary.tgz --strip-components=1 \
        -C "/opt/opencode/node_modules/${primary_name}" \
    && if [ -n "${fallback_url}" ]; then \
        curl -fsSL "${fallback_url}" -o /tmp/opencode-fallback.tgz; \
        echo "${fallback_sha}  /tmp/opencode-fallback.tgz" | sha512sum --check --strict; \
        mkdir -p "/opt/opencode/node_modules/${fallback_name}"; \
        tar -xzf /tmp/opencode-fallback.tgz --strip-components=1 \
          -C "/opt/opencode/node_modules/${fallback_name}"; \
      fi \
    && printf '%s\n' \
        '#!/bin/sh' \
        'set -eu' \
        'case "$(uname -m)" in' \
        '  x86_64|amd64)' \
        '    if grep -qw avx2 /proc/cpuinfo; then' \
        '      binary=/opt/opencode/node_modules/opencode-linux-x64/bin/opencode' \
        '    else' \
        '      binary=/opt/opencode/node_modules/opencode-linux-x64-baseline/bin/opencode' \
        '    fi' \
        '    ;;' \
        '  aarch64|arm64)' \
        '    binary=/opt/opencode/node_modules/opencode-linux-arm64/bin/opencode' \
        '    ;;' \
        '  *) echo "unsupported runtime architecture: $(uname -m)" >&2; exit 2 ;;' \
        'esac' \
        'exec "$binary" "$@"' \
        > /usr/local/bin/opencode \
    && chmod 0755 /usr/local/bin/opencode \
    && rm -f /tmp/opencode.tgz /tmp/opencode-primary.tgz \
        /tmp/opencode-fallback.tgz

LABEL org.opencontainers.image.source="https://github.com/KE7/helix" \
      io.helix.runner.backend="opencode" \
      io.helix.runner.cli-version="${CLI_VERSION}" \
      io.helix.runner.cli-tarball="${CLI_TARBALL}" \
      io.helix.runner.cli-sha512="${CLI_SHA512}" \
      io.helix.runner.cli-amd64-sha512="${CLI_AMD64_SHA512}" \
      io.helix.runner.cli-amd64-fallback-sha512="${CLI_AMD64_FALLBACK_SHA512}" \
      io.helix.runner.cli-arm64-sha512="${CLI_ARM64_SHA512}"

WORKDIR /workspace
