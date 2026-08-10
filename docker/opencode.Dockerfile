ARG BASE_IMAGE=helix-runner-base:latest
FROM ${BASE_IMAGE}

ARG TARGETARCH
ARG CLI_VERSION=1.18.5
ARG CLI_TARBALL=https://registry.npmjs.org/opencode-ai/-/opencode-ai-1.18.5.tgz
ARG CLI_SHA512=4348e55f88a19fbbde31e62c2d7ddce0f60500a214454dc62295edd459e1371367deff3e469219f33f6e986e43d2bf20f129a9f5f64b1a380b7bff6627837263
ARG CLI_AMD64_TARBALL=https://registry.npmjs.org/opencode-linux-x64/-/opencode-linux-x64-1.18.5.tgz
ARG CLI_AMD64_SHA512=69afd8eb1ce7686b44d24036b1929d75d63b88d070ff2ad7587d2603a7c6b381a0ad873d8dc5d2f945cbc0b64b599c0b2d804e158a00a32796889a046746daeb
ARG CLI_AMD64_FALLBACK_TARBALL=https://registry.npmjs.org/opencode-linux-x64-baseline/-/opencode-linux-x64-baseline-1.18.5.tgz
ARG CLI_AMD64_FALLBACK_SHA512=e1e5ce2421bff0a98f52515edaf2534fbbb6e06385f1530b3c794d8eaaf0bb7d42c654fe72c124dbcee52346b23d7dc9d11d696c6ef51cb7092eab28911fb0bf
ARG CLI_ARM64_TARBALL=https://registry.npmjs.org/opencode-linux-arm64/-/opencode-linux-arm64-1.18.5.tgz
ARG CLI_ARM64_SHA512=727bf62008bf4e98ad60952ae384c2d028b9cd50a2f50b29c7fecfda90cd729e481fd7fa539f2df688bd5a91f256d905b6101073609d4880726f71ee76eb1c02

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
