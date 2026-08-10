ARG BASE_IMAGE=helix-runner-base:latest
FROM ${BASE_IMAGE}

ARG TARGETARCH
ARG CLI_VERSION=1.18.5
ARG CLI_TARBALL=https://registry.npmjs.org/opencode-ai/-/opencode-ai-1.18.5.tgz
ARG CLI_SHA512=4348e55f88a19fbbde31e62c2d7ddce0f60500a214454dc62295edd459e1371367deff3e469219f33f6e986e43d2bf20f129a9f5f64b1a380b7bff6627837263
ARG CLI_AMD64_TARBALL=https://registry.npmjs.org/opencode-linux-x64-baseline/-/opencode-linux-x64-baseline-1.18.5.tgz
ARG CLI_AMD64_SHA512=e1e5ce2421bff0a98f52515edaf2534fbbb6e06385f1530b3c794d8eaaf0bb7d42c654fe72c124dbcee52346b23d7dc9d11d696c6ef51cb7092eab28911fb0bf
ARG CLI_ARM64_TARBALL=https://registry.npmjs.org/opencode-linux-arm64/-/opencode-linux-arm64-1.18.5.tgz
ARG CLI_ARM64_SHA512=727bf62008bf4e98ad60952ae384c2d028b9cd50a2f50b29c7fecfda90cd729e481fd7fa539f2df688bd5a91f256d905b6101073609d4880726f71ee76eb1c02

# amd64 uses Bun's *baseline* build, which runs on every x86-64 CPU. The AVX2
# build was also shipped, chosen at runtime by grepping /proc/cpuinfo, so image
# behaviour depended on the host CPU -- in an image whose whole point is that a
# digest means one thing. The CLI's wall-clock is dominated by waiting on LLM
# API calls, so the vectorisation delta is not measurable here.
RUN case "${TARGETARCH}" in \
        amd64) name=opencode-linux-x64-baseline; url="${CLI_AMD64_TARBALL}"; sha="${CLI_AMD64_SHA512}" ;; \
        arm64) name=opencode-linux-arm64; url="${CLI_ARM64_TARBALL}"; sha="${CLI_ARM64_SHA512}" ;; \
        *) echo "unsupported TARGETARCH: ${TARGETARCH}" >&2; exit 2 ;; \
    esac \
    && curl -fsSL "${CLI_TARBALL}" -o /tmp/opencode.tgz \
    && curl -fsSL "${url}" -o /tmp/opencode-binary.tgz \
    && echo "${CLI_SHA512}  /tmp/opencode.tgz" | sha512sum --check --strict \
    && echo "${sha}  /tmp/opencode-binary.tgz" | sha512sum --check --strict \
    && mkdir -p /opt/opencode "/opt/opencode/node_modules/${name}" \
    && tar -xzf /tmp/opencode.tgz --strip-components=1 -C /opt/opencode \
    && tar -xzf /tmp/opencode-binary.tgz --strip-components=1 \
        -C "/opt/opencode/node_modules/${name}" \
    && ln -s "/opt/opencode/node_modules/${name}/bin/opencode" \
        /usr/local/bin/opencode \
    && rm -f /tmp/opencode.tgz /tmp/opencode-binary.tgz

LABEL org.opencontainers.image.source="https://github.com/KE7/helix" \
      io.helix.runner.backend="opencode" \
      io.helix.runner.cli-version="${CLI_VERSION}"

WORKDIR /workspace
