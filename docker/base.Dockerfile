ARG NODE_BASE=node:22-bookworm-slim@sha256:6c74791e557ce11fc957704f6d4fe134a7bc8d6f5ca4403205b2966bd488f6b3
FROM ${NODE_BASE}

ARG NODE_BASE
ENV DEBIAN_FRONTEND=noninteractive
ARG TARGETARCH
ARG DEBIAN_SNAPSHOT=20260720T000000Z
ARG UV_VERSION=0.11.7
ARG UV_AMD64_WHEEL=https://files.pythonhosted.org/packages/83/eb/4e1557daf6693cb446ed28185664ad6682fd98c6dbac9e433cbc35df450a/uv-0.11.7-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
ARG UV_AMD64_SHA256=4e4d5e31bea86e1b6e0f5a0f95e14e80018e6f6c0129256d2915a4b3d793644d
ARG UV_ARM64_WHEEL=https://files.pythonhosted.org/packages/f2/7f/fbed29775b0612f4f5679d3226268f1a347161abc1727b4080fb41d9f46f/uv-0.11.7-py3-none-manylinux_2_17_aarch64.manylinux2014_aarch64.musllinux_1_1_aarch64.whl
ARG UV_ARM64_SHA256=5985a15a92bd9a170fc1947abb1fbc3e9828c5a430ad85b5bed8356c20b67a71

RUN rm -f /etc/apt/sources.list /etc/apt/sources.list.d/* \
    && printf '%s\n' \
        "deb [check-valid-until=no] http://snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT}/ bookworm main" \
        "deb [check-valid-until=no] http://snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT}/ bookworm-updates main" \
        "deb [check-valid-until=no] http://snapshot.debian.org/archive/debian-security/${DEBIAN_SNAPSHOT}/ bookworm-security main" \
        > /etc/apt/sources.list \
    && apt-get -o Acquire::Check-Valid-Until=false update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        git \
        python3 \
        python3-pip \
        python3-venv \
        ripgrep \
    && rm -rf /var/lib/apt/lists/* /var/log/apt/* \
    && rm -f /var/log/dpkg.log

RUN case "${TARGETARCH}" in \
        amd64) uv_wheel="${UV_AMD64_WHEEL}"; uv_sha="${UV_AMD64_SHA256}" ;; \
        arm64) uv_wheel="${UV_ARM64_WHEEL}"; uv_sha="${UV_ARM64_SHA256}" ;; \
        *) echo "unsupported TARGETARCH: ${TARGETARCH}" >&2; exit 2 ;; \
    esac \
    && uv_path="/tmp/${uv_wheel##*/}" \
    && curl -fsSL "$uv_wheel" -o "$uv_path" \
    && echo "${uv_sha}  ${uv_path}" | sha256sum --check --strict \
    && python3 -m pip install --break-system-packages --no-cache-dir "$uv_path" \
    && rm -f "$uv_path" \
    && ln -s /usr/bin/python3 /usr/local/bin/python

LABEL org.opencontainers.image.source="https://github.com/KE7/helix" \
      org.opencontainers.image.description="HELIX mutation-agent runner base" \
      io.helix.runner.node-base="${NODE_BASE}" \
      io.helix.runner.debian-snapshot="${DEBIAN_SNAPSHOT}" \
      io.helix.runner.uv-version="${UV_VERSION}" \
      io.helix.runner.uv-amd64-sha256="${UV_AMD64_SHA256}" \
      io.helix.runner.uv-arm64-sha256="${UV_ARM64_SHA256}"

WORKDIR /workspace
