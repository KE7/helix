# Mutation-agent image: Claude Code pinned to the version whose --effort help
# lists xhigh, on the shared numpy/scipy base.
ARG BASE_IMAGE=helix-circle-clean/base:1
FROM ${BASE_IMAGE}

ARG CLAUDE_CODE_VERSION=2.1.274
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION} \
    && npm cache clean --force

LABEL helix-circle-clean="1" \
      helix-circle-clean.role="agent-claude" \
      helix-circle-clean.claude-code-version="${CLAUDE_CODE_VERSION}"

WORKDIR /workspace
