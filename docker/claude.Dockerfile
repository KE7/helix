ARG BASE_IMAGE=helix-runner-base:latest
FROM ${BASE_IMAGE}

RUN npm install -g @anthropic-ai/claude-code

WORKDIR /workspace
