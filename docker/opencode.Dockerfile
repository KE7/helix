ARG BASE_IMAGE=helix-runner-base:latest
FROM ${BASE_IMAGE}

RUN npm install -g opencode-ai

WORKDIR /workspace
