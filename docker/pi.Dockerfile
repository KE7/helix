ARG BASE_IMAGE=helix-runner-base:latest
FROM ${BASE_IMAGE}

# pi's own containerization guide installs it this way; the package needs no
# install scripts.
RUN npm install -g --ignore-scripts @earendil-works/pi-coding-agent

WORKDIR /workspace
