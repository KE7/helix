FROM ghcr.io/ke7/helix-evo-runner-codex:0.2.0

USER root
LABEL org.opencontainers.image.title="HELIX LiveBench-Math disposable evaluator"
LABEL org.helix.demo="livebench-math"

COPY requirements-evaluator.txt /opt/livebench-math/requirements-evaluator.txt
RUN python3 -m venv /opt/livebench-venv && \
    /opt/livebench-venv/bin/pip install --no-cache-dir \
      -r /opt/livebench-math/requirements-evaluator.txt && \
    /opt/livebench-venv/bin/pip install --no-cache-dir --no-deps \
      "livebench @ git+https://github.com/LiveBench/LiveBench@1de6a43e82a137beeeaf2b92d683eedb67f0cf97"
ENV PATH="/opt/livebench-venv/bin:${PATH}"

COPY constants.py dataset.py scoring.py prepare_data.py server.py smoke_score.py /opt/livebench-math/
RUN cd /opt/livebench-math && \
    HF_HOME=/tmp/hf-home python3 prepare_data.py && \
    rm -rf /tmp/hf-home

WORKDIR /workspace
USER node
