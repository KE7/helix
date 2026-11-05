# The image each trial's solver runs in. evaluate.py builds it on first use
# and starts one throwaway container per trial with no network, a read-only
# filesystem and only a copy of solver/ mounted, so solver code cannot touch
# the simulator, the scorer or anything else on the host.
FROM python:3.12.16-slim

RUN pip install --no-cache-dir numpy==2.5.4 \
    && useradd --create-home --uid 10001 solver

USER solver
