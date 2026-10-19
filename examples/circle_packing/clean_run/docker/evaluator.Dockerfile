# Evaluator sidecar and evaluator-runner image.
#   /opt/cpeval/private   service code and scoring rules (root only, 0700)
#   /opt/cpeval/harness   harness the candidate user runs (world-readable)
#   /opt/cpeval/runner    client HELIX's evaluator command runs (world-readable)
#   /srv/cpeval/ledger    mount point of the private ledger (parent root only, 0700)
# The candidate user can read neither the private code nor the ledger.  The
# ledger is a host bind mount, and Docker Desktop's file sharing does not
# enforce Linux permissions on bind mounts, so the ledger is only protected
# by its root-only parent directory, which lives in the image and blocks path
# traversal before the mount is reached.
ARG BASE_IMAGE=helix-circle-clean/base:1
FROM ${BASE_IMAGE}

RUN groupadd --gid 2000 candidate \
    && useradd --uid 2000 --gid 2000 --no-create-home --home-dir /nonexistent \
        --shell /usr/sbin/nologin candidate

COPY evaluator/evaluator_server.py evaluator/packing_rules.py /opt/cpeval/private/
COPY evaluator/candidate_harness.py /opt/cpeval/harness/
COPY evaluator/evaluator_client.py /opt/cpeval/runner/

RUN chown -R root:root /opt/cpeval \
    && chmod 0711 /opt/cpeval \
    && chmod 0700 /opt/cpeval/private && chmod 0600 /opt/cpeval/private/* \
    && chmod 0755 /opt/cpeval/harness /opt/cpeval/runner \
    && chmod 0644 /opt/cpeval/harness/* /opt/cpeval/runner/* \
    && mkdir -p /work /srv/cpeval/ledger && chmod 0711 /work && chmod 0700 /srv/cpeval

LABEL helix-circle-clean="1" \
      helix-circle-clean.role="evaluator"

WORKDIR /
CMD ["python3", "/opt/cpeval/private/evaluator_server.py"]
