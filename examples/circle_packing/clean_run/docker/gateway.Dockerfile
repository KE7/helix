# Egress gateway: forwards only allowed Anthropic Messages API requests.
ARG PYTHON_IMAGE=python:3.13-alpine
FROM ${PYTHON_IMAGE}

COPY gateway/anthropic_gateway.py /opt/gateway/anthropic_gateway.py
RUN chmod 0755 /opt/gateway && chmod 0644 /opt/gateway/anthropic_gateway.py \
    && mkdir -p /logs && chown nobody:nobody /logs

LABEL helix-circle-clean="1" \
      helix-circle-clean.role="gateway"

USER 1000:1000
CMD ["python3", "/opt/gateway/anthropic_gateway.py"]
