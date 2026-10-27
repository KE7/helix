# The agent's only route out: anthropic_gateway.py.  Build context: examples/.
FROM python:3.13-alpine

COPY circle_packing_sandbox/anthropic_gateway.py /opt/gateway/
RUN mkdir -p /logs && chown 1000:1000 /logs

USER 1000:1000
CMD ["python3", "/opt/gateway/anthropic_gateway.py"]
