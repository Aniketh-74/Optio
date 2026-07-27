# Demo image (M4-5). Built from the repo root so it installs the working tree,
# not a published wheel -- the demo must exercise the code in this checkout.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependency metadata first so a change to the demo scripts does not invalidate
# the dependency layer.
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

# The OTLP exporter is a demo-only dependency: agentmeter rides the user's
# existing OTel pipeline and does not ship an exporter of its own (SC-1).
RUN pip install --no-cache-dir . opentelemetry-exporter-otlp-proto-grpc

COPY examples/demo/ ./demo/

RUN chmod +x ./demo/entrypoint.sh

# Unprivileged: nothing here needs root, and a demo that tells people to run
# containers as root is teaching the wrong habit.
RUN useradd --create-home --uid 10001 demo && chown -R demo:demo /app
USER demo

ENTRYPOINT ["./demo/entrypoint.sh"]
