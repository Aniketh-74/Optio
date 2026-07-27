#!/bin/sh
# Demo entrypoint (M4-5): wait for the collector, then run the demo.
#
# The wait matters. The OTLP exporter buffers and retries, so spans sent before
# the collector is listening are not lost immediately -- they are lost quietly,
# on shutdown, after the demo has already printed its summary. Waiting for the
# port turns a confusing empty trace into a two-second pause.
#
# `set -e` so a failure here fails the container rather than running the demo
# against a collector that never came up.
set -eu

ENDPOINT_HOST="${OTEL_COLLECTOR_HOST:-collector}"
ENDPOINT_PORT="${OTEL_COLLECTOR_PORT:-4317}"
ATTEMPTS="${OTEL_WAIT_ATTEMPTS:-30}"

echo "waiting for collector at ${ENDPOINT_HOST}:${ENDPOINT_PORT} ..."

i=0
while [ "$i" -lt "$ATTEMPTS" ]; do
    # Pure-Python probe: the slim image has no nc, and adding a package to run
    # one TCP connect would be a dependency for nothing.
    if python -c "
import socket, sys
s = socket.socket()
s.settimeout(1)
sys.exit(0 if s.connect_ex(('${ENDPOINT_HOST}', ${ENDPOINT_PORT})) == 0 else 1)
" 2>/dev/null; then
        echo "collector is up"
        exec python demo/run_demo.py
    fi
    i=$((i + 1))
    sleep 1
done

echo "collector did not become reachable after ${ATTEMPTS}s" >&2
exit 1
