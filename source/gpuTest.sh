#!/bin/bash
#
# Usage: gpuTest.sh [blocks]
#
# Environment variables:
#   ITERATIONS     number of test runs before exiting, 0 = run forever (default: 0)
#   SLEEP_SECONDS  pause between runs (default: 60)

set -uo pipefail

blocks=${1:-512}
iterations=${ITERATIONS:-0}
sleep_seconds=${SLEEP_SECONDS:-60}

# bash as PID 1 ignores SIGTERM by default; exit promptly when the pod is stopped
trap 'echo "Received termination signal, exiting"; exit 0' TERM INT

run=0
while true; do
    run=$((run + 1))
    echo -e "=== Run ${run} ===\nnvidia-smi:\n"
    nvidia-smi || echo "WARNING: nvidia-smi failed or is not available in this container"
    echo -e "\ncalculate pi:\n"
    if ! /opt/carl/cudaPi "${blocks}"; then
        echo "ERROR: cudaPi failed on run ${run}" >&2
        exit 1
    fi
    if [ "${iterations}" -gt 0 ] && [ "${run}" -ge "${iterations}" ]; then
        echo "Completed ${run} run(s) successfully"
        exit 0
    fi
    # run sleep in the background so the TERM trap fires immediately
    sleep "${sleep_seconds}" &
    wait $!
done
