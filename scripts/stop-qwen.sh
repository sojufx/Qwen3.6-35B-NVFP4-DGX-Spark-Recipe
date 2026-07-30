#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8000}"

if command -v lsof >/dev/null 2>&1; then
  pids="$(lsof -ti tcp:"${PORT}" || true)"
  if [[ -n "${pids}" ]]; then
    kill ${pids}
  fi
fi

echo "Stopped ${CONTAINER_NAME}"
