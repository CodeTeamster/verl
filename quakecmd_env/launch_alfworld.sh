#!/usr/bin/env bash
set -euo pipefail

# Quake starts one entry process per local device. Only rank zero is the Verl
# driver; its Ray actors schedule work on every node supplied by LAUNCH_RAY=1.
LOCAL_RANK="${LOCAL_RANK:-${RANK:-0}}"
if [[ "${LOCAL_RANK}" != "0" ]]; then
    exit 0
fi

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Ensure `python -m verl` resolves to this checkout, not a version bundled in
# the training image. Dependencies were installed by Quake beforehand.
python3 -m pip install --no-deps -e .

exec bash quakecmd_env/run_alfworld_grpo.sh "$@"
