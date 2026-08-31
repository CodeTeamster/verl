#!/usr/bin/env bash
# Convert a verl FSDP actor checkpoint into a Hugging Face model directory.
#
# Usage:
#   bash lab/convert_fsdp_checkpoint_to_hf.sh SOURCE_DIR OUTPUT_DIR

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

usage() {
    echo "Usage: bash lab/convert_fsdp_checkpoint_to_hf.sh SOURCE_DIR OUTPUT_DIR"
}

if (( $# != 2 )); then
    usage >&2
    exit 2
fi

SOURCE_DIR="$1"
OUTPUT_DIR="$2"

if [[ ! -d "${SOURCE_DIR}" ]]; then
    echo "FSDP checkpoint directory does not exist: ${SOURCE_DIR}" >&2
    exit 1
fi
SOURCE_DIR="$(cd -- "${SOURCE_DIR}" && pwd -P)"

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd -- "${OUTPUT_DIR}" && pwd -P)"

args=(
    -m verl.model_merger merge
    --backend fsdp
    --local_dir "${SOURCE_DIR}"
    --target_dir "${OUTPUT_DIR}"
)

cd "${PROJECT_ROOT}"
echo "Converting FSDP checkpoint: ${SOURCE_DIR}"
echo "Hugging Face output:       ${OUTPUT_DIR}"
python3 "${args[@]}"
