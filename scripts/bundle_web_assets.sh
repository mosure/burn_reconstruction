#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_ROOT="${1:-${ROOT_DIR}/assets/models}"
DST_ROOT="${ROOT_DIR}/www/assets/models"

mkdir -p "${DST_ROOT}"

if [[ ! -d "${SRC_ROOT}" ]]; then
  echo "[bundle] source models directory not found: ${SRC_ROOT}" >&2
  exit 1
fi

echo "[bundle] source=${SRC_ROOT}"
echo "[bundle] dest=${DST_ROOT}"

find "${SRC_ROOT}" -type f \
  \( \
    -name "*.bpk" -o \
    -name "*.bpk.parts.json" -o \
    -name "*.bpk.part-*" -o \
    -name "*.json" \
  \) | while IFS= read -r src; do
    rel="${src#"${SRC_ROOT}/"}"
    dst="${DST_ROOT}/${rel}"
    mkdir -p "$(dirname "${dst}")"
    cp -f "${src}" "${dst}"
    echo "[bundle] copied ${rel}"
  done

echo "[bundle] complete"
