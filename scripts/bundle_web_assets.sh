#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_ROOT="${1:-${ROOT_DIR}/assets/models}"
DST_ROOT="${2:-${ROOT_DIR}/www/assets/model/yono}"
IMAGE_SRC_ROOT="${ROOT_DIR}/assets/images"
IMAGE_DST_ROOT="${ROOT_DIR}/www/assets/images"
STRICT="${BURN_RECONSTRUCTION_WEB_BUNDLE_STRICT:-0}"

MODELS=(
  "yono_backbone"
  "yono_head"
)
PRECISION_SUFFIXES=(
  "_f16"
)

mkdir -p "${DST_ROOT}"

# Keep the destination deterministic and storage-minimal for deployment.
# Remove previously bundled model artifacts (including stale f32 files).
find "${DST_ROOT}" -maxdepth 1 -type f \
  \( -name "*.bpk" -o -name "*.bpk.parts.json" -o -name "*.bpk.part-*" -o -name "manifest.json" \) \
  -delete

if [[ ! -d "${SRC_ROOT}" ]]; then
  if [[ "${STRICT}" == "1" ]]; then
    echo "[bundle] source models directory not found: ${SRC_ROOT}" >&2
    exit 1
  fi
  echo "[bundle] source models directory not found (skipping): ${SRC_ROOT}" >&2
  exit 0
fi

echo "[bundle] source=${SRC_ROOT}"
echo "[bundle] dest=${DST_ROOT}"

copy_count=0
missing_count=0
manifest_entries=()

copy_file() {
  local src="$1"
  local rel="$2"
  local dst="${DST_ROOT}/${rel}"
  mkdir -p "$(dirname "${dst}")"
  cp -f "${src}" "${dst}"
  echo "[bundle] copied ${rel}"
  copy_count=$((copy_count + 1))

  local size
  local sha
  size="$(stat -c%s "${dst}")"
  sha="$(sha256sum "${dst}" | awk '{print $1}')"
  manifest_entries+=("${rel}|${size}|${sha}")
}

for model in "${MODELS[@]}"; do
  found_any=0
  for suffix in "${PRECISION_SUFFIXES[@]}"; do
    stem="${model}${suffix}"
    parts_manifest="${SRC_ROOT}/${stem}.bpk.parts.json"
    if [[ ! -f "${parts_manifest}" ]]; then
      continue
    fi

    mapfile -t parts < <(find "${SRC_ROOT}" -maxdepth 1 -type f -name "${stem}.bpk.part-*" | sort)
    if [[ "${#parts[@]}" -eq 0 ]]; then
      echo "[bundle] skipping ${stem}: missing part shards for ${parts_manifest}" >&2
      continue
    fi
    found_any=1

    copy_file "${parts_manifest}" "${stem}.bpk.parts.json"

    for part in "${parts[@]}"; do
      rel="$(basename "${part}")"
      copy_file "${part}" "${rel}"
    done
  done

  if [[ "${found_any}" -eq 0 ]]; then
    echo "[bundle] missing required model parts bundle (f16): ${model}_f16.bpk.parts.json + shards" >&2
    missing_count=$((missing_count + 1))
  fi
done

if [[ "${missing_count}" -ne 0 ]]; then
  if [[ "${STRICT}" == "1" ]]; then
    echo "[bundle] required burnpack parts bundles are missing. Generate them first, for example:" >&2
    echo "[bundle] cargo run --features cli --bin import -- --component both --format bpk --precision f16" >&2
    exit 1
  fi
  echo "[bundle] required parts bundles are missing (skipping absent artifacts)." >&2
fi

if [[ "${copy_count}" -eq 0 ]]; then
  if [[ "${STRICT}" == "1" ]]; then
    echo "[bundle] no files copied" >&2
    exit 1
  fi
  echo "[bundle] no files copied" >&2
  exit 0
fi

manifest_path="${DST_ROOT}/manifest.json"
timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
{
  echo "{"
  echo "  \"model\": \"yono\","
  echo "  \"generated_at_utc\": \"${timestamp}\","
  echo "  \"files\": ["
  for i in "${!manifest_entries[@]}"; do
    IFS='|' read -r rel size sha <<<"${manifest_entries[$i]}"
    suffix=","
    if [[ "$i" -eq $((${#manifest_entries[@]} - 1)) ]]; then
      suffix=""
    fi
    echo "    {\"path\": \"${rel}\", \"bytes\": ${size}, \"sha256\": \"${sha}\"}${suffix}"
  done
  echo "  ]"
  echo "}"
} > "${manifest_path}"

echo "[bundle] wrote manifest ${manifest_path}"
echo "[bundle] complete (${copy_count} files)"

if [[ -d "${IMAGE_SRC_ROOT}" ]]; then
  mkdir -p "${IMAGE_DST_ROOT}"
  rm -rf "${IMAGE_DST_ROOT:?}/"*
  cp -r "${IMAGE_SRC_ROOT}/." "${IMAGE_DST_ROOT}/"
  image_count="$(find "${IMAGE_DST_ROOT}" -type f | wc -l | tr -d ' ')"
  echo "[bundle] copied image assets to ${IMAGE_DST_ROOT} (${image_count} files)"
else
  if [[ "${STRICT}" == "1" ]]; then
    echo "[bundle] image source directory not found: ${IMAGE_SRC_ROOT}" >&2
    exit 1
  fi
  echo "[bundle] image source directory not found (skipping): ${IMAGE_SRC_ROOT}" >&2
fi
