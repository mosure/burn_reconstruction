#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${1:-${ROOT_DIR}/dist/cdn/model}"
YONO_SRC="${BURN_RECONSTRUCTION_YONO_MODEL_SRC:-${ROOT_DIR}/assets/models}"
ZIPSPLAT_SRC="${BURN_RECONSTRUCTION_ZIPSPLAT_MODEL_SRC:-${HOME}/.burn_reconstruction/models/zipsplat}"
LINK_MODE="${BURN_RECONSTRUCTION_CDN_BUNDLE_LINK_MODE:-hardlink}"

if [[ "${OUT_ROOT}" != /* ]]; then
  OUT_ROOT="${ROOT_DIR}/${OUT_ROOT}"
fi

case "${LINK_MODE}" in
  copy | hardlink) ;;
  *)
    echo "[cdn-bundle] invalid BURN_RECONSTRUCTION_CDN_BUNDLE_LINK_MODE=${LINK_MODE}; expected copy or hardlink" >&2
    exit 1
    ;;
esac

copy_or_link() {
  local src="$1"
  local dst="$2"
  mkdir -p "$(dirname "${dst}")"
  if [[ "${LINK_MODE}" == "hardlink" ]]; then
    ln -f "${src}" "${dst}" 2>/dev/null || cp -f "${src}" "${dst}"
  else
    cp -f "${src}" "${dst}"
  fi
}

json_string() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  printf '"%s"' "${value}"
}

bundle_model() {
  local model="$1"
  local src_root="$2"
  shift 2
  local stems=("$@")
  local dst_root="${OUT_ROOT}/${model}"
  local manifest_entries=()

  if [[ ! -d "${src_root}" ]]; then
    echo "[cdn-bundle] missing source directory for ${model}: ${src_root}" >&2
    exit 1
  fi

  rm -rf "${dst_root}"
  mkdir -p "${dst_root}"

  for stem in "${stems[@]}"; do
    local manifest="${src_root}/${stem}.bpk.parts.json"
    if [[ ! -f "${manifest}" ]]; then
      echo "[cdn-bundle] missing manifest for ${model}: ${manifest}" >&2
      exit 1
    fi

    copy_or_link "${manifest}" "${dst_root}/$(basename "${manifest}")"
    manifest_entries+=("$(basename "${manifest}")")

    mapfile -t parts < <(find "${src_root}" -maxdepth 1 -type f -name "${stem}.bpk.part-*.bpk" | sort)
    if [[ "${#parts[@]}" -eq 0 ]]; then
      echo "[cdn-bundle] missing shards for ${model}: ${stem}.bpk.part-*.bpk" >&2
      exit 1
    fi

    for part in "${parts[@]}"; do
      copy_or_link "${part}" "${dst_root}/$(basename "${part}")"
      manifest_entries+=("$(basename "${part}")")
    done
  done

  local manifest_path="${dst_root}/manifest.json"
  local timestamp
  timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  {
    echo "{"
    echo "  \"model\": $(json_string "${model}"),"
    echo "  \"upload_prefix\": $(json_string "/model/${model}/"),"
    echo "  \"generated_at_utc\": $(json_string "${timestamp}"),"
    echo "  \"link_mode\": $(json_string "${LINK_MODE}"),"
    echo "  \"files\": ["
    for i in "${!manifest_entries[@]}"; do
      local rel="${manifest_entries[$i]}"
      local path="${dst_root}/${rel}"
      local size
      local sha
      size="$(stat -c%s "${path}")"
      sha="$(sha256sum "${path}" | awk '{print $1}')"
      local suffix=","
      if [[ "${i}" -eq $((${#manifest_entries[@]} - 1)) ]]; then
        suffix=""
      fi
      echo "    {\"path\": $(json_string "${rel}"), \"bytes\": ${size}, \"sha256\": $(json_string "${sha}")}${suffix}"
    done
    echo "  ]"
    echo "}"
  } > "${manifest_path}"

  echo "[cdn-bundle] ${model}: ${#manifest_entries[@]} files -> ${dst_root}"
}

rm -rf "${OUT_ROOT}"
mkdir -p "${OUT_ROOT}"

bundle_model "yono" "${YONO_SRC}" "yono_backbone_f16" "yono_head_f16"
bundle_model "zipsplat" "${ZIPSPLAT_SRC}" "zipsplat_f16"

echo "[cdn-bundle] upload contents of ${OUT_ROOT} to s3://<bucket>/model/"
