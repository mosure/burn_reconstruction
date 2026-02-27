#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/www/out"
WASM_PATH="${ROOT_DIR}/target/wasm32-unknown-unknown/wasm-release/burn_gaussian_splatting.wasm"

unset RUSTFLAGS
unset CARGO_ENCODED_RUSTFLAGS
unset CARGO_BUILD_RUSTFLAGS
unset CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS
unset RUSTDOCFLAGS

mkdir -p "${OUT_DIR}"

echo "[web] build wasm"
cargo build \
  --lib \
  --target wasm32-unknown-unknown \
  --profile wasm-release

echo "[web] bindgen"
wasm-bindgen \
  "${WASM_PATH}" \
  --out-dir "${OUT_DIR}" \
  --target web \
  --no-typescript

test -f "${OUT_DIR}/burn_gaussian_splatting.js"
test -f "${OUT_DIR}/burn_gaussian_splatting_bg.wasm"

echo "[web] done"
