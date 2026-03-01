#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/www/out"
BEVY_WASM_PATH="${ROOT_DIR}/target/wasm32-unknown-unknown/wasm-release/bevy_reconstruction.wasm"
CORE_WASM_PATH="${ROOT_DIR}/target/wasm32-unknown-unknown/wasm-release/burn_reconstruction.wasm"

unset RUSTFLAGS
unset CARGO_ENCODED_RUSTFLAGS
unset CARGO_BUILD_RUSTFLAGS
unset CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS
unset RUSTDOCFLAGS

mkdir -p "${OUT_DIR}"

echo "[web] build wasm"
cargo build \
  -p bevy_reconstruction \
  --bin bevy_reconstruction \
  -p burn_reconstruction \
  --lib \
  --target wasm32-unknown-unknown \
  --profile wasm-release

echo "[web] bindgen bevy_reconstruction"
wasm-bindgen \
  "${BEVY_WASM_PATH}" \
  --out-dir "${OUT_DIR}" \
  --target web \
  --no-typescript

echo "[web] bindgen burn_reconstruction"
wasm-bindgen \
  "${CORE_WASM_PATH}" \
  --out-dir "${OUT_DIR}" \
  --target web \
  --no-typescript

test -f "${OUT_DIR}/bevy_reconstruction.js"
test -f "${OUT_DIR}/bevy_reconstruction_bg.wasm"
test -f "${OUT_DIR}/burn_reconstruction.js"
test -f "${OUT_DIR}/burn_reconstruction_bg.wasm"

echo "[web] done"
