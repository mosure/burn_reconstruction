# burn_gaussian_splatting 🔥🌌

[![test](https://github.com/mosure/burn_gaussian_splatting/workflows/test/badge.svg)](https://github.com/Mosure/burn_gaussian_splatting/actions?query=workflow%3Atest)
[![GitHub License](https://img.shields.io/github/license/mosure/burn_gaussian_splatting)](https://raw.githubusercontent.com/mosure/burn_gaussian_splatting/main/LICENSE)
[![crates.io](https://img.shields.io/crates/v/burn_gaussian_splatting.svg)](https://crates.io/crates/burn_gaussian_splatting)


burn feed-forward gaussian splatting


![Alt text](docs/teaser.png)


## capabilities

- [x] multi-view -> 3dgs


## multi-image to GLB

Run the CLI (requires `cli` feature):

```bash
cargo run --features cli --bin splat_glb -- \
  --images view0.png view1.png \
  --output outputs/gaussians.glb \
  --image-size 224 \
  --weights-format safetensors \
  --quality balanced \
  --profile
```

This exports a GLB with `KHR_gaussian_splatting` extension metadata and gaussian attributes.
You can override export policy with:
- `--max-gaussians`
- `--opacity-threshold`
- `--sort-mode {opacity|index}`
- `--weights-format {safetensors|bpk}`

Quality presets:
- `fast`: lower gaussian budget, quick outputs.
- `balanced`: default quality/perf tradeoff.
- `high`: full gaussian export


## license
licensed under either of

 - Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
 - MIT license (http://opensource.org/licenses/MIT)

at your option.
