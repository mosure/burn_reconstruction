# burn_reconstruction 🕊️🔥🌌

[![test](https://github.com/mosure/burn_reconstruction/workflows/test/badge.svg)](https://github.com/Mosure/burn_reconstruction/actions?query=workflow%3Atest)
[![GitHub License](https://img.shields.io/github/license/mosure/burn_reconstruction)](https://raw.githubusercontent.com/mosure/burn_reconstruction/main/LICENSE)
[![crates.io](https://img.shields.io/crates/v/burn_reconstruction.svg)](https://crates.io/crates/burn_reconstruction)

bevy / burn feed-forward gaussian splatting. view the [wasm example](https://mosure.github.io/burn_reconstruction/?image=re10k/0.png&image=re10k/1.png&image=re10k/2.png)

![teaser](docs/teaser.png)


## features

- [x] multi-view images -> 3d gaussians (YoNoSplat path)
- [x] multi-model API/CLI/wasm/bevy selection surface (YoNoSplat + ZipSplat)
- [x] CLI export to GLB (`KHR_gaussian_splatting`)
- [x] bevy UI (native + wasm)
- [x] model auto-bootstrap + cache (parts-first burnpack)
- [x] native Burn/Rust ZipSplat forward path (DA3/DINO backbone + ZipSplat fusion/head)
- [x] official ZipSplat checkpoint auto-cache/import to native burnpack
- [x] ZipSplat upstream-output numerical parity fixture/test path
- [ ] [bevy_synth](https://github.com/mosure/bevy_synth) integration
- [ ] [bevy_zeroverse](https://github.com/mosure/bevy_zeroverse) fine-tuning/training


## setup

### install

```bash
# cli
cargo install burn_reconstruction

# bevy app
cargo install bevy_reconstruction
```

### usage

```bash
burn_reconstruction \
  --images view0.png view1.png view2.png \
  --output /tmp/scene.glb
```

```bash
burn_reconstruction \
  --model zipsplat \
  --weights-format bpk \
  --quality compact \
  --zipsplat-r 4 \
  --images view0.png view1.png view2.png \
  --output /tmp/scene_zipsplat.glb
```

ZipSplat runs through the native Burn/Rust pipeline. When `--zipsplat-weights` is omitted on
native targets, the CLI downloads the official ZipSplat checkpoint to
`~/.burn_reconstruction/models/zipsplat`, imports it to `zipsplat.bpk`, converts the requested
precision (`zipsplat_f16.bpk` by default), and reuses that cache on later runs. To import manually:

```bash
cargo run -p burn_reconstruction --bin import -- \
  --model zipsplat \
  --zipsplat-weights ~/.burn_reconstruction/models/zipsplat/zipsplat-da3g-252p.tar \
  --zipsplat-output ~/.burn_reconstruction/models/zipsplat/zipsplat \
  --precision both \
  --parts true
```

Useful native ZipSplat flags:

- `--weights-format bpk|safetensors` selects the native checkpoint format.
- `--zipsplat-weights <path>` selects an explicit converted ZipSplat checkpoint instead of auto-cache.
- `--zipsplat-r <N>` controls the retained-token reduction factor; higher values emit fewer Gaussians.

ZipSplat numerical parity uses the official upstream PyTorch implementation only as offline
reference-export tooling:

```bash
git clone --depth 1 https://github.com/cvg/ZipSplat /tmp/ZipSplat
python3 -m venv .venv-zipsplat-ref
.venv-zipsplat-ref/bin/python -m pip install torch safetensors einops numpy pillow
.venv-zipsplat-ref/bin/python tool/export_zipsplat_reference.py
cargo test -p burn_reconstruction --features correctness --test parity -- --nocapture zipsplat
```

The runtime ZipSplat path remains native Burn/Rust and does not call Python.

```bash
bevy_reconstruction -- \
  --image assets/images/re10k/0.png \
  --image assets/images/re10k/1.png \
  --image assets/images/re10k/2.png
```

> note, input images are optional for bevy_reconstruction
> ZipSplat browser/wasm inference requires hosted `zipsplat_f16.bpk.parts.json` and matching parts.


## license

licensed under either of:

- Apache License, Version 2.0
- MIT license

at your option.

> note: model weights have their own license


## references

- [bevy_gaussian_splatting](https://github.com/mosure/bevy_gaussian_splatting)
- [burn_dino](https://github.com/mosure/burn_dino)
- [YoNoSplat](https://github.com/cvg/YoNoSplat)
- [ZipSplat](https://github.com/cvg/ZipSplat)
