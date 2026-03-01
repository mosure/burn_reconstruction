# burn_reconstruction 🔥🌌

[![test](https://github.com/mosure/burn_reconstruction/workflows/test/badge.svg)](https://github.com/Mosure/burn_reconstruction/actions?query=workflow%3Atest)
[![GitHub License](https://img.shields.io/github/license/mosure/burn_reconstruction)](https://raw.githubusercontent.com/mosure/burn_reconstruction/main/LICENSE)
[![crates.io](https://img.shields.io/crates/v/burn_reconstruction.svg)](https://crates.io/crates/burn_reconstruction)

bevy / burn feed-forward gaussian splatting.

![teaser](docs/teaser.png)


## features

- [x] multi-view images -> 3d gaussians (YoNoSplat path)
- [x] CLI export to GLB (`KHR_gaussian_splatting`)
- [x] bevy UI (native + wasm)
- [x] model auto-bootstrap + cache (parts-first burnpack)


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
bevy_reconstruction -- \
  --image assets/images/re10k/0.png \
  --image assets/images/re10k/1.png \
  --image assets/images/re10k/2.png
```

> note, input images are optional for bevy_reconstruction


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
