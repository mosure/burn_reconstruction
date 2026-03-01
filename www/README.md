# bevy_reconstruction web

## build wasm + bindings

```bash
./scripts/build_web.sh
```

This writes:

- `www/out/bevy_reconstruction.js`
- `www/out/bevy_reconstruction_bg.wasm`
- `www/out/burn_reconstruction.js`
- `www/out/burn_reconstruction_bg.wasm`

## local preview

```bash
python3 -m http.server 4173 -d www
```

Then open:

- `http://127.0.0.1:4173/index.html`
- `http://127.0.0.1:4173/infer.html`

This web entrypoint validates wasm startup and asset loading for the Bevy app shell.
`infer.html` is a simplified non-Bevy inference page (file input -> wasm API -> GLB download).
The web app now runs inference and, by default, pulls model parts from:
- `https://aberration.technology/model/yono`

Override source at runtime (before wasm init) with browser globals:
- `window.BURN_RECONSTRUCTION_MODEL_BASE_URL`
- `window.BURN_RECONSTRUCTION_YONO_REMOTE_ROOT`

## optional model bundle copy

If you want local model asset mirrors for service-worker caching, copy selected files into
`www/assets/model/yono` and demo startup images into `www/assets/images`:

```bash
./scripts/bundle_web_assets.sh
```

Set `BURN_RECONSTRUCTION_WEB_BUNDLE_STRICT=1` to fail if required `.bpk` files are missing.

The bundler keeps deployment payload constrained to web-runtime files:

- f16 parts manifests only: `yono_backbone_f16.bpk.parts.json`, `yono_head_f16.bpk.parts.json`
- matching part shards only: `*.bpk.part-*`
- `manifest.json` with size + sha256 metadata
- image fixtures copied from `assets/images/**` (for concise URL startup args)

Example startup URL (concise image args):

```text
http://127.0.0.1:4173/index.html?image=re10k/0.png&image=re10k/1.png&image=re10k/2.png
```

`re10k/<n>.png` is resolved to `assets/images/re10k/<n>.png` in wasm startup logic.

## github pages deploy

CI deploy intentionally does **not** bundle model weights.
Pages publishes wasm/web assets + `assets/images/**`; model files are expected from remote hosting
(for example S3/CloudFront via `https://aberration.technology/model/...`).

## service worker

Source is `www/burn_reconstruction_sw.js`.

During deploy, copy it to site root as `burn_reconstruction_sw.js` so root pages are in scope
without relying on `Service-Worker-Allowed` headers.

`www/index.html` registers the service worker before wasm init; parts requests
(`*.bpk.parts.json`, `*.bpk.part-*`) are cacheable by default.
