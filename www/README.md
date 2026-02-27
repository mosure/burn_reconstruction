# burn_gaussian_splatting web

## build wasm + bindings

```bash
./scripts/build_web.sh
```

This writes:

- `www/out/burn_gaussian_splatting.js`
- `www/out/burn_gaussian_splatting_bg.wasm`

## local preview

```bash
python3 -m http.server 4173 -d www
```

Then open:

- `http://127.0.0.1:4173/index.html`

## optional model bundle copy

If you want local model asset mirrors for service-worker caching, copy selected files into
`www/assets/models`:

```bash
./scripts/bundle_web_assets.sh
```

The bundler keeps deployment payload constrained to web-runtime files:

- `*.bpk`
- `*.bpk.parts.json`
- `*.bpk.part-*`
- `*.json`

## service worker

Source is `www/burn_gaussian_splatting_sw.js`.

During deploy, copy it to site root as `burn_gaussian_splatting_sw.js` so root pages are in scope
without relying on `Service-Worker-Allowed` headers.
