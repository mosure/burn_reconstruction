#![cfg(target_arch = "wasm32")]

use std::{cell::RefCell, sync::Once};

use burn_yono::glb::{encode_export_gaussians_to_glb_bytes, GlbExportOptions, GlbSortMode};
use js_sys::{Array, Reflect, Uint8Array};
use serde::Deserialize;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

use crate::{
    backend::{default_device, ensure_wasm_wgpu_runtime},
    pipeline::{ImageToGaussianPipeline, PipelineConfig, PipelineInputImage},
};

static PANIC_HOOK_ONCE: Once = Once::new();
const DEFAULT_MODEL_BASE_URL: &str = "https://aberration.technology/model";
const DEFAULT_MODEL_REMOTE_ROOT: &str = "yono";
const BACKBONE_BURNPACK_FILE: &str = "yono_backbone_f16.bpk";
const HEAD_BURNPACK_FILE: &str = "yono_head_f16.bpk";

thread_local! {
    static PIPELINE_CACHE: RefCell<Option<CachedPipeline>> = const { RefCell::new(None) };
}

struct CachedPipeline {
    image_size: usize,
    pipeline: ImageToGaussianPipeline,
}

#[derive(Debug, Deserialize)]
struct WasmPartsManifest {
    parts: Vec<WasmPartEntry>,
}

#[derive(Debug, Deserialize)]
struct WasmPartEntry {
    path: String,
}

#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct WasmInferOptions {
    image_size: u32,
    max_gaussians: u32,
    opacity_threshold: f32,
    sort_by_opacity: bool,
    model_base_url: Option<String>,
    model_remote_root: Option<String>,
}

impl Default for WasmInferOptions {
    fn default() -> Self {
        Self {
            image_size: 224,
            max_gaussians: 4096,
            opacity_threshold: 0.01,
            sort_by_opacity: true,
            model_base_url: None,
            model_remote_root: None,
        }
    }
}

#[wasm_bindgen]
impl WasmInferOptions {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_image_size(&mut self, image_size: u32) {
        let size = image_size.max(14);
        self.image_size = size - (size % 14);
    }

    pub fn set_max_gaussians(&mut self, max_gaussians: u32) {
        self.max_gaussians = max_gaussians.max(1);
    }

    pub fn set_opacity_threshold(&mut self, opacity_threshold: f32) {
        self.opacity_threshold = opacity_threshold.clamp(0.0, 1.0);
    }

    pub fn set_sort_by_opacity(&mut self, sort_by_opacity: bool) {
        self.sort_by_opacity = sort_by_opacity;
    }

    pub fn set_model_base_url(&mut self, model_base_url: String) {
        self.model_base_url = Some(model_base_url);
    }

    pub fn set_model_remote_root(&mut self, model_remote_root: String) {
        self.model_remote_root = Some(model_remote_root);
    }
}

#[wasm_bindgen]
pub fn webgpu_available() -> bool {
    let Some(window) = web_sys::window() else {
        return false;
    };
    let window_js: JsValue = window.into();
    let Ok(navigator) = Reflect::get(&window_js, &JsValue::from_str("navigator")) else {
        return false;
    };
    if navigator.is_null() || navigator.is_undefined() {
        return false;
    }
    let Ok(gpu) = Reflect::get(&navigator, &JsValue::from_str("gpu")) else {
        return false;
    };
    !(gpu.is_null() || gpu.is_undefined())
}

/// Generates a Gaussian-splat GLB from multiple input image bytes.
///
/// Input must be an array of `Uint8Array` values.
#[wasm_bindgen]
pub async fn infer_glb_from_image_bytes_multi(
    images: Array,
    options: Option<WasmInferOptions>,
) -> Result<Uint8Array, JsValue> {
    PANIC_HOOK_ONCE.call_once(console_error_panic_hook::set_once);

    if !webgpu_available() {
        return Err(JsValue::from_str(
            "WebGPU is unavailable in this browser context.",
        ));
    }

    let opts = options.unwrap_or_default();
    if images.length() < 2 {
        return Err(JsValue::from_str(
            "at least 2 images are required for multi-view inference",
        ));
    }

    let image_bytes = collect_image_bytes(images)?;
    ensure_pipeline_loaded(opts.image_size as usize, &opts).await?;

    let names = (0..image_bytes.len())
        .map(|index| format!("view_{index:02}.png"))
        .collect::<Vec<_>>();
    let inputs = names
        .iter()
        .zip(image_bytes.iter())
        .map(|(name, bytes)| PipelineInputImage {
            name: name.as_str(),
            bytes: bytes.as_slice(),
        })
        .collect::<Vec<_>>();

    let entry = PIPELINE_CACHE
        .with(|cell| cell.borrow_mut().take())
        .ok_or_else(|| JsValue::from_str("pipeline cache is unexpectedly empty"))?;
    let run_result = entry
        .pipeline
        .run_image_bytes_timed_with_cameras_async(inputs.as_slice(), true)
        .await;
    PIPELINE_CACHE.with(|cell| {
        *cell.borrow_mut() = Some(entry);
    });
    let run_output =
        run_result.map_err(|err| JsValue::from_str(format!("inference failed: {err}").as_str()))?;

    let export_options = GlbExportOptions {
        max_gaussians: opts.max_gaussians as usize,
        opacity_threshold: opts.opacity_threshold,
        sort_mode: if opts.sort_by_opacity {
            GlbSortMode::Opacity
        } else {
            GlbSortMode::Index
        },
    };
    let selected =
        burn_yono::glb::select_export_gaussians_async(&run_output.gaussians, &export_options)
            .await
            .map_err(|err| JsValue::from_str(format!("export selection failed: {err}").as_str()))?;
    let bytes = encode_export_gaussians_to_glb_bytes(&selected)
        .map_err(|err| JsValue::from_str(format!("glb encode failed: {err}").as_str()))?;
    Ok(Uint8Array::from(bytes.as_slice()))
}

#[wasm_bindgen]
pub async fn infer_glb_from_image_bytes_multi_async(
    images: Array,
    options: Option<WasmInferOptions>,
) -> Result<Uint8Array, JsValue> {
    infer_glb_from_image_bytes_multi(images, options).await
}

fn collect_image_bytes(images: Array) -> Result<Vec<Vec<u8>>, JsValue> {
    let mut out = Vec::with_capacity(images.length() as usize);
    for index in 0..images.length() {
        let item = images.get(index);
        if item.is_null() || item.is_undefined() {
            return Err(JsValue::from_str(
                format!("image[{index}] is null/undefined").as_str(),
            ));
        }

        let bytes = if item.is_instance_of::<Uint8Array>() {
            Uint8Array::new(&item).to_vec()
        } else if let Some(buffer) = item.dyn_ref::<js_sys::ArrayBuffer>() {
            Uint8Array::new(buffer).to_vec()
        } else {
            return Err(JsValue::from_str(
                format!("image[{index}] must be Uint8Array or ArrayBuffer").as_str(),
            ));
        };
        if bytes.is_empty() {
            return Err(JsValue::from_str(
                format!("image[{index}] is empty").as_str(),
            ));
        }
        out.push(bytes);
    }
    Ok(out)
}

async fn ensure_pipeline_loaded(
    image_size: usize,
    options: &WasmInferOptions,
) -> Result<(), JsValue> {
    let needs_reload = PIPELINE_CACHE.with(|cell| {
        let cache = cell.borrow();
        match cache.as_ref() {
            Some(entry) => entry.image_size != image_size,
            None => true,
        }
    });
    if !needs_reload {
        return Ok(());
    }

    let model_root = resolve_model_root_url(options);
    let backbone_url = join_url(model_root.as_str(), BACKBONE_BURNPACK_FILE);
    let head_url = join_url(model_root.as_str(), HEAD_BURNPACK_FILE);
    let backbone_parts = fetch_parts_bundle(backbone_url.as_str()).await?;
    let head_parts = fetch_parts_bundle(head_url.as_str()).await?;

    let cfg = PipelineConfig {
        image_size,
        ..PipelineConfig::default()
    };
    let device = default_device();
    ensure_wasm_wgpu_runtime(&device).await;
    let (pipeline, _report) = ImageToGaussianPipeline::load_from_yono_parts(
        device,
        cfg,
        backbone_parts.as_slice(),
        head_parts.as_slice(),
    )
    .map_err(|err| JsValue::from_str(format!("failed to initialize pipeline: {err}").as_str()))?;

    PIPELINE_CACHE.with(|cell| {
        *cell.borrow_mut() = Some(CachedPipeline {
            image_size,
            pipeline,
        });
    });
    Ok(())
}

fn resolve_model_root_url(options: &WasmInferOptions) -> String {
    let model_base = options
        .model_base_url
        .clone()
        .filter(|entry| !entry.trim().is_empty())
        .or_else(|| js_window_string("BURN_RECONSTRUCTION_MODEL_BASE_URL"))
        .unwrap_or_else(|| DEFAULT_MODEL_BASE_URL.to_string());
    let remote_root = options
        .model_remote_root
        .clone()
        .filter(|entry| !entry.trim().is_empty())
        .or_else(|| js_window_string("BURN_RECONSTRUCTION_YONO_REMOTE_ROOT"))
        .unwrap_or_else(|| DEFAULT_MODEL_REMOTE_ROOT.to_string());

    if remote_root.starts_with("http://") || remote_root.starts_with("https://") {
        return remote_root;
    }
    join_url(model_base.as_str(), remote_root.as_str())
}

fn js_window_string(key: &str) -> Option<String> {
    let window = web_sys::window()?;
    let window_js = JsValue::from(window);
    let value = Reflect::get(&window_js, &JsValue::from_str(key)).ok()?;
    value
        .as_string()
        .map(|entry| entry.trim().to_string())
        .filter(|entry| !entry.is_empty())
}

fn join_url(base: &str, child: &str) -> String {
    if child.starts_with("http://") || child.starts_with("https://") {
        return child.to_string();
    }
    let left = base.trim_end_matches('/');
    let right = child.trim_start_matches('/');
    if left.is_empty() {
        return format!("/{right}");
    }
    format!("{left}/{right}")
}

async fn fetch_parts_bundle(base_burnpack_url: &str) -> Result<Vec<Vec<u8>>, JsValue> {
    let manifest_url = format!("{base_burnpack_url}.parts.json");
    let manifest_bytes = fetch_url_bytes(manifest_url.as_str()).await?;
    let manifest: WasmPartsManifest =
        serde_json::from_slice(manifest_bytes.as_slice()).map_err(|err| {
            JsValue::from_str(
                format!("failed to parse parts manifest {manifest_url}: {err}").as_str(),
            )
        })?;
    if manifest.parts.is_empty() {
        return Err(JsValue::from_str(
            format!("parts manifest is empty: {manifest_url}").as_str(),
        ));
    }

    let mut parts = Vec::with_capacity(manifest.parts.len());
    for part in manifest.parts {
        let url = if part.path.starts_with("http://") || part.path.starts_with("https://") {
            part.path
        } else {
            let manifest_parent = manifest_url
                .rsplit_once('/')
                .map(|(parent, _)| parent)
                .unwrap_or(manifest_url.as_str());
            join_url(manifest_parent, part.path.as_str())
        };
        parts.push(fetch_url_bytes(url.as_str()).await?);
    }
    Ok(parts)
}

async fn fetch_url_bytes(url: &str) -> Result<Vec<u8>, JsValue> {
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("window is unavailable"))?;
    let response_value = JsFuture::from(window.fetch_with_str(url))
        .await
        .map_err(|err| JsValue::from_str(format!("fetch failed for {url}: {err:?}").as_str()))?;
    let response: web_sys::Response = response_value.dyn_into().map_err(|_| {
        JsValue::from_str(format!("fetch returned non-response value for {url}").as_str())
    })?;
    if !response.ok() {
        return Err(JsValue::from_str(
            format!("HTTP {} while fetching {url}", response.status()).as_str(),
        ));
    }
    let array_buffer = JsFuture::from(response.array_buffer().map_err(|err| {
        JsValue::from_str(format!("failed to read response bytes for {url}: {err:?}").as_str())
    })?)
    .await
    .map_err(|err| JsValue::from_str(format!("arrayBuffer failed for {url}: {err:?}").as_str()))?;
    Ok(js_sys::Uint8Array::new(&array_buffer).to_vec())
}
