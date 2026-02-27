#![cfg(target_arch = "wasm32")]

use std::sync::Once;

use burn_yono::glb::{encode_export_gaussians_to_glb_bytes, ExportGaussians};
use image::imageops::FilterType;
use js_sys::{Array, Reflect, Uint8Array};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

static PANIC_HOOK_ONCE: Once = Once::new();

#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct WasmInferOptions {
    image_size: u32,
    max_gaussians: u32,
    samples_per_image: u32,
    opacity_threshold: f32,
    z_separation: f32,
    point_scale: f32,
}

impl Default for WasmInferOptions {
    fn default() -> Self {
        Self {
            image_size: 224,
            max_gaussians: 4096,
            samples_per_image: 2048,
            opacity_threshold: 0.02,
            z_separation: 0.2,
            point_scale: 0.015,
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
        self.image_size = image_size.max(32);
    }

    pub fn set_max_gaussians(&mut self, max_gaussians: u32) {
        self.max_gaussians = max_gaussians.max(1);
    }

    pub fn set_samples_per_image(&mut self, samples_per_image: u32) {
        self.samples_per_image = samples_per_image.max(64);
    }

    pub fn set_opacity_threshold(&mut self, opacity_threshold: f32) {
        self.opacity_threshold = opacity_threshold.clamp(0.0, 1.0);
    }

    pub fn set_z_separation(&mut self, z_separation: f32) {
        self.z_separation = z_separation.max(0.0);
    }

    pub fn set_point_scale(&mut self, point_scale: f32) {
        self.point_scale = point_scale.clamp(0.001, 0.2);
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

/// Generates a Gaussian-splat GLB from multiple image files.
///
/// Input must be an array of `Uint8Array` values.
#[wasm_bindgen]
pub fn infer_glb_from_image_bytes_multi(
    images: Array,
    options: Option<WasmInferOptions>,
) -> Result<Uint8Array, JsValue> {
    PANIC_HOOK_ONCE.call_once(console_error_panic_hook::set_once);

    let opts = options.unwrap_or_default();
    if images.length() < 2 {
        return Err(JsValue::from_str(
            "at least 2 images are required for multi-view inference",
        ));
    }

    let image_bytes = collect_image_bytes(images)?;
    let export = synthesize_export_gaussians(image_bytes.as_slice(), &opts)
        .map_err(|err| JsValue::from_str(err.as_str()))?;
    let bytes = encode_export_gaussians_to_glb_bytes(&export)
        .map_err(|err| JsValue::from_str(err.to_string().as_str()))?;
    Ok(Uint8Array::from(bytes.as_slice()))
}

fn collect_image_bytes(images: Array) -> Result<Vec<Vec<u8>>, JsValue> {
    let mut out = Vec::with_capacity(images.length() as usize);
    for index in 0..images.length() {
        let item = images.get(index);
        if item.is_null() || item.is_undefined() {
            return Err(JsValue::from_str(&format!(
                "image[{index}] is null/undefined"
            )));
        }

        let bytes = if item.is_instance_of::<Uint8Array>() {
            Uint8Array::new(&item).to_vec()
        } else if let Some(buffer) = item.dyn_ref::<js_sys::ArrayBuffer>() {
            Uint8Array::new(buffer).to_vec()
        } else {
            return Err(JsValue::from_str(&format!(
                "image[{index}] must be Uint8Array or ArrayBuffer"
            )));
        };
        if bytes.is_empty() {
            return Err(JsValue::from_str(&format!("image[{index}] is empty")));
        }
        out.push(bytes);
    }
    Ok(out)
}

fn synthesize_export_gaussians(
    images: &[Vec<u8>],
    options: &WasmInferOptions,
) -> Result<ExportGaussians, String> {
    let views = images.len().max(1);
    let image_size = options.image_size.max(32) as usize;
    let per_image_target = ((options.max_gaussians as usize)
        .div_ceil(views)
        .max(options.samples_per_image as usize))
    .max(64);
    let sample_axis = (per_image_target as f64).sqrt().ceil() as usize;
    let step = (image_size / sample_axis.max(1)).max(1);

    let max_total = options.max_gaussians.max(1) as usize;
    let mut positions = Vec::with_capacity(max_total * 3);
    let mut colors = Vec::with_capacity(max_total * 3);
    let mut scales = Vec::with_capacity(max_total * 3);
    let mut rotations = Vec::with_capacity(max_total * 4);
    let mut opacities = Vec::with_capacity(max_total);

    for (view_idx, bytes) in images.iter().enumerate() {
        let image = image::load_from_memory(bytes)
            .map_err(|err| format!("failed to decode image[{view_idx}]: {err}"))?;
        let resized = image
            .resize_exact(image_size as u32, image_size as u32, FilterType::Triangle)
            .to_rgb8();

        for y in (0..image_size).step_by(step) {
            for x in (0..image_size).step_by(step) {
                if opacities.len() >= max_total {
                    break;
                }
                let px = resized.get_pixel(x as u32, y as u32).0;
                let r = px[0] as f32 / 255.0;
                let g = px[1] as f32 / 255.0;
                let b = px[2] as f32 / 255.0;
                let lum = (0.2126 * r + 0.7152 * g + 0.0722 * b).clamp(0.0, 1.0);
                if lum < options.opacity_threshold {
                    continue;
                }

                let nx = (x as f32 / image_size as f32) * 2.0 - 1.0;
                let ny = 1.0 - (y as f32 / image_size as f32) * 2.0;
                let nz = (view_idx as f32 - (views as f32 - 1.0) * 0.5) * options.z_separation;

                positions.extend_from_slice([nx, ny, nz].as_slice());
                colors.extend_from_slice([r, g, b].as_slice());
                scales.extend_from_slice(
                    [
                        options.point_scale,
                        options.point_scale,
                        options.point_scale,
                    ]
                    .as_slice(),
                );
                rotations.extend_from_slice([0.0, 0.0, 0.0, 1.0].as_slice());
                opacities.push((0.2 + 0.8 * lum).clamp(0.0, 1.0));
            }
            if opacities.len() >= max_total {
                break;
            }
        }
    }

    if opacities.is_empty() {
        return Err(
            "all sampled points were filtered out; lower opacity threshold or use brighter images"
                .to_string(),
        );
    }

    Ok(ExportGaussians {
        positions,
        colors,
        scales,
        rotations,
        opacities,
    })
}
