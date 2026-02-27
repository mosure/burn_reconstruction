use std::cmp::Ordering;

#[cfg(feature = "io")]
use std::{path::Path, time::Instant};

use burn::prelude::*;

use crate::model::gaussian::FlatGaussians;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GlbSortMode {
    Opacity,
    Index,
}

#[derive(Clone, Debug)]
pub struct GlbExportOptions {
    pub max_gaussians: usize,
    pub opacity_threshold: f32,
    pub sort_mode: GlbSortMode,
}

impl Default for GlbExportOptions {
    fn default() -> Self {
        Self {
            max_gaussians: 4096,
            opacity_threshold: 0.01,
            sort_mode: GlbSortMode::Opacity,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GlbExportReport {
    pub selected_gaussians: usize,
    pub select_millis: f64,
    pub write_millis: f64,
}

#[derive(Debug)]
pub struct ExportGaussians {
    pub positions: Vec<f32>,
    pub colors: Vec<f32>,
    pub scales: Vec<f32>,
    pub rotations: Vec<f32>,
    pub opacities: Vec<f32>,
}

#[derive(Debug, thiserror::Error)]
pub enum GlbExportError {
    #[error("gaussian harmonics tensor has zero SH channels")]
    MissingHarmonics,
    #[error("failed to read gaussian tensors from backend: {0}")]
    TensorData(String),
    #[error("no gaussians survived opacity filtering")]
    NoGaussians,
    #[cfg(feature = "io")]
    #[error("failed to write glb: {0}")]
    Io(#[from] std::io::Error),
    #[cfg(feature = "io")]
    #[error("failed to serialize gltf json: {0}")]
    Json(#[from] serde_json::Error),
}

pub fn select_export_gaussians<B: Backend>(
    gaussians: &FlatGaussians<B>,
    options: &GlbExportOptions,
) -> Result<ExportGaussians, GlbExportError> {
    let packed = pack_export_rows(gaussians)?;
    let row_width = 14usize;
    let rows = packed.len() / row_width;

    let mut indices: Vec<usize> = (0..rows)
        .filter(|&idx| packed[idx * row_width + 13] >= options.opacity_threshold)
        .collect();

    if matches!(options.sort_mode, GlbSortMode::Opacity) {
        indices.sort_by(|lhs, rhs| {
            packed[rhs * row_width + 13]
                .partial_cmp(&packed[lhs * row_width + 13])
                .unwrap_or(Ordering::Equal)
        });
    }

    indices.truncate(options.max_gaussians.max(1));
    if indices.is_empty() {
        return Err(GlbExportError::NoGaussians);
    }

    let mut out = ExportGaussians {
        positions: Vec::with_capacity(indices.len() * 3),
        colors: Vec::with_capacity(indices.len() * 3),
        scales: Vec::with_capacity(indices.len() * 3),
        rotations: Vec::with_capacity(indices.len() * 4),
        opacities: Vec::with_capacity(indices.len()),
    };

    for idx in indices {
        let off = idx * row_width;
        out.positions.extend_from_slice(&packed[off..off + 3]);
        out.colors.extend_from_slice(
            [
                packed[off + 3].clamp(0.0, 1.0),
                packed[off + 4].clamp(0.0, 1.0),
                packed[off + 5].clamp(0.0, 1.0),
            ]
            .as_slice(),
        );
        out.scales.extend_from_slice(&packed[off + 6..off + 9]);
        out.rotations.extend_from_slice(&packed[off + 9..off + 13]);
        out.opacities.push(packed[off + 13].clamp(0.0, 1.0));
    }

    Ok(out)
}

#[cfg(feature = "io")]
pub fn save_gaussians_to_glb<B: Backend>(
    path: &Path,
    gaussians: &FlatGaussians<B>,
    options: &GlbExportOptions,
) -> Result<usize, GlbExportError> {
    let report = save_gaussians_to_glb_timed(path, gaussians, options)?;
    Ok(report.selected_gaussians)
}

#[cfg(feature = "io")]
pub fn save_gaussians_to_glb_timed<B: Backend>(
    path: &Path,
    gaussians: &FlatGaussians<B>,
    options: &GlbExportOptions,
) -> Result<GlbExportReport, GlbExportError> {
    let select_start = Instant::now();
    let selected = select_export_gaussians(gaussians, options)?;
    let select_millis = select_start.elapsed().as_secs_f64() * 1000.0;

    let write_start = Instant::now();
    write_khr_gaussian_glb(path, &selected)?;
    let write_millis = write_start.elapsed().as_secs_f64() * 1000.0;

    Ok(GlbExportReport {
        selected_gaussians: selected.opacities.len(),
        select_millis,
        write_millis,
    })
}

fn pack_export_rows<B: Backend>(gaussians: &FlatGaussians<B>) -> Result<Vec<f32>, GlbExportError> {
    let [b, g, _] = gaussians.means.shape().dims::<3>();
    let d_sh = gaussians.harmonics.shape().dims::<4>()[3];
    if d_sh == 0 {
        return Err(GlbExportError::MissingHarmonics);
    }

    // One packed readback to minimize GPU -> host transfers during export.
    let colors = gaussians
        .harmonics
        .clone()
        .slice([0..b as i32, 0..g as i32, 0..3, 0..1])
        .reshape([b as i32, g as i32, 3]);

    let packed = Tensor::cat(
        vec![
            gaussians.means.clone(),
            colors,
            gaussians.scales.clone(),
            gaussians.rotations.clone(),
            gaussians.opacities.clone().unsqueeze_dim(2),
        ],
        2,
    );

    packed
        .into_data()
        .to_vec::<f32>()
        .map_err(|err| GlbExportError::TensorData(format!("{err:?}")))
}

#[cfg(feature = "io")]
#[derive(Debug, Clone, Copy)]
struct BufferSlice {
    byte_offset: usize,
    byte_length: usize,
}

#[cfg(feature = "io")]
fn write_khr_gaussian_glb(path: &Path, gaussians: &ExportGaussians) -> Result<(), GlbExportError> {
    use std::{fs::File, io::Write};

    use serde_json::json;

    let count = gaussians.opacities.len();
    let (min_pos, max_pos) = min_max_vec3(gaussians.positions.as_slice());

    let mut bin = Vec::<u8>::new();
    let pos_slice = push_f32_data(&mut bin, gaussians.positions.as_slice());
    let color_slice = push_f32_data(&mut bin, gaussians.colors.as_slice());
    let scale_slice = push_f32_data(&mut bin, gaussians.scales.as_slice());
    let rot_slice = push_f32_data(&mut bin, gaussians.rotations.as_slice());
    let opa_slice = push_f32_data(&mut bin, gaussians.opacities.as_slice());
    align4(&mut bin, 0);

    let buffer_views = vec![
        json!({
            "buffer": 0,
            "byteOffset": pos_slice.byte_offset,
            "byteLength": pos_slice.byte_length,
            "target": 34962
        }),
        json!({
            "buffer": 0,
            "byteOffset": color_slice.byte_offset,
            "byteLength": color_slice.byte_length,
            "target": 34962
        }),
        json!({
            "buffer": 0,
            "byteOffset": scale_slice.byte_offset,
            "byteLength": scale_slice.byte_length,
            "target": 34962
        }),
        json!({
            "buffer": 0,
            "byteOffset": rot_slice.byte_offset,
            "byteLength": rot_slice.byte_length,
            "target": 34962
        }),
        json!({
            "buffer": 0,
            "byteOffset": opa_slice.byte_offset,
            "byteLength": opa_slice.byte_length,
            "target": 34962
        }),
    ];

    let accessors = vec![
        json!({
            "bufferView": 0,
            "componentType": 5126,
            "count": count,
            "type": "VEC3",
            "min": min_pos,
            "max": max_pos
        }),
        json!({
            "bufferView": 1,
            "componentType": 5126,
            "count": count,
            "type": "VEC3"
        }),
        json!({
            "bufferView": 2,
            "componentType": 5126,
            "count": count,
            "type": "VEC3"
        }),
        json!({
            "bufferView": 3,
            "componentType": 5126,
            "count": count,
            "type": "VEC4"
        }),
        json!({
            "bufferView": 4,
            "componentType": 5126,
            "count": count,
            "type": "SCALAR"
        }),
    ];

    let gltf = json!({
        "asset": {
            "version": "2.0",
            "generator": "burn_yono::glb"
        },
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{
            "primitives": [{
                "mode": 0,
                "attributes": {
                    "POSITION": 0,
                    "COLOR_0": 1
                },
                "extensions": {
                    "KHR_gaussian_splatting": {
                        "attributes": {
                            "SCALE": 2,
                            "ROTATION": 3,
                            "OPACITY": 4
                        }
                    }
                }
            }]
        }],
        "buffers": [{"byteLength": bin.len()}],
        "bufferViews": buffer_views,
        "accessors": accessors,
        "extensionsUsed": ["KHR_gaussian_splatting"],
        "extensionsRequired": ["KHR_gaussian_splatting"]
    });

    let mut json_bytes = serde_json::to_vec(&gltf)?;
    align4(&mut json_bytes, 0x20);

    let mut file = File::create(path)?;

    let total_len = 12 + 8 + json_bytes.len() + 8 + bin.len();
    file.write_all(&0x4654_6C67u32.to_le_bytes())?;
    file.write_all(&2u32.to_le_bytes())?;
    file.write_all(&(total_len as u32).to_le_bytes())?;

    file.write_all(&(json_bytes.len() as u32).to_le_bytes())?;
    file.write_all(&0x4E4F_534Au32.to_le_bytes())?;
    file.write_all(json_bytes.as_slice())?;

    file.write_all(&(bin.len() as u32).to_le_bytes())?;
    file.write_all(&0x004E_4942u32.to_le_bytes())?;
    file.write_all(bin.as_slice())?;

    Ok(())
}

#[cfg(feature = "io")]
fn push_f32_data(buffer: &mut Vec<u8>, values: &[f32]) -> BufferSlice {
    align4(buffer, 0);
    let offset = buffer.len();
    for value in values {
        buffer.extend_from_slice(&value.to_le_bytes());
    }
    BufferSlice {
        byte_offset: offset,
        byte_length: std::mem::size_of_val(values),
    }
}

#[cfg(feature = "io")]
fn align4(buffer: &mut Vec<u8>, pad: u8) {
    while !buffer.len().is_multiple_of(4) {
        buffer.push(pad);
    }
}

#[cfg(feature = "io")]
fn min_max_vec3(values: &[f32]) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];

    for chunk in values.chunks_exact(3) {
        for axis in 0..3 {
            min[axis] = min[axis].min(chunk[axis]);
            max[axis] = max[axis].max(chunk[axis]);
        }
    }

    (min, max)
}
