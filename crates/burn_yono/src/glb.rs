use std::cmp::Ordering;

#[cfg(feature = "io")]
use std::{path::Path, time::Instant};

use burn::prelude::*;
use nalgebra::{Matrix3, Quaternion, SymmetricEigen, UnitQuaternion, Vector3};

use crate::model::gaussian::FlatGaussians;

const VIEWER_MIN_SCALE: f32 = 1e-3;
const VIEWER_MAX_SCALE: f32 = 0.3;
const VIEWER_MAX_SCALE_RATIO: f32 = 256.0;

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

#[derive(Debug, Clone)]
pub struct PackedGaussianRows {
    pub rows: usize,
    pub d_sh: usize,
    pub row_width: usize,
    pub values: Vec<f32>,
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CanonicalGaussianTransform {
    pub position: [f32; 3],
    pub rotation_wxyz: [f32; 4],
    pub scale: [f32; 3],
    pub opacity: f32,
}

pub fn select_export_gaussians<B: Backend>(
    gaussians: &FlatGaussians<B>,
    options: &GlbExportOptions,
) -> Result<ExportGaussians, GlbExportError> {
    let packed = pack_export_rows(gaussians)?;
    select_export_gaussians_from_packed(packed, options)
}

#[cfg(target_arch = "wasm32")]
pub async fn select_export_gaussians_async<B: Backend>(
    gaussians: &FlatGaussians<B>,
    options: &GlbExportOptions,
) -> Result<ExportGaussians, GlbExportError> {
    let packed = pack_export_rows_async(gaussians).await?;
    select_export_gaussians_from_packed(packed, options)
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
    let bytes_report = encode_gaussians_to_glb_bytes_timed(gaussians, options)?;

    let write_start = Instant::now();
    std::fs::write(path, bytes_report.bytes.as_slice())?;
    let write_millis = write_start.elapsed().as_secs_f64() * 1000.0;

    Ok(GlbExportReport {
        selected_gaussians: bytes_report.selected_gaussians,
        select_millis: bytes_report.select_millis,
        write_millis,
    })
}

#[cfg(feature = "io")]
#[derive(Debug, Clone)]
pub struct GlbEncodeReport {
    pub selected_gaussians: usize,
    pub select_millis: f64,
    pub bytes: Vec<u8>,
}

#[cfg(feature = "io")]
pub fn encode_gaussians_to_glb_bytes<B: Backend>(
    gaussians: &FlatGaussians<B>,
    options: &GlbExportOptions,
) -> Result<Vec<u8>, GlbExportError> {
    Ok(encode_gaussians_to_glb_bytes_timed(gaussians, options)?.bytes)
}

#[cfg(feature = "io")]
pub fn encode_gaussians_to_glb_bytes_timed<B: Backend>(
    gaussians: &FlatGaussians<B>,
    options: &GlbExportOptions,
) -> Result<GlbEncodeReport, GlbExportError> {
    let select_start = Instant::now();
    let selected = select_export_gaussians(gaussians, options)?;
    let select_millis = select_start.elapsed().as_secs_f64() * 1000.0;
    let selected_gaussians = selected.opacities.len();
    let bytes = encode_export_gaussians_to_glb_bytes(&selected)?;

    Ok(GlbEncodeReport {
        selected_gaussians,
        select_millis,
        bytes,
    })
}

#[derive(Clone, Copy, Debug)]
enum HarmonicsPackMode {
    DcOnly,
    Full,
}

pub fn pack_gaussian_rows_full<B: Backend>(
    gaussians: &FlatGaussians<B>,
) -> Result<PackedGaussianRows, GlbExportError> {
    pack_gaussian_rows(gaussians, HarmonicsPackMode::Full)
}

#[cfg(target_arch = "wasm32")]
pub async fn pack_gaussian_rows_full_async<B: Backend>(
    gaussians: &FlatGaussians<B>,
) -> Result<PackedGaussianRows, GlbExportError> {
    pack_gaussian_rows_async(gaussians, HarmonicsPackMode::Full).await
}

fn pack_export_rows<B: Backend>(gaussians: &FlatGaussians<B>) -> Result<Vec<f32>, GlbExportError> {
    Ok(pack_gaussian_rows(gaussians, HarmonicsPackMode::DcOnly)?.values)
}

fn pack_gaussian_rows<B: Backend>(
    gaussians: &FlatGaussians<B>,
    harmonics_mode: HarmonicsPackMode,
) -> Result<PackedGaussianRows, GlbExportError> {
    let [b, g, _] = gaussians.means.shape().dims::<3>();
    let d_sh = gaussians.harmonics.shape().dims::<4>()[3];
    if d_sh == 0 {
        return Err(GlbExportError::MissingHarmonics);
    }

    // One packed readback to minimize GPU -> host transfers.
    let harmonics = match harmonics_mode {
        HarmonicsPackMode::DcOnly => gaussians
            .harmonics
            .clone()
            .slice([0..b as i32, 0..g as i32, 0..3, 0..1])
            .reshape([b as i32, g as i32, 3]),
        HarmonicsPackMode::Full => {
            gaussians
                .harmonics
                .clone()
                .reshape([b as i32, g as i32, (3 * d_sh) as i32])
        }
    };

    let packed = Tensor::cat(
        vec![
            gaussians.means.clone(),
            harmonics,
            gaussians
                .covariances
                .clone()
                .reshape([b as i32, g as i32, 9]),
            gaussians.scales.clone(),
            gaussians.rotations.clone(),
            gaussians.opacities.clone().unsqueeze_dim(2),
        ],
        2,
    );

    let values = packed
        .into_data()
        .to_vec::<f32>()
        .map_err(|err| GlbExportError::TensorData(format!("{err:?}")))?;

    let harmonics_width = match harmonics_mode {
        HarmonicsPackMode::DcOnly => 3,
        HarmonicsPackMode::Full => 3 * d_sh,
    };

    Ok(PackedGaussianRows {
        rows: b * g,
        d_sh,
        row_width: 3 + harmonics_width + 9 + 3 + 4 + 1,
        values,
    })
}

#[cfg(target_arch = "wasm32")]
async fn pack_export_rows_async<B: Backend>(
    gaussians: &FlatGaussians<B>,
) -> Result<Vec<f32>, GlbExportError> {
    Ok(
        pack_gaussian_rows_async(gaussians, HarmonicsPackMode::DcOnly)
            .await?
            .values,
    )
}

#[cfg(target_arch = "wasm32")]
async fn pack_gaussian_rows_async<B: Backend>(
    gaussians: &FlatGaussians<B>,
    harmonics_mode: HarmonicsPackMode,
) -> Result<PackedGaussianRows, GlbExportError> {
    let [b, g, _] = gaussians.means.shape().dims::<3>();
    let d_sh = gaussians.harmonics.shape().dims::<4>()[3];
    if d_sh == 0 {
        return Err(GlbExportError::MissingHarmonics);
    }

    // One packed readback to minimize GPU -> host transfers.
    let harmonics = match harmonics_mode {
        HarmonicsPackMode::DcOnly => gaussians
            .harmonics
            .clone()
            .slice([0..b as i32, 0..g as i32, 0..3, 0..1])
            .reshape([b as i32, g as i32, 3]),
        HarmonicsPackMode::Full => {
            gaussians
                .harmonics
                .clone()
                .reshape([b as i32, g as i32, (3 * d_sh) as i32])
        }
    };

    let packed = Tensor::cat(
        vec![
            gaussians.means.clone(),
            harmonics,
            gaussians
                .covariances
                .clone()
                .reshape([b as i32, g as i32, 9]),
            gaussians.scales.clone(),
            gaussians.rotations.clone(),
            gaussians.opacities.clone().unsqueeze_dim(2),
        ],
        2,
    );

    let values = packed
        .into_data_async()
        .await
        .to_vec::<f32>()
        .map_err(|err| GlbExportError::TensorData(format!("{err:?}")))?;

    let harmonics_width = match harmonics_mode {
        HarmonicsPackMode::DcOnly => 3,
        HarmonicsPackMode::Full => 3 * d_sh,
    };

    Ok(PackedGaussianRows {
        rows: b * g,
        d_sh,
        row_width: 3 + harmonics_width + 9 + 3 + 4 + 1,
        values,
    })
}

fn select_export_gaussians_from_packed(
    packed: Vec<f32>,
    options: &GlbExportOptions,
) -> Result<ExportGaussians, GlbExportError> {
    let row_width = 23usize;
    let rows = packed.len() / row_width;

    let mut indices: Vec<usize> = (0..rows)
        .filter(|&idx| packed[idx * row_width + 22] >= options.opacity_threshold)
        .collect();

    if matches!(options.sort_mode, GlbSortMode::Opacity) {
        indices.sort_by(|lhs, rhs| {
            packed[rhs * row_width + 22]
                .partial_cmp(&packed[lhs * row_width + 22])
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
        let transform = canonicalize_gaussian_transform_cv(
            [packed[off], packed[off + 1], packed[off + 2]],
            [
                packed[off + 6],
                packed[off + 7],
                packed[off + 8],
                packed[off + 9],
                packed[off + 10],
                packed[off + 11],
                packed[off + 12],
                packed[off + 13],
                packed[off + 14],
            ],
            [packed[off + 15], packed[off + 16], packed[off + 17]],
            [
                packed[off + 18],
                packed[off + 19],
                packed[off + 20],
                packed[off + 21],
            ],
            packed[off + 22],
        );

        out.positions
            .extend_from_slice(transform.position.as_slice());
        out.colors.extend_from_slice(
            [
                packed[off + 3].clamp(0.0, 1.0),
                packed[off + 4].clamp(0.0, 1.0),
                packed[off + 5].clamp(0.0, 1.0),
            ]
            .as_slice(),
        );
        out.scales.extend_from_slice(transform.scale.as_slice());
        out.rotations
            .extend_from_slice(transform.rotation_wxyz.as_slice());
        out.opacities.push(transform.opacity);
    }

    Ok(out)
}

pub fn canonicalize_gaussian_transform_cv(
    mean_xyz: [f32; 3],
    covariance_row_major_cv: [f32; 9],
    fallback_scale_xyz: [f32; 3],
    fallback_rotation_xyzw: [f32; 4],
    opacity: f32,
) -> CanonicalGaussianTransform {
    let position = [mean_xyz[0], -mean_xyz[1], -mean_xyz[2]];
    let fallback_rotation_wxyz = cv_xyzw_to_canonical_wxyz(fallback_rotation_xyzw);
    let fallback_scale = sanitize_scale_for_viewer(fallback_scale_xyz);
    let (rotation_wxyz, scale) = rotation_scale_from_covariance_cv(covariance_row_major_cv)
        .unwrap_or((fallback_rotation_wxyz, fallback_scale));

    CanonicalGaussianTransform {
        position,
        rotation_wxyz,
        scale,
        opacity: opacity.clamp(0.0, 1.0),
    }
}

pub fn cv_xyzw_to_canonical_wxyz(q_xyzw: [f32; 4]) -> [f32; 4] {
    let x = q_xyzw[0];
    let y = q_xyzw[1];
    let z = q_xyzw[2];
    let w = q_xyzw[3];
    let norm = (x * x + y * y + z * z + w * w).sqrt();
    let (x, y, z, w) = if norm.is_finite() && norm > f32::EPSILON {
        (x / norm, y / norm, z / norm, w / norm)
    } else {
        (0.0, 0.0, 0.0, 1.0)
    };

    let q = UnitQuaternion::from_quaternion(Quaternion::new(w, x, y, z));
    let basis = Matrix3::new(
        1.0, 0.0, 0.0, //
        0.0, -1.0, 0.0, //
        0.0, 0.0, -1.0,
    );
    let r_canonical = basis * q.to_rotation_matrix().into_inner() * basis;
    let q_canonical = UnitQuaternion::from_matrix(&r_canonical);
    let q_inner = q_canonical.quaternion();
    [q_inner.w, q_inner.i, q_inner.j, q_inner.k]
}

pub fn sanitize_scale_for_viewer(scale_xyz: [f32; 3]) -> [f32; 3] {
    let mut scale = [0.0f32; 3];
    for (dst, src) in scale.iter_mut().zip(scale_xyz.iter()) {
        let value = if src.is_finite() {
            *src
        } else {
            VIEWER_MIN_SCALE
        };
        *dst = value.clamp(VIEWER_MIN_SCALE, VIEWER_MAX_SCALE);
    }

    let min_scale = scale
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min)
        .max(VIEWER_MIN_SCALE);
    let max_allowed = (min_scale * VIEWER_MAX_SCALE_RATIO).min(VIEWER_MAX_SCALE);
    for component in &mut scale {
        *component = component.min(max_allowed);
    }

    scale
}

pub fn rotation_scale_from_covariance_cv(
    covariance_row_major_cv: [f32; 9],
) -> Option<([f32; 4], [f32; 3])> {
    let cov_cv = mat3_from_row_major(covariance_row_major_cv);
    if !matrix3_is_finite(&cov_cv) {
        return None;
    }
    let basis = Matrix3::new(
        1.0, 0.0, 0.0, //
        0.0, -1.0, 0.0, //
        0.0, 0.0, -1.0,
    );
    let cov_canonical = basis * cov_cv * basis;
    rotation_scale_from_covariance_canonical(cov_canonical)
}

fn rotation_scale_from_covariance_canonical(
    covariance: Matrix3<f32>,
) -> Option<([f32; 4], [f32; 3])> {
    if !matrix3_is_finite(&covariance) {
        return None;
    }

    let rows = mat3_to_rows(covariance);
    let cov = Matrix3::new(
        rows[0][0], rows[0][1], rows[0][2], //
        rows[1][0], rows[1][1], rows[1][2], //
        rows[2][0], rows[2][1], rows[2][2],
    );
    let eigen = SymmetricEigen::new(cov);
    let mut order = [0usize, 1usize, 2usize];
    order.sort_by(|lhs, rhs| {
        eigen.eigenvalues[*rhs]
            .partial_cmp(&eigen.eigenvalues[*lhs])
            .unwrap_or(Ordering::Equal)
    });

    let mut columns = order.map(|index| {
        let axis = Vector3::new(
            eigen.eigenvectors[(0, index)],
            eigen.eigenvectors[(1, index)],
            eigen.eigenvectors[(2, index)],
        );
        let norm = axis.norm();
        if norm.is_finite() && norm > f32::EPSILON {
            axis / norm
        } else {
            match index {
                0 => Vector3::new(1.0, 0.0, 0.0),
                1 => Vector3::new(0.0, 1.0, 0.0),
                _ => Vector3::new(0.0, 0.0, 1.0),
            }
        }
    });

    let mut eig = order.map(|index| eigen.eigenvalues[index].max(0.0));
    let mut canonical_from_axes = Matrix3::from_columns(&[columns[0], columns[1], columns[2]]);
    if canonical_from_axes.determinant() < 0.0 {
        columns[2] = -columns[2];
        canonical_from_axes = Matrix3::from_columns(&[columns[0], columns[1], columns[2]]);
    }

    let q = UnitQuaternion::from_matrix(&canonical_from_axes);
    let q_inner = q.quaternion();

    eig[0] = eig[0].max(0.0);
    eig[1] = eig[1].max(0.0);
    eig[2] = eig[2].max(0.0);
    let scale = sanitize_scale_for_viewer([eig[0].sqrt(), eig[1].sqrt(), eig[2].sqrt()]);
    Some(([q_inner.w, q_inner.i, q_inner.j, q_inner.k], scale))
}

fn mat3_from_row_major(values: [f32; 9]) -> Matrix3<f32> {
    Matrix3::new(
        values[0], values[1], values[2], //
        values[3], values[4], values[5], //
        values[6], values[7], values[8],
    )
}

fn mat3_to_rows(matrix: Matrix3<f32>) -> [[f32; 3]; 3] {
    [
        [matrix[(0, 0)], matrix[(0, 1)], matrix[(0, 2)]],
        [matrix[(1, 0)], matrix[(1, 1)], matrix[(1, 2)]],
        [matrix[(2, 0)], matrix[(2, 1)], matrix[(2, 2)]],
    ]
}

fn matrix3_is_finite(matrix: &Matrix3<f32>) -> bool {
    matrix.iter().all(|value| value.is_finite())
}

#[cfg(feature = "io")]
#[derive(Debug, Clone, Copy)]
struct BufferSlice {
    byte_offset: usize,
    byte_length: usize,
}

#[cfg(feature = "io")]
pub fn encode_export_gaussians_to_glb_bytes(
    gaussians: &ExportGaussians,
) -> Result<Vec<u8>, GlbExportError> {
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

    let total_len = 12 + 8 + json_bytes.len() + 8 + bin.len();
    let mut out = Vec::with_capacity(total_len);
    out.extend_from_slice(&0x4654_6C67u32.to_le_bytes());
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total_len as u32).to_le_bytes());
    out.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&0x4E4F_534Au32.to_le_bytes());
    out.extend_from_slice(json_bytes.as_slice());
    out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    out.extend_from_slice(&0x004E_4942u32.to_le_bytes());
    out.extend_from_slice(bin.as_slice());
    Ok(out)
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::prelude::Backend;

    #[test]
    fn canonicalize_gaussian_transform_converts_cv_basis() {
        let transform = canonicalize_gaussian_transform_cv(
            [1.0, 2.0, 3.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [0.01, 0.02, 0.03],
            [0.0, 0.0, 0.0, 1.0],
            1.2,
        );

        assert_eq!(transform.position, [1.0, -2.0, -3.0]);
        assert_eq!(transform.rotation_wxyz, [1.0, 0.0, 0.0, 0.0]);
        assert!(transform.opacity <= 1.0);
    }

    #[test]
    fn covariance_decomposition_identity_is_stable() {
        let cov = [1.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 9.0];
        let Some((rotation_wxyz, scale)) = rotation_scale_from_covariance_cv(cov) else {
            panic!("expected decomposition");
        };

        assert!(rotation_wxyz.iter().all(|value| value.is_finite()));
        let q_norm = rotation_wxyz
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        assert!((q_norm - 1.0).abs() < 1e-4);
        assert!((scale[0] - 0.3).abs() < 1e-6);
        assert!((scale[1] - 0.3).abs() < 1e-6);
        assert!((scale[2] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn sanitize_scale_enforces_limits() {
        let scale = sanitize_scale_for_viewer([0.0, 0.3, 1e-6]);
        assert!(scale.iter().all(|value| value.is_finite()));
        assert!(scale.iter().all(|value| *value >= VIEWER_MIN_SCALE));
        assert!(scale.iter().all(|value| *value <= VIEWER_MAX_SCALE));
    }

    #[test]
    fn full_packed_rows_include_all_harmonics_channels() {
        type TestBackend = NdArray<f32>;
        let device = <TestBackend as Backend>::Device::default();

        let means = Tensor::<TestBackend, 1>::from_floats([1.0f32, 2.0, 3.0].as_slice(), &device)
            .reshape([1, 1, 3]);
        let harmonics = Tensor::<TestBackend, 1>::from_floats(
            [
                0.0f32, 1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0, 7.0, //
                8.0, 9.0, 10.0, 11.0,
            ]
            .as_slice(),
            &device,
        )
        .reshape([1, 1, 3, 4]);
        let covariances = Tensor::<TestBackend, 1>::from_floats(
            [
                1.0f32, 0.0, 0.0, //
                0.0, 2.0, 0.0, //
                0.0, 0.0, 3.0,
            ]
            .as_slice(),
            &device,
        )
        .reshape([1, 1, 3, 3]);
        let scales = Tensor::<TestBackend, 1>::from_floats([0.1f32, 0.2, 0.3].as_slice(), &device)
            .reshape([1, 1, 3]);
        let rotations =
            Tensor::<TestBackend, 1>::from_floats([0.0f32, 0.0, 0.0, 1.0].as_slice(), &device)
                .reshape([1, 1, 4]);
        let opacities =
            Tensor::<TestBackend, 1>::from_floats([0.8f32].as_slice(), &device).reshape([1, 1]);

        let gaussians = FlatGaussians {
            means,
            covariances,
            harmonics,
            opacities,
            rotations,
            scales,
        };

        let packed = pack_gaussian_rows_full(&gaussians).expect("full packed rows");
        assert_eq!(packed.rows, 1);
        assert_eq!(packed.d_sh, 4);
        assert_eq!(packed.row_width, 32);
        assert_eq!(packed.values.len(), 32);
        assert_eq!(&packed.values[0..3], [1.0, 2.0, 3.0].as_slice());
        assert_eq!(
            &packed.values[3..15],
            [
                0.0f32, 1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0, 7.0, //
                8.0, 9.0, 10.0, 11.0,
            ]
            .as_slice()
        );
        assert_eq!(&packed.values[31..32], [0.8].as_slice());
    }
}
